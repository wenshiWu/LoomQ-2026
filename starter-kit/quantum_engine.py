#!/usr/bin/env python3
"""LoomQ unified quantum engine: OpenQASM 2.0 parser + state-vector simulator.

This module is the "universal middle layer" of the L1 entry. It parses the
OpenQASM 2.0 subset used by the competition (the 12-gate whitelist plus
register declarations and measurement) into an internal gate IR, then executes
it with a dependency-free state-vector simulator.

Design notes
------------
* Bit ordering follows the competition contract: the binary index of a qubit
  q[k] is the k-th least significant bit of the state index. State index
  ``index`` has qubit q[k] = (index >> k) & 1. This matches Qiskit and the
  `bit_order: "little"` convention in the unified result schema.
* Only gates from the official whitelist are accepted:
    h x s sdg t tdg rz(θ) ry(θ) cx cu1(θ) swap ccx
  plus `measure`. Any other gate raises ParseError, which mirrors the
  guarantee that evaluation circuits never exceed the whitelist.
* Complex math is done with the standard library only (``cmath`` / ``math``),
  so the engine runs anywhere Python 3.10+ runs, with no third-party deps.
"""

from __future__ import annotations

import cmath
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PI = math.pi
SQRT2 = math.sqrt(2.0)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when an OpenQASM 2.0 program cannot be parsed."""


# ---------------------------------------------------------------------------
# Parsed circuit representation
# ---------------------------------------------------------------------------


@dataclass
class GateOp:
    """One gate application, stored in execution order."""

    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)


@dataclass
class MeasureOp:
    """One classical measurement: qubit -> classical bit."""

    qubit: int
    cbit: int


@dataclass
class Circuit:
    """The parsed quantum circuit."""

    num_qubits: int
    num_cbits: int
    gates: List[GateOp] = field(default_factory=list)
    measures: List[MeasureOp] = field(default_factory=list)
    measured_qubits: Optional[List[int]] = None  # None => all qubits measured


# ---------------------------------------------------------------------------
# Tokenizer + parser for the OpenQASM 2.0 subset
# ---------------------------------------------------------------------------

# Simple permissive tokenizer: strips comments, then splits on whitespace,
# parens, and commas. Handles negative numbers and scientific notation.
_TOKEN_RE = re.compile(
    r"""
    -?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?   # number literal
    | [A-Za-z_][A-Za-z0-9_]*           # identifier
    | ->                               # measurement arrow
    | [(),;[\]]                        # punctuation
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"//[^\n]*", "", text)  # strip line comments
    return _TOKEN_RE.findall(text)


def parse_qasm(text: str) -> Circuit:
    """Parse an OpenQASM 2.0 program into a :class:`Circuit`.

    Raises :class:`ParseError` on malformed input or off-whitelist gates.
    """
    tokens = _tokenize(text)

    qregs: Dict[str, int] = {}   # name -> size
    cregs: Dict[str, int] = {}
    qsizes: Dict[str, int] = {}   # name -> size (filled once we know total)
    circ = Circuit(num_qubits=0, num_cbits=0)
    measure_qubits: List[int] = []
    measure_cbits: List[int] = []

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        if tok == "OPENQASM":
            while i < n and tokens[i] != ";":
                i += 1
            i += 1  # consume ';'
            continue
        if tok == "include":
            while i < n and tokens[i] != ";":
                i += 1
            i += 1  # consume ';'
            continue

        if tok == "qreg":
            # qreg name[size];
            if i + 3 >= n or tokens[i + 2] != "[":
                raise ParseError("malformed qreg declaration")
            name = tokens[i + 1]
            size = int(tokens[i + 3])
            qregs[name] = size
            while i < n and tokens[i] != ";":
                i += 1
            i += 1  # consume ';'
            continue

        if tok == "creg":
            # creg name[size];
            if i + 3 >= n or tokens[i + 2] != "[":
                raise ParseError("malformed creg declaration")
            name = tokens[i + 1]
            size = int(tokens[i + 3])
            cregs[name] = size
            while i < n and tokens[i] != ";":
                i += 1
            i += 1  # consume ';'
            continue

        if tok == "measure":
            # measure q;  OR  measure q[k];  OR  measure q -> c[k];  OR  measure q -> c;
            i += 1
            qname, qindex, qconsumed = _read_operand(tokens, i)
            i += qconsumed
            if qindex is None:
                if qname not in qregs:
                    raise ParseError("unknown qreg in measure: %s" % qname)
                mq = list(range(qregs[qname]))
            else:
                mq = [qindex]
            if i < n and tokens[i] == "->":
                i += 1
                cname, cindex, cconsumed = _read_operand(tokens, i)
                i += cconsumed
                if cindex is None:
                    if cname not in cregs:
                        raise ParseError("unknown creg in measure: %s" % cname)
                    mc = list(range(cregs[cname]))
                else:
                    mc = [cindex]
            else:
                # measure q;  with no classical target: write into implicit bits
                mc = list(range(len(mq)))
            if len(mq) != len(mc):
                raise ParseError("measure register width mismatch")
            for q, c in zip(mq, mc):
                measure_qubits.append(q)
                measure_cbits.append(c)
            # consume trailing ';'
            while i < n and tokens[i] != ";":
                i += 1
            i += 1
            continue

        # A gate application.
        gate_name = tok
        i += 1
        params: List[float] = []
        if i < n and tokens[i] == "(":
            i += 1
            while i < n and tokens[i] != ")":
                if tokens[i] != ",":
                    params.append(float(tokens[i]))
                i += 1
            i += 1  # consume ')'
        # operand list
        qubits: List[int] = []
        while i < n and tokens[i] != ";":
            if tokens[i] == ",":
                i += 1
                continue
            qname, qindex, consumed = _read_operand(tokens, i)
            i += consumed
            if qname not in qregs:
                raise ParseError("unknown qreg %s" % qname)
            if qindex is None:
                raise ParseError("gate operand must be an indexed qubit: %s" % qname)
            qubits.append(qindex)
        if i < n and tokens[i] == ";":
            i += 1

        if gate_name not in _GATE_ARITY:
            raise ParseError("gate not on whitelist: %s" % gate_name)
        expected = _GATE_ARITY[gate_name]
        if len(qubits) != expected:
            raise ParseError(
                "gate %s expects %d qubits, got %d"
                % (gate_name, expected, len(qubits))
            )
        circ.gates.append(GateOp(name=gate_name, qubits=qubits, params=params))

    circ.num_qubits = sum(qregs.values())
    circ.num_cbits = sum(cregs.values())
    circ.measures = [
        MeasureOp(q, c) for q, c in zip(measure_qubits, measure_cbits)
    ]
    if measure_qubits:
        circ.measured_qubits = measure_qubits
    return circ


def _read_operand(tokens: List[str], i: int) -> Tuple[str, Optional[int], int]:
    """Read a ``name`` or ``name[k]`` operand at index i.

    Returns ``(name, index_or_None, tokens_consumed)``.
    """
    if i >= len(tokens):
        raise ParseError("unexpected end of operands")
    name = tokens[i]
    index = None
    consumed = 1
    if i + 2 < len(tokens) and tokens[i + 1] == "[":
        index = int(tokens[i + 2])
        consumed = 4  # name [ k ] (']' and any following ',' handled by caller)
    return name, index, consumed


_GATE_ARITY = {
    "h": 1,
    "x": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "rz": 1,
    "ry": 1,
    "cx": 2,
    "cu1": 2,
    "swap": 2,
    "ccx": 3,
}


# ---------------------------------------------------------------------------
# State-vector simulator
# ---------------------------------------------------------------------------


def _phase(theta: float) -> complex:
    return cmath.exp(1j * theta)


class StateVectorSimulator:
    """Dependency-free state-vector simulator for the whitelisted gate set."""

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.dim = 1 << circuit.num_qubits
        self.state = [0.0 + 0.0j] * self.dim
        self.state[0] = 1.0 + 0.0j

    def _apply_1q(self, gate: GateOp) -> None:
        q = gate.qubits[0]
        bit = 1 << q
        if gate.name == "h":
            f = 1.0 / SQRT2
            for base in range(self.dim):
                if base & bit:
                    continue
                j = base | bit
                a, b = self.state[base], self.state[j]
                self.state[base] = f * (a + b)
                self.state[j] = f * (a - b)
        elif gate.name == "x":
            for base in range(self.dim):
                if base & bit:
                    continue
                j = base | bit
                self.state[base], self.state[j] = self.state[j], self.state[base]
        elif gate.name in ("s", "sdg"):
            phase = 1j if gate.name == "s" else -1j
            for base in range(self.dim):
                if base & bit:
                    self.state[base] *= phase
        elif gate.name in ("t", "tdg"):
            phase = cmath.exp(1j * PI / 4.0) if gate.name == "t" else cmath.exp(-1j * PI / 4.0)
            for base in range(self.dim):
                if base & bit:
                    self.state[base] *= phase
        elif gate.name == "rz":
            theta = gate.params[0]
            phase = _phase(-theta / 2.0)
            for base in range(self.dim):
                if base & bit:
                    self.state[base] *= phase
                else:
                    self.state[base] *= cmath.exp(1j * theta / 2.0)
        elif gate.name == "ry":
            theta = gate.params[0]
            c = math.cos(theta / 2.0)
            s = math.sin(theta / 2.0)
            for base in range(self.dim):
                if base & bit:
                    continue
                j = base | bit
                a, b = self.state[base], self.state[j]
                self.state[base] = c * a - s * b
                self.state[j] = s * a + c * b
        else:
            raise ParseError("unsupported single-qubit gate: %s" % gate.name)

    def _apply_2q(self, gate: GateOp) -> None:
        q0, q1 = gate.qubits
        b0, b1 = 1 << q0, 1 << q1
        if gate.name == "cx":
            # control = q0, target = q1: flip target iff control is |1>.
            for base in range(self.dim):
                if (base & b0) == 0:
                    continue
                j = base ^ b1  # flip the target bit
                if j > base:
                    self.state[base], self.state[j] = self.state[j], self.state[base]
        elif gate.name == "cu1":
            theta = gate.params[0]
            phase = _phase(theta)
            for base in range(self.dim):
                if (base & b0) and (base & b1):
                    self.state[base] *= phase
        elif gate.name == "swap":
            for base in range(self.dim):
                j = base
                if base & b0:
                    j &= ~b0
                else:
                    j |= b0
                if base & b1:
                    j &= ~b1
                else:
                    j |= b1
                if j > base:
                    self.state[base], self.state[j] = self.state[j], self.state[base]
        else:
            raise ParseError("unsupported two-qubit gate: %s" % gate.name)

    def _apply_3q(self, gate: GateOp) -> None:
        q0, q1, q2 = gate.qubits
        if gate.name != "ccx":
            raise ParseError("unsupported three-qubit gate: %s" % gate.name)
        b0, b1, b2 = 1 << q0, 1 << q1, 1 << q2
        for base in range(self.dim):
            if (base & b0) and (base & b1):
                j = base ^ b2
                if j > base:
                    self.state[base], self.state[j] = self.state[j], self.state[base]

    def apply(self, gate: GateOp) -> None:
        arity = _GATE_ARITY[gate.name]
        if arity == 1:
            self._apply_1q(gate)
        elif arity == 2:
            self._apply_2q(gate)
        else:
            self._apply_3q(gate)

    def probabilities(self) -> List[float]:
        return [abs(x) ** 2 for x in self.state]

    def sample(self, shots: int, rng: Optional[random.Random] = None) -> Dict[str, int]:
        """Sample measurement outcomes, returning counts keyed by little-endian
        bitstring over the measured qubits.

        If no measurement instructions exist, every qubit is measured.
        """
        probs = self.probabilities()
        if shots <= 0:
            raise ValueError("shots must be positive")
        rng = rng or random.Random()
        measured = self.circuit.measured_qubits
        counts: Dict[str, int] = {}
        for _ in range(shots):
            r = rng.random()
            acc = 0.0
            idx = 0
            for idx, p in enumerate(probs):
                acc += p
                if r <= acc:
                    break
            key = self._format_index(idx, measured)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _format_index(self, index: int, measured: Optional[List[int]]) -> str:
        if measured is None:
            measured = list(range(self.circuit.num_qubits))
        # bit_order little: leftmost char is the highest measured qubit index,
        # so we build a bitstring with qubit k at position k.
        width = max(measured) + 1 if measured else 0
        bits = ["0"] * width
        for k in measured:
            bits[k] = "1" if (index >> k) & 1 else "0"
        # Only keep measured qubits, ordered high -> low (little-endian string).
        kept = [bits[k] for k in sorted(measured, reverse=True)]
        return "".join(kept)


def simulate(qasm_text: str, shots: int, seed: Optional[int] = None) -> Dict[str, int]:
    """Parse a QASM 2.0 program and sample ``shots`` outcomes.

    Convenience wrapper used by :func:`run` in ``adapter.py``.
    """
    circ = parse_qasm(qasm_text)
    sim = StateVectorSimulator(circ)
    for gate in circ.gates:
        sim.apply(gate)
    rng = random.Random(seed)
    return sim.sample(shots, rng=rng)
