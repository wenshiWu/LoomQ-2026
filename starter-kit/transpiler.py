#!/usr/bin/env python3
"""Transpilation from OpenQASM 2.0 to the three target IRs.

The ``transpile()`` output is parsed and simulated by the organizer's evaluator,
so it must be a complete, semantically faithful program in the target IR
(see ``target_ir_contract.md``):

* ``spinq``   -> OpenQASM 2.0 (qelib1 supports all 12 whitelisted gates natively)
* ``originq`` -> OriginIR text (QINIT / CREG / gate lines / MEASURE)
* ``braket``  -> OpenQASM 3.0 (stdgates.inc; ``cu1`` -> ``cp``, ``swap`` -> 3x cx,
                 since ``cu1``/``swap`` are not part of the OQ3 standard library)
"""

from __future__ import annotations

from typing import List

from quantum_engine import Circuit, GateOp, MeasureOp

_PI_SYMBOL = "pi"


def _fmt_angle(value: float) -> str:
    """Format an angle; use a compact decimal to stay parser-friendly."""
    return repr(round(value, 12))


def _fmt_operand(q: int) -> str:
    return "q[%d]" % q


# ---------------------------------------------------------------------------
# Target: spinq (OpenQASM 2.0)
# ---------------------------------------------------------------------------


def transpile_spinq(circ: Circuit) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    lines.append("qreg q[%d];" % circ.num_qubits)
    if circ.num_cbits:
        lines.append("creg c[%d];" % circ.num_cbits)
    for gate in circ.gates:
        lines.append(_spinq_gate(gate))
    for m in circ.measures:
        lines.append("measure q[%d] -> c[%d];" % (m.qubit, m.cbit))
    return "\n".join(lines) + "\n"


def _spinq_gate(gate: GateOp) -> str:
    args = ", ".join(_fmt_operand(q) for q in gate.qubits)
    if gate.params:
        params = ", ".join(_fmt_angle(p) for p in gate.params)
        return "%s(%s) %s;" % (gate.name, params, args)
    return "%s %s;" % (gate.name, args)


# ---------------------------------------------------------------------------
# Target: originq (OriginIR)
# ---------------------------------------------------------------------------

_ORIGINQ_GATE = {
    "h": "H",
    "x": "X",
    "s": "S",
    "sdg": "SDAG",
    "t": "T",
    "tdg": "TDAG",
    "rz": "RZ",
    "ry": "RY",
    "cx": "CNOT",
    "cu1": "CU1",
    "swap": "SWAP",
    "ccx": "TOFFOLI",
}


def transpile_originq(circ: Circuit) -> str:
    lines = ["QINIT %d" % circ.num_qubits]
    if circ.num_cbits:
        lines.append("CREG %d" % circ.num_cbits)
    for gate in circ.gates:
        name = _ORIGINQ_GATE[gate.name]
        args = ", ".join(_fmt_operand(q) for q in gate.qubits)
        if gate.params:
            params = ", ".join(_fmt_angle(p) for p in gate.params)
            lines.append("%s(%s) %s" % (name, params, args))
        else:
            lines.append("%s %s" % (name, args))
    for m in circ.measures:
        lines.append("MEASURE q[%d], c[%d]" % (m.qubit, m.cbit))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Target: braket (OpenQASM 3.0)
# ---------------------------------------------------------------------------


def transpile_braket(circ: Circuit) -> str:
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    lines.append("qubit[%d] q;" % circ.num_qubits)
    if circ.num_cbits:
        lines.append("bit[%d] c;" % circ.num_cbits)
    for gate in circ.gates:
        lines.append(_braket_gate(gate))
    for m in circ.measures:
        lines.append("c[%d] = measure q[%d];" % (m.cbit, m.qubit))
    return "\n".join(lines) + "\n"


def _braket_gate(gate: GateOp) -> str:
    args = ", ".join(_fmt_operand(q) for q in gate.qubits)
    name = gate.name
    # cu1 == controlled-phase == cp in OpenQASM 3 standard library.
    if gate.name == "cu1":
        name = "cp"
    elif gate.name == "swap":
        # swap is not part of stdgates.inc; decompose into 3 cnots.
        a, b = gate.qubits
        return "cnot %s, %s;\ncnot %s, %s;\ncnot %s, %s;" % (
            _fmt_operand(a), _fmt_operand(b),
            _fmt_operand(b), _fmt_operand(a),
            _fmt_operand(a), _fmt_operand(b),
        )
    if gate.params:
        params = ", ".join(_fmt_angle(p) for p in gate.params)
        return "%s(%s) %s;" % (name, params, args)
    return "%s %s;" % (name, args)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def transpile(qasm_text: str, target: str) -> str:
    """Translate OpenQASM 2.0 into the requested target's native representation."""
    circ = _parse(qasm_text)
    if target == "spinq":
        return transpile_spinq(circ)
    if target == "originq":
        return transpile_originq(circ)
    if target == "braket":
        return transpile_braket(circ)
    raise ValueError("unknown target: %s (expected one of spinq/originq/braket)" % target)


def _parse(qasm_text: str) -> Circuit:
    from quantum_engine import parse_qasm

    return parse_qasm(qasm_text)
