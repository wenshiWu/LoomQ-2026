#!/usr/bin/env python3
"""Self-verification for every gate identity in ``gate_identities.md``.

Method
------
For each identity, build two OpenQASM 2.0 circuits that are identical except
for the gate under test: one uses the original gate, the other uses the
identity-based decomposition. Both are executed on the dependency-free
state-vector simulator (``quantum_engine.py``) and the final state vectors are
compared exactly (allowing a global-phase difference, which never affects
measurement outcomes).

This simultaneously regression-tests the simulator itself: if a gate's matrix
is implemented wrongly, one or more identities will fail.

Usage
-----
    python verify_gate_identities.py

Exit code 0 => every identity holds; 1 => at least one failed.
"""

from __future__ import annotations

import cmath
import math
import sys

from quantum_engine import StateVectorSimulator, parse_qasm

PI = math.pi

# ---------------------------------------------------------------------------
# Simulator helpers
# ---------------------------------------------------------------------------


def run_to_state(qasm_text: str) -> list:
    circ = parse_qasm(qasm_text)
    sim = StateVectorSimulator(circ)
    for gate in circ.gates:
        sim.apply(gate)
    return sim.state


def equal_up_to_phase(a: list, b: list, tol: float = 1e-9) -> float:
    """Return the max absolute difference between two state vectors,
    allowing for an arbitrary global phase.

    Computes ``min_phi || a - e^{i*phi} b ||_inf`` by rotating b to align
    phase at the first non-zero amplitude.
    """
    if len(a) != len(b):
        return float("inf")
    # Find a reference index where a is not ~zero.
    ref = None
    for i, amp in enumerate(a):
        if abs(amp) > 1e-12:
            ref = i
            break
    if ref is None:
        # both should be the all-zero-ish state; compare directly
        return max(abs(x - y) for x, y in zip(a, b))

    phi = cmath.phase(a[ref] / b[ref])
    rotated = [x * cmath.exp(1j * phi) for x in b]
    return max(abs(x - y) for x, y in zip(a, rotated))


def make_circuit(qubit_count: int, body_lines: list) -> str:
    """Wrap a list of gate lines into a complete, parseable OpenQASM 2.0."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    lines.append("qreg q[%d];" % qubit_count)
    lines.append("creg c[%d];" % qubit_count)
    lines.extend(body_lines)
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Identity table
# ---------------------------------------------------------------------------
# Each entry: (name, qubit_count, prep_lines, original_lines, decomposed_lines)
# ``prep_lines`` is executed before the gate under test so that states are
# non-trivial superpositions (catches phase errors, not just bit flips).

IDENTITIES = []

# --- 1. Phase gates are special cases of u1 --------------------------------
for original, decomposed in [
    ("z q[0];", "u1(%r) q[0];" % PI),
    ("s q[0];", "u1(%r) q[0];" % (PI / 2)),
    ("sdg q[0];", "u1(%r) q[0];" % (-PI / 2)),
    ("t q[0];", "u1(%r) q[0];" % (PI / 4)),
    ("tdg q[0];", "u1(%r) q[0];" % (-PI / 4)),
]:
    IDENTITIES.append(
        (
            "phase->u1: %s == %s" % (original.strip(), decomposed.strip()),
            1,
            ["h q[0];"],
            [original],
            [decomposed],
        )
    )

# --- 2. rz == u1 (single-qubit, up to global phase) ------------------------
IDENTITIES.append(
    (
        "rz(theta) == u1(theta) as standalone gate",
        1,
        ["h q[0];"],
        ["rz(0.7) q[0];"],
        ["u1(0.7) q[0];"],
    )
)

# --- 3. swap == 3 cx --------------------------------------------------------
IDENTITIES.append(
    (
        "swap == cx(a,b) cx(b,a) cx(a,b)",
        2,
        ["h q[0];", "h q[1];"],
        ["swap q[0], q[1];"],
        ["cx q[0], q[1];", "cx q[1], q[0];", "cx q[0], q[1];"],
    )
)

# --- 4. cu1 decomposition ---------------------------------------------------
theta = 1.3
IDENTITIES.append(
    (
        "cu1(theta) == 5-gate decomposition",
        2,
        ["h q[0];", "h q[1];"],
        ["cu1(%r) q[0], q[1];" % theta],
        [
            "u1(%r) q[0];" % (theta / 2),
            "cx q[0], q[1];",
            "u1(%r) q[1];" % (-theta / 2),
            "cx q[0], q[1];",
            "u1(%r) q[1];" % (theta / 2),
        ],
    )
)

# --- 5. ccx (Toffoli) qelib1 decomposition ----------------------------------
IDENTITIES.append(
    (
        "ccx == qelib1 15-gate decomposition",
        3,
        ["h q[0];", "h q[1];", "h q[2];"],
        ["ccx q[0], q[1], q[2];"],
        [
            "h q[2];",
            "cx q[1], q[2];",
            "tdg q[2];",
            "cx q[0], q[2];",
            "t q[2];",
            "cx q[1], q[2];",
            "tdg q[2];",
            "cx q[0], q[2];",
            "t q[1];",
            "t q[2];",
            "h q[2];",
            "cx q[0], q[1];",
            "t q[0];",
            "tdg q[1];",
            "cx q[0], q[1];",
        ],
    )
)

# --- 6. ry fallback ---------------------------------------------------------
IDENTITIES.append(
    (
        "ry(theta) == sdg h rz(theta) h s",
        1,
        ["h q[0];"],
        ["ry(0.7) q[0];"],
        ["sdg q[0];", "h q[0];", "rz(0.7) q[0];", "h q[0];", "s q[0];"],
    )
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    print("Verifying %d gate identities ...\n" % len(IDENTITIES))
    for name, nq, prep, original, decomposed in IDENTITIES:
        original_circuit = make_circuit(nq, prep + original)
        decomposed_circuit = make_circuit(nq, prep + decomposed)
        try:
            v1 = run_to_state(original_circuit)
            v2 = run_to_state(decomposed_circuit)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("[FAIL] %s" % name)
            print("       error during simulation: %s: %s" % (type(exc).__name__, exc))
            continue

        diff = equal_up_to_phase(v1, v2)
        ok = diff < 1e-8
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print("[%s] %s" % (status, name))
        print("       max |v_orig - e^{i phi} v_decomp| = %.3e" % diff)
        if not ok:
            print("       original circuit:\n%s" % original_circuit)
            print("       decomposed circuit:\n%s" % decomposed_circuit)

    print("")
    if failures:
        print("%d/%d identities FAILED" % (failures, len(IDENTITIES)))
        return 1
    print("All %d gate identities verified successfully." % len(IDENTITIES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
