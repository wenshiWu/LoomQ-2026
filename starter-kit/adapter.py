#!/usr/bin/env python3
"""LoomQ submission adapter.

Implements the competition contract (v1.0) on top of the dependency-free
``quantum_engine`` and ``transpiler`` modules:

* ``transpile`` - OpenQASM 2.0 -> target native IR (spinq/originq/braket)
* ``run``       - execute a circuit on our universal state-vector simulator and
                  return the unified result schema (bit_order: little)
* ``agent_chat``- optional L2 entry point (DeepSeek via LOOMQ_LLM_* env)
* ``compile_hybrid`` - optional L3 entry point
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from quantum_engine import ParseError, parse_qasm, simulate
from transpiler import transpile as _transpile


SUPPORTED_TARGETS = ("spinq", "originq", "braket")

_META_BACKEND = {
    "spinq": "spinq_taurus_simulator",
    "originq": "originq_local_simulator",
    "braket": "braket_local_simulator",
}


def transpile(qasm_str: str, target: str) -> str:
    """Translate OpenQASM 2.0 into the target backend's native representation."""
    return _transpile(qasm_str, target)


def run(qasm_str: str, target: str, shots: int) -> Dict[str, Any]:
    """Execute a circuit and return the unified result schema from the rules.

    Execution uses the built-in state-vector simulator so the adapter has no
    third-party runtime dependency and behaves identically everywhere.
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError("unknown target: %s" % target)
    if shots <= 0:
        raise ValueError("shots must be positive")

    circ = parse_qasm(qasm_str)
    counts = simulate(qasm_str, shots)

    return {
        "backend": _META_BACKEND[target],
        "job_id": _job_id(qasm_str, target, shots),
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": {
            "transpiled_gates": len(circ.gates),
            "qubits": circ.num_qubits,
        },
    }


def _job_id(qasm_str: str, target: str, shots: int) -> str:
    import hashlib

    digest = hashlib.sha256(
        ("%s|%s|%d" % (qasm_str.strip(), target, shots)).encode("utf-8")
    ).hexdigest()[:16]
    return "loomq-%s-%s" % (target, digest)


# ---------------------------------------------------------------------------
# L2: agent_chat
# ---------------------------------------------------------------------------


def agent_chat(prompt: str) -> str:
    """L2 entry point.

    Reads ``LOOMQ_LLM_BASE_URL`` / ``LOOMQ_LLM_API_KEY`` / ``LOOMQ_LLM_MODEL``
    from the environment and returns the model's response text.
    """
    from l2_agent import agent_chat as _agent_chat

    return _agent_chat(prompt)


# ---------------------------------------------------------------------------
# L3: compile_hybrid
# ---------------------------------------------------------------------------


def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:
    """Optional L3 entry point. Not implemented in this scaffold."""
    raise NotImplementedError("L3 is optional; implement compile_hybrid to enter")
