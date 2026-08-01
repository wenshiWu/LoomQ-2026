#!/usr/bin/env python3
"""L2 agent: turn natural language into verified quantum workflows.

Strategy
--------
1. Ask the LLM to answer with a strict JSON object describing the task type and
   its payload (this makes parsing robust regardless of wording).
2. For QASM-generation and code-correction tasks, verify the produced program
   with our own dependency-free simulator (parses, uses only the 12-gate
   whitelist, contains a measurement). If verification fails, feed the exact
   error back to the LLM and retry, up to a bounded number of attempts.
3. For backend-selection tasks, hand the official backend_capabilities.json to
   the LLM so it reasons over real constraints and must output the canonical
   backend id.

The final answer text is reformatted by us, so the response the evaluator sees
contains a clean OpenQASM 2.0 program (for QASM tasks) or the canonical backend
identifier (for selection tasks).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from quantum_engine import ParseError, parse_qasm

MAX_ATTEMPTS = 3

_SYSTEM_PROMPT = """你是一个量子计算助手，帮助完全没有量子背景的用户。

你接收三类任务，必须用如下严格 JSON 回复（不要输出 JSON 之外的任何文字）：

## 任务 1：生成量子电路
用户用自然语言描述想要制备的量子态。你需要输出正确的 OpenQASM 2.0 代码。
规则：
- 只能使用这 12 个门：h, x, s, sdg, t, tdg, rz(θ), ry(θ), cx, cu1(θ), swap, ccx
- 必须包含 qreg/creg 声明，以及 measure 语句
- 测量目标写成 `measure q -> c;`
- 回复格式：{"type": "qasm", "qasm": "OPENQASM 2.0;\\n..."}
  其中 qasm 字段是完整、可运行的 OpenQASM 2.0 程序（用 \\n 表示换行，不要用 ``` 代码块）。

## 任务 2：修正量子电路代码
用户给出一段有语法或语义错误的 OpenQASM 2.0 代码，并明确说明他想要制备的目标态。
你需要在保持其意图的前提下修复代码。规则同任务 1。
回复格式：{"type": "qasm", "qasm": "修复后的完整 OpenQASM 2.0 程序"}

## 任务 3：量子后端选型
用户给出电路比特数和约束（排队、费用等）。你根据以下官方后端能力表选择最合适的后端，
并只返回规范标识（id 字段）：
%s
规则：回复格式 {"type": "backend", "backend": "规范 id 或 'none'"}。
如果没有任何后端满足约束，backend 填 "none"，并在 "reason" 中说明。
必须使用表里的规范 id（如 braket_local_simulator），不要自创名字。

## 其它问题
如果是简单咨询，回复 {"type": "text", "text": "简短回答"}。
"""


def _load_backend_capabilities() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "backend_capabilities.json")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _call_llm(messages: List[Dict[str, str]]) -> str:
    from llm_client import chat_completion

    reply = chat_completion(messages)
    content = reply["choices"][0]["message"]["content"]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        content = "".join(parts)
    return content


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced {...} object out of the model's reply."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_qasm(qasm_str: str) -> Optional[str]:
    """Return None if the program is valid, else an error message."""
    if not isinstance(qasm_str, str) or not qasm_str.strip():
        return "QASM 内容为空"
    try:
        circ = parse_qasm(qasm_str)
    except ParseError as exc:
        return "QASM 解析失败：%s" % exc
    if circ.num_qubits <= 0:
        return "缺少 qreg 声明"
    if not circ.measures:
        return "缺少 measure 测量语句"
    return None


def _run_agent(prompt: str) -> Dict[str, Any]:
    system = _SYSTEM_PROMPT % _load_backend_capabilities()
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_ATTEMPTS):
        raw = _call_llm(messages)
        parsed = _extract_json(raw)
        if parsed is None:
            # Fall back: tell the model it must emit JSON and retry.
            messages.append({"role": "user", "content": "你刚才的回复不是有效的 JSON。请严格按格式输出 JSON，不要包含其他文字。"})
            continue

        task_type = parsed.get("type")
        if task_type == "qasm":
            qasm_text = parsed.get("qasm", "")
            err = _validate_qasm(qasm_text)
            if err is None:
                return {"type": "qasm", "qasm": qasm_text, "attempts": attempt + 1}
            if attempt < MAX_ATTEMPTS - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": "你生成的 QASM 自检未通过：%s。请修复后重新以 JSON 输出（type=qasm, qasm=完整程序）。"
                        % err,
                    }
                )
                continue
            return {"type": "qasm", "qasm": qasm_text, "error": err, "attempts": attempt + 1}

        if task_type == "backend":
            return {
                "type": "backend",
                "backend": parsed.get("backend", "none"),
                "reason": parsed.get("reason", ""),
                "attempts": attempt + 1,
            }

        if task_type == "text":
            return {"type": "text", "text": parsed.get("text", "")}

        messages.append(
            {"role": "user", "content": "type 字段必须为 qasm / backend / text 之一，请重新以 JSON 输出。"}
        )

    return {"type": "text", "text": "抱歉，我暂时无法完成这个请求。"}


def agent_chat(prompt: str) -> str:
    """L2 entry point: return the agent's answer text for the given prompt."""
    result = _run_agent(prompt)
    if result["type"] == "qasm":
        qasm_text = result["qasm"].strip()
        # Include a short label then the program; evaluator extracts the QASM.
        return "以下是修复/生成后的 OpenQASM 2.0 程序：\n\n%s\n" % qasm_text
    if result["type"] == "backend":
        backend = result.get("backend", "none")
        if backend == "none":
            return "没有后端满足所有约束。%s" % result.get("reason", "")
        return "推荐后端：%s" % backend
    return result.get("text", "")


def extract_qasm_from_response(text: str) -> Optional[str]:
    """Extract an OpenQASM 2.0 program from an agent response (for local tests)."""
    match = re.search(r"OPENQASM\s+2\.0;.*?(?=^\s*```|\Z)", text, re.DOTALL | re.MULTILINE)
    return match.group(0).strip() if match else None
