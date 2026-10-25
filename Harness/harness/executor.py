"""M7 - Tool executor. Execute only what passes every check.

1 look up the tool (unknown name -> reject, list valid tools)
2 validate arguments against the schema
3 check permission for the current context
4 run with a timeout, catch exceptions
5 return ok/err as an observation and log it
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import Any

from pydantic import ValidationError

from .tools import TOOLS

_pool = ThreadPoolExecutor(max_workers=4)


def ok(result: Any) -> dict:
    return {"ok": True, "result": result}


def err(message: str) -> dict:
    return {"ok": False, "error": message}


def run_with_timeout(fn, args, timeout: float):
    fut = _pool.submit(fn, args)
    try:
        return fut.result(timeout=timeout)
    except FutTimeout:
        raise TimeoutError(f"tool timed out after {timeout}s")


def execute(call: dict, *, allowed: set[str] = frozenset({"read"}), logger=None,
            run_id: str | None = None) -> dict:
    """Execute one tool call proposed by the model. Never raises."""
    t0 = time.perf_counter()
    fn = call.get("function", {})
    name = fn.get("name", "")
    raw = fn.get("arguments", {})
    out = _execute(name, raw, allowed)
    if logger is not None:
        logger.tool_call(run_id=run_id, name=name, arguments=raw, ok=out["ok"],
                         error=out.get("error"), duration_s=time.perf_counter() - t0)
    return out


def _execute(name: str, raw: Any, allowed: set[str]) -> dict:
    # 1 look up
    if name not in TOOLS:
        return err(f"Unknown tool {name!r}. Valid tools: {sorted(TOOLS)}")
    spec = TOOLS[name]
    # 2 validate arguments (arguments may arrive as a JSON string)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return err("Arguments are not valid JSON")
    try:
        args = spec.args.model_validate(raw)
    except ValidationError as e:
        details = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return err(f"Bad arguments for {name}: {details}. "
                   f"Expected schema: {json.dumps(spec.args.model_json_schema()['properties'])}")
    # 3 permission
    if spec.permission not in allowed:
        return err(f"Permission denied: {name} needs '{spec.permission}' permission")
    # 4 run
    try:
        return ok(run_with_timeout(spec.fn, args, spec.timeout))
    except Exception as e:  # noqa: BLE001 - every failure becomes an observation
        return err(f"{type(e).__name__}: {e}")


def tool_message(call: dict, result: dict) -> dict:
    """The role: tool message appended to the history after execution."""
    return {"role": "tool", "tool_name": call.get("function", {}).get("name", ""),
            "content": json.dumps(result, ensure_ascii=False, default=str)}
