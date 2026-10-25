"""M14 - Logging. One JSONL line per LLM call and per tool call.

This log is the raw data for the experiment session. Load it with
``pandas.read_json("logs/calls.jsonl", lines=True)``.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def new_run_id() -> str:
    return "r-" + uuid.uuid4().hex[:8]


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec

    def llm_call(self, *, run_id, profile, result, prompt_id=None, valid=None,
                 error=None, retry=0, **extra):
        return self.write(
            "llm_call", run_id=run_id, profile=profile, model=result.model,
            prompt=prompt_id, in_tokens=result.prompt_tokens, out_tokens=result.output_tokens,
            ttft_s=round(result.ttft_s, 3), gen_s=round(result.gen_s, 3),
            wall_s=round(result.wall_s, 3), tool_calls=len(result.tool_calls),
            valid=valid, error=error, retry=retry, **extra)

    def tool_call(self, *, run_id, name, arguments, ok, duration_s, error=None, **extra):
        return self.write("tool_call", run_id=run_id, tool=name, arguments=arguments,
                          ok=ok, error=error, duration_s=round(duration_s, 4), **extra)
