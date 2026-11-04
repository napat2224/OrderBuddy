"""Logging: one readable log for people, next to the JSONL data the harness keeps.

Harness/logs/app.log       one line per step, watch it with: tail -f Harness/logs/app.log
Harness/logs/calls.jsonl   the same events in full, for scripts and the eval

Every event goes through JsonlLogger.write, so overriding that one method covers routes,
states, model calls and tools without touching the harness files. TracedHarness also adds
the model's reply ("llm_trace"), which the harness does not record.
"""
from __future__ import annotations

import json
import logging

from harness import Harness
from harness.logger import JsonlLogger

log = logging.getLogger("orderbuddy")

APP_LOG_NAME = "app.log"
FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_HIDDEN = {"run_id", "trace", "user"}   # trace is printed on its own lines


def short(text: object, limit: int = 400) -> str:
    """One line in app.log; calls.jsonl keeps the full text."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit] + "..."


def setup_file_log(log_dir, *extra_loggers: str) -> None:
    """Send the orderbuddy logger (and any extra loggers) to app.log. Safe to call twice."""
    path = log_dir / APP_LOG_NAME
    log.setLevel(logging.INFO)
    if any(isinstance(x, logging.FileHandler) for x in log.handlers):
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    for name in ("orderbuddy", *extra_loggers):
        logging.getLogger(name).addHandler(handler)


class ReadableLogger(JsonlLogger):
    """Writes the JSONL record as usual and one readable line to the orderbuddy logger."""

    def write(self, event, **fields):
        rec = super().write(event, **fields)
        # a valid llm_call only repeats llm_trace; a rejected one explains why
        if event == "llm_call" and fields.get("valid") is not False:
            return rec
        shown = {k: v for k, v in fields.items() if k not in _HIDDEN}
        text = short(json.dumps(shown, ensure_ascii=False, default=str))
        line = f"[{fields.get('run_id')}] {event} {text}"
        if event == "agent_error":
            log.error("%s\n%s", line, fields.get("trace", ""))
        elif event == "safe_failure" or fields.get("error"):
            log.warning(line)
        else:
            log.info(line)
        return rec


class TracedHarness(Harness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = ReadableLogger(self.log.path)

    def llm(self, profile, messages, *, run_id=None, prompt_id=None, **kwargs):
        user = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")
        fields = {"run_id": run_id, "profile": profile, "prompt": prompt_id, "user": user}
        try:
            out = super().llm(profile, messages, run_id=run_id, prompt_id=prompt_id, **kwargs)
        except Exception as e:  # noqa: BLE001 - record it, the caller still decides what to do
            self.log.write("llm_trace", **fields, error=f"{type(e).__name__}: {e}")
            raise
        # done_reason "length" means num_predict cut the reply off: the usual cause of bad JSON
        self.log.write("llm_trace", **fields, model=out.model, wall_s=round(out.wall_s, 2),
                       done_reason=out.raw.get("done_reason"), reply=out.content,
                       tool_calls=out.tool_calls)
        return out
