"""M10-M11 - Router (STAND-IN: owner ฟลุ๊ค improves prompts and rules).

Two jobs:
1 classify()     newest message -> RouteResult (intent + complexity), router profile
2 pick_profile() RouteResult -> which profile extracts the order

Contract used by the state machine: classify(...) -> RouteResult (never raises)
and pick_profile(route, has_image) -> "fast" | "reasoning" | "vision".
"""
from __future__ import annotations

import json
import re

from harness import ask_structured, build_messages

from .schemas import DraftOrder, RouteResult

_CONFIRM = re.compile(r"^(ยืนยัน|ok|โอเค|ได้เลย|ตกลง|ถูกต้อง|yes|confirm)", re.I)
_CANCEL = re.compile(r"(ยกเลิก|ไม่เอาแล้ว|cancel)", re.I)


def classify(h, text: str, draft: DraftOrder, last_bot: str, *, run_id: str,
             profile: str = "router", retry: bool = True) -> RouteResult:
    p = h.prompts.get("route")
    open_order = (json.dumps([f"{i.name} x{i.qty}" for i in draft.items], ensure_ascii=False)
                  if draft.items else "none")
    system = p.render(open_order=open_order, last_bot=last_bot or "none")
    res = ask_structured(h, build_messages(system, user=text), RouteResult,
                         profiles=(profile,), max_retry=2 if retry else 0, repair=retry,
                         prompt_id=p.id, run_id=run_id)
    if res.ok:
        return res.value
    return keyword_route(text, draft)      # safe fallback, logged as safe_failure above


def keyword_route(text: str, draft: DraftOrder) -> RouteResult:
    t = text.strip()
    if _CANCEL.search(t):
        return RouteResult(intent="cancel", complexity="simple")
    if _CONFIRM.match(t):
        return RouteResult(intent="confirm", complexity="simple")
    return RouteResult(intent="modify_order" if draft.items else "new_order",
                       complexity="simple")


def pick_profile(route: RouteResult, has_image: bool) -> str:
    if has_image:
        return "vision"
    if route.intent == "modify_order" or route.complexity == "complex":
        return "reasoning"
    return "fast"
