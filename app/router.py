"""M10-M11 - Router (owner ฟลุ๊ค).

Two jobs:
1 classify()     newest message -> RouteResult (intent + complexity)
2 pick_profile() RouteResult -> which profile extracts the order

classify() runs three layers, cheapest first, and code always has the last word:
  rules  - a whole-message match ("ยืนยัน", "ยกเลิก", "สวัสดี") skips the model
  model  - the router profile fills RouteResult (validated, retried, repaired)
  guard  - confirm/cancel that also names food or numbers becomes modify_order
           (so "ยกเลิกชาเย็น" never cancels the whole order), and complexity is
           raised (never lowered) by deterministic cues
If the model fails or Ollama is down, keyword_route() answers instead.

Contract used by the state machine: classify(...) -> RouteResult (never raises)
and pick_profile(route, has_image) -> "fast" | "reasoning" | "vision".
"""
from __future__ import annotations

import json
import re
from typing import Optional

from harness import ask_structured, build_messages

from .catalog import catalog
from .schemas import DraftOrder, RouteResult

# whole-message phrases, compared after _normalize()
_CONFIRM_WORDS = {"ยืนยัน", "ok", "okay", "โอเค", "ได้เลย", "ตกลง", "ถูกต้อง",
                  "ใช่", "yes", "confirm", "สั่งเลย"}
_CANCEL_WORDS = {"ยกเลิก", "ยกเลิกออเดอร์", "ยกเลิกออร์เดอร์", "ยกเลิกทั้งหมด", "ไม่เอาแล้ว",
                 "ไม่สั่งแล้ว", "cancel"}
_GREET_WORDS = {"สวัสดี", "hello", "hi", "หวัดดี"}

_PARTICLES = re.compile(r"(ครับ|คับ|ค่ะ|คะ|ค่า|จ้า|จ้ะ|นะ|ผม|หน่อย)+$")
_PUNCT = re.compile(r"[\s.!?,~:;\"'()\-_/]+")
# words that turn "ok" into an edit, e.g. "ok แต่เปลี่ยนเป็นข้าวผัด"
_EDIT_CUES = re.compile(r"เพิ่ม|เปลี่ยน|แทน|ไม่เอา|เอาออก|ลด|แต่")
# words that make a change to an open order worth the reasoning model
_CHANGE_CUES = re.compile(r"เปลี่ยน|แทน|ไม่เอา|เอาออก|ลด|เหลือ|เพิ่มเป็น|ยกเลิก")
_CONDITION = re.compile(r"ถ้า")


def classify(h, text: str, draft: DraftOrder, last_bot: str, *, run_id: str,
             profile: str = "router", retry: bool = True) -> RouteResult:
    route, source = quick_route(text), "rule"
    if route is None:
        route, source = _ask_model(h, text, draft, last_bot, run_id=run_id,
                                   profile=profile, retry=retry), "model"
    if route is None:
        route, source = keyword_route(text, draft), "fallback"
    checked = guard(route, text, draft)
    if source == "model" and checked != route:
        source = "guard"
    h.log.write("route_source", run_id=run_id, source=source, intent=checked.intent,
                complexity=checked.complexity)
    return checked


def _ask_model(h, text, draft, last_bot, *, run_id, profile, retry) -> Optional[RouteResult]:
    """Router model call. None = no valid answer (safe_failure is logged by the harness)."""
    p = h.prompts.get("route")
    open_order = (json.dumps([f"{i.name} x{i.qty}" for i in draft.items], ensure_ascii=False)
                  if draft.items else "none")
    system = p.render(open_order=open_order, last_bot=last_bot or "none")
    try:
        res = ask_structured(h, build_messages(system, user=text), RouteResult,
                             profiles=(profile,), max_retry=2 if retry else 0, repair=retry,
                             prompt_id=p.id, run_id=run_id)
    except Exception as e:  # noqa: BLE001 - Ollama down or timeout: classify never raises
        h.log.write("safe_failure", run_id=run_id, prompt=p.id,
                    reason=f"{type(e).__name__}: {e}"[:200], attempts=0)
        return None
    return res.value if res.ok else None


def quick_route(text: str) -> Optional[RouteResult]:
    """Rules for short, unambiguous messages. None = ask the model."""
    t = _normalize(text)
    if not t or t in _GREET_WORDS:
        return RouteResult(intent="other", complexity="simple")
    if t in _CONFIRM_WORDS:
        return RouteResult(intent="confirm", complexity="simple")
    if t in _CANCEL_WORDS:
        return RouteResult(intent="cancel", complexity="simple")
    return None


def keyword_route(text: str, draft: DraftOrder) -> RouteResult:
    """Fallback when the model gives no valid answer: rules, else treat it as an order."""
    route = quick_route(text) or RouteResult(
        intent="modify_order" if draft.items else "new_order", complexity="simple")
    return guard(route, text, draft)


def guard(route: RouteResult, text: str, draft: DraftOrder) -> RouteResult:
    """Deterministic checks on top of any route. Only makes the result safer."""
    intent, complexity = route.intent, route.complexity
    named = menu_mentions(text)
    has_digit = bool(re.search(r"\d", text))
    if intent == "cancel" and (named or has_digit):
        intent = "modify_order"          # "ยกเลิกชาเย็น" removes one item, not the order
    if intent == "confirm" and (named or has_digit or _EDIT_CUES.search(text)):
        intent = "modify_order"          # "ok แต่เพิ่มชาเย็น 1" still changes the order
    if intent == "modify_order" and not draft.items:
        intent = "new_order"
    if intent in ("new_order", "modify_order") and (
            len(named) >= 4 or _CONDITION.search(text)
            or (draft.items and _CHANGE_CUES.search(text))):
        complexity = "complex"
    if (intent, complexity) == (route.intent, route.complexity):
        return route
    return RouteResult(intent=intent, complexity=complexity)


def menu_mentions(text: str) -> set[str]:
    """SKUs whose name or alias appears in the text."""
    t = re.sub(r"\s+", "", text.lower())
    return {it.sku for it in catalog().values()
            if any(re.sub(r"\s+", "", n.lower()) in t for n in (it.name, *it.aliases))}


def _normalize(text: str) -> str:
    t = _PUNCT.sub("", (text or "").lower())
    return _PARTICLES.sub("", t) or t


def pick_profile(route: RouteResult, has_image: bool) -> str:
    """vision for photos, reasoning for complex messages, otherwise the fast 4B model.

    A simple answer to the bot ("เผ็ดน้อย", a phone number) stays on fast; real changes
    to an open order are marked complex by guard(), so they go to reasoning.
    """
    if has_image:
        return "vision"
    if route.complexity == "complex":
        return "reasoning"
    return "fast"
