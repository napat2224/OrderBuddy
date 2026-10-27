"""M5 - Domain checks on model output (owner ฟลุ๊ค).

The schema (Pydantic) says the output has the right shape; these checks say it is
believable. A failed check raises DomainError inside ask_structured, which sends the
error back to the model (repair), then escalates fast -> reasoning, then gives up
safely. Nothing here calls a model.

extract_check()  ExtractedOrder from fast / reasoning / vision
clean_reply()    free-text clarify / answer messages shown to the customer
"""
from __future__ import annotations

import re
from typing import Callable

from harness import DomainError

from .schemas import DraftOrder, ExtractedOrder

MAX_LINES = 15
MAX_REPLY = 500


def extract_check(text: str, draft: DraftOrder, history: list[dict],
                  has_image: bool) -> Callable[[ExtractedOrder], None]:
    """Build the domain check for one EXTRACT call.

    A phone number must come from the customer (this message, recent user messages
    or the draft); 4B models like to invent one. Photos are skipped: only the vision
    model can see the number there.
    """
    sources = [text, draft.customer.phone or ""]
    sources += [m["content"] for m in history if m.get("role") == "user"]
    known_digits = " ".join(_digits(s) for s in sources)

    def check(order: ExtractedOrder) -> None:
        if len(order.items) > MAX_LINES:
            raise DomainError(f"too many items: at most {MAX_LINES} lines")
        if any(not i.text.strip() for i in order.items):
            raise DomainError("every item needs a menu name in text")
        phone = _digits(order.customer.phone or "")
        if phone and not has_image and phone not in known_digits:
            raise DomainError(f"customer.phone {order.customer.phone} is not in the customer's "
                              "messages; copy it exactly or use null")
    return check


def clean_reply(raw: str, system_prompt: str, fallback: str) -> str:
    """Return a safe customer message, or the template fallback.

    Any "<number> บาท" must already be in the system prompt (menu prices); otherwise the
    model made up money and the fallback is used. The reply ends with ครับ (shop style).
    """
    reply = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    if not reply:
        return fallback
    allowed = set(re.findall(r"\d+", system_prompt))
    if any(n not in allowed for n in re.findall(r"(\d+)\s*บาท", reply.replace(",", ""))):
        return fallback
    reply = reply[:MAX_REPLY].rstrip()
    reply = re.sub(r"(ค่ะ|คะ|ค่า|คับ)([\s.!~]*)$", r"ครับ\2", reply)
    if "ครับ" not in reply[-12:]:
        reply += "ครับ"
    return reply


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)
