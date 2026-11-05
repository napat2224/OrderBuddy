"""M5 - Domain checks on model output (owner ฟลุ๊ค).

The schema (Pydantic) says the output has the right shape; these checks say it is
believable. A failed check raises DomainError inside ask_structured, which sends the
error back to the model (repair), then escalates fast -> reasoning, then gives up
safely. Nothing here calls a model.

extract_check()    ExtractedOrder from fast / reasoning / vision
clean_extracted()  removes what the customer refused or never asked for
clean_reply()      free-text clarify / answer messages shown to the customer
"""
from __future__ import annotations

import re
from typing import Callable

from harness import DomainError

from .catalog import catalog
from .schemas import DraftOrder, ExtractedOrder

MAX_LINES = 15
POSTCODE = re.compile(r"(?<!\d)\d{5}(?!\d)")   # no \b: Thai letters count as word characters
CHANGE_WORDS = ("ไม่", "ยกเลิก", "แทน", "ออก", "เปลี่ยน")   # a named dish may be leaving the order
FULFILLMENT_WORDS = {"delivery": ("ส่ง", "เดลิเวอรี่"), "pickup": ("รับ", "ที่ร้าน", "หน้าร้าน")}
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
        if not has_image:
            for name in _named_but_missing(text, order):
                raise DomainError(f"the newest message orders {name} but items does not have it; "
                                  "add it and keep the other items")
    return check


def _named_but_missing(text: str, order: ExtractedOrder) -> list[str]:
    """Menu names in the newest message that are not in the order ("เอาชาเย็น 1" was ignored)."""
    said = _squash(text)
    if any(w in said for w in CHANGE_WORDS):
        return []
    got = [_squash(i.text) for i in order.items]
    return [m.name for m in catalog().values() if _squash(m.name) in said
            and not _zero_qty(m.name, said)
            and not any(g and (g in _squash(m.name) or _squash(m.name) in g) for g in got)]


def clean_extracted(order: ExtractedOrder, text: str, draft: DraftOrder, history: list[dict],
                    has_image: bool) -> tuple[ExtractedOrder, list[str]]:
    """Keep only what the customer really asked for.

    A guessed fulfillment is undone too: it would ask a pickup customer for an address.
    "ไม่เอาชาไทย" must remove the item, but even qwen3:4b sometimes keeps it, and the
    customer would pay for it, so an explicit refusal in the newest message wins.

    Small models copy the menu line into the answer ("ไม่เผ็ด/เผ็ดน้อย/...", "ไข่ดาว +10").
    That passes the schema, then either sends a plain order to CLARIFY or charges for an
    addon nobody wanted, so code removes it here instead of paying for a model retry.
    Returns the order and what was removed (logged in the "extract" event). Photos are
    skipped: there is no text to compare with.
    """
    if has_image:
        return order, []
    kept = [v for line in draft.items for v in (*line.options.values(), *line.addons)]
    users = [m["content"] for m in history if m.get("role") == "user"]
    said = _squash(" ".join([text, *users, *kept]))

    newest = _squash(text)

    def refused(name: str) -> bool:
        n = _squash(name)
        return (any(p in newest for p in (f"ไม่เอา{n}", f"ไม่ใส่{n}", f"ยกเลิก{n}", f"เอา{n}ออก"))
                or _zero_qty(name, newest))

    def asked_for(value: str) -> str | None:
        names = [v.split("+")[0].strip() for v in value.split("/")]
        hits = [n for n in names if n and _squash(n) in said and not refused(n)]
        return hits[0] if len(hits) == 1 else None

    items, dropped = [], []
    for item in order.items:
        if refused(item.text):
            dropped.append(f"{item.text} refused")
            continue
        options, addons = {}, []
        for key, value in item.options.items():
            hit = asked_for(value)
            if hit:
                options[key] = hit
            else:
                dropped.append(f"{item.text} option {key}={value}")
        for value in item.addons:
            hit = asked_for(value)
            if hit:
                addons.append(hit)
            else:
                dropped.append(f"{item.text} addon {value}")
        items.append(item.model_copy(update={"options": options, "addons": addons}))
    update = {"items": items}
    address = order.customer.address or draft.customer.address
    code = POSTCODE.search(text)
    if address and code and code.group() not in address:
        # models keep the old address when only the postcode arrives ("รหัสไปรษณีย์ 10110")
        new = POSTCODE.sub(code.group(), address) if POSTCODE.search(address) \
            else f"{address} {code.group()}"
        update["customer"] = order.customer.model_copy(update={"address": new})
    fulfillment = order.fulfillment
    if fulfillment and fulfillment != draft.fulfillment:
        address = _squash(order.customer.address or "")
        by_address = fulfillment == "delivery" and address and address in said
        if not by_address and not any(w in said for w in FULFILLMENT_WORDS[fulfillment]):
            dropped.append(f"fulfillment {fulfillment}")
            update["fulfillment"] = draft.fulfillment
    return order.model_copy(update=update), dropped


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


def _zero_qty(name: str, said: str) -> bool:
    """True for "ชาไทย0แก้ว" in squashed text: zero means none, and the schema has no qty 0.

    (?!\\d) keeps a phone right after the name ("ชาไทย0812345678") from counting as zero.
    """
    return re.search(re.escape(_squash(name)) + r"(?:x|จำนวน)?(?:0|ศูนย์)(?!\d)", said,
                     flags=re.I) is not None


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)
