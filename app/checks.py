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
from difflib import SequenceMatcher
from typing import Callable, Optional

from harness import DomainError

from .catalog import catalog
from .schemas import DraftOrder, ExtractedOrder

MAX_LINES = 15
POSTCODE = re.compile(r"(?<!\d)\d{5}(?!\d)")   # no \b: Thai letters count as word characters
CHANGE_WORDS = ("ไม่", "ยกเลิก", "แทน", "ออก", "เปลี่ยน")   # a named dish may be leaving the order
FULFILLMENT_WORDS = {"delivery": ("ส่ง", "เดลิเวอรี่"), "pickup": ("รับ", "ที่ร้าน", "หน้าร้าน")}
MAX_REPLY = 500
MENU_COPY = 5          # a photo "order" with this many different dishes at qty 1 is the menu
FUZZY = 0.8            # typo tolerance when a dish name is looked for in the customer's words
# the newest message decides pickup / delivery ("เดะไปรับ" = I will pick it up)
PICKUP_CUES = re.compile(r"รับเอง|มารับ|ไปรับ|รับที่ร้าน|รับหน้าร้าน|เข้าไปรับ")
DELIVERY_CUES = re.compile(r"ส่งที่|มาส่ง|ให้ส่ง|ไปส่ง|ส่งด้วย|ส่งให้|เดลิเวอรี่|เดลิเวอรี|delivery", re.I)
# removal: "ไม่เอาข้าวผัดแล้ว" / "ลบ ชาเย็น" up to the next "เอา", "แต่" or comma; "เอา X กับ Y ออก"
_REMOVE_CLAUSE = re.compile(r"(?:ไม่เอา|ไม่ใส่|ไม่สั่ง|ยกเลิก|ลบ)((?:(?!เอา|แต่|,).)*)")
_TAKE_OUT = re.compile(r"เอา((?:(?!เอา).)*?)ออก")
_FILLER = re.compile(r"แล้ว|ด้วย|ครับ|คับ|ค่ะ|คะ|นะ|จ้า|ละ|อะ|เลย|ทั้งหมด|อันนี้")
# a contact-only message: digits plus these labels and nothing else
_CONTACT_LABELS = re.compile(r"เบอร์โทรศัพท์|เบอร์โทร|เบอร์|เบอ|โทรศัพท์|โทร|รหัสไปรษณีย์|รหัสไปรษณี|รหัสไปนี"
                             r"|รหัสปณ|ปณ\.?|รหัส|ไปรษณีย์|tel|phone|และ|กับ|ครับ|ค่ะ|คะ|นะ", re.I)
_LETTER = re.compile(r"[ก-๙A-Za-z]")      # an item text needs at least one letter
_ADDRESS_WORDS = re.compile(r"\d|ตึก|ซอย|ซ\.|ถนน|ถ\.|อาคาร|ห้อง|แขวง|เขต|หมู่|ม\.|ตำบล|อำเภอ|จังหวัด"
                            r"|คณะ|มหาวิทยาลัย|คอนโด|หมู่บ้าน|บ้านเลขที่")


def extract_check(text: str, draft: DraftOrder, history: list[dict],
                  has_image: bool) -> Callable[[ExtractedOrder], None]:
    """Build the domain check for one EXTRACT call.

    A phone number must come from the customer (this message, recent user messages
    or the draft); 4B models like to invent one. Photos are skipped: only the vision
    model can see the number there.
    """
    sources = [text, draft.customer.phone or ""]
    users = [m["content"] for m in history if m.get("role") == "user"]
    sources += users
    known_digits = " ".join(_digits(s) for s in sources)
    said = _squash(" ".join([text, *users]))
    in_draft = {line.name for line in draft.items}

    def check(order: ExtractedOrder) -> None:
        if len(order.items) > MAX_LINES:
            raise DomainError(f"too many items: at most {MAX_LINES} lines")
        if any(not i.text.strip() for i in order.items):
            raise DomainError("every item needs a menu name in text")
        if has_image:
            # a real order has quantities; a menu copy is many dishes, every one at 1 or none
            # (a 6-dish handwritten order with "ข้าวผัดกุ้ง 2" was refused as a menu copy)
            names = {i.text for i in order.items}
            all_one = all(i.qty in (None, 1) for i in order.items)
            if all_one and len(names & {m.name for m in catalog().values()}) >= MENU_COPY:
                raise DomainError("you copied the menu; return only the items written in the "
                                  "image, or items [] when the image is not an order")
            return
        for item in order.items:
            # 4B models swap an unknown dish for a menu dish ("แกงจืดเต้าหู้" -> ต้มยำกุ้ง),
            # pick one of two teas for "ชา", or invent a whole order for "อะไรก็ได้"
            if item.text not in in_draft and not mentioned(item.text, said):
                raise DomainError(f"{item.text} is not in the customer's messages; write the "
                                  "customer's own words as text, or leave the item out")
        phone = _digits(order.customer.phone or "")
        if phone and not has_image and phone not in known_digits:
            raise DomainError(f"customer.phone {order.customer.phone} is not in the customer's "
                              "messages; copy it exactly or use null")
        if not has_image:
            for name in _named_but_missing(text, order):
                raise DomainError(f"the newest message orders {name} but items does not have it; "
                                  "add it and keep the other items")
    return check


def mentioned(name: str, said: str) -> bool:
    """Did the customer write this menu dish? Name, alias, a part of the name that fits only
    this dish ("กะเพรา"), or a typo of it ("กระเพา"). Names that are not on the menu pass:
    RESOLVE handles them and the customer is told "ไม่พบเมนู"."""
    menu = list(catalog().values())
    m = next((x for x in menu if name in (x.name, *x.aliases)), None)
    if m is None:
        return True
    names = [_squash(n) for n in (m.name, *m.aliases)]
    if any(n in said for n in names):
        return True
    others = [_squash(n) for x in menu if x is not m for n in (x.name, *x.aliases)]
    for n in names:
        for size in range(len(n), 3, -1):            # a part of the name, at least 4 chars
            for i in range(len(n) - size + 1):
                part = n[i:i + size]
                if part in said and not any(part in o for o in others):
                    return True
        for size in range(5, len(n)):                   # a typo of the start ("กระเพา")
            if any(SequenceMatcher(None, n[:size], said[i:i + size]).ratio() > FUZZY
                   for i in range(len(said) - size + 1)):
                return True
        for size in (len(n) - 1, len(n), len(n) + 1):   # a typo of the whole name
            for i in range(max(len(said) - size + 1, 1)):
                if SequenceMatcher(None, n, said[i:i + size]).ratio() >= FUZZY:
                    return True
    return False


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
    dropped: list[str] = []
    junk = [i.text for i in order.items if not _LETTER.search(i.text)]
    if junk:                       # {"text": ": "}: not a dish, never "ไม่พบเมนู ': '"
        dropped += [f"item {t!r} not a name" for t in junk]
        order = order.model_copy(update={"items": [i for i in order.items
                                                   if _LETTER.search(i.text)]})
    if has_image:
        return _image_customer(order, dropped), dropped
    kept = [v for line in draft.items for v in (*line.options.values(), *line.addons)]
    users = [m["content"] for m in history if m.get("role") == "user"]
    said = _squash(" ".join([text, *users, *kept]))

    newest = _squash(text)
    clauses = [_FILLER.sub("", c) for c in
               _REMOVE_CLAUSE.findall(newest) + _TAKE_OUT.findall(newest)]
    order_names = [i.text for i in order.items]

    def refused(name: str) -> bool:
        if _zero_qty(name, newest):
            return True
        names = [_squash(n) for n in _all_names(name)]
        for c in clauses:
            if any(n in c for n in names):
                return True
            # "ไม่เอาชาแล้ว": a short word that starts exactly one dish in the order
            if len(c) >= 2 and c in names[0][:len(c)] and \
                    sum(_squash(o).startswith(c) for o in order_names) == 1:
                return True
        return False

    def asked_for(value: str) -> str | None:
        names = [v.split("+")[0].strip() for v in value.split("/")]
        hits = [n for n in names if n and _squash(n) in said and not refused(n)]
        return hits[0] if len(hits) == 1 else None

    items = []
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
    customer = _ground_customer(order, text, draft, users, dropped)
    if _contact_only(text):
        customer = _contact_fields(text, draft, customer)
    update["customer"] = customer
    address = customer.address or draft.customer.address
    code = POSTCODE.search(text)
    if address and code and code.group() not in address:
        # models keep the old address when only the postcode arrives ("รหัสไปรษณีย์ 10110")
        new = POSTCODE.sub(code.group(), address) if POSTCODE.search(address) \
            else f"{address} {code.group()}"
        update["customer"] = customer.model_copy(update={"address": new})
    fulfillment = order.fulfillment
    cue = ("pickup" if PICKUP_CUES.search(newest) else
           "delivery" if DELIVERY_CUES.search(newest) or newest == "ส่ง" else None)
    if cue:
        if fulfillment != cue:
            dropped.append(f"fulfillment {fulfillment} -> {cue}")
        update["fulfillment"] = cue
    elif fulfillment and fulfillment != draft.fulfillment:
        address = _squash(order.customer.address or "")
        by_address = fulfillment == "delivery" and address and address in said
        if not by_address and not any(w in said for w in FULFILLMENT_WORDS[fulfillment]):
            dropped.append(f"fulfillment {fulfillment}")
            update["fulfillment"] = draft.fulfillment
    return order.model_copy(update=update), dropped


def _image_customer(order: ExtractedOrder, dropped: list[str]) -> ExtractedOrder:
    """The vision model fills customer fields with order lines: name "กะเพราหมูกรอบ 1",
    phone "1", address "ข้าวสวย 2". A field holding a dish name, or a phone that is too
    short, is removed (null keeps what the draft already has)."""
    c = order.customer
    dishes = [_squash(n) for m in catalog().values() for n in (m.name, *m.aliases)]
    update = {}
    for field in ("name", "address"):
        value = _squash(getattr(c, field) or "")
        if value and any(d in value for d in dishes):
            dropped.append(f"customer.{field} {getattr(c, field)}")
            update[field] = None
    if c.phone and len(_digits(c.phone)) < 9:
        dropped.append(f"customer.phone {c.phone}")
        update["phone"] = None
    if not update:
        return order
    return order.model_copy(update={"customer": c.model_copy(update=update)})


def _all_names(name: str) -> list[str]:
    """The dish name first, then its catalog name and aliases ("ข้าวผัดกุ้ง" -> also "ข้าวผัด")."""
    m = next((x for x in catalog().values() if name in (x.name, *x.aliases)), None)
    return [name, *((m.name, *m.aliases) if m else ())]


def _ground_customer(order: ExtractedOrder, text: str, draft: DraftOrder, users: list[str],
                     dropped: list[str]):
    """A new name or address must come from the customer. A URL gave "สมชาย ดวงดี" and an
    address on Sukhumvit; "ตึก 3 วิศวะจุฬา 11111" was split into name and address."""
    c = order.customer
    said = _squash(" ".join([text, *users]))
    name, address = c.name, c.address
    if name and name != draft.customer.name:
        if _squash(name) not in said:
            dropped.append(f"customer.name {name}")
            name = draft.customer.name
        elif _ADDRESS_WORDS.search(name):          # an address written into the name
            dropped.append(f"customer.name {name} -> address")
            if not address or re.fullmatch(r"[\d\s]*", address):
                address = f"{name} {address or ''}".strip()
            name = draft.customer.name
    if address and address != draft.customer.address:
        words = [_squash(w) for w in re.split(r"[\s,]+", address) if len(_squash(w)) >= 2]
        found = sum(w in said for w in words)
        if words and found / len(words) < 0.6:
            dropped.append(f"customer.address {address}")
            address = draft.customer.address
    return c.model_copy(update={"name": name, "address": address})


def _contact_only(text: str) -> bool:
    """"10890", "เบอร์ 0812345678 รหัสไปรษณีย์ 10110": digits and labels, nothing else."""
    rest = _CONTACT_LABELS.sub("", re.sub(r"[\d\s\-.:,/()]+", " ", text or ""))
    return bool(re.search(r"\d", text or "")) and not rest.strip()


def _contact_fields(text: str, draft: DraftOrder, customer):
    """Code reads the phone and postcode itself; the model garbled the address with them
    ("10890" -> "10890/220/22 จุฬา", "11111" replaced the whole address)."""
    phone, code = draft.customer.phone, None
    for token in re.findall(r"\d[\d\-]*\d|\d", text):
        digits = token.replace("-", "")
        if len(digits) == 5:
            code = digits
        elif len(digits) >= 9 and digits.startswith("0"):
            phone = digits
        elif len(digits) >= 9:
            phone = digits                     # wrong number: address_validator says so
    address = draft.customer.address
    if code and address:
        address = POSTCODE.sub(code, address) if POSTCODE.search(address) else f"{address} {code}"
    return customer.model_copy(update={"phone": phone, "address": address,
                                       "name": draft.customer.name})


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
