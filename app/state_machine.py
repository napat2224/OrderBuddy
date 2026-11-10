"""M9 - State machine: one LINE message in, a list of reply texts out.

    agent = OrderAgent()
    replies = agent.handle(line_user_id, text="ขอกะเพราหมูกรอบ 2 ส่ง ...")
    replies = agent.handle(line_user_id, image_b64="...")      # photo of an order

States (spec table): ROUTE -> EXTRACT -> RESOLVE -> PRICE -> ADDRESS -> CONFIRM -> DONE,
with CLARIFY when something is missing and ERROR when anything raises.
Every state change is logged as a "state" event in Harness/logs/calls.jsonl.

Safety (M19):
* the model only ever sees the read tools product_search and stock_checker
* draft_order_manager (write) is called by code, only in DONE or on cancel
* DONE needs the previous state to be CONFIRM, no missing fields AND a
  "confirm" intent in the newest message; the tool checks the same again
"""
from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from harness import Harness, LLMError, ask_structured, build_messages, new_run_id

from . import faq, router, tools
from .agent import MIN_SCORE, call_tool, resolve_items
from .catalog import catalog, menu_text, stock
from .checks import CHANGE_WORDS, clean_extracted, clean_reply, extract_check, mentioned
from .schemas import DraftOrder, ExtractedItem, ExtractedOrder, OrderLine, RouteResult
from .session import SessionStore
from .trace import TracedHarness

READ = frozenset({"read"})
READ_WRITE = frozenset({"read", "write"})

MSG = {
    "greet": "สวัสดีครับ พิมพ์เมนูและจำนวนที่ต้องการ หรือส่งรูปรายการอาหารมาได้เลยครับ",
    "error": "ขออภัยครับ ระบบขัดข้องชั่วคราว รบกวนส่งข้อความอีกครั้ง หรือรอพนักงานติดต่อกลับครับ",
    "not_understood": "ขอโทษครับ ยังไม่เข้าใจรายการ รบกวนพิมพ์ชื่อเมนูและจำนวนอีกครั้งได้ไหมครับ",
    "ask_items": "ต้องการสั่งเมนูอะไรครับ",
    "cancelled": "ยกเลิกออร์เดอร์ {order_id} แล้วครับ",
    "nothing_to_cancel": "ตอนนี้ยังไม่มีออร์เดอร์ที่ต้องยกเลิกครับ",
    "submitted": "รับออร์เดอร์ {order_id} แล้วครับ ยอดรวม {total} บาท ขอบคุณครับ",
    "sold_out_now": "ขออภัยครับ มีเมนูในออร์เดอร์เพิ่งหมด รบกวนพิมพ์รายการที่ต้องการแก้ได้เลยครับ",
    # never show the raw error to the customer (it showed "FileExistsError: ..."); the
    # tool_call event in the log keeps the reason
    "submit_failed": "ขออภัยครับ ยังส่งออร์เดอร์ไม่ได้ รบกวนพิมพ์ \"ยืนยัน\" อีกครั้ง หรือรอพนักงานติดต่อกลับครับ",
    "nothing_to_confirm": "ตอนนี้ยังไม่มีออร์เดอร์ที่ต้องยืนยันครับ พิมพ์เมนูและจำนวนที่ต้องการได้เลยครับ",
    "no_open_order": "ตอนนี้ยังไม่มีออร์เดอร์ที่กำลังสั่งครับ พิมพ์เมนูและจำนวนที่ต้องการได้เลยครับ",
    "confirm_typo": "หมายถึง \"ยืนยัน\" ใช่ไหมครับ ถ้าใช่ พิมพ์ \"ยืนยัน\" อีกครั้งได้เลยครับ",
    "photo_unread": ("ยังอ่านรายการจากรูปไม่ได้ครับ รบกวนพิมพ์ชื่อเมนูและจำนวน "
                     "หรือส่งรูปที่เห็นตัวหนังสือชัด ๆ อีกครั้งได้เลยครับ"),
    "old_order": ("ออร์เดอร์ที่ยืนยันแล้วดูหรือแก้ในแชตนี้ไม่ได้ครับ "
                  "เดี๋ยวพนักงานจะช่วยตรวจสอบให้ทาง LINE นี้ครับ"),
}

PROBLEM_TEXT = {
    "fulfillment": "ยังไม่ทราบว่าจะรับเองที่ร้านหรือให้ไปส่ง",
    "phone": "ยังไม่มีเบอร์โทรที่ถูกต้อง (10 หลัก)",
    "address": "ยังไม่มีที่อยู่จัดส่ง (บ้านเลขที่)",
    "postcode": "ที่อยู่ยังไม่มีรหัสไปรษณีย์ (5 หลัก)",
}


@dataclass
class AgentConfig:
    """Harness configs compared in the eval (Part C). C1 = single-model baseline,
    built in eval/ because it skips this state machine."""
    name: str = "C3"
    use_router: bool = True          # False: every step uses the reasoning profile
    validate_retry: bool = True      # False: one attempt, no repair, no escalation
    resolve_with_model: bool = True  # False: unclear names go straight to CLARIFY
    max_tool_rounds: int = 4


CONFIGS = {
    "C2": AgentConfig(name="C2", validate_retry=False),
    "C3": AgentConfig(name="C3"),
}


class OrderAgent:
    def __init__(self, h: Optional[Harness] = None, store: Optional[SessionStore] = None,
                 config: Optional[AgentConfig] = None):
        self.h = h or TracedHarness()
        self.store = store or SessionStore()
        self.cfg = config or AgentConfig()
        tools.configure(self.store)

    # ================================================================ entry
    def handle(self, line_user_id: str, text: str = "",
               image_b64: Optional[str] = None) -> list[str]:
        run_id = new_run_id()
        text = (text or "").strip()
        with self.store.lock(line_user_id):
            draft = self.store.get_or_create_draft(line_user_id)
            history = self.store.get_history(line_user_id)
            self.h.log.write("message", run_id=run_id, user=line_user_id, config=self.cfg.name,
                             state=draft.state, text=text, image=bool(image_b64))
            try:
                replies, closed = self._step(run_id, draft, history, text, image_b64)
            except Exception as e:  # noqa: BLE001 - ERROR state: tell the user, keep the old draft
                self.h.log.write("agent_error", run_id=run_id, user=line_user_id,
                                 state=draft.state, error=f"{type(e).__name__}: {e}",
                                 trace=traceback.format_exc())
                self._log_state(run_id, draft.state, "ERROR")
                self._log_reply(run_id, line_user_id, "ERROR", [MSG["error"]])
                return [MSG["error"]]

            self._log_reply(run_id, line_user_id, draft.state, replies)

            if closed:                       # submitted or cancelled
                # the id stays reserved: a cancelled ORD-...-0007 was given to the next order
                self.store.clear_history(line_user_id)
            else:
                if draft.items or draft.state != "IDLE":
                    self.store.save_draft(draft)   # greetings/questions leave no empty draft
                else:
                    self.store.release_order_id(draft.order_id)
                self.store.append_history(line_user_id, "user", text or "[รูปภาพ]")
                for r in replies:
                    self.store.append_history(line_user_id, "assistant", r)
            return replies

    # ================================================================ ROUTE
    def _step(self, run_id, draft: DraftOrder, history, text, image_b64):
        prev = draft.state
        last_bot = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        self._goto(run_id, draft, "ROUTE")

        if image_b64:   # a photo is always an order; the router model cannot see images
            route = RouteResult(intent="modify_order" if draft.items else "new_order",
                                complexity="simple")
        else:
            route = router.classify(self.h, text, draft, last_bot, run_id=run_id,
                                    profile="router" if self.cfg.use_router else "reasoning",
                                    retry=self.cfg.validate_retry)
        self.h.log.write("route", run_id=run_id, intent=route.intent,
                         complexity=route.complexity, prev_state=prev)
        intent = route.intent

        if intent == "cancel":
            if not draft.items:
                draft.state = "IDLE"
                return [MSG["nothing_to_cancel"]], False
            call_tool(self.h, "draft_order_manager",
                      {"action": "cancel", "line_user_id": draft.line_user_id},
                      run_id=run_id, allowed=READ_WRITE)
            draft.state = "IDLE"
            return [MSG["cancelled"].format(order_id=draft.order_id)], True

        if intent == "confirm":
            if prev == "CONFIRM" and not draft.missing:
                return self._done(run_id, draft)
            if not draft.items:
                draft.state = "IDLE"
                return [MSG["nothing_to_confirm"]], False
            intent = "modify_order"          # "ok" as an answer to a question

        if prev == "CONFIRM" and intent != "confirm" and not image_b64 and _confirm_typo(text):
            draft.state = prev                # "ยืรยัน": never submitted, only a hint
            return [MSG["confirm_typo"]], False

        # "สรุปออเดอร์ของผม", "ร้านรับคนละครึ่งไหม": answered by code from the draft and
        # data/*.json, whatever the router said, as long as no dish or number is in the text
        if intent != "question" and not image_b64 and self._plain_question(text):
            reply = self._code_answer(draft, text)
            if reply:
                draft.state = prev
                return [reply], False

        if intent == "question":
            draft.state = prev                # a question never moves the order
            return [self._code_answer(draft, text) or self._answer(run_id, text)], False

        if intent == "other" and not draft.items:
            draft.state = "IDLE"
            return [MSG["greet"]], False

        if self.cfg.use_router:
            profile = router.pick_profile(route, bool(image_b64))
        else:
            profile = "vision" if image_b64 else "reasoning"
        return self._order_flow(run_id, draft, history, text, image_b64, profile), False

    @staticmethod
    def _plain_question(text: str) -> bool:
        """No dish and no number (an order id aside): the message cannot change the order."""
        t = faq._ORDER_ID.sub("", text or "")
        return not router.menu_mentions(t) and not re.search(r"\d", t)

    def _code_answer(self, draft: DraftOrder, text: str):
        """Reply built by code, or None. The open order comes first, then app/faq.py."""
        if _PHOTO_ASK.search(text or ""):   # "เข้าใจรายการในรูปว่าอะไร": the model made up an answer
            if not draft.items:
                return MSG["photo_unread"]
            return "\n".join([f"ออร์เดอร์ {draft.order_id} ตอนนี้มี", *_item_rows(draft),
                              "ถ้าไม่ถูกต้อง พิมพ์รายการที่ต้องการแก้ได้เลยครับ"])
        if faq.wants_order_summary(text) or faq.mentions_old_order(text):
            if faq.mentions_old_order(text) and draft.order_id not in text:
                return MSG["old_order"]
            if not draft.items:
                return MSG["no_open_order"]
            if draft.state == "CONFIRM" or (not draft.missing and draft.total):
                return self._summary(draft)
            rows = [f"ออร์เดอร์ {draft.order_id} ตอนนี้มี"]
            rows += [f"- {l.name} x{l.qty}" for l in draft.items]
            if draft.missing:
                rows.append("ยังขาดข้อมูลอยู่ พิมพ์ส่วนที่ขาดมาได้เลยครับ")
            return "\n".join(rows)
        return faq.faq_answer(text)

    # ======================================== EXTRACT -> RESOLVE -> PRICE -> ADDRESS
    def _order_flow(self, run_id, draft, history, text, image_b64, profile) -> list[str]:
        self._goto(run_id, draft, "EXTRACT")
        if not image_b64 and draft.items and _ANY_IS_FINE.search(text):
            ext = self._from_draft(run_id, draft)
        else:
            ext = self._extract(run_id, draft, history, text, image_b64, profile)
        if ext is None:                       # safe failure after retries
            self._goto(run_id, draft, "CLARIFY")
            return [MSG["not_understood"]]
        self._merge_customer(draft, ext)
        if not image_b64:
            _drop_guessed_qty(ext.items, text, draft)
            _drop_decimal_qty(ext.items, text)
        ext.items = [it for it in ext.items if it.qty != 0]    # qty 0 = customer removed it
        if not ext.items:
            draft.items, draft.missing = [], ["items"]
            self._goto(run_id, draft, "CLARIFY")
            return [MSG["ask_items"]]

        # ---- RESOLVE: code matches names; the model only picks between equal matches
        self._goto(run_id, draft, "RESOLVE")
        mapping = resolve_items(self.h, ext.items, run_id=run_id,
                                use_model=self.cfg.resolve_with_model,
                                max_rounds=self.cfg.max_tool_rounds)
        lines, missing, problems, suggestions = self._build_lines(run_id, ext.items, mapping)
        draft.items = lines

        if not missing:
            # ---- PRICE (code calls the tool; the model never computes money)
            self._goto(run_id, draft, "PRICE")
            r = call_tool(self.h, "price_calculator", {
                "items": [{"sku": l.sku, "qty": l.qty, "addons": l.addons} for l in lines],
                "fulfillment": draft.fulfillment or "pickup"}, run_id=run_id)
            if not r["ok"]:
                raise RuntimeError(f"price_calculator failed: {r['error']}")
            self._apply_price(draft, r["result"])

            # ---- ADDRESS
            self._goto(run_id, draft, "ADDRESS")
            if draft.fulfillment is None:
                missing.append("fulfillment")
            else:
                need_addr = draft.fulfillment == "delivery"
                r = call_tool(self.h, "address_validator", {
                    "address": draft.customer.address or "", "phone": draft.customer.phone or "",
                    "need_address": need_addr}, run_id=run_id)
                if not r["ok"]:
                    raise RuntimeError(f"address_validator failed: {r['error']}")
                res = r["result"]
                missing += res["missing"]
                if res["normalized"]["phone"] and "phone" not in res["missing"]:
                    draft.customer.phone = res["normalized"]["phone"]
                draft.customer.address_valid = need_addr and res["valid"]
            problems += [PROBLEM_TEXT[m] for m in missing if m in PROBLEM_TEXT]

        draft.missing = missing
        if missing:
            self._goto(run_id, draft, "CLARIFY")
            return [self._clarify(problems, suggestions, _item_rows(draft, skip_stock=True))]
        self._goto(run_id, draft, "CONFIRM")
        return [self._summary(draft)]

    # --------------------------------------------------------------- EXTRACT
    def _extract(self, run_id, draft, history, text, image_b64, profile):
        current = json.dumps({
            "items": [{"text": l.name, "qty": None if _qty_unknown(draft, l) else l.qty,
                       "options": l.options, "addons": l.addons} for l in draft.items],
            "customer": {"name": draft.customer.name, "phone": draft.customer.phone,
                         "address": draft.customer.address},
            "fulfillment": draft.fulfillment}, ensure_ascii=False)
        retry = self.cfg.validate_retry
        if image_b64:
            p = self.h.prompts.get("extract_image")
            system = p.render(menu=menu_text(), current=current, caption=text or "none")
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": "Read the order in this image.",
                     "images": [image_b64]}]
            profiles = ("vision",)            # only the vision model can see the image
        else:
            p = self.h.prompts.get("extract")
            system = p.render(menu=menu_text(), current=current)
            msgs = build_messages(system, history=history[-6:], user=text)
            profiles = ((profile, "reasoning") if retry and profile != "reasoning"
                        else (profile,))
        res = ask_structured(self.h, msgs, ExtractedOrder, profiles=profiles,
                             max_retry=2 if retry else 0, repair=retry,
                             domain_check=extract_check(text, draft, history[-6:], bool(image_b64)),
                             prompt_id=p.id, run_id=run_id)
        if not res.ok:                        # the harness already logged safe_failure
            return None
        order, dropped = clean_extracted(res.value, text, draft, history[-6:], bool(image_b64))
        if not image_b64:
            dropped += _keep_unmentioned(order, draft, text)
            dropped += _replace_sold_out(order, draft, text)
        self.h.log.write("extract", run_id=run_id, prompt=p.id, profile=res.profile_used,
                         attempts=res.attempts, errors=res.errors, dropped=dropped,
                         result=order.model_dump())
        return order

    def _from_draft(self, run_id, draft: DraftOrder) -> ExtractedOrder:
        """"อะไรก็ได้" answers an option question: keep the order as it is (a wrong option was
        already left out). The model invented ': ' and once replaced the order with ข้าวสวย x1."""
        order = ExtractedOrder(
            items=[ExtractedItem(text=l.name, qty=None if _qty_unknown(draft, l) else l.qty,
                                 options=l.options, addons=l.addons, note=l.note)
                   for l in draft.items],
            customer=draft.customer.model_dump(include={"name", "phone", "address"}),
            fulfillment=draft.fulfillment)
        self.h.log.write("extract", run_id=run_id, prompt=None, profile="code", attempts=0,
                         errors=[], dropped=[], result=order.model_dump())
        return order

    @staticmethod
    def _merge_customer(draft: DraftOrder, ext: ExtractedOrder) -> None:
        for field in ("name", "phone", "address"):
            new = getattr(ext.customer, field)
            if new and new != getattr(draft.customer, field):
                setattr(draft.customer, field, new)
                if field == "address":
                    draft.customer.address_valid = False
        if ext.fulfillment:
            draft.fulfillment = ext.fulfillment

    # --------------------------------------------------------------- RESOLVE
    def _build_lines(self, run_id, items: list[ExtractedItem], mapping):
        lines, missing, problems, suggestions = [], [], [], []
        menu = catalog()
        in_order = {mapping.get(i) for i in range(len(items))}   # never suggest these again
        for i, it in enumerate(items):
            sku = mapping.get(i)
            if sku not in menu:                       # model answer re-checked by code
                missing.append(f"item:{it.text}")
                r = call_tool(self.h, "product_search", {"query": it.text, "top_k": 3},
                              run_id=run_id)
                hits = r.get("result") or []
                close = [x["name"] + (" (หมด)" if stock().get(x["sku"], 0) <= 0 else "")
                         for x in hits if x["score"] >= MIN_SCORE]
                if len(close) > 1:                    # "ชา": ชาเย็น or ชาไทย, ask which one
                    problems.append(f"'{it.text}' หมายถึงเมนูไหนครับ: {', '.join(close)}")
                    continue
                problems.append(f"ไม่พบเมนู '{it.text}'")
                suggestions += [x["name"] for x in hits
                                if x.get("sku") not in in_order and stock().get(x["sku"], 0) > 0]
                continue
            m = menu[sku]
            options, addons, note, bad = _sort_options(m, it)
            if bad:
                missing.append(f"option:{sku}")
                problems.append(_option_problem(m, bad))
            # qty None: placeholder 1, marked missing, so it is never priced or summarised
            line = OrderLine(sku=sku, name=m.name, qty=it.qty or 1, note=note,
                             options=options, addons=addons)
            lines.append(line)
            if it.qty is None:
                missing.append(f"qty:{sku}")
                problems.append(f"ยังไม่ทราบจำนวน '{m.name}'")
                continue
            r = call_tool(self.h, "stock_checker", {"sku": sku, "qty": it.qty}, run_id=run_id)
            if not r["ok"] or not r["result"]["enough"]:
                left = r["result"]["available"] if r["ok"] else 0
                missing.append(f"stock:{sku}")
                problems.append(f"'{m.name}' เหลือ {left} ที่" if left else f"'{m.name}' หมด")
                suggestions += [x.name for x in menu.values()
                                if x.category == m.category and x.sku not in in_order
                                and stock().get(x.sku, 0) >= it.qty][:2]
        return lines, missing, problems, list(dict.fromkeys(suggestions))

    @staticmethod
    def _apply_price(draft: DraftOrder, price: dict) -> None:
        by_sku = {}
        for p in price["lines"]:
            by_sku.setdefault(p["sku"], []).append(p)
        for line in draft.items:
            p = by_sku[line.sku].pop(0)
            line.unit_price, line.line_total = p["unit_price"], p["line_total"]
        draft.subtotal = price["subtotal"]
        draft.delivery_fee = price["delivery_fee"]
        draft.total = price["total"]

    # ------------------------------------------------------- CLARIFY / answer
    @staticmethod
    def _clarify(problems, suggestions, items=()) -> str:
        """Written by code: the customer must see exactly what is wrong. With a model, a postcode
        the customer had sent was reported as missing and invented order statuses appeared.
        The current items are listed too: "ข้าวผัดเหลือ 1" got the same reply as before, so
        the customer could not see that the change was made (or, in that log, later lost)."""
        rows = ["ออร์เดอร์ยังไม่ครบครับ", *(f"- {p}" for p in problems)]
        if suggestions:
            rows.append("เมนูที่แนะนำ: " + ", ".join(suggestions))
        if items:
            rows.append("ตอนนี้ในออร์เดอร์:")
            rows += items
        rows.append("รบกวนแจ้งกลับด้วยนะครับ")
        return "\n".join(rows)

    def _answer(self, run_id, text) -> str:
        p = self.h.prompts.get("answer")
        left = stock()
        menu = "\n".join(line + (" (หมด)" if left.get(sku, 0) <= 0 else "")
                         for sku, line in zip(catalog(), menu_text().splitlines()))
        return self._say(run_id, "fast", p.id, p.render(menu=menu, shop=faq.shop_facts()), text,
                         "ขอโทษครับ ตอบคำถามนี้ไม่ได้ เดี๋ยวพนักงานจะตอบกลับครับ")

    def _say(self, run_id, profile, prompt_id, system, user, fallback) -> str:
        """Free-text reply from the model; the template is used if the model fails."""
        try:
            out = self.h.llm(profile, [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                             run_id=run_id, prompt_id=prompt_id)
        except LLMError:
            return fallback
        return clean_reply(out.content, system, fallback)   # price + ending checks

    # --------------------------------------------------------- CONFIRM / DONE
    @staticmethod
    def _summary(d: DraftOrder) -> str:
        rows = [f"สรุปออร์เดอร์ {d.order_id}"]
        for l in d.items:
            extra = [*l.options.values(), *(f"+{a}" for a in l.addons)]
            rows.append(f"- {l.name}{' (' + ', '.join(extra) + ')' if extra else ''}"
                        f" x{l.qty} = {l.line_total} บาท")
        if d.delivery_fee:
            rows.append(f"ค่าส่ง {d.delivery_fee} บาท")
        rows.append(f"รวม {d.total} บาท")
        rows.append(f"ส่งที่: {d.customer.address}" if d.fulfillment == "delivery"
                    else "รับเองที่ร้าน")
        rows.append(f"โทร: {d.customer.phone}")
        rows.append('พิมพ์ "ยืนยัน" เพื่อสั่ง หรือพิมพ์สิ่งที่ต้องการแก้ได้เลยครับ')
        return "\n".join(rows)

    def _done(self, run_id, draft: DraftOrder):
        self._goto(run_id, draft, "DONE")
        r = call_tool(self.h, "draft_order_manager",
                      {"action": "submit", "line_user_id": draft.line_user_id},
                      run_id=run_id, allowed=READ_WRITE)
        if not r["ok"]:
            draft.state = "CONFIRM"
            if "out_of_stock" in r["error"]:
                return [MSG["sold_out_now"]], False
            return [MSG["submit_failed"]], False
        return [MSG["submitted"].format(order_id=r["result"]["order_id"],
                                        total=r["result"]["total"])], True

    # ------------------------------------------------------------- helpers
    def _goto(self, run_id, draft: DraftOrder, state: str) -> None:
        self._log_state(run_id, draft.state, state)
        draft.state = state

    def _log_state(self, run_id, src, dst) -> None:
        self.h.log.write("state", run_id=run_id, src=src, dst=dst)

    def _log_reply(self, run_id, line_user_id, state, replies) -> None:
        """Pairs with the "message" event (same run_id) so each turn can be reviewed later."""
        self.h.log.write("reply", run_id=run_id, user=line_user_id, state=state, replies=replies)


_QTY = re.compile(r"(?<!\d)\d{1,2}(?![\d/])|หนึ่ง|นึง|เดียว|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ")
_NOT_ITEM = re.compile(r"ส่ง|รับ|โทร|เบอร์|ที่อยู่|บ้านเลขที่")   # text after this is not a quantity


def _drop_guessed_qty(items: list[ExtractedItem], text: str, draft: DraftOrder) -> None:
    """The model writes qty 1 even when no number was given. For a new item whose name is in
    the message, look for a number between the name and the next item name; none -> None."""
    if "แทน" in text:
        return                        # "เอา X แทน": a replacement keeps the old quantity
    menu = catalog()
    names = {n: x for x in menu.values() for n in [x.name, *x.aliases]}
    hits = sorted((text.find(n), n) for n in names if n in text)
    ordered = {l.name for l in draft.items if not _qty_unknown(draft, l)}
    asked = [c[len("item:"):] for c in draft.missing if c.startswith("item:")]
    for it in items:
        if it.qty in (None, 0) or it.text in ordered or it.text not in names:
            continue
        if any(a and a in it.text for a in asked):
            continue                  # "ชา 2" -> "ชาไทย": the number came with the question
        own = {names[it.text].name, *names[it.text].aliases}
        pos = [(p, n) for p, n in hits if n in own]
        if not pos:
            continue                  # name written differently: trust the model
        start = pos[0][0] + len(max((n for p, n in pos if p == pos[0][0]), key=len))
        end = min([p for p, n in hits if p >= start and n not in own] + [len(text)])
        segment = _NOT_ITEM.split(text[start:end])[0]
        if pos[0][0] == hits[0][0]:   # first item: "เอา 2 กะเพรา" puts the number before it
            segment += " " + text[:pos[0][0]]
        if not _QTY.search(segment):
            it.qty = None


_DECIMAL = r"\s*(?:x|จำนวน)?\s*\d*\.\d+"


def _drop_decimal_qty(items: list[ExtractedItem], text: str) -> None:
    """"ชาไทย 1.5 แก้ว" was ordered as 1 without asking; a decimal means: ask how many."""
    menu = catalog()
    for it in items:
        m = next((x for x in menu.values() if it.text in (x.name, *x.aliases)), None)
        names = [m.name, *m.aliases] if m else [it.text]
        if any(re.search(re.escape(n) + _DECIMAL, text) for n in names):
            it.qty = None


_ANY_IS_FINE = re.compile(r"(?:อะ)?ไร(?:ก็|ก้)ได้|แล้วแต่ร้าน")
_PHOTO_ASK = re.compile(r"ในรูป|จากรูป|รูปนี้|รูปที่ส่ง|รูปเมื่อกี้")
# a dish the newest message does not name keeps its line, unless the customer is changing
# the order in a way that needs no dish name ("เอาแค่ชาไทย", "เปลี่ยนเป็น...")
_KEEP_UNLESS = (*CHANGE_WORDS, "แค่", "เหลือ", "เฉพาะ")


def _sku_of(text: str) -> Optional[str]:
    return next((sku for sku, x in catalog().items() if text in (x.name, *x.aliases)), None)


def _keep_unmentioned(order: ExtractedOrder, draft: DraftOrder, text: str) -> list[str]:
    """The model may only change dishes the newest message names. After "ข้าวผัดเหลือ 1",
    an address message brought back qty 2 from history; "เอาไรก็ได้" dropped all lines but
    ข้าวสวย. Lines without a known qty are left to the model (the customer may answer "2").
    Changes order.items in place; returns notes for the "extract" log."""
    said = _squash(text)
    may_drop = any(w in said for w in _KEEP_UNLESS)
    if len(draft.items) == 1 and _bare_qty(text):
        return []                     # "2" / "เอา0อัน" with one dish: that dish's new quantity
    notes = []
    for line in draft.items:
        if _qty_unknown(draft, line) or mentioned(line.name, said):
            continue
        same = [it for it in order.items if _sku_of(it.text) == line.sku]
        for it in same:
            if it.qty != line.qty and not (it.qty == 0 and may_drop):
                notes.append(f"restored {line.name} qty {it.qty} -> {line.qty}")
                it.qty = line.qty
        if not same and not may_drop:
            notes.append(f"restored {line.name} x{line.qty}")
            order.items.append(ExtractedItem(text=line.name, qty=line.qty, options=line.options,
                                             addons=line.addons, note=line.note))
    return notes


def _replace_sold_out(order: ExtractedOrder, draft: DraftOrder, text: str) -> list[str]:
    """"ชาเย็น หมด, แนะนำ ชาไทย" -> "ชาไทย 1 ก็ได้": the customer took the other dish, but the
    model kept ชาเย็น too and the bot said "ชาเย็น หมด" again. A sold-out dish the newest
    message does not name leaves when it names another dish of the same category."""
    menu, left, said = catalog(), stock(), _squash(text)
    notes = []
    for code in draft.missing:
        sku = code[len("stock:"):] if code.startswith("stock:") else None
        if sku not in menu or left.get(sku, 0) > 0 or mentioned(menu[sku].name, said):
            continue
        others = [s for s in map(_sku_of, (it.text for it in order.items))
                  if s and s != sku and menu[s].category == menu[sku].category
                  and mentioned(menu[s].name, said)]
        if others:
            order.items[:] = [it for it in order.items if _sku_of(it.text) != sku]
            notes.append(f"{menu[sku].name} sold out, replaced by {menu[others[0]].name}")
    return notes


def _sort_options(m, it: ExtractedItem) -> tuple[dict, list, str, list]:
    """Put each value where the dish has it. The vision model wrote "น้ำใส" as an addon and
    the customer read "'ต้มยำกุ้ง' ไม่มีตัวเลือก น้ำใส (เลือกได้: ... น้ำใส/น้ำข้น)"; it also wrote
    {"addon": "ไข่ดาว"}. A value under a key the dish does not have ({"note": "ไม่ต้องหุง"})
    is a free request, not an error. Returns options, addons, note, bad values."""
    options, addons, notes, bad = {}, [], [it.note], []
    for key, value in [*it.options.items(), *((None, a) for a in it.addons)]:
        own = next((k for k, vals in m.options.items() if value in vals), None)
        if own and own not in options:
            options[own] = value
        elif value in m.addons:
            if value not in addons:
                addons.append(value)
        elif key is not None and key not in m.options:
            notes.append(value)
        else:
            bad.append(value)
    return options, addons, " ".join(n for n in notes if n).strip(), bad


def _item_rows(draft: DraftOrder, skip_stock: bool = False) -> list[str]:
    """"- ข้าวผัดกุ้ง x1" for every line; a sold-out line is already listed as a problem."""
    rows = []
    for l in draft.items:
        if skip_stock and f"stock:{l.sku}" in draft.missing:
            continue
        qty = "(ยังไม่ระบุจำนวน)" if _qty_unknown(draft, l) else f"x{l.qty}"
        rows.append(f"- {l.name} {qty}")
    return rows


def _bare_qty(text: str) -> bool:
    """A short message with a quantity and no address or phone ("2", "เพิ่มเป็น 3 จาน")."""
    return len(_squash(text)) <= 20 and not _NOT_ITEM.search(text) and bool(_QTY.search(text))


def _confirm_typo(text: str) -> bool:
    """"ยืรยัน" is close to ยืนยัน but is not a confirm word (M19), so it gets a hint."""
    t = _squash(text)
    return 3 <= len(t) <= 8 and t != "ยืนยัน" and \
        SequenceMatcher(None, t, "ยืนยัน").ratio() >= 0.6


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _option_problem(m, bad: list[str]) -> str:
    """Readable for the customer: values joined, and what the dish does offer."""
    allowed = ["/".join(v) for v in m.options.values()] + [f"+{a}" for a in m.addons]
    tail = f" (เลือกได้: {', '.join(allowed)})" if allowed else " (เมนูนี้ไม่มีตัวเลือกเพิ่ม)"
    return f"'{m.name}' ไม่มีตัวเลือก {', '.join(bad)}{tail}"


def _qty_unknown(draft: DraftOrder, line: OrderLine) -> bool:
    """The line's qty is a placeholder: the customer has not said how many yet."""
    return f"qty:{line.sku}" in draft.missing
