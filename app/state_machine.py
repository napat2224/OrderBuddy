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
from dataclasses import dataclass
from typing import Optional

from harness import Harness, LLMError, ask_structured, build_messages, new_run_id

from . import router, tools
from .agent import call_tool, resolve_with_code, resolve_with_model
from .catalog import catalog, menu_text, stock
from .checks import clean_reply, extract_check
from .schemas import DraftOrder, ExtractedItem, ExtractedOrder, OrderLine, RouteResult
from .session import SessionStore

READ = frozenset({"read"})
READ_WRITE = frozenset({"read", "write"})

MSG = {
    "greet": "สวัสดีครับ พิมพ์เมนูและจำนวนที่ต้องการ หรือส่งรูปรายการอาหารมาได้เลยครับ",
    "error": "ขออภัยครับ ระบบขัดข้องชั่วคราว รบกวนส่งข้อความอีกครั้ง หรือรอพนักงานติดต่อกลับครับ",
    "not_understood": "ขอโทษครับ ยังไม่เข้าใจรายการ รบกวนพิมพ์ชื่อเมนูและจำนวนอีกครั้งได้ไหมครับ",
    "ask_items": "ต้องการสั่งเมนูอะไรครับ",
    "cancelled": "ยกเลิกออร์เดอร์ {order_id} แล้วครับ",
    "nothing_to_cancel": "ตอนนี้ยังไม่มีออร์เดอร์ที่ต้องยกเลิกครับ",
    "submitted": "รับออร์เดอร์ {order_id} แล้วครับ รอร้านยืนยันสักครู่ จะแจ้งกลับทาง LINE นี้ครับ",
    "submit_failed": "ขออภัยครับ ยังส่งออร์เดอร์ไม่ได้ ({reason}) รบกวนตรวจสอบรายการอีกครั้งครับ",
}

PROBLEM_TEXT = {
    "fulfillment": "ยังไม่ทราบว่าจะรับเองที่ร้านหรือให้ไปส่ง",
    "phone": "ยังไม่มีเบอร์โทรที่ถูกต้อง (10 หลัก)",
    "address": "ยังไม่มีที่อยู่จัดส่ง (บ้านเลขที่)",
    "postcode": "ที่อยู่ยังไม่มีรหัสไปรษณีย์",
    "zone": "ที่อยู่อยู่นอกพื้นที่จัดส่ง",
}


@dataclass
class AgentConfig:
    """Harness configs compared in the eval (Part C). C1 = single-model baseline,
    built in eval/ because it skips this state machine."""
    name: str = "C3"
    use_router: bool = True          # False: every step uses the reasoning profile
    validate_retry: bool = True      # False: one attempt, no repair, no escalation
    resolve_with_model: bool = True  # False: code matching only
    max_tool_rounds: int = 4


CONFIGS = {
    "C2": AgentConfig(name="C2", validate_retry=False),
    "C3": AgentConfig(name="C3"),
}


class OrderAgent:
    def __init__(self, h: Optional[Harness] = None, store: Optional[SessionStore] = None,
                 config: Optional[AgentConfig] = None):
        self.h = h or Harness()
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
                                 state=draft.state, error=f"{type(e).__name__}: {e}")
                self._log_state(run_id, draft.state, "ERROR")
                return [MSG["error"]]

            if closed:                       # submitted or cancelled
                self.store.clear_history(line_user_id)
            else:
                if draft.items or draft.state != "IDLE":
                    self.store.save_draft(draft)   # greetings/questions leave no empty draft
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
            return [MSG["cancelled"].format(order_id=draft.order_id)], True

        if intent == "confirm":
            if prev == "CONFIRM" and not draft.missing:
                return self._done(run_id, draft)
            if not draft.items:
                draft.state = "IDLE"
                return [MSG["greet"]], False
            intent = "modify_order"          # "ok" as an answer to a question

        if intent == "question":
            draft.state = prev                # a question never moves the order
            return [self._answer(run_id, text)], False

        if intent == "other" and not draft.items:
            draft.state = "IDLE"
            return [MSG["greet"]], False

        if self.cfg.use_router:
            profile = router.pick_profile(route, bool(image_b64))
        else:
            profile = "vision" if image_b64 else "reasoning"
        return self._order_flow(run_id, draft, history, text, image_b64, profile), False

    # ======================================== EXTRACT -> RESOLVE -> PRICE -> ADDRESS
    def _order_flow(self, run_id, draft, history, text, image_b64, profile) -> list[str]:
        self._goto(run_id, draft, "EXTRACT")
        ext = self._extract(run_id, draft, history, text, image_b64, profile)
        if ext is None:                       # safe failure after retries
            self._goto(run_id, draft, "CLARIFY")
            return [MSG["not_understood"]]
        self._merge_customer(draft, ext)
        if not ext.items:
            draft.items, draft.missing = [], ["items"]
            self._goto(run_id, draft, "CLARIFY")
            return [MSG["ask_items"]]

        # ---- RESOLVE: model picks skus with tools, code re-checks everything
        self._goto(run_id, draft, "RESOLVE")
        mapping = None
        if self.cfg.resolve_with_model:
            mapping = resolve_with_model(self.h, ext.items, run_id=run_id,
                                         max_rounds=self.cfg.max_tool_rounds)
        if mapping is None:
            mapping = resolve_with_code(self.h, ext.items, run_id=run_id)
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
                if need_addr and not res["in_zone"]:
                    missing.append("zone")
                if res["normalized"]["phone"] and "phone" not in res["missing"]:
                    draft.customer.phone = res["normalized"]["phone"]
                draft.customer.address_valid = need_addr and res["valid"]
            problems += [PROBLEM_TEXT[m] for m in missing if m in PROBLEM_TEXT]

        draft.missing = missing
        if missing:
            self._goto(run_id, draft, "CLARIFY")
            return [self._clarify(run_id, draft, problems, suggestions)]
        self._goto(run_id, draft, "CONFIRM")
        return [self._summary(draft)]

    # --------------------------------------------------------------- EXTRACT
    def _extract(self, run_id, draft, history, text, image_b64, profile):
        current = json.dumps({
            "items": [{"text": l.name, "qty": l.qty, "options": l.options, "addons": l.addons}
                      for l in draft.items],
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
        return res.value if res.ok else None

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
        for i, it in enumerate(items):
            sku = mapping.get(i)
            if sku not in menu:                       # model answer re-checked by code
                missing.append(f"item:{it.text}")
                problems.append(f"ไม่พบเมนู '{it.text}'")
                r = call_tool(self.h, "product_search", {"query": it.text, "top_k": 3},
                              run_id=run_id)
                suggestions += [x["name"] for x in r.get("result") or []]
                continue
            m = menu[sku]
            bad_opts = {k: v for k, v in it.options.items() if v not in m.options.get(k, [])}
            bad_addons = [a for a in it.addons if a not in m.addons]
            if bad_opts or bad_addons:
                missing.append(f"option:{sku}")
                problems.append(f"'{m.name}' ไม่มีตัวเลือก {list(bad_opts.values()) + bad_addons}")
            line = OrderLine(sku=sku, name=m.name, qty=it.qty, note=it.note,
                             options={k: v for k, v in it.options.items() if k not in bad_opts},
                             addons=[a for a in it.addons if a in m.addons])
            lines.append(line)
            r = call_tool(self.h, "stock_checker", {"sku": sku, "qty": it.qty}, run_id=run_id)
            if not r["ok"] or not r["result"]["enough"]:
                left = r["result"]["available"] if r["ok"] else 0
                missing.append(f"stock:{sku}")
                problems.append(f"'{m.name}' เหลือ {left} ที่" if left else f"'{m.name}' หมด")
                suggestions += [x.name for x in menu.values()
                                if x.category == m.category and x.sku != sku
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
    def _clarify(self, run_id, draft, problems, suggestions) -> str:
        fallback = "ขอข้อมูลเพิ่มนิดนึงครับ: " + "; ".join(problems)
        if suggestions:
            fallback += " (แนะนำ: " + ", ".join(suggestions) + ")"
        p = self.h.prompts.get("clarify")
        system = p.render(order=", ".join(f"{l.name} x{l.qty}" for l in draft.items) or "none",
                          problems="; ".join(problems), suggestions=", ".join(suggestions) or "none")
        return self._say(run_id, "reasoning", p.id, system, "เขียนข้อความถึงลูกค้า", fallback)

    def _answer(self, run_id, text) -> str:
        p = self.h.prompts.get("answer")
        return self._say(run_id, "fast", p.id, p.render(menu=menu_text()), text,
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
            return [MSG["submit_failed"].format(reason=r["error"][:80])], False
        self._log_state(run_id, "DONE", "STAFF_REVIEW")
        return [MSG["submitted"].format(order_id=r["result"]["order_id"])], True

    # ------------------------------------------------------------- helpers
    def _goto(self, run_id, draft: DraftOrder, state: str) -> None:
        self._log_state(run_id, draft.state, state)
        draft.state = state

    def _log_state(self, run_id, src, dst) -> None:
        self.h.log.write("state", run_id=run_id, src=src, dst=dst)

