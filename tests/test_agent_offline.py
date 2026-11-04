"""Offline tests for the whole order flow: no Ollama needed.

A FakeClient plays the models. It answers by looking at what is asked
(schema name, tools, prompt), so the real state machine, tools, executor,
validation and session store all run for real.

    python -m unittest discover -s tests -v        (from the OrderBuddy folder)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  (puts Harness/ on sys.path)
from harness import ChatResult, Harness  # noqa: E402

from app.session import SessionStore  # noqa: E402
from app.state_machine import CONFIGS, OrderAgent  # noqa: E402

UID = "U" + "0" * 32
NAME_TO_SKU = {"กะเพราหมูกรอบ": "F001", "ข้าวผัดกุ้ง": "F002", "ต้มยำกุ้ง": "F003",
               "ข้าวสวย": "F004", "ชาเย็น": "F005", "ชาไทย": "F006"}


def _text(msg):
    return msg.get("content", "") if msg else ""


class FakeClient:
    """extract: {user text: ExtractedOrder dict or list of raw outputs (one per attempt)}"""

    def __init__(self, extract, write_tool_attack=False, say="ข้อความจากโมเดลครับ",
                 resolve_null=(), route_confirm=()):
        self.extract = extract
        self.route_confirm = route_confirm        # messages the router wrongly labels "confirm"
        self.resolve_null = resolve_null          # item names the model "fails" to match
        self.write_tool_attack = write_tool_attack
        self.say = say                            # clarify / answer text
        self.calls = []

    def chat(self, model, messages, schema=None, tools=None, **kw):
        title = (schema or {}).get("title")
        self.calls.append((model, title, bool(tools)))
        system = _text(messages[0])
        user = next(_text(m) for m in reversed(messages) if m["role"] == "user")

        if title == "RouteResult":
            if "ยืนยัน" in user or user in self.route_confirm:
                intent = "confirm"
            elif "ยกเลิก" in user:
                intent = "cancel"
            elif "มีเมนู" in user:
                intent = "question"
            elif "Open order: none" in system:
                intent = "new_order"
            else:
                intent = "modify_order"
            return self._out(model, {"intent": intent, "complexity": "simple"})

        if title == "ExtractedOrder":
            # on a repair attempt the newest user message is the harness error note,
            # so look for the customer's message further back
            key = next(_text(m) for m in reversed(messages)
                       if m["role"] == "user" and _text(m) in self.extract)
            out = self.extract[key]
            if isinstance(out, list):            # scripted attempts: bad then good
                out = out.pop(0)
            return self._out(model, out)

        if tools:                                 # RESOLVE tool loop
            if messages[-1]["role"] == "tool":    # tools already ran: stop calling
                return self._out(model, "done")
            items = [l.split(". ", 1)[1].rsplit(" x", 1)[0]
                     for l in _text(messages[-1]).splitlines()]
            calls = [{"function": {"name": "product_search", "arguments": {"query": t}}}
                     for t in items]
            if self.write_tool_attack:            # model tries to submit by itself
                calls.append({"function": {"name": "draft_order_manager",
                                           "arguments": {"action": "submit",
                                                         "line_user_id": UID}}})
            return ChatResult(model=model, content="", tool_calls=calls,
                              message={"role": "assistant", "content": "", "tool_calls": calls})

        if title == "ResolveResult":
            listing = next(_text(m) for m in messages if m["role"] == "user")
            items = [l.split(". ", 1)[1].rsplit(" x", 1)[0] for l in listing.splitlines()]
            return self._out(model, {"items": [
                {"index": i, "sku": None if t in self.resolve_null else NAME_TO_SKU.get(t)}
                for i, t in enumerate(items)]})

        return self._out(model, self.say)

    @staticmethod
    def _out(model, content):
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return ChatResult(model=model, content=content,
                          message={"role": "assistant", "content": content}, wall_s=0.01)


def order(items, phone=None, address=None, fulfillment=None):
    return {"items": [{"text": t, "qty": q, **extra} for t, q, extra in items],
            "customer": {"name": None, "phone": phone, "address": address},
            "fulfillment": fulfillment}


ADDR = "99/1 ถ.สุขุมวิท 10110"
MSG_ORDER = "ขอกะเพราหมูกรอบ 2 เผ็ดน้อย ไข่ดาว ส่ง 99/1 ถ.สุขุมวิท 10110 โทร 0812345678"
GOOD_ORDER = order([("กะเพราหมูกรอบ", 2, {"options": {"spice": "เผ็ดน้อย"}, "addons": ["ไข่ดาว"]})],
                   phone="0812345678", address=ADDR, fulfillment="delivery")


class Base(unittest.TestCase):
    def make(self, extract, config="C3", **kw):
        tmp = Path(tempfile.mkdtemp())
        self.client = FakeClient(extract, **kw)
        h = Harness(client=self.client, log_file=tmp / "calls.jsonl")
        self.store = SessionStore(tmp)
        self.log = tmp / "calls.jsonl"
        return OrderAgent(h, self.store, CONFIGS[config])

    def events(self, name):
        rows = [json.loads(l) for l in self.log.read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if r["event"] == name]


class TestHappyPath(Base):
    def test_order_confirm_submit(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        r = a.handle(UID, MSG_ORDER)[0]
        self.assertIn("รวม 170 บาท", r)            # (60 + 10 ไข่ดาว) x 2 + 30 delivery
        self.assertEqual(self.store.get_draft(UID).state, "CONFIRM")

        r = a.handle(UID, "ยืนยัน")[0]
        self.assertIn("รับออร์เดอร์", r)
        self.assertIsNone(self.store.get_draft(UID))
        orders = list((self.store.orders_dir).glob("*.json"))
        self.assertEqual(len(orders), 1)
        saved = json.loads(orders[0].read_text(encoding="utf-8"))
        self.assertEqual((saved["status"], saved["total"]), ("pending_staff", 170))

    def test_models_used_per_state(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        used = {(t or ("tools" if tools else "text")): m for m, t, tools in self.client.calls}
        self.assertEqual(used["RouteResult"], "qwen3:4b-q4_K_M")
        self.assertEqual(used["ExtractedOrder"], "qwen3:4b-q4_K_M")
        self.assertEqual(used["tools"], "qwen3:4b-q4_K_M")
        names = [e["dst"] for e in self.events("state")]
        self.assertEqual(names, ["ROUTE", "EXTRACT", "RESOLVE", "PRICE", "ADDRESS", "CONFIRM"])

    def test_input_and_reply_logged(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        replies = a.handle(UID, MSG_ORDER)
        msg, reply = self.events("message")[0], self.events("reply")[0]
        self.assertEqual(msg["text"], MSG_ORDER)
        self.assertEqual(reply["run_id"], msg["run_id"])
        self.assertEqual((reply["state"], reply["replies"]), ("CONFIRM", replies))


class TestClarify(Base):
    def test_out_of_stock_then_replace(self):
        m1, m2 = "ชาเย็น 1 รับเอง โทร 0899999999", "เอาชาไทยแทน"
        a = self.make({m1: order([("ชาเย็น", 1, {})], phone="0899999999", fulfillment="pickup"),
                       m2: order([("ชาไทย", 1, {})], phone="0899999999", fulfillment="pickup")})
        a.handle(UID, m1)
        d = self.store.get_draft(UID)
        self.assertEqual(d.state, "CLARIFY")
        self.assertIn("stock:F005", d.missing)
        r = a.handle(UID, m2)[0]
        self.assertIn("รวม 35 บาท", r)
        self.assertIn("รับเองที่ร้าน", r)

    def test_clarify_reply_that_swaps_item_is_replaced(self):
        # the model says the wrong drink is out: code sends its own template instead
        m = "ชาเย็น 1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาเย็น", 1, {})], phone="0899999999", fulfillment="pickup")},
                      say="ขออภัยครับ ชาไทยหมด แนะนำชาไทยครับ")
        r = a.handle(UID, m)[0]
        self.assertIn("'ชาเย็น' หมด", r)
        self.assertIn("แนะนำ: ชาไทย", r)
        self.assertEqual(len(self.events("safe_failure")), 1)

    def test_clarify_reply_with_right_item_is_kept(self):
        m = "ชาเย็น 1 รับเอง โทร 0899999999"
        good = "ขออภัยครับ ชาเย็นหมดแล้ว รับเป็นชาไทยแทนไหมครับ"
        a = self.make({m: order([("ชาเย็น", 1, {})], phone="0899999999", fulfillment="pickup")},
                      say=good)
        self.assertEqual(a.handle(UID, m)[0], good)

    def test_no_suggestion_of_item_already_in_order(self):
        m = "ชาเย็น 2 ชาไทย 1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาเย็น", 2, {}), ("ชาไทย", 1, {})],
                                phone="0899999999", fulfillment="pickup")})
        r = a.handle(UID, m)[0]                   # fallback: FakeClient text names no item
        self.assertIn("'ชาเย็น' หมด", r)
        self.assertNotIn("แนะนำ", r)              # ชาไทย is the only other drink, already ordered

    def test_clarify_reply_suggesting_ordered_item_is_replaced(self):
        m = "ชาเย็น 2 ชาไทย 1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาเย็น", 2, {}), ("ชาไทย", 1, {})],
                                phone="0899999999", fulfillment="pickup")},
                      say="ขออภัยครับ ชาเย็นหมดแล้ว รับเป็นชาไทยแทนไหมครับ")
        r = a.handle(UID, m)[0]
        self.assertNotIn("ชาไทย", r)
        self.assertIn("'ชาเย็น' หมด", r)

    def test_out_of_stock_item_is_not_called_unknown(self):
        # the model answers null for ชาเย็น after seeing it is out of stock
        m = "ชาเย็น 1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาเย็น", 1, {})], phone="0899999999", fulfillment="pickup")},
                      resolve_null=("ชาเย็น",))
        r = a.handle(UID, m)[0]
        self.assertEqual(self.store.get_draft(UID).missing, ["stock:F005"])
        self.assertIn("'ชาเย็น' หมด", r)
        self.assertNotIn("ไม่พบเมนู", r)

    def test_qty_zero_removes_item(self):
        m1, m2 = "ชาไทย 1 รับเอง โทร 0899999999", "แก้ไข เอา0อัน"
        a = self.make({m1: order([("ชาไทย", 1, {})], phone="0899999999", fulfillment="pickup"),
                       m2: order([("ชาไทย", 0, {})], phone="0899999999", fulfillment="pickup")})
        a.handle(UID, m1)
        self.assertEqual(self.store.get_draft(UID).state, "CONFIRM")
        r = a.handle(UID, m2)[0]
        d = self.store.get_draft(UID)
        self.assertEqual((d.items, d.missing), ([], ["items"]))
        self.assertEqual(r, "ต้องการสั่งเมนูอะไรครับ")

    def test_no_number_asks_qty_before_price(self):
        m1, m2 = "ชาไทย รับเอง โทร 0899999999", "2"
        a = self.make({m1: order([("ชาไทย", None, {})], phone="0899999999", fulfillment="pickup"),
                       m2: order([("ชาไทย", 2, {})], phone="0899999999", fulfillment="pickup")})
        r = a.handle(UID, m1)[0]
        d = self.store.get_draft(UID)
        self.assertEqual((d.state, d.missing), ("CLARIFY", ["qty:F006"]))
        self.assertNotIn("รวม", r)                    # not priced with a guessed qty
        self.assertIn("ยังไม่ทราบจำนวน 'ชาไทย'", r)
        r = a.handle(UID, m2)[0]
        self.assertIn("ชาไทย x2 = 70 บาท", r)
        self.assertEqual(self.store.get_draft(UID).state, "CONFIRM")

    def test_model_guessed_qty_is_not_trusted(self):
        # the model writes qty 1 although no number was written: code asks instead
        m = "ชาไทย รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาไทย", 1, {})], phone="0899999999", fulfillment="pickup")})
        a.handle(UID, m)
        self.assertEqual(self.store.get_draft(UID).missing, ["qty:F006"])

    def test_written_qty_is_kept(self):
        m = "เอา 2 ชาไทย กับ ข้าวผัดกุ้ง1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาไทย", 2, {}), ("ข้าวผัดกุ้ง", 1, {})],
                                phone="0899999999", fulfillment="pickup")})
        a.handle(UID, m)
        d = self.store.get_draft(UID)
        self.assertEqual((d.state, [l.qty for l in d.items]), ("CONFIRM", [2, 1]))

    def test_number_answer_is_not_confirm(self):
        # on the summary screen "2" must not submit the order, whatever the router says
        m1 = "ชาไทย 1 รับเอง โทร 0899999999"
        a = self.make({m1: order([("ชาไทย", 1, {})], phone="0899999999", fulfillment="pickup"),
                       "2": order([("ชาไทย", 2, {})], phone="0899999999", fulfillment="pickup")},
                      route_confirm=("2",))
        a.handle(UID, m1)
        r = a.handle(UID, "2")[0]
        self.assertFalse(list(self.store.orders_dir.glob("*.json")))
        self.assertIn("ชาไทย x2", r)

    def test_missing_phone(self):
        m = "ข้าวผัดกุ้ง 1 ส่ง 99/1 ถ.สุขุมวิท 10110"
        a = self.make({m: order([("ข้าวผัดกุ้ง", 1, {})], address=ADDR, fulfillment="delivery")})
        a.handle(UID, m)
        self.assertEqual(self.store.get_draft(UID).missing, ["phone"])

    def test_unknown_item(self):
        m = "ขอพิซซ่า 1"
        a = self.make({m: order([("พิซซ่า", 1, {})])})
        a.handle(UID, m)
        self.assertIn("item:พิซซ่า", self.store.get_draft(UID).missing)


class TestSafety(Base):
    def test_confirm_before_summary_does_not_submit(self):
        m = "ข้าวผัดกุ้ง 1"
        a = self.make({m: order([("ข้าวผัดกุ้ง", 1, {})]), "ยืนยัน": order([("ข้าวผัดกุ้ง", 1, {})])})
        a.handle(UID, m)                          # CLARIFY: no phone, no fulfillment
        a.handle(UID, "ยืนยัน")
        self.assertEqual(list(self.store.orders_dir.glob("*.json")) if
                         self.store.orders_dir.exists() else [], [])

    def test_model_cannot_call_write_tool(self):
        a = self.make({MSG_ORDER: GOOD_ORDER}, write_tool_attack=True)
        a.handle(UID, MSG_ORDER)
        denied = [e for e in self.events("tool_call") if e["tool"] == "draft_order_manager"]
        self.assertTrue(denied and not denied[0]["ok"])
        self.assertIn("Permission denied", denied[0]["error"])
        self.assertFalse(self.store.orders_dir.exists())

    def test_cancel(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        self.assertIn("ยกเลิก", a.handle(UID, "ยกเลิก")[0])
        self.assertIsNone(self.store.get_draft(UID))

    def test_question_keeps_state(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        a.handle(UID, "มีเมนูอะไรบ้าง")
        self.assertEqual(self.store.get_draft(UID).state, "CONFIRM")


class TestReliability(Base):
    def test_c3_recovers_from_bad_json(self):
        a = self.make({MSG_ORDER: ["not json", "{\"items\": 5}", json.dumps(GOOD_ORDER)]})
        self.assertIn("รวม 170 บาท", a.handle(UID, MSG_ORDER)[0])

    def test_c2_has_no_retry(self):
        a = self.make({MSG_ORDER: ["not json", json.dumps(GOOD_ORDER)]}, config="C2")
        self.assertIn("ไม่เข้าใจ", a.handle(UID, MSG_ORDER)[0])


if __name__ == "__main__":
    unittest.main()
