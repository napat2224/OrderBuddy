"""Offline tests for the whole order flow: no Ollama needed.

A FakeClient plays the models. It answers by looking at what is asked
(schema name, tools, prompt), so the real state machine, tools, executor,
validation and session store all run for real.

    python -m unittest discover -s tests -v        (from the OrderBuddy folder)
"""
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  (puts Harness/ on sys.path)
from harness import ChatResult, Harness, LLMConnectionError  # noqa: E402

from app.session import SessionStore  # noqa: E402
from app import catalog  # noqa: E402
from app.debug import format_event, last_run_ids  # noqa: E402
from app.state_machine import CONFIGS, OrderAgent  # noqa: E402
from app.trace import TracedHarness  # noqa: E402

UID = "U" + "0" * 32
NAME_TO_SKU = {"กะเพราหมูกรอบ": "F001", "ข้าวผัดกุ้ง": "F002", "ต้มยำกุ้ง": "F003",
               "ข้าวสวย": "F004", "ชาเย็น": "F005", "ชาไทย": "F006",
               "ชา": "F006",      # unclear name: the model picks one of the two teas
               "กุ้ง": "F004"}    # unclear name: the model answers with an item that does not fit


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
    def setUp(self):
        # confirming an order writes stock.json: work on a copy of data/
        data = Path(tempfile.mkdtemp())
        shutil.copytree(catalog.DATA_DIR, data, dirs_exist_ok=True)
        self.addCleanup(setattr, catalog, "DATA_DIR", catalog.DATA_DIR)
        catalog.DATA_DIR = data

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
        self.assertEqual((saved["status"], saved["total"]), ("confirmed", 170))
        self.assertIn("รวม 170 บาท", r)
        self.assertEqual(catalog.stock()["F001"], 18)      # 20 - 2, cut when confirmed

    def test_no_stock_is_cut_before_confirm(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        self.assertEqual(catalog.stock()["F001"], 20)

    def test_sold_out_while_waiting_for_confirm(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        catalog._save_stock({**catalog.stock(), "F001": 1})
        self.assertIn("เพิ่งหมด", a.handle(UID, "ยืนยัน")[0])
        self.assertEqual(catalog.stock()["F001"], 1)       # nothing cut
        self.assertFalse(self.store.orders_dir.exists())   # nothing saved
        self.assertEqual(self.store.get_draft(UID).state, "CONFIRM")   # the customer can still edit

    def test_stock_is_all_or_nothing(self):
        before = catalog.stock()
        self.assertEqual(catalog.take_stock({"F001": 1, "F005": 1}), ["F005"])   # F005 is sold out
        self.assertEqual(catalog.stock(), before)

    def test_stock_returns_when_the_order_cannot_be_saved(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        order_id = self.store.get_draft(UID).order_id
        self.store.orders_dir.mkdir(parents=True)
        (self.store.orders_dir / f"{order_id}.json").write_text("{}", encoding="utf-8")
        a.handle(UID, "ยืนยัน")
        self.assertEqual(catalog.stock()["F001"], 20)

    def test_models_used_per_state(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        a.handle(UID, MSG_ORDER)
        used = {(t or ("tools" if tools else "text")): m for m, t, tools in self.client.calls}
        model = a.h.config.profile          # read config.yaml so swapping models keeps tests green
        self.assertEqual(used["RouteResult"], model("router").model)
        self.assertEqual(used["ExtractedOrder"], model("fast").model)
        self.assertNotIn("tools", used)     # a clear name needs no tool-calling loop
        names = [e["dst"] for e in self.events("state")]
        self.assertEqual(names, ["ROUTE", "EXTRACT", "RESOLVE", "PRICE", "ADDRESS", "CONFIRM"])

    def test_input_and_reply_logged(self):
        a = self.make({MSG_ORDER: GOOD_ORDER})
        replies = a.handle(UID, MSG_ORDER)
        msg, reply = self.events("message")[0], self.events("reply")[0]
        self.assertEqual(msg["text"], MSG_ORDER)
        self.assertEqual(reply["run_id"], msg["run_id"])
        self.assertEqual((reply["state"], reply["replies"]), ("CONFIRM", replies))


class TestDebugLog(Base):
    """The log must be enough to see why a customer got a given answer."""

    def traced(self, client):
        self.log = Path(tempfile.mkdtemp()) / "calls.jsonl"
        h = TracedHarness(client=client, log_file=self.log)
        return OrderAgent(h, SessionStore(self.log.parent), CONFIGS["C3"])

    def test_raw_reply_result_and_customer_reply_are_logged(self):
        menu_copy = order([("กะเพราหมูกรอบ", 2, {"addons": ["ไข่เจียว +15"]})])
        m = "ขอกะเพราหมูกรอบ 2"
        replies = self.traced(FakeClient({m: menu_copy})).handle(UID, m)
        raw = [e for e in self.events("llm_trace") if "ไข่เจียว" in (e.get("reply") or "")]
        self.assertEqual(len(raw), 1)                     # the model's words, before checks
        extract = self.events("extract")[0]
        self.assertEqual(extract["result"]["items"][0]["addons"], [])
        self.assertEqual(extract["dropped"], ["กะเพราหมูกรอบ addon ไข่เจียว +15"])
        self.assertEqual(self.events("reply")[0]["replies"], replies)

    def test_model_failure_is_logged_with_traceback(self):
        class Down(FakeClient):
            def chat(self, model, messages, schema=None, tools=None, **kw):
                if (schema or {}).get("title") == "ExtractedOrder":
                    raise LLMConnectionError("Is Ollama running?")
                return super().chat(model, messages, schema=schema, tools=tools, **kw)

        a = self.traced(Down({}))
        with self.assertLogs("orderbuddy", level="ERROR"):
            a.handle(UID, "ขอกะเพราหมูกรอบ 2")
        self.assertIn("Is Ollama running?", self.events("llm_trace")[-1]["error"])
        error = self.events("agent_error")[0]
        self.assertIn("LLMConnectionError", error["trace"])
        self.assertIn("LLMConnectionError", format_event(error))

    def test_every_step_reaches_the_readable_log(self):
        m = "ขอกะเพราหมูกรอบ 2"
        a = self.traced(FakeClient({m: order([("กะเพราหมูกรอบ", 2, {})])}))
        with self.assertLogs("orderbuddy", level="INFO") as seen:
            a.handle(UID, m)
        text = "\n".join(seen.output)
        for event in ("message", "route", "state", "llm_trace", "extract", "tool_call", "reply"):
            self.assertIn(f"] {event} ", text)
        self.assertNotIn("] llm_call ", text)             # valid ones only repeat llm_trace

    def test_last_run_ids_only_counts_customer_messages(self):
        events = [{"event": "message", "run_id": "a"}, {"event": "llm_call", "run_id": "x"},
                  {"event": "message", "run_id": "b"}]
        self.assertEqual(last_run_ids(events, 1), ["b"])
        self.assertEqual(last_run_ids(events, 5), ["a", "b"])


class TestResolve(Base):
    def lines(self, text, qty=1, **kw):
        m = f"ขอ{text} {qty} รับเอง โทร 0812345678"
        a = self.make({m: order([(text, qty, {})], phone="0812345678", fulfillment="pickup")}, **kw)
        replies = a.handle(UID, m)
        return self.store.get_draft(UID), replies

    def model_called(self):
        return any(tools for _, _, tools in self.client.calls)

    def test_clear_names_need_no_model(self):
        for text, sku in [("ข้าวผัดกุ้ง", "F002"), ("กระเพราหมูกรอบ", "F001"),      # exact, alias
                          ("กะเพรา", "F001"), ("ชาเยน", "F005"), ("ข้าวผัดกุ้งสด", "F002")]:
            with self.subTest(text):
                d, _ = self.lines(text, qty=1)
                self.assertEqual([l.sku for l in d.items], [sku])
                self.assertFalse(self.model_called())

    def test_other_dishes_are_not_swapped(self):
        for text in ["ผัดไทยกุ้งสด", "ข้าวผัดหมู", "ข้าวผัดปู", "ต้มยำไก่", "ต้มข่ากุ้ง", "กะเพราไก่",
                     "พิซซ่า"]:
            with self.subTest(text):
                d, _ = self.lines(text)
                self.assertEqual(d.items, [])
                self.assertEqual(d.missing, [f"item:{text}"])
                self.assertFalse(self.model_called())

    def test_sold_out_is_not_called_unknown(self):
        d, _ = self.lines("ชาเย็น", qty=2)            # stock of F005 is 0
        self.assertEqual(d.missing, ["stock:F005"])
        self.assertEqual([l.sku for l in d.items], ["F005"])

    def test_unclear_name_goes_to_the_model(self):
        d, _ = self.lines("ชา")
        self.assertTrue(self.model_called())
        self.assertEqual([l.sku for l in d.items], ["F006"])
        self.assertEqual(self.events("resolve")[0]["unclear"], ["ชา"])

    def test_model_answer_outside_the_matches_is_rejected(self):
        d, _ = self.lines("กุ้ง")                      # fits F002 and F003, model says F004
        self.assertEqual(d.items, [])
        self.assertEqual(d.missing, ["item:กุ้ง"])
        self.assertEqual(self.events("resolve")[0]["rejected"], ["กุ้ง: F004"])

    def test_unclear_name_without_the_model_asks_the_customer(self):
        m = "ชา 1"
        a = self.make({m: order([("ชา", 1, {})])})
        a.cfg = replace(a.cfg, resolve_with_model=False)   # CONFIGS is shared between tests
        a.handle(UID, m)
        self.assertEqual(self.store.get_draft(UID).missing, ["item:ชา"])
        self.assertFalse(self.model_called())


class TestClarifyMessage(Base):
    def test_each_problem_is_stated_without_a_model(self):
        m = "ข้าวผัดกุ้ง 1 ส่ง 12/3 ซอยประชา 12312 โทร 099999999"
        a = self.make({m: order([("ข้าวผัดกุ้ง", 1, {})], phone="099999999",
                                address="12/3 ซอยประชา 12312", fulfillment="delivery")})
        reply = a.handle(UID, m)[0]
        self.assertIn("เบอร์โทร", reply)
        self.assertNotIn("รหัสไปรษณีย์", reply)         # the customer did send one
        self.assertTrue(reply.endswith("ครับ"))
        self.assertFalse([c for c in self.client.calls if c[1] is None and not c[2]])

    def test_any_postcode_can_get_delivery(self):
        m = "ข้าวผัดกุ้ง 1 ส่ง 12/3 ซอยประชา 99999 โทร 0812345678"
        a = self.make({m: order([("ข้าวผัดกุ้ง", 1, {})], phone="0812345678",
                                address="12/3 ซอยประชา 99999", fulfillment="delivery")})
        self.assertIn("รวม 100 บาท", a.handle(UID, m)[0])
        d = self.store.get_draft(UID)
        self.assertEqual((d.state, d.missing), ("CONFIRM", []))
        self.assertTrue(d.customer.address_valid)

    def test_suggestions_are_listed(self):
        m = "ขอผัดไทยกุ้งสด 1"
        reply = self.make({m: order([("ผัดไทยกุ้งสด", 1, {})])}).handle(UID, m)[0]
        self.assertIn("ไม่พบเมนู 'ผัดไทยกุ้งสด'", reply)
        self.assertIn("เมนูที่แนะนำ: ข้าวผัดกุ้ง", reply)


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

    def test_clarify_never_uses_model_wording(self):
        # a model that says the wrong drink is out is never asked: code writes the message
        m = "ชาเย็น 1 รับเอง โทร 0899999999"
        a = self.make({m: order([("ชาเย็น", 1, {})], phone="0899999999", fulfillment="pickup")},
                      say="ขออภัยครับ ชาไทยหมด แนะนำชาไทยครับ")
        r = a.handle(UID, m)[0]
        self.assertIn("'ชาเย็น' หมด", r)
        self.assertIn("แนะนำ: ชาไทย", r)
        self.assertNotIn("ชาไทยหมด", r)

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
        a = self.make({"ชา 1": order([("ชา", 1, {})])}, write_tool_attack=True)
        a.handle(UID, "ชา 1")                     # an unclear name, so the model loop runs
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

    def test_invented_phone_is_repaired(self):
        m = "ข้าวผัดกุ้ง 1 รับเอง โทร 0812345678"
        bad = order([("ข้าวผัดกุ้ง", 1, {})], phone="0899999999", fulfillment="pickup")
        good = order([("ข้าวผัดกุ้ง", 1, {})], phone="0812345678", fulfillment="pickup")
        a = self.make({m: [json.dumps(bad), json.dumps(good)]})
        self.assertIn("โทร: 0812345678", a.handle(UID, m)[0])
        self.assertTrue(any("not in the customer" in (e.get("error") or "")
                            for e in self.events("llm_call")))


if __name__ == "__main__":
    unittest.main()
