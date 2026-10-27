"""Offline tests for Part B: router, domain checks and prompts. No Ollama needed.

    python -m unittest discover -s tests -v        (from the OrderBuddy folder)
"""
import json
import string
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  (puts Harness/ on sys.path)
from harness import ChatResult, DomainError, Harness, LLMConnectionError  # noqa: E402

from app import router  # noqa: E402
from app.checks import clean_reply, extract_check  # noqa: E402
from app.schemas import Customer, DraftOrder, ExtractedOrder, OrderLine, RouteResult  # noqa: E402

UID = "U" + "0" * 32


class ScriptedClient:
    """Returns the scripted outputs in order; an Exception in the script is raised."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat(self, model, messages, schema=None, tools=None, **kw):
        self.calls += 1
        out = self.outputs.pop(0) if self.outputs else "not json"
        if isinstance(out, Exception):
            raise out
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False)
        return ChatResult(model=model, content=out,
                          message={"role": "assistant", "content": out}, wall_s=0.01)


def draft(*items):
    """A draft holding (name, sku, qty) lines."""
    return DraftOrder(order_id="ORD-TEST", line_user_id=UID,
                      items=[OrderLine(sku=s, name=n, qty=q) for n, s, q in items])


EMPTY = draft()
OPEN = draft(("ชาเย็น", "F005", 2))


class RouterBase(unittest.TestCase):
    def classify(self, text, d=EMPTY, *outputs):
        tmp = Path(tempfile.mkdtemp())
        self.client = ScriptedClient(*outputs)
        self.log = tmp / "calls.jsonl"
        h = Harness(client=self.client, log_file=self.log)
        return router.classify(h, text, d, "", run_id="t")

    def source(self):
        rows = [json.loads(l) for l in self.log.read_text(encoding="utf-8").splitlines()]
        return [r["source"] for r in rows if r["event"] == "route_source"][-1]


class TestRules(RouterBase):
    def test_confirm_skips_model(self):
        r = self.classify("ยืนยันครับ", OPEN)
        self.assertEqual((r.intent, self.client.calls, self.source()), ("confirm", 0, "rule"))

    def test_cancel_whole_order(self):
        for t in ("ยกเลิก", "ไม่เอาแล้วค่ะ", "Cancel!"):
            self.assertEqual(self.classify(t, OPEN).intent, "cancel", t)

    def test_greeting(self):
        self.assertEqual(self.classify("สวัสดีครับ").intent, "other")


class TestGuard(RouterBase):
    def test_cancel_one_item_is_modify(self):
        r = self.classify("ยกเลิกชาเย็น", OPEN, {"intent": "cancel", "complexity": "simple"})
        self.assertEqual((r.intent, r.complexity), ("modify_order", "complex"))
        self.assertEqual(self.source(), "guard")

    def test_ok_with_change_is_modify(self):
        r = self.classify("ok เพิ่มชาไทย 1", OPEN, {"intent": "confirm", "complexity": "simple"})
        self.assertEqual(r.intent, "modify_order")

    def test_four_items_is_complex(self):
        r = self.classify("กะเพราหมูกรอบ 1 ข้าวผัดกุ้ง 1 ต้มยำกุ้ง 1 ชาไทย 2", EMPTY,
                          {"intent": "new_order", "complexity": "simple"})
        self.assertEqual(r.complexity, "complex")

    def test_never_downgrades(self):
        r = self.classify("ขอข้าวผัดกุ้ง", EMPTY, {"intent": "new_order", "complexity": "complex"})
        self.assertEqual((r.complexity, self.source()), ("complex", "model"))

    def test_simple_answer_stays_simple(self):
        r = self.classify("เผ็ดน้อย", OPEN, {"intent": "modify_order", "complexity": "simple"})
        self.assertEqual(router.pick_profile(r, False), "fast")


class TestFallback(RouterBase):
    def test_bad_json_uses_keywords(self):
        r = self.classify("ขอข้าวผัดกุ้ง 1", EMPTY, "not json", "{}", "[1]")
        self.assertEqual((r.intent, self.source()), ("new_order", "fallback"))
        self.assertEqual(self.client.calls, 3)            # 1 try + 2 repairs

    def test_ollama_down_never_raises(self):
        r = self.classify("ยกเลิกชาเย็น", OPEN, LLMConnectionError("down"))
        self.assertEqual((r.intent, self.source()), ("modify_order", "fallback"))


class TestPickProfile(unittest.TestCase):
    def test_matrix(self):
        cases = [("new_order", "simple", False, "fast"),
                 ("modify_order", "simple", False, "fast"),
                 ("modify_order", "complex", False, "reasoning"),
                 ("new_order", "complex", True, "vision"),
                 ("new_order", "simple", True, "vision")]
        for intent, cx, img, want in cases:
            got = router.pick_profile(RouteResult(intent=intent, complexity=cx), img)
            self.assertEqual(got, want, (intent, cx, img))


def ext(phone=None, n=1):
    return ExtractedOrder(items=[{"text": "ชาไทย", "qty": 1}] * n, customer=Customer(phone=phone))


class TestExtractCheck(unittest.TestCase):
    def test_invented_phone_rejected(self):
        check = extract_check("ชาไทย 1 รับเอง", EMPTY, [], has_image=False)
        with self.assertRaises(DomainError):
            check(ext("0811111111"))

    def test_phone_from_text_history_or_draft(self):
        extract_check("โทร 081-111-1111", EMPTY, [], False)(ext("0811111111"))
        extract_check("รับเอง", EMPTY, [{"role": "user", "content": "0822222222"}],
                      False)(ext("0822222222"))
        d = draft(("ชาไทย", "F006", 1))
        d.customer.phone = "0833333333"
        extract_check("เพิ่มชาไทย", d, [], False)(ext("0833333333"))

    def test_image_skips_phone_check(self):
        extract_check("", EMPTY, [], has_image=True)(ext("0844444444"))

    def test_too_many_lines(self):
        with self.assertRaises(DomainError):
            extract_check("x", EMPTY, [], False)(ext(n=16))


class TestCleanReply(unittest.TestCase):
    SYSTEM = "- ชาไทย 35 บาท"
    FALLBACK = "template ครับ"

    def test_invented_price_uses_fallback(self):
        self.assertEqual(clean_reply("ชาไทยแก้วละ 25 บาทครับ", self.SYSTEM, self.FALLBACK),
                         self.FALLBACK)

    def test_menu_price_kept(self):
        self.assertEqual(clean_reply("ชาไทย 35 บาทครับ", self.SYSTEM, self.FALLBACK),
                         "ชาไทย 35 บาทครับ")

    def test_think_stripped_and_ending_fixed(self):
        self.assertEqual(clean_reply("<think>hmm</think> มีชาไทยค่ะ", self.SYSTEM, self.FALLBACK),
                         "มีชาไทยครับ")
        self.assertEqual(clean_reply("มีชาไทย", self.SYSTEM, self.FALLBACK), "มีชาไทยครับ")

    def test_empty_uses_fallback(self):
        self.assertEqual(clean_reply("<think>x</think>", self.SYSTEM, self.FALLBACK),
                         self.FALLBACK)


class TestPrompts(unittest.TestCase):
    """AGENTS.md contract 4: each prompt uses exactly these variables."""
    VARS = {"route": {"open_order", "last_bot"}, "extract": {"menu", "current"},
            "extract_image": {"menu", "current", "caption"},
            "clarify": {"order", "problems", "suggestions"}, "answer": {"menu"},
            "resolve": set()}

    def test_variables_match_contract(self):
        prompts = Harness(client=ScriptedClient(),
                          log_file=Path(tempfile.mkdtemp()) / "c.jsonl").prompts
        for name, want in self.VARS.items():
            t = string.Template(prompts.get(name).template)
            found = {m.group("named") or m.group("braced")
                     for m in t.pattern.finditer(t.template) if m.group("invalid") is None}
            found.discard(None)
            self.assertEqual(found, want, name)
            t.substitute({v: "x" for v in want})          # no stray "$"

    def test_latest_versions(self):
        prompts = Harness(client=ScriptedClient(),
                          log_file=Path(tempfile.mkdtemp()) / "c.jsonl").prompts
        for name in ("route", "extract", "extract_image", "clarify"):
            self.assertEqual(prompts.get(name).version, 2, name)


if __name__ == "__main__":
    unittest.main()
