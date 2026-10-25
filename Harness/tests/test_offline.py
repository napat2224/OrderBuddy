"""Offline tests: no Ollama needed. A FakeClient replays scripted model outputs.

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import harness.sample_tools  # noqa: E402,F401
from harness import (ChatResult, Harness, Ticket, ask_structured, build_messages,  # noqa: E402
                     check_ticket, execute, tool_defs, tool_message)
from harness.prompts import PromptManager  # noqa: E402


class FakeClient:
    """Returns queued outputs in order; records every request."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def chat(self, model, messages, **kw):
        self.requests.append({"model": model, "messages": messages, **kw})
        out = self.outputs.pop(0)
        if isinstance(out, dict):  # tool calls
            return ChatResult(model=model, content="", tool_calls=out["tool_calls"],
                              message={"role": "assistant", "content": "", **out},
                              wall_s=0.01)
        return ChatResult(model=model, content=out, message={"role": "assistant", "content": out},
                          prompt_tokens=10, output_tokens=5, wall_s=0.01)


def make_harness(outputs):
    tmp = tempfile.mkdtemp()
    return Harness(client=FakeClient(outputs), log_file=Path(tmp) / "calls.jsonl")


GOOD = json.dumps({"category": "billing", "priority": 2,
                   "summary": "Customer was charged twice.", "customer_id": "C-104"})


class TestPrompts(unittest.TestCase):
    def test_latest_and_pinned(self):
        pm = PromptManager()
        self.assertEqual(pm.get("extract_ticket").id, "extract_ticket@v2")
        self.assertEqual(pm.get("extract_ticket@v1").id, "extract_ticket@v1")

    def test_order_stable_to_variable(self):
        m = build_messages("SYS", policies="POL", history=[{"role": "user", "content": "old"}],
                           user="new")
        self.assertEqual(m[0]["content"], "SYS\n\nPOL")
        self.assertEqual(m[-1]["content"], "new")


class TestValidation(unittest.TestCase):
    def test_first_try_valid(self):
        h = make_harness([GOOD])
        r = ask_structured(h, [], Ticket, domain_check=check_ticket)
        self.assertTrue(r.ok and r.first_try_valid)
        self.assertEqual(r.value.customer_id, "C-104")
        self.assertEqual(r.attempts, 1)

    def test_repair_sends_error_back(self):
        bad = json.dumps({"category": "billing", "priority": 5, "summary": "Charged twice."})
        h = make_harness([bad, GOOD])
        r = ask_structured(h, [{"role": "user", "content": "x"}], Ticket)
        self.assertTrue(r.ok)
        self.assertFalse(r.first_try_valid)
        repair_msgs = h.client.requests[1]["messages"]
        self.assertIn("Invalid output", repair_msgs[-1]["content"])
        self.assertIn("priority", repair_msgs[-1]["content"])

    def test_domain_rule_unknown_customer(self):
        bad = json.dumps({"category": "billing", "priority": 2, "summary": "Overcharged.",
                          "customer_id": "C-999"})
        h = make_harness([bad, GOOD])
        r = ask_structured(h, [], Ticket, domain_check=check_ticket)
        self.assertTrue(r.ok)
        self.assertIn("does not exist", r.errors[0])

    def test_fallback_then_safe_failure(self):
        h = make_harness(["not json"] * 9)
        r = ask_structured(h, [], Ticket, profiles=("fast", "reasoning", "fallback"), max_retry=2)
        self.assertFalse(r.ok)
        self.assertEqual(r.attempts, 9)
        models = [q["model"] for q in h.client.requests]
        self.assertEqual(models[0], h.config.profile("fast").model)
        self.assertEqual(models[-1], h.config.profile("fallback").model)
        log = [json.loads(l) for l in h.log.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(log[-1]["event"], "safe_failure")


class TestTools(unittest.TestCase):
    def call(self, name, args):
        return {"function": {"name": name, "arguments": args}}

    def test_defs_exported(self):
        names = [d["function"]["name"] for d in tool_defs()]
        self.assertIn("lookup_customer", names)
        self.assertIn("customer_id", tool_defs(["lookup_customer"])[0]["function"]["parameters"]["properties"])

    def test_ok(self):
        r = execute(self.call("lookup_customer", {"customer_id": "C-104"}))
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["available_credit_thb"], 440000)

    def test_arguments_as_json_string(self):
        r = execute(self.call("calculator", '{"expression": "750000 - 310000"}'))
        self.assertEqual(r["result"]["value"], 440000)

    def test_unknown_tool(self):
        r = execute(self.call("get_credit_limit", {}))
        self.assertFalse(r["ok"])
        self.assertIn("Valid tools", r["error"])

    def test_bad_arguments(self):
        r = execute(self.call("lookup_customer", {"id": "C-104"}))
        self.assertIn("Bad arguments", r["error"])

    def test_permission_denied(self):
        r = execute(self.call("update_credit_limit", {"customer_id": "C-104", "new_limit_thb": 1}))
        self.assertIn("Permission denied", r["error"])
        r2 = execute(self.call("update_credit_limit", {"customer_id": "C-104", "new_limit_thb": 1}),
                     allowed={"read", "write"})
        self.assertTrue(r2["ok"])

    def test_calculator_is_safe(self):
        r = execute(self.call("calculator", {"expression": "__import__('os').getcwd()"}))
        self.assertFalse(r["ok"])

    def test_tool_error_becomes_observation(self):
        r = execute(self.call("lookup_customer", {"customer_id": "C-999"}))
        self.assertFalse(r["ok"])
        self.assertTrue(r["error"].startswith("LookupError"))

    def test_round_trip(self):
        call = self.call("lookup_customer", {"customer_id": "C-104"})
        h = make_harness([{"tool_calls": [call]}, "C-104 has a credit limit of 750,000 THB."])
        r1 = h.llm("fast", [{"role": "user", "content": "limit of C-104?"}], tools=tool_defs())
        msgs = [r1.message, tool_message(call, execute(call))]
        r2 = h.llm("fast", msgs, tools=tool_defs())
        self.assertIn("750,000", r2.content)
        self.assertEqual(msgs[-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
