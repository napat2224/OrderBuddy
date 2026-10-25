"""Live demo + Hands-on 3a: one tool-calling round trip (no loop yet).

    model -> tool_calls -> executor -> role:tool message -> model -> final answer

    python scripts/demo_tool_call.py
    python scripts/demo_tool_call.py "What is 18% of 245,000 baht?"
    python scripts/demo_tool_call.py --profile reasoning "Which tier is C-103?"
    python scripts/demo_tool_call.py --failure      # show how the executor rejects bad calls
"""
import argparse
import json

import _path  # noqa: F401

import harness.sample_tools  # noqa: F401  (registers the sample tools)
from harness import Harness, build_messages, execute, new_run_id, tool_defs, tool_message

DEFAULT = "What is the credit limit of customer C-104?"


def show_failures(h):
    """Calls a small model might produce, and what the executor answers."""
    bad = [
        {"function": {"name": "get_credit_limit", "arguments": {"customer_id": "C-104"}}},
        {"function": {"name": "lookup_customer", "arguments": {"id": "C-104"}}},
        {"function": {"name": "lookup_customer", "arguments": {"customer_id": "C-999"}}},
        {"function": {"name": "calculator", "arguments": {"expression": "__import__('os')"}}},
        {"function": {"name": "update_credit_limit",
                      "arguments": {"customer_id": "C-104", "new_limit_thb": 900000}}},
    ]
    for c in bad:
        print(f"\ncall   : {c['function']['name']}({json.dumps(c['function']['arguments'])})")
        print(f"result : {json.dumps(execute(c, logger=h.log, run_id='failure-demo'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=DEFAULT)
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--failure", action="store_true")
    a = ap.parse_args()

    h = Harness()
    if a.failure:
        show_failures(h)
        return

    run_id = new_run_id()
    system = h.prompts.get("tool_assistant")
    msgs = build_messages(system.render(), user=a.prompt)
    tools = tool_defs()

    # 1) model proposes tool calls
    r1 = h.llm(a.profile, msgs, tools=tools, run_id=run_id, prompt_id=system.id)
    msgs.append(r1.message)
    print(f"user        : {a.prompt}")
    if not r1.tool_calls:
        print(f"model       : (no tool call) {r1.content}")
        return

    # 2) harness executes each call and appends a tool message
    for call in r1.tool_calls:
        f = call["function"]
        print(f"tool call   : {f['name']}({json.dumps(f.get('arguments'), ensure_ascii=False)})")
        result = execute(call, logger=h.log, run_id=run_id)
        print(f"tool result : {json.dumps(result, ensure_ascii=False)}")
        msgs.append(tool_message(call, result))

    # 3) model writes the final answer
    r2 = h.llm(a.profile, msgs, tools=tools, run_id=run_id, prompt_id=system.id)
    print(f"answer      : {r2.content.strip()}")
    print(f"time        : {r1.wall_s + r2.wall_s:.2f}s   run_id={run_id}   log: {h.log.path}")


if __name__ == "__main__":
    main()
