"""Homework: tool-call success rate per model on data/tool_prompts.jsonl.

A prompt counts as a success when the model's FIRST response
  1. calls the expected tool,
  2. with arguments that pass validation (the executor returns ok, or the tool itself
     reports a data error such as an unknown customer), and
  3. matches expected_args when given.

    python scripts/tool_call_eval.py
    python scripts/tool_call_eval.py --profiles fast,reasoning --prompt tool_assistant@v1
    python scripts/tool_call_eval.py --tools lookup_customer,calculator   # offer fewer tools
"""
import argparse
import csv
import json
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

import harness.sample_tools  # noqa: F401
from harness import Harness, build_messages, execute, new_run_id, tool_defs, tool_message

READ_TOOLS = ["calculator", "lookup_customer", "current_datetime"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="fast,reasoning")
    ap.add_argument("--prompt", default="tool_assistant")
    ap.add_argument("--tools", default=",".join(READ_TOOLS))
    ap.add_argument("--answer", action="store_true", help="also ask for the final answer")
    ap.add_argument("--out", default=str(ROOT / "logs" / "tool_call_results.csv"))
    a = ap.parse_args()

    h = Harness()
    system = h.prompts.get(a.prompt)
    tools = tool_defs(a.tools.split(","))
    with open(ROOT / "data" / "tool_prompts.jsonl", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    rows = []
    for profile in a.profiles.split(","):
        model = h.config.profile(profile).model
        print(f"\n=== profile {profile} ({model}) ===")
        for it in items:
            run_id = new_run_id()
            msgs = build_messages(system.render(), user=it["prompt"])
            r = h.llm(profile, msgs, tools=tools, run_id=run_id, prompt_id=system.id)
            call = r.tool_calls[0] if r.tool_calls else None
            name = call["function"]["name"] if call else ""
            args = call["function"].get("arguments") if call else None
            result = execute(call, logger=h.log, run_id=run_id) if call else None

            right_tool = name == it["expected_tool"]
            data_error = bool(result and not result["ok"]
                              and result["error"].startswith("LookupError"))
            args_valid = bool(result and (result["ok"] or data_error))
            exp = it.get("expected_args")
            args_match = exp is None or (isinstance(args, dict) and all(
                str(args.get(k, "")).upper() == str(v).upper() for k, v in exp.items()))
            success = right_tool and args_valid and args_match

            answer = ""
            if a.answer and call:
                msgs += [r.message, tool_message(call, result)]
                answer = h.llm(profile, msgs, tools=tools, run_id=run_id,
                               prompt_id=system.id).content.strip()

            rows.append({"profile": profile, "model": model, "id": it["id"],
                         "called": name or "(none)", "expected": it["expected_tool"],
                         "right_tool": right_tool, "args_valid": args_valid,
                         "args_match": args_match, "success": success,
                         "extra_calls": max(0, len(r.tool_calls) - 1),
                         "wall_s": round(r.wall_s, 3),
                         "error": (result or {}).get("error", "") if result else "no tool call",
                         "answer": answer})
            print(f"  {it['id']} {'OK  ' if success else 'FAIL'} called={name or '(none)':<17} "
                  f"args={json.dumps(args, ensure_ascii=False)[:60]}"
                  + (f"\n        answer: {answer[:100]}" if answer else ""))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== tool-call success rate ===")
    for profile in a.profiles.split(","):
        r = [x for x in rows if x["profile"] == profile]
        s = sum(x["success"] for x in r)
        print(f"{profile:<10} {r[0]['model']:<24} {s}/{len(r)} = {100 * s / len(r):.0f}%   "
              f"right tool {sum(x['right_tool'] for x in r)}/{len(r)}   "
              f"valid args {sum(x['args_valid'] for x in r)}/{len(r)}")
    print(f"\nPer-prompt results: {a.out}")


if __name__ == "__main__":
    main()
