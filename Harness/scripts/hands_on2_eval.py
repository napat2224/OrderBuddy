"""Hands-on 2: validated structured output on 20 test inputs, compared across models.

For each profile (model) and each input in data/ticket_inputs.jsonl:
  * first-try validity (schema + domain rules, no retry)
  * validity after retry/repair (up to --max-retry extra attempts, same model only)
  * field accuracy against the expected values
  * wall time

    python scripts/hands_on2_eval.py
    python scripts/hands_on2_eval.py --profiles fast,reasoning --prompt extract_ticket@v1
    python scripts/hands_on2_eval.py --no-repair      # plain retry, for comparison
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

from harness import Harness, Ticket, ask_structured, build_messages, check_ticket, new_run_id


def load_inputs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="fast,reasoning")
    ap.add_argument("--prompt", default="extract_ticket")
    ap.add_argument("--max-retry", type=int, default=2)
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N inputs")
    ap.add_argument("--out", default=str(ROOT / "logs" / "hands_on2_results.csv"))
    a = ap.parse_args()

    h = Harness()
    prompt = h.prompts.get(a.prompt)
    items = load_inputs(ROOT / "data" / "ticket_inputs.jsonl")
    if a.limit:
        items = items[:a.limit]

    rows = []
    for profile in a.profiles.split(","):
        model = h.config.profile(profile).model
        print(f"\n=== profile {profile} ({model}) · prompt {prompt.id} ===")
        for it in items:
            msgs = build_messages(prompt.render(), user=it["text"])
            res = ask_structured(h, msgs, Ticket, profiles=(profile,), max_retry=a.max_retry,
                                 repair=not a.no_repair, domain_check=check_ticket,
                                 prompt_id=prompt.id, run_id=new_run_id())
            exp = it["expected"]
            got = res.value.model_dump() if res.ok else {}
            match = {k: got.get(k) == exp[k] for k in ("category", "priority", "customer_id")}
            rows.append({
                "profile": profile, "model": model, "prompt": prompt.id, "id": it["id"],
                "first_try_valid": res.first_try_valid, "final_valid": res.ok,
                "attempts": res.attempts, "wall_s": round(res.total_wall_s, 3),
                "category_ok": match["category"], "priority_ok": match["priority"],
                "customer_ok": match["customer_id"],
                "last_error": res.reason if not res.first_try_valid else "",
            })
            mark = "OK " if res.ok else "FAIL"
            print(f"  {it['id']} {mark} attempts={res.attempts} {res.total_wall_s:5.2f}s "
                  f"cat={'Y' if match['category'] else 'n'} pri={'Y' if match['priority'] else 'n'} "
                  f"cust={'Y' if match['customer_id'] else 'n'}"
                  + (f"  <- {res.reason[:70]}" if res.errors else ""))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== summary ===")
    print(f"{'profile':<10} {'model':<24} {'first-try':>9} {'after-repair':>12} "
          f"{'category':>8} {'priority':>8} {'customer':>8} {'mean s':>7}")
    for profile in a.profiles.split(","):
        r = [x for x in rows if x["profile"] == profile]
        pct = lambda k: f"{100 * sum(x[k] for x in r) / len(r):.0f}%"  # noqa: E731
        print(f"{profile:<10} {r[0]['model']:<24} {pct('first_try_valid'):>9} {pct('final_valid'):>12} "
              f"{pct('category_ok'):>8} {pct('priority_ok'):>8} {pct('customer_ok'):>8} "
              f"{st.mean(x['wall_s'] for x in r):7.2f}")
    print(f"\nPer-input results: {a.out}\nCall log:          {h.log.path}")


if __name__ == "__main__":
    main()
