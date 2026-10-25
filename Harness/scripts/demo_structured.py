"""Demo M1-M5: extract one validated Ticket from a message.

    python scripts/demo_structured.py
    python scripts/demo_structured.py "C-107: portal down since 8am, cannot take orders!"
    python scripts/demo_structured.py --prompt extract_ticket@v1 --no-repair "..."
"""
import argparse

import _path  # noqa: F401

from harness import Harness, Ticket, ask_structured, build_messages, check_ticket

DEFAULT = "Hi, this is C-104. Our invoice was charged twice on the credit card. Please refund one."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=DEFAULT)
    ap.add_argument("--prompt", default="extract_ticket", help="prompt id, e.g. extract_ticket@v1")
    ap.add_argument("--profiles", default="fast,reasoning,fallback")
    ap.add_argument("--no-repair", action="store_true")
    a = ap.parse_args()

    h = Harness()
    prompt = h.prompts.get(a.prompt)
    msgs = build_messages(prompt.render(), user=a.text)

    res = ask_structured(h, msgs, Ticket, profiles=tuple(a.profiles.split(",")),
                         repair=not a.no_repair, domain_check=check_ticket, prompt_id=prompt.id)

    print(f"prompt   : {prompt.id}")
    print(f"input    : {a.text}")
    if res.ok:
        print(f"result   : OK via profile '{res.profile_used}' after {res.attempts} attempt(s)")
        print(res.value.model_dump_json(indent=2))
    else:
        print(f"result   : SAFE FAILURE after {res.attempts} attempts")
    for e in res.errors:
        print(f"  error  : {e}")
    print(f"wall time: {res.total_wall_s:.2f}s   log: {h.log.path}")


if __name__ == "__main__":
    main()
