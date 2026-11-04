"""Debug commands: try the agent without LINE and read what happened.

    python -m app.debug extract "ขอกะเพราหมูกรอบ 2 จาน"   only the EXTRACT step, one message
    python -m app.debug chat                               full flow in the terminal
    python -m app.debug last [n]                           the last n messages from the log
    python -m app.debug run r-1234abcd                     one message by its run_id

extract and chat need Ollama and also write to Harness/logs/app.log. chat keeps drafts,
orders and stock in a temp folder, so nothing it does reaches the real drafts/, orders/ or
data/stock.json. last and run only read the log.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from harness import new_run_id

from . import catalog
from .schemas import DraftOrder
from .session import SessionStore
from .state_machine import OrderAgent
from .trace import TracedHarness, setup_file_log

DEBUG_USER = "debug-user"


def load_events(harness) -> list[dict]:
    path = harness.log.path
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_event(e: dict) -> str:
    fields = {k: v for k, v in e.items() if k not in ("ts", "event", "run_id", "trace")}
    row = f"{e['ts'][11:]} {e['event']:<12} {json.dumps(fields, ensure_ascii=False)}"
    return row + ("\n" + e["trace"] if e.get("trace") else "")


def show_run(harness, run_id: str) -> None:
    print(f"--- {run_id}")
    for e in load_events(harness):
        if e.get("run_id") == run_id:
            print(format_event(e))


def last_run_ids(events: list[dict], n: int) -> list[str]:
    """Runs that start with a customer message, newest last (course-script runs are skipped)."""
    return [e["run_id"] for e in events if e["event"] == "message"][-n:]


def make_agent() -> OrderAgent:
    h = TracedHarness()
    setup_file_log(h.log.path.parent)
    # confirming an order cuts stock: a practice order must not touch the real file
    catalog.DATA_DIR = Path(shutil.copytree(catalog.DATA_DIR, tempfile.mkdtemp(),
                                            dirs_exist_ok=True))
    logging.getLogger("orderbuddy").propagate = False   # log to the file, keep the chat readable
    return OrderAgent(h=h, store=SessionStore(tempfile.mkdtemp()))


def cmd_extract(args) -> None:
    agent = make_agent()
    run_id = new_run_id()
    draft = DraftOrder(order_id="DEBUG", line_user_id=DEBUG_USER)
    order = agent._extract(run_id, draft, [], args.text, None, args.profile)
    show_run(agent.h, run_id)
    print("\nRESULT:" if order else "\nRESULT: safe failure (see the errors above)")
    if order:
        print(json.dumps(order.model_dump(), ensure_ascii=False, indent=2))


def cmd_chat(_args) -> None:
    agent = make_agent()
    print("Type as the customer. Ctrl+C to stop.")
    try:
        while text := input("\nคุณ> ").strip():
            for reply in agent.handle(DEBUG_USER, text=text):
                print(f"ร้าน> {reply}")
    except (KeyboardInterrupt, EOFError):
        print()


def cmd_last(args) -> None:
    h = TracedHarness()
    run_ids = last_run_ids(load_events(h), args.n)
    if not run_ids:
        print(f"No customer messages in {h.log.path} yet.")
    for run_id in run_ids:
        show_run(h, run_id)


def cmd_run(args) -> None:
    show_run(TracedHarness(), args.run_id)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m app.debug", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="run only the EXTRACT step on one message")
    p.add_argument("text")
    p.add_argument("--profile", default="fast", choices=["fast", "reasoning"])
    p.set_defaults(func=cmd_extract)
    sub.add_parser("chat", help="full flow in the terminal").set_defaults(func=cmd_chat)
    p = sub.add_parser("last", help="show the last n messages from the log")
    p.add_argument("n", nargs="?", type=int, default=1)
    p.set_defaults(func=cmd_last)
    p = sub.add_parser("run", help="show one message by run_id")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
