"""M8 - RESOLVE: match what the customer wrote to menu SKUs.

Code decides whatever it can: one clear match is used, no match goes to CLARIFY. That
is most messages, and it costs no model call (the loop took 4 calls, 13-25 s per message).

Only a name that fits several items equally ("ชา", "ข้าว") reaches the agent loop. There
the model gets ONLY product_search and stock_checker, calls them in a bounded loop
(max_rounds), then returns its choice as JSON. Code accepts the answer only if it is one
of those items, so a wrong model answer can never reach the price or the order. If the
loop fails, the customer is asked to choose.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from harness import ask_structured, build_messages, execute, tool_defs, tool_message

from .catalog import catalog
from .schemas import SKU_PATTERN, ExtractedItem
from .tools import MODEL_TOOLS

MIN_SCORE = 0.85   # below this a name is only a suggestion: "ผัดไทยกุ้งสด" is not ข้าวผัดกุ้ง


class ResolvedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    sku: Optional[str] = Field(default=None, pattern=SKU_PATTERN)


class ResolveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ResolvedItem]


def call_tool(h, name: str, args: dict, *, run_id: str,
              allowed: frozenset = frozenset({"read"})) -> dict:
    """Code-side tool call: same executor, permission check and log as the model's."""
    return execute({"function": {"name": name, "arguments": args}},
                   allowed=set(allowed), logger=h.log, run_id=run_id)


def resolve_with_model(h, items: list[ExtractedItem], *, run_id: str,
                       max_rounds: int = 4) -> Optional[dict[int, Optional[str]]]:
    """Tool-calling loop -> {item index: sku or None}. None = the loop failed."""
    p = h.prompts.get("resolve")
    listing = "\n".join(f"{i}. {it.text} x{it.qty or 1}" for i, it in enumerate(items))
    msgs = build_messages(p.render(), user=listing)
    defs = tool_defs(MODEL_TOOLS)

    for _ in range(max_rounds):
        out = h.llm("fast", msgs, tools=defs, run_id=run_id, prompt_id=p.id)
        msgs.append(out.message)
        if not out.tool_calls:
            break
        for call in out.tool_calls:
            # allowed={"read"}: even if the model names a write tool, it is refused
            res = execute(call, allowed={"read"}, logger=h.log, run_id=run_id)
            msgs.append(tool_message(call, res))

    msgs.append({"role": "user", "content":
                 "Now return JSON: for every item index, the chosen sku, or null when "
                 "nothing on the menu matches."})
    res = ask_structured(h, msgs, ResolveResult, profiles=("fast",), max_retry=1,
                         prompt_id=p.id, run_id=run_id)
    if not res.ok:
        return None
    return {r.index: r.sku for r in res.value.items if 0 <= r.index < len(items)}


def _generic(text: str, skus: set[str]) -> bool:
    """The customer's word is part of the name of every candidate, so it cannot pick one."""
    menu = catalog()
    q = re.sub(r"\s+", "", text.lower())
    return bool(q) and all(
        any(q in re.sub(r"\s+", "", n.lower()) for n in (menu[s].name, *menu[s].aliases))
        for s in skus if s in menu)


def resolve_items(h, items: list[ExtractedItem], *, run_id: str, use_model: bool = True,
                  max_rounds: int = 4) -> dict[int, Optional[str]]:
    """{item index: sku, or None when the customer has to be asked}.

    Stock plays no part here: a sold-out item keeps its sku so that the state machine
    can say it is sold out instead of "not on the menu".
    """
    mapping: dict[int, Optional[str]] = {}
    unclear: dict[int, set[str]] = {}
    for i, it in enumerate(items):
        r = call_tool(h, "product_search", {"query": it.text, "top_k": 5}, run_id=run_id)
        hits = [x for x in (r["result"] if r["ok"] else []) if x["score"] >= MIN_SCORE]
        best = [x["sku"] for x in hits if x["score"] == hits[0]["score"]]
        mapping[i] = best[0] if len(best) == 1 else None
        if len(best) > 1:
            unclear[i] = set(best)

    rejected = []
    if use_model and unclear:
        order = list(unclear)
        answer = resolve_with_model(h, [items[i] for i in order], run_id=run_id,
                                    max_rounds=max_rounds) or {}
        for pos, sku in answer.items():
            if sku in unclear[order[pos]] and _generic(items[order[pos]].text, unclear[order[pos]]):
                # "ชา" fits ชาเย็น and ชาไทย alike: the model picked ชาไทย because ชาเย็น was
                # sold out, and the customer was never asked. Nothing in the words decides it.
                rejected.append(f"{items[order[pos]].text}: {sku} (generic)")
            elif sku in unclear[order[pos]]:
                mapping[order[pos]] = sku
            elif sku:
                rejected.append(f"{items[order[pos]].text}: {sku}")
    h.log.write("resolve", run_id=run_id, mapping={items[i].text: s for i, s in mapping.items()},
                unclear=[items[i].text for i in unclear], rejected=rejected)
    return mapping
