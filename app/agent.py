"""M8 - Agent loop used in the RESOLVE state.

The model gets ONLY product_search and stock_checker, calls them in a bounded
loop (max_rounds), then returns its choice as JSON. Code then re-checks every
sku, so a wrong model answer can never reach the price or the order.

If the loop fails (bad JSON, no answer) we fall back to code matching, so the
order flow never stops because of the model.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from harness import ask_structured, build_messages, execute, tool_defs, tool_message

from .schemas import SKU_PATTERN, ExtractedItem
from .tools import MODEL_TOOLS


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
    listing = "\n".join(f"{i}. {it.text} x{it.qty}" for i, it in enumerate(items))
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


def resolve_with_code(h, items: list[ExtractedItem], *, run_id: str,
                      min_score: float = 0.8) -> dict[int, Optional[str]]:
    """Deterministic fallback: best product_search hit with a high score."""
    out = {}
    for i, it in enumerate(items):
        r = call_tool(h, "product_search", {"query": it.text, "top_k": 1}, run_id=run_id)
        hit = r["result"][0] if r["ok"] and r["result"] else None
        out[i] = hit["sku"] if hit and hit["score"] >= min_score else None
    return out
