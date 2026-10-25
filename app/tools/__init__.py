"""M6 - The 5 OrderBuddy tools (STAND-INS: owner บีม replaces the bodies).

Contract that must stay the same (state machine depends on it):
* names, Args classes and permissions below
* result shapes in the docstrings
* errors are raised as ValueError("<code>: <detail>"); the harness executor
  turns them into {"ok": false, "error": "ValueError: <code>: <detail>"}

Import this package once (app.state_machine does) to register the tools.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal, Optional

from pydantic import BaseModel, Field

from harness import tool

from .. import catalog as cat
from ..schemas import SKU_PATTERN, DraftOrder

_store = None  # SessionStore, set by configure()


def configure(store) -> None:
    """Give draft_order_manager access to the session store."""
    global _store
    _store = store


# ------------------------------------------------------------ product_search
class SearchArgs(BaseModel):
    query: str = Field(min_length=1, description="menu item name as the customer wrote it")
    top_k: int = Field(default=3, ge=1, le=5)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _score(query: str, candidate: str) -> float:
    q, c = _norm(query), _norm(candidate)
    if q == c:
        return 1.0
    if q in c or c in q:
        return 0.85
    return SequenceMatcher(None, q, c).ratio()


@tool(permission="read")
def product_search(args: SearchArgs) -> list[dict]:
    """Find menu items by name. Returns up to top_k matches with sku, name, price and
    score (0-1). An empty list means nothing on the menu matches."""
    hits = []
    for it in cat.catalog().values():
        s = max(_score(args.query, n) for n in [it.name, *it.aliases])
        if s >= 0.6:
            hits.append({"sku": it.sku, "name": it.name, "price": it.price, "score": round(s, 2)})
    return sorted(hits, key=lambda h: -h["score"])[: args.top_k]


# ------------------------------------------------------------- stock_checker
class StockArgs(BaseModel):
    sku: str = Field(pattern=SKU_PATTERN, description="sku from product_search, e.g. F001")
    qty: int = Field(ge=1, le=20)


@tool(permission="read")
def stock_checker(args: StockArgs) -> dict:
    """Check whether a menu item (sku) has enough stock for qty. Use after product_search."""
    if args.sku not in cat.catalog():
        raise ValueError(f"unknown_sku: {args.sku}")
    n = cat.stock().get(args.sku, 0)
    return {"sku": args.sku, "available": n, "enough": n >= args.qty}


# ---------------------------------------------------------- price_calculator
class PriceItem(BaseModel):
    sku: str = Field(pattern=SKU_PATTERN)
    qty: int = Field(ge=1, le=20)
    addons: list[str] = Field(default_factory=list)


class PriceArgs(BaseModel):
    items: list[PriceItem] = Field(min_length=1)
    fulfillment: Literal["delivery", "pickup"]


@tool(permission="read")
def price_calculator(args: PriceArgs) -> dict:
    """Compute line totals, delivery fee and total in THB. Called by code, not the model."""
    lines, subtotal = [], 0
    for it in args.items:
        menu = cat.catalog().get(it.sku)
        if menu is None:
            raise ValueError(f"unknown_sku: {it.sku}")
        bad = [a for a in it.addons if a not in menu.addons]
        if bad:
            raise ValueError(f"bad_addon: {it.sku} {bad}")
        unit = menu.price + sum(menu.addons[a] for a in it.addons)
        lines.append({"sku": it.sku, "unit_price": unit, "line_total": unit * it.qty})
        subtotal += unit * it.qty
    fee = cat.shop()["delivery_fee"] if args.fulfillment == "delivery" else 0
    return {"lines": lines, "subtotal": subtotal, "delivery_fee": fee, "total": subtotal + fee}


# --------------------------------------------------------- address_validator
class AddressArgs(BaseModel):
    address: str = ""
    phone: str = ""
    need_address: bool = True


_POSTCODE = re.compile(r"\b(\d{5})\b")


@tool(permission="read")
def address_validator(args: AddressArgs) -> dict:
    """Check a Thai phone number and delivery address. Returns valid, normalized
    address/phone, missing fields and whether the postcode is in the delivery zone."""
    missing, phone = [], re.sub(r"[\s\-]", "", args.phone)
    if not re.fullmatch(r"0\d{9}", phone):
        missing.append("phone")
    in_zone = True
    if args.need_address:
        addr = " ".join(args.address.split())
        m = _POSTCODE.search(addr)
        if not addr or not re.search(r"\d", addr):
            missing.append("address")
        elif not m:
            missing.append("postcode")
        else:
            in_zone = m.group(1) in cat.shop()["delivery_zones"]
    return {"valid": not missing and in_zone, "missing": missing, "in_zone": in_zone,
            "normalized": {"address": " ".join(args.address.split()), "phone": phone}}


# ------------------------------------------------------- draft_order_manager
class DraftArgs(BaseModel):
    action: Literal["get", "submit", "cancel"]
    line_user_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")


@tool(permission="write")
def draft_order_manager(args: DraftArgs) -> Optional[dict]:
    """Read, submit or cancel the customer's draft order. submit turns a confirmed
    draft into orders/<order_id>.json with status pending_staff."""
    if _store is None:
        raise RuntimeError("tools.configure(store) was not called")
    draft: Optional[DraftOrder] = _store.get_draft(args.line_user_id)
    if draft is None:
        raise ValueError(f"no_draft: {args.line_user_id}")
    if args.action == "get":
        return draft.model_dump(mode="json")
    if args.action == "cancel":
        _store.delete_draft(args.line_user_id)
        return {"cancelled": draft.order_id}
    # submit: the two safety checks live here as well as in the state machine
    if draft.state != "CONFIRM" or draft.missing or not draft.items:
        raise ValueError(f"not_ready: state={draft.state} missing={draft.missing}")
    return _store.submit(draft).model_dump(mode="json")


ALL_TOOLS = ["product_search", "stock_checker", "price_calculator",
             "address_validator", "draft_order_manager"]
MODEL_TOOLS = ["product_search", "stock_checker"]   # the only tools the model ever sees
