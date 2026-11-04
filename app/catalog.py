"""Read-only access to data/catalog.json, data/stock.json and data/shop.json.

Shared by the tools (Part C) and the state machine (prompts, order summary).
Sample data only: บีม replaces data/*.json with the real 20-30 item menu.
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from .schemas import CatalogItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def catalog() -> dict[str, CatalogItem]:
    return {i.sku: i for i in (CatalogItem.model_validate(x) for x in _load("catalog.json"))}


_stock_lock = threading.Lock()


def stock() -> dict[str, int]:
    """Read every time: stock changes while the server runs."""
    return _load("stock.json")


def _save_stock(s: dict[str, int]) -> None:
    path = DATA_DIR / "stock.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def take_stock(qty_by_sku: dict[str, int]) -> list[str]:
    """Cut stock for one order, all or nothing. Returns the skus that are short (nothing is cut
    then). The lock covers check and cut together, so two customers cannot get the last one."""
    with _stock_lock:
        s = stock()
        short = [sku for sku, qty in qty_by_sku.items() if s.get(sku, 0) < qty]
        if not short:
            _save_stock({**s, **{sku: s[sku] - qty for sku, qty in qty_by_sku.items()}})
        return short


def return_stock(qty_by_sku: dict[str, int]) -> None:
    """Undo take_stock when the order could not be saved."""
    with _stock_lock:
        s = stock()
        _save_stock({**s, **{sku: s.get(sku, 0) + qty for sku, qty in qty_by_sku.items()}})


@lru_cache(maxsize=1)
def shop() -> dict:
    return _load("shop.json")


def menu_text() -> str:
    """Compact menu for prompts: one line per item, stable order (prefix caching)."""
    lines = []
    for it in catalog().values():
        parts = [f"{it.name} {it.price} บาท"]
        for key, values in it.options.items():
            parts.append(f"{key}: {'/'.join(values)}")
        if it.addons:
            parts.append("addon: " + ", ".join(f"{a} +{p}" for a, p in it.addons.items()))
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)
