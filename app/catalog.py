"""Read-only access to data/catalog.json, data/stock.json and data/shop.json.

Shared by the tools (Part C) and the state machine (prompts, order summary).
Sample data only: บีม replaces data/*.json with the real 20-30 item menu.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schemas import CatalogItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def catalog() -> dict[str, CatalogItem]:
    return {i.sku: i for i in (CatalogItem.model_validate(x) for x in _load("catalog.json"))}


def stock() -> dict[str, int]:
    """Read every time: stock changes while the server runs."""
    return _load("stock.json")


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
