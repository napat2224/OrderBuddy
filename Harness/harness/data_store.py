"""Tiny read-only data layer over data/customers.csv (used by tools and domain rules)."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUSTOMERS_CSV = ROOT / "data" / "customers.csv"


@lru_cache(maxsize=1)
def _customers() -> dict[str, dict]:
    with open(CUSTOMERS_CSV, newline="", encoding="utf-8") as f:
        return {row["customer_id"]: row for row in csv.DictReader(f)}


def get_customer(customer_id: str) -> dict | None:
    row = _customers().get(customer_id.strip().upper())
    if row is None:
        return None
    return {**row, "credit_limit_thb": int(row["credit_limit_thb"]),
            "balance_thb": int(row["balance_thb"])}


def customer_exists(customer_id: str) -> bool:
    return customer_id.strip().upper() in _customers()


def all_customer_ids() -> list[str]:
    return list(_customers())
