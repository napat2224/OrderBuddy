"""M4 - Schemas shared by every module (router, models, tools, state machine).

Rules (same as Harness/harness/schemas.py):
* extra="forbid" so the model cannot invent fields
* Literal instead of Enum, small and flat for 4B models
* Schemas the model fills (RouteResult, ExtractedOrder) stay short;
  schemas filled by code (DraftOrder, FinalOrder) can be richer.

Prices are integer THB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SKU_PATTERN = r"^F\d{3}$"

Intent = Literal["new_order", "modify_order", "confirm", "cancel", "question", "other"]
Fulfillment = Literal["delivery", "pickup"]
State = Literal["IDLE", "ROUTE", "EXTRACT", "RESOLVE", "CLARIFY", "PRICE",
                "ADDRESS", "CONFIRM", "DONE", "STAFF_REVIEW", "ERROR"]
OrderStatus = Literal["pending_staff", "approved", "rejected"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------- 1. catalog
class CatalogItem(_Strict):
    """One menu item in data/catalog.json. Stock lives in data/stock.json."""
    sku: str = Field(pattern=SKU_PATTERN)
    name: str
    aliases: list[str] = Field(default_factory=list)
    category: Literal["main", "side", "drink", "dessert"]
    price: int = Field(ge=0)
    options: dict[str, list[str]] = Field(
        default_factory=dict, description="e.g. {'spice': ['ไม่เผ็ด', 'เผ็ดน้อย']}")
    addons: dict[str, int] = Field(
        default_factory=dict, description="addon name -> extra price, e.g. {'ไข่ดาว': 10}")


# ---------------------------------------------------------- 2. router output
class RouteResult(_Strict):
    """Filled by the router profile (qwen3:4b)."""
    intent: Intent
    complexity: Literal["simple", "complex"] = Field(
        description="complex = more than 3 items, conditions, or unclear wording")


# --------------------------------------------------------- 3. model extraction
class ExtractedItem(_Strict):
    text: str = Field(min_length=1, description="item name as the customer wrote it")
    qty: int = Field(ge=1, le=20)
    options: dict[str, str] = Field(default_factory=dict)
    addons: list[str] = Field(default_factory=list)
    note: str = ""


class Customer(_Strict):
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None


class ExtractedOrder(_Strict):
    """Filled by fast / reasoning / vision. Not yet matched to SKUs."""
    items: list[ExtractedItem] = Field(default_factory=list)
    customer: Customer = Field(default_factory=Customer)
    fulfillment: Optional[Fulfillment] = None


# ------------------------------------------------------------- 4. draft order
class OrderLine(_Strict):
    sku: str = Field(pattern=SKU_PATTERN)
    name: str
    qty: int = Field(ge=1, le=20)
    options: dict[str, str] = Field(default_factory=dict)
    addons: list[str] = Field(default_factory=list)
    note: str = ""
    unit_price: int = Field(default=0, ge=0)
    line_total: int = Field(default=0, ge=0)


class DraftCustomer(Customer):
    address_valid: bool = False


class DraftOrder(_Strict):
    """One open draft per LINE user, stored in drafts/<line_user_id>.json."""
    order_id: str
    line_user_id: str
    state: State = "IDLE"
    items: list[OrderLine] = Field(default_factory=list)
    subtotal: int = 0
    delivery_fee: int = 0
    total: int = 0
    customer: DraftCustomer = Field(default_factory=DraftCustomer)
    fulfillment: Optional[Fulfillment] = None
    missing: list[str] = Field(default_factory=list,
                               description="what is still needed, e.g. ['phone']")
    updated_at: datetime = Field(default_factory=datetime.now)


# ------------------------------------------------------------- 5. final order
class FinalOrder(_Strict):
    """Written to orders/<order_id>.json after the customer confirms."""
    order_id: str
    line_user_id: str
    status: OrderStatus = "pending_staff"
    items: list[OrderLine]
    subtotal: int
    delivery_fee: int
    total: int
    customer: DraftCustomer
    fulfillment: Fulfillment
    confirmed_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None

    @classmethod
    def from_draft(cls, d: DraftOrder) -> "FinalOrder":
        data = d.model_dump(exclude={"state", "missing", "updated_at"})
        return cls(**data)
