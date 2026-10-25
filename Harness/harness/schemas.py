"""M4 - Structured output schemas.

Keep schemas small and flat for small models; use Literal enums for categories.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    """A support ticket extracted from a free-text customer message."""
    model_config = ConfigDict(extra="forbid")

    category: Literal["billing", "technical", "other"]
    priority: int = Field(ge=1, le=3, description="1 low, 2 normal, 3 urgent")
    summary: str = Field(min_length=5, max_length=200)
    customer_id: Optional[str] = Field(default=None, pattern=r"^C-\d{3}$")


def schema_of(model: type[BaseModel]) -> dict:
    """JSON schema to send in Ollama's `format` field."""
    return model.model_json_schema()
