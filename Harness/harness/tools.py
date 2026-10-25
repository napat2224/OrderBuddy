"""M6 - Tool registry.

Every tool is declared, typed and permissioned:

    class LookupArgs(BaseModel):
        customer_id: str

    @tool(permission="read")
    def lookup_customer(args: LookupArgs) -> dict:
        '''Return the customer record for an ID. Use when the user names a customer.'''
        ...

`tool_defs()` exports the registry in the format Ollama's `tools` field expects.
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

PERMISSIONS = ("read", "write", "dangerous")


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    args: type[BaseModel]
    description: str
    permission: str = "read"
    timeout: float = 10.0

    def definition(self) -> dict:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": schema}}


TOOLS: dict[str, ToolSpec] = {}


def tool(permission: str = "read", timeout: float = 10.0, name: str | None = None):
    """Decorator that registers a function taking one Pydantic `args` parameter."""
    if permission not in PERMISSIONS:
        raise ValueError(f"permission must be one of {PERMISSIONS}")

    def wrap(fn: Callable) -> Callable:
        params = list(inspect.signature(fn).parameters)
        hints = typing.get_type_hints(fn)
        arg_type = hints.get(params[0]) if len(params) == 1 else None
        if not (isinstance(arg_type, type) and issubclass(arg_type, BaseModel)):
            raise TypeError(f"{fn.__name__}: tools take exactly one Pydantic `args` parameter")
        doc = inspect.getdoc(fn) or ""
        if not doc:
            raise ValueError(f"{fn.__name__}: a description (docstring) is required")
        spec = ToolSpec(name=name or fn.__name__, fn=fn, args=arg_type,
                        description=" ".join(doc.split()), permission=permission,
                        timeout=timeout)
        TOOLS[spec.name] = spec
        return fn
    return wrap


def tool_defs(names: list[str] | None = None) -> list[dict]:
    """Tool definitions for the model. Pass `names` to offer only a subset."""
    specs = TOOLS.values() if names is None else [TOOLS[n] for n in names]
    return [s.definition() for s in specs]
