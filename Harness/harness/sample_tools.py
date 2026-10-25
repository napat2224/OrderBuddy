"""Three sample tools for Hands-on 3a: calculator, customer lookup, current date/time.

Importing this module registers them in the tool registry.
Add your own tools here (or in a new module) with the @tool decorator.
"""
from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .data_store import get_customer
from .tools import tool

# ------------------------------------------------------------------ calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


class CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '750000 - 310000' or '(12*3.5)/2'")


@tool(permission="read")
def calculator(args: CalcArgs) -> dict:
    """Evaluate an arithmetic expression exactly. Use for any calculation instead of doing math yourself."""
    value = _eval(ast.parse(args.expression.replace(",", ""), mode="eval"))
    return {"expression": args.expression, "value": round(value, 6)}


# ------------------------------------------------------------- customer lookup
class LookupArgs(BaseModel):
    customer_id: str = Field(description="Customer ID in the format C-123")


@tool(permission="read")
def lookup_customer(args: LookupArgs) -> dict:
    """Return the customer record (name, tier, credit limit, current balance in THB) for a customer ID.
    Use when the user mentions a customer ID."""
    rec = get_customer(args.customer_id)
    if rec is None:
        raise LookupError(f"No customer with ID {args.customer_id}")
    rec["available_credit_thb"] = rec["credit_limit_thb"] - rec["balance_thb"]
    return rec


# ------------------------------------------------------------ current datetime
class TimeArgs(BaseModel):
    timezone: str = Field(default="Asia/Bangkok", description="IANA time zone, e.g. Asia/Bangkok")


@tool(permission="read")
def current_datetime(args: TimeArgs) -> dict:
    """Return the current date, time and weekday. Use when the answer depends on today's date or time."""
    now = datetime.now(ZoneInfo(args.timezone))
    return {"iso": now.isoformat(timespec="seconds"), "date": now.date().isoformat(),
            "weekday": now.strftime("%A"), "timezone": args.timezone}


# ----------------------------------------- example of a non-read tool (blocked)
class CreditArgs(BaseModel):
    customer_id: str
    new_limit_thb: int = Field(gt=0)


@tool(permission="write")
def update_credit_limit(args: CreditArgs) -> dict:
    """Change a customer's credit limit. Requires write permission."""
    return {"customer_id": args.customer_id, "new_limit_thb": args.new_limit_thb,
            "status": "not implemented in the starter repo"}
