"""M5 - Validation & recovery.

LLM output -> schema check (Pydantic) -> domain check (business rules)
  valid   -> continue
  invalid -> 1 retry -> 2 repair (send the error back) -> 3 fallback profile
             -> 4 safe failure (never execute invalid output)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .data_store import customer_exists
from .logger import new_run_id
from .schemas import Ticket

T = TypeVar("T", bound=BaseModel)


class DomainError(ValueError):
    """Output has the right shape but breaks a business rule."""


@dataclass
class StructuredResult(Generic[T]):
    ok: bool
    value: Optional[T] = None
    attempts: int = 0
    first_try_valid: bool = False
    profile_used: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    total_wall_s: float = 0.0

    @property
    def reason(self) -> str:
        return self.errors[-1] if self.errors else ""


# ----------------------------------------------------------- domain rules
def check_ticket(t: Ticket) -> None:
    """Business rules for tickets (schema already passed)."""
    if t.customer_id and not customer_exists(t.customer_id):
        raise DomainError(f"customer_id {t.customer_id} does not exist")
    if t.priority == 3 and t.category == "other":
        raise DomainError("priority 3 (urgent) is only allowed for billing or technical tickets")


# ------------------------------------------------------------- recovery loop
def _short(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc'])) or 'root'}: {x['msg']}"
                         for x in e.errors())[:400]
    return str(e)[:400]


def ask_structured(h, messages: list[dict], model_cls: type[T], *,
                   profiles: tuple[str, ...] = ("fast", "reasoning", "fallback"),
                   max_retry: int = 2, repair: bool = True,
                   domain_check: Callable[[T], None] | None = None,
                   prompt_id: str | None = None, run_id: str | None = None
                   ) -> StructuredResult[T]:
    """Get a validated object or a safe failure.

    For each profile: first attempt, then up to `max_retry` more attempts.
    With repair=True the previous bad output and its error are sent back.
    With repair=False the same prompt is simply retried.
    """
    run_id = run_id or new_run_id()
    schema = model_cls.model_json_schema()
    res: StructuredResult[T] = StructuredResult(ok=False)

    for profile in profiles:
        msgs = list(messages)
        for attempt in range(max_retry + 1):
            res.attempts += 1
            out = h.llm(profile, msgs, schema=schema, run_id=run_id, prompt_id=prompt_id,
                        log=False)
            res.total_wall_s += out.wall_s
            try:
                obj = model_cls.model_validate_json(out.content)
                if domain_check:
                    domain_check(obj)
            except (ValidationError, DomainError, json.JSONDecodeError) as e:
                err = _short(e)
                res.errors.append(f"{profile}#{attempt}: {err}")
                h.log.llm_call(run_id=run_id, profile=profile, result=out, prompt_id=prompt_id,
                               valid=False, error=err, retry=attempt)
                if repair:
                    msgs = msgs + [
                        {"role": "assistant", "content": out.content},
                        {"role": "user", "content": f"Invalid output: {err}. "
                                                    "Fix it and return JSON only, matching the schema."}]
                continue
            h.log.llm_call(run_id=run_id, profile=profile, result=out, prompt_id=prompt_id,
                           valid=True, retry=attempt)
            res.ok, res.value, res.profile_used = True, obj, profile
            res.first_try_valid = res.attempts == 1
            return res

    # 4. safe failure: report, never execute
    h.log.write("safe_failure", run_id=run_id, prompt=prompt_id, reason=res.reason,
                attempts=res.attempts)
    return res
