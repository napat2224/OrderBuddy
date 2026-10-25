"""M3 - Prompt manager.

Prompts are versioned template files in ``prompts/`` named ``<name>@v<N>.txt``.
``get("extract_ticket")`` returns the highest version; ``get("extract_ticket@v1")``
pins one. The prompt id (e.g. ``extract_ticket@v2``) is logged with every call.

``build_messages`` orders content from MOST STABLE to MOST VARIABLE so that
identical beginnings can be reused by prefix caching (Session 3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "prompts"
_NAME = re.compile(r"^(?P<name>[\w\-]+)@v(?P<ver>\d+)\.txt$")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **variables) -> str:
        # $variable syntax; raises KeyError if a variable is missing
        return Template(self.template).substitute(**variables)


class PromptManager:
    def __init__(self, directory: str | Path = PROMPT_DIR):
        self.directory = Path(directory)
        self._prompts: dict[str, dict[int, Prompt]] = {}
        for f in sorted(self.directory.glob("*.txt")):
            m = _NAME.match(f.name)
            if not m:
                continue
            p = Prompt(m["name"], int(m["ver"]), f.read_text(encoding="utf-8").strip())
            self._prompts.setdefault(p.name, {})[p.version] = p

    def get(self, ref: str) -> Prompt:
        name, _, ver = ref.partition("@v")
        versions = self._prompts.get(name)
        if not versions:
            raise KeyError(f"No prompt named {name!r} in {self.directory}")
        return versions[int(ver)] if ver else versions[max(versions)]

    def names(self) -> list[str]:
        return [p.id for v in self._prompts.values() for p in v.values()]


def build_messages(system: str, *, policies: str = "", static_context: str = "",
                   history: list[dict] | None = None, observations: list[dict] | None = None,
                   user: str | None = None) -> list[dict]:
    """Assemble messages from most stable to most variable.

    system prompt -> policies -> static domain context -> history
    -> tool observations -> newest user input
    (Tool definitions and output schemas are sent as request fields.)
    """
    head = "\n\n".join(x for x in (system, policies, static_context) if x)
    msgs = [{"role": "system", "content": head}]
    msgs += history or []
    msgs += observations or []
    if user is not None:
        msgs.append({"role": "user", "content": user})
    return msgs
