"""M2 - Model config.

Code asks for a *profile* ("fast", "reasoning", "fallback"), never a model tag.
Swapping a model is a change in config.yaml, not in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# Keys sent as Ollama "options"; everything else is handled separately.
OPTION_KEYS = {"temperature", "num_ctx", "num_predict", "seed", "top_p", "top_k"}


@dataclass
class Profile:
    name: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    think: bool | None = None
    keep_alive: str | None = None


@dataclass
class Config:
    base_url: str
    api: str
    timeout: float
    profiles: dict[str, Profile]
    log_file: Path

    def profile(self, name: str) -> Profile:
        if name not in self.profiles:
            raise KeyError(f"Unknown profile {name!r}. Known: {list(self.profiles)}")
        return self.profiles[name]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rt = data.get("runtime", {})
    profiles = {}
    for name, p in data["profiles"].items():
        profiles[name] = Profile(
            name=name,
            model=p["model"],
            options={k: v for k, v in p.items() if k in OPTION_KEYS},
            think=p.get("think"),
            keep_alive=p.get("keep_alive"),
        )
    log_file = ROOT / data.get("logging", {}).get("file", "logs/calls.jsonl")
    return Config(
        base_url=rt.get("base_url", "http://localhost:11434"),
        api=rt.get("api", "native"),
        timeout=float(rt.get("timeout_s", 300)),
        profiles=profiles,
        log_file=log_file,
    )
