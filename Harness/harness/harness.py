"""The harness facade: ties connector (M1), config (M2), prompts (M3) and logging (M14).

    h = Harness()
    res = h.llm("fast", messages, schema=..., prompt_id="extract_ticket@v2")
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, load_config, DEFAULT_CONFIG
from .llm_client import ChatResult, LLMClient
from .logger import JsonlLogger, new_run_id
from .prompts import PromptManager


class Harness:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, client: LLMClient | None = None,
                 log_file: str | Path | None = None):
        self.config: Config = load_config(config_path)
        self.client = client or LLMClient(self.config.base_url, self.config.api,
                                          self.config.timeout)
        self.prompts = PromptManager()
        self.log = JsonlLogger(log_file or self.config.log_file)

    def llm(self, profile: str, messages: list[dict], *, schema: dict | None = None,
            tools: list[dict] | None = None, run_id: str | None = None,
            prompt_id: str | None = None, log: bool = True, **overrides) -> ChatResult:
        """Call the model behind a profile. Per-call overrides win over config."""
        p = self.config.profile(profile)
        options = {**p.options, **overrides}
        result = self.client.chat(p.model, messages, schema=schema, tools=tools,
                                  think=p.think, keep_alive=p.keep_alive, **options)
        if log:
            self.log.llm_call(run_id=run_id or new_run_id(), profile=profile,
                              result=result, prompt_id=prompt_id)
        return result
