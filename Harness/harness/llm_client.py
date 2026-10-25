"""M1 - Runtime connector.

One small client for every model call. Supports two APIs:

* ``native``  - Ollama ``/api/chat`` (full options + timing metadata)
* ``openai``  - OpenAI-compatible ``/v1/chat/completions`` (works with Ollama,
                vLLM, or any compatible server; fewer timing fields)

Everything else in the harness calls :meth:`LLMClient.chat`, which returns a
:class:`ChatResult` with the same fields regardless of the API used.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class LLMError(RuntimeError):
    """Base error for runtime problems."""


class LLMConnectionError(LLMError):
    """The runtime is not reachable (is Ollama running?)."""


class LLMTimeoutError(LLMError):
    """The request took longer than the timeout."""


class ModelNotFoundError(LLMError):
    """The model tag is not installed (run `ollama pull <tag>`)."""


NS = 1e9  # Ollama durations are in nanoseconds


@dataclass
class ChatResult:
    model: str
    content: str
    message: dict                     # assistant message, ready to append to history
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_s: float = 0.0
    prompt_s: float = 0.0
    gen_s: float = 0.0
    wall_s: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def ttft_s(self) -> float:
        """Approximate time to first token for a non-streaming call."""
        return self.load_s + self.prompt_s

    @property
    def gen_tps(self) -> float:
        return self.output_tokens / self.gen_s if self.gen_s else 0.0


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434", api: str = "native",
                 timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.timeout = timeout

    # ------------------------------------------------------------------ public
    def chat(self, model: str, messages: list[dict], schema: dict | None = None,
             tools: list[dict] | None = None, think: bool | None = None,
             keep_alive: str | None = None, **options: Any) -> ChatResult:
        """Send one chat request.

        schema   - JSON schema to constrain the output (structured output)
        tools    - tool definitions (tool calling)
        think    - reasoning on/off for models that support it (e.g. qwen3)
        options  - runtime options: temperature, num_ctx, num_predict, seed ...
        """
        if self.api == "openai":
            return self._chat_openai(model, messages, schema, tools, **options)
        return self._chat_native(model, messages, schema, tools, think, keep_alive, **options)

    def list_models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m["name"] for m in data.get("models", [])]

    # ----------------------------------------------------------------- native
    def _chat_native(self, model, messages, schema, tools, think, keep_alive, **options):
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                "options": options}
        if schema is not None:
            body["format"] = schema
        if tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = think
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        t0 = time.perf_counter()
        data = self._post("/api/chat", body, model)
        wall = time.perf_counter() - t0

        msg = data.get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        return ChatResult(
            model=model,
            content=msg.get("content", "") or "",
            message={k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")},
            tool_calls=calls,
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            load_s=data.get("load_duration", 0) / NS,
            prompt_s=data.get("prompt_eval_duration", 0) / NS,
            gen_s=data.get("eval_duration", 0) / NS,
            wall_s=wall,
            raw=data,
        )

    # ----------------------------------------------------------------- openai
    def _chat_openai(self, model, messages, schema, tools, **options):
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if "temperature" in options:
            body["temperature"] = options["temperature"]
        if "num_predict" in options:
            body["max_tokens"] = options["num_predict"]
        if "seed" in options:
            body["seed"] = options["seed"]
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": schema}}
        if tools:
            body["tools"] = tools

        t0 = time.perf_counter()
        data = self._post("/v1/chat/completions", body, model)
        wall = time.perf_counter() - t0

        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    pass  # keep the raw string; the executor will reject it
            calls.append({"id": c.get("id"), "function": {"name": c["function"]["name"],
                                                          "arguments": args}})
        usage = data.get("usage", {})
        return ChatResult(
            model=model,
            content=msg.get("content") or "",
            message={"role": "assistant", "content": msg.get("content") or "",
                     **({"tool_calls": calls} if calls else {})},
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            wall_s=wall,
            raw=data,
        )

    # ------------------------------------------------------------------- http
    def _post(self, path: str, body: dict, model: str) -> dict:
        try:
            r = requests.post(self.base_url + path, json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot reach {self.base_url}. Is Ollama running? ({e})") from e
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(f"Request to {model} timed out after {self.timeout}s") from e
        if r.status_code == 404 and "not found" in r.text.lower():
            raise ModelNotFoundError(f"Model {model!r} not found. Run: ollama pull {model}")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str) -> dict:
        try:
            r = requests.get(self.base_url + path, timeout=30)
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Cannot reach {self.base_url}. Is Ollama running?") from e
        r.raise_for_status()
        return r.json()
