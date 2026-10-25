"""Check that Ollama is running and every model in config.yaml is installed.

    python scripts/check_setup.py
"""
import _path  # noqa: F401

from harness import Harness, LLMConnectionError


def main():
    h = Harness()
    print(f"Runtime: {h.config.base_url} (api={h.config.api})")
    try:
        installed = set(h.client.list_models())
    except LLMConnectionError as e:
        print(f"[FAIL] {e}")
        print("       Start Ollama (open the Ollama app or run `ollama serve`) and try again.")
        return 1
    print(f"[ OK ] Ollama reachable, {len(installed)} models installed")

    missing = 0
    for name, p in h.config.profiles.items():
        tag = p.model if ":" in p.model else p.model + ":latest"
        if tag in installed:
            print(f"[ OK ] profile {name:<10} -> {p.model}")
        else:
            missing += 1
            print(f"[MISS] profile {name:<10} -> {p.model}   run: ollama pull {p.model}")

    print(f"[ OK ] prompts: {', '.join(h.prompts.names())}")
    if missing:
        print(f"\n{missing} model(s) missing. Pull them or edit config.yaml.")
        return 1

    r = h.llm("fast", [{"role": "user", "content": "Reply with the single word: ready"}],
              num_predict=16)
    print(f"[ OK ] test call: {r.content.strip()!r}  ({r.wall_s:.2f}s, {r.gen_tps:.1f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
