# Harness: Session 2 starter code

Source code for **Session 2: Harness Foundation & Tools** (Sat 10 Oct 2026).
It implements modules **M1–M7** and **M14 (logging)** on top of a local Ollama model.
Session 3 adds the agent loop, state machine and router on top of this code.

## Setup (Windows, PowerShell)

```powershell
cd "Local LLM\Harness"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen3:4b-q4_K_M
ollama pull qwen3:8b-q4_K_M
ollama pull gemma3:12b-it-q4_K_M      # fallback profile (optional, slow on 8 GB)

python scripts/check_setup.py          # checks Ollama, models, prompts, one test call
python -m unittest discover -s tests   # offline tests, no Ollama needed
```

Requires Python 3.10+ and a running Ollama.
To use other models, edit `config.yaml`. No code changes are needed.

## Folder layout

| Path | Module | Slide |
|---|---|---|
| `harness/llm_client.py` | **M1** Runtime connector: native `/api/chat` or OpenAI-compatible `/v1` | M1 |
| `config.yaml`, `harness/config.py` | **M2** Model config: profiles `fast`, `reasoning`, `fallback` | M2 |
| `prompts/*.txt`, `harness/prompts.py` | **M3** Prompt manager: versioned templates `name@vN.txt`, ordered from stable to variable | M3 |
| `harness/schemas.py` | **M4** Structured output: `Ticket` Pydantic model → JSON schema in `format` | M4 |
| `harness/validation.py` | **M5** Validation & recovery: schema check → domain rules → retry / repair / fallback / safe failure | M5 |
| `harness/logger.py` | **M14** Logging: one JSONL line per LLM call and per tool call (`logs/calls.jsonl`) | Logging |
| `harness/tools.py` | **M6** Tool registry: `@tool(permission=...)`, `tool_defs()` | M6 |
| `harness/executor.py` | **M7** Tool executor: look up → validate → permission → timeout → ok/err | M7 |
| `harness/sample_tools.py` | `calculator`, `lookup_customer`, `current_datetime`, plus `update_credit_limit` (write, blocked by default) | Hands-on 3a |
| `harness/harness.py` | `Harness` facade: `h.llm(profile, messages, schema=..., tools=...)` | — |
| `harness/data_store.py`, `data/customers.csv` | Sample customer data used by tools and domain rules | — |
| `data/ticket_inputs.jsonl` | 20 test inputs with expected values (Hands-on 2) | Hands-on 2 |
| `data/tool_prompts.jsonl` | 10 tool-calling prompts with the expected tool (homework) | Homework |
| `scripts/` | Demos, hands-on and homework runners (below) | — |
| `tests/test_offline.py` | 15 offline tests using a fake client | — |

## Live demos (teacher)

```powershell
python scripts/demo_structured.py                      # M1–M5 on one message
python scripts/demo_structured.py "c-105: refund still not received after 30 days"
python scripts/demo_structured.py --prompt extract_ticket@v1 --no-repair "..."   # weaker prompt, no repair

python scripts/demo_tool_call.py                       # one tool-calling round trip
python scripts/demo_tool_call.py --profile reasoning "Which tier is C-103?"
python scripts/demo_tool_call.py --failure             # made-up tool, bad args, unknown customer,
                                                       # unsafe expression, permission denied
```

## Hands-on 2: validated structured output (30 min)

1. Read `llm_client.py`, `config.yaml` and `validation.py`.
2. Run the evaluation on 20 inputs with two models:
   ```powershell
   python scripts/hands_on2_eval.py                          # fast (4B) vs reasoning (8B)
   python scripts/hands_on2_eval.py --no-repair              # plain retry, to compare
   python scripts/hands_on2_eval.py --prompt extract_ticket@v1   # weaker prompt, to compare
   ```
3. Report **first-try validity**, **validity after repair**, field accuracy and mean latency.
   Per-input results are saved to `logs/hands_on2_results.csv`.
4. Extension: add one domain rule of your own to `check_ticket()` in `validation.py`.

Inputs `t14`, `t15` and `t19` are designed to trigger the domain rules or the ID pattern.

**Checkpoint:** show the TA one validated output and your `logs/calls.jsonl`.

## Hands-on 3a: your first tool call (20 min)

1. Read `tools.py`, `executor.py` and `sample_tools.py`.
2. Run `python scripts/demo_tool_call.py` with several prompts and both profiles.
3. Add a tool of your own in `sample_tools.py` (a Pydantic `args` class, a docstring and `@tool`).

**Checkpoint:** one end-to-end tool call, with `tool_call` events visible in the log.

## Homework: reliable tool calling (due before Session 3, Sat 17 Oct)

```powershell
python scripts/tool_call_eval.py                        # success rate per model on 10 prompts
python scripts/tool_call_eval.py --answer               # also generate the final answers
python scripts/tool_call_eval.py --tools lookup_customer,calculator   # offer fewer tools
```

Try at least two fixes (better tool descriptions, simpler argument schemas, prompt changes,
fewer tools, a larger model) and write a short note on which helped most.
Results are saved to `logs/tool_call_results.csv`.

## Reading the log

```python
import pandas as pd
df = pd.read_json("logs/calls.jsonl", lines=True)
df[df.event == "llm_call"].groupby("model")[["ttft_s", "gen_s", "wall_s"]].mean()
df[df.event == "llm_call"].groupby("model")["valid"].mean()
```

Events: `llm_call`, `tool_call`, `safe_failure`. Group rows from one task with `run_id`.

## Notes

- `think: false` in `config.yaml` turns off Qwen3 reasoning for faster, cleaner JSON.
  Remove the key for models without a reasoning mode (for example Gemma 3).
- Ollama durations are in nanoseconds. The client converts them to seconds.
  With non-streaming calls, `ttft_s` is approximated as load time + prompt time.
- Tool arguments may arrive as a dict (Ollama native) or a JSON string (OpenAI-compatible API).
  The executor accepts both.
