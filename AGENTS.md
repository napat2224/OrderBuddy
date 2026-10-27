# AGENTS.md: OrderBuddy

Read this before you change anything. It is for teammates and for AI coding agents
(Claude Code, Codex, Cursor...). Keep it up to date when you change a contract below.

## What this project is

A local LINE chat-to-order agent for a Thai restaurant (Agentic AI class, Group 01).
A customer sends text or a photo in LINE -> the agent builds an order with local
models (Ollama) and 5 tools -> the customer confirms -> staff approve.

```
LINE -> ngrok -> app/main.py (/webhook) -> app/state_machine.py
     -> Harness (models via Ollama, tools via executor) -> drafts/, orders/ (JSON files)
```

Models (profiles in `Harness/config.yaml`): `router` + `fast` = qwen3:4b,
`reasoning` = qwen3:8b, `vision` = gemma3:4b. There is no database.

## Layout and owners

| Path | What | Owner |
|---|---|---|
| `Harness/harness/` | Course harness M1-M7, M14. **Do not edit.** | teacher |
| `Harness/config.yaml` | Model profiles (M2) | เปรม |
| `Harness/prompts/*@vN.txt` | Prompts (M3) | ฟลุ๊ค |
| `app/schemas.py` | Shared Pydantic schemas (M4) | เปรม |
| `app/state_machine.py` | States, transitions, permissions (M9, M19) | เปรม |
| `app/agent.py` | Tool-calling loop in RESOLVE (M8) | เปรม |
| `app/session.py` | Drafts, chat history, orders on disk (M12) | เปรม |
| `app/main.py`, `app/line_adapter.py` | FastAPI + LINE (M17) | เปรม |
| `app/router.py` | Intent + profile choice: rules, model, guard (M10-M11) | ฟลุ๊ค |
| `app/checks.py` | Domain checks on model output, reply guard (M5) | ฟลุ๊ค |
| `app/tools/__init__.py` | The 5 tools (M6) | บีม |
| `app/catalog.py`, `data/*.json` | Menu, stock, shop settings | บีม |
| `eval/` | Test tasks, metrics, C1-C3 comparison (M20) | บีม |
| `tests/` | Offline tests (no Ollama, no LINE) | everyone |

Full spec: the "Chat-to-Order Agent: System Spec (Part A)" doc.

## Setup and commands

Run from the `OrderBuddy/` folder. Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v                # our tests, no Ollama needed
(cd Harness && python -m unittest discover -s tests)   # course tests must still pass

# live (needs Ollama + LINE keys in .env, see .env.example)
uvicorn app.main:app --port 8000
ngrok http 8000
```

**Before you push: both test commands must pass.** If you change behaviour, add or
update a test in `tests/`. The FakeClient in `tests/test_agent_offline.py` plays the
models, so you can test the whole flow without a GPU.

## Contracts: do not break these

Other people's code depends on them. To change one, tell the team and update this
file, the spec doc and the tests in the same commit.

1. **Schemas** (`app/schemas.py`): field names and types of `RouteResult`,
   `ExtractedOrder`, `DraftOrder`, `FinalOrder`. All use `extra="forbid"`.
2. **Tool names, Args and permissions** (`app/tools/__init__.py`):

   | Tool | Permission | Called by |
   |---|---|---|
   | `product_search(query, top_k)` | read | model (RESOLVE) + code |
   | `stock_checker(sku, qty)` | read | model (RESOLVE) + code |
   | `price_calculator(items, fulfillment)` | read | code only |
   | `address_validator(address, phone, need_address)` | read | code only |
   | `draft_order_manager(action, line_user_id)` | **write** | code only |

   Result shapes are in each tool's docstring. Errors: `raise ValueError("<code>: <detail>")`.
3. **Router API** (`app/router.py`): `classify(...) -> RouteResult` never raises;
   `pick_profile(route, has_image) -> "fast" | "reasoning" | "vision"`.
4. **Prompt variables**: `route` ($open_order, $last_bot), `extract` ($menu, $current),
   `extract_image` ($menu, $current, $caption), `clarify` ($order, $problems,
   $suggestions), `answer` ($menu), `resolve` (none).
5. **Profile names** in `config.yaml`: `router`, `fast`, `reasoning`, `vision`.
6. **Agent entry point**: `OrderAgent.handle(line_user_id, text="", image_b64=None) -> list[str]`.

## Safety rules (M19): never weaken

- The model only ever gets `tool_defs(MODEL_TOOLS)` = product_search + stock_checker.
  Never give it `draft_order_manager` or any write tool.
- Code passes `allowed={"read"}` to `execute()` by default; `{"read", "write"}` only in
  DONE (submit) and cancel.
- An order is written to `orders/` only when the previous state is CONFIRM, `missing`
  is empty AND the newest message has intent `confirm`. The tool checks it again.
- Money and stock are computed by code (`price_calculator`, `stock_checker`), never by
  a model. Every sku from a model is re-checked against the catalog.
- Stock is cut only when staff approve (`POST /orders/{id}/approve`).

## How to change things

- **Prompts**: never edit a version that has eval results. Copy `extract@v1.txt` to
  `extract@v2.txt` and change the copy; the newest version is used automatically.
  Use `$name` only for variables (it is `string.Template`); don't write a `$` anywhere else.
- **Tools**: one Pydantic `args` parameter, a docstring (the model reads it as the
  tool description), `@tool(permission=...)`. Keep tools deterministic and fast (< 10 s).
- **Schemas the model fills** (`RouteResult`, `ExtractedOrder`): keep them small and
  flat; 4B models fail on long nested schemas. Use `Literal`, not Enum.
- **Models**: change `Harness/config.yaml`, not code. Keep `temperature: 0`.
- **Menu**: edit `data/catalog.json` (validated by `CatalogItem`) and `data/stock.json`.
  SKUs look like `F001`.
- **Calls to the model**: always through `h.llm(...)` or `ask_structured(...)` so they
  are logged. Call tools from code through `agent.call_tool(...)` so they are logged
  and permission-checked.
- **New state**: add it to `State` in `schemas.py`, the state table in the spec doc and
  a test.

## Do not commit

`.env` (LINE keys), `drafts/`, `sessions/`, `orders/` (customer data), `logs/`,
`.venv/`, model files. They are in `.gitignore`. Use made-up names, phones and
addresses in tests and examples, never real customer data.

## Style

- Python: type hints, small functions, docstrings say *why*. Comments in English.
- Customer-facing text is Thai and ends with ครับ.
- Prices are integer THB.
- Logs: `Harness/logs/calls.jsonl`; one `run_id` per LINE message. Events: `message`,
  `route`, `route_source` (rule / model / guard / fallback), `state`, `llm_call`,
  `tool_call`, `safe_failure`, `agent_error`.
