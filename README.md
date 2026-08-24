# requisite-demo

A standalone project that installs [`requisite-ai`](https://pypi.org/project/requisite-ai/)
0.30.0 fresh from PyPI (not from the framework's own source checkout)
and exercises its features end-to-end, the way an external consumer
actually would. Built to verify the 0.30.0 release -- including the
[18 fixes from the ADR-0031 code review pass](https://github.com/requisite-ai/requisite-ai/blob/main/docs/adr/0031-code-review-fixes.md) --
against the real published package, and to double as a working
"how do I actually use this" reference.

## Setup

The virtual environment and dependencies are already installed. If you
ever need to recreate it from scratch:

```bash
python -m venv .venv
.venv\Scripts\pip install "requisite-ai[all]"
```

`[all]` pulls in every optional provider/backend except CrewAI (CrewAI's
own dependencies hard-pin an older `mcp` version that conflicts with
this package's `mcp>=2.0` requirement -- see the main repo's README for
details). That's OpenAI, Gemini, Anthropic, Ollama, LangGraph, MCP,
Pinecone, Weaviate, Redis, OpenTelemetry, and AutoGen.

### Add your API key

1. Open `.env` in this folder.
2. Fill in `GEMINI_API_KEY=` with your key.
3. Save.

Every demo defaults to the Gemini provider (`DEFAULT_PROVIDER=gemini` in
`.env`) since that's the key this project was set up with. To use a
different provider instead, add its key to `.env` (see `.env.example`
for the full list) and either change `DEFAULT_PROVIDER`/`MODEL`, or edit
an individual demo's `Agent(provider=..., model=...)` calls directly.

**Rate limiting**: every demo builds its `Agent`/`AI` instances through
`demos/_shared.py`'s `make_agent()`/`make_ai()`, which share *one*
`RateLimiter(requests_per_minute=8)` for the whole run. That single
shared instance is the point: constructing `Agent()`/`AI()` directly
instead would give each one its own private limiter (built from `.env`'s
`RATE_LIMIT_RPM`), and several separately-paced limiters can each look
fine on their own while their combined real call rate against one API
key still exceeds the actual free-tier quota. `run_all.py` imports and
runs every demo in one continuous process rather than as separate
subprocesses, specifically so this one `RateLimiter` has complete,
accurate visibility into every real call made during the whole run.
Raise `requests_per_minute` in `demos/_shared.py` if you're on a paid
tier with a higher quota.

## Running the demos

Run everything, in order, with a pass/fail summary at the end:

```bash
.venv\Scripts\python run_all.py
```

Or run any single demo directly:

```bash
.venv\Scripts\python demos\01_quickstart.py
```

Demos that don't need a live model call at all (`demos/10_recent_fixes.py`,
and the ADR-0031 checks inside `demos/05_rag.py`/`demos/09_prompts.py`)
will pass even before you add a key. Everything else needs
`GEMINI_API_KEY` set.

## What each demo covers

| File | Covers |
|---|---|
| `demos/01_quickstart.py` | `AI` (one-shot chat), `Agent` + `@tool` (tool-calling loop) |
| `demos/02_workflows_native.py` | Multi-agent orchestration on the default backend: `sequential`, `parallel`, `supervisor`, `hierarchical` (nested `Workflow` as a delegate), `reflection`, `graph` (developer-declared routing) |
| `demos/03_workflows_langgraph.py` | The same strategies, switched to the LangGraph execution backend via `.use_langgraph()` -- one-line config change, same `.add()`/`.run()` API |
| `demos/04_memory.py` | `InProcessMemory`, `SQLiteMemory` (persists across restarts), `VectorMemory` (semantic recall via `load_relevant()`) |
| `demos/05_rag.py` | `Retriever` (dense/embedding), `BM25Retriever` (keyword), `HybridRetriever` (fused via RRF), `LLMReranker`, `LLMContextCompressor`, exposing a retriever as an agent capability |
| `demos/06_mcp_server.py` | A self-contained MCP server exposing Requisite tools + an agent (not run directly -- spawned by `07`) |
| `demos/07_mcp_client.py` | `MCPClient` in both default (per-call-reconnect) and opt-in persistent-session mode, plus a timing comparison between the two |
| `demos/08_capabilities.py` | `agent.requires(...)` -- built-in `weather`/`internet_search`/`github` capabilities, and overriding one with a higher-priority provider |
| `demos/09_prompts.py` | `PromptTemplate`/`ChatPromptTemplate`, plus two ADR-0031 regression checks (dotted-field validation, `partial()` injection) |
| `demos/10_recent_fixes.py` | Structural verification of the four most severe ADR-0031 fixes, using scripted (no live model call needed) providers -- see below |

## About `demos/10_recent_fixes.py`

This one is worth calling out specifically: it re-verifies the four
most severe bugs found and fixed in the 0.30.0 code review pass,
*against the real installed package* rather than the framework's own
internal test suite. It uses scripted fake providers (no live model
call, so it runs even before you add a Gemini key) to deterministically
reproduce each original bug scenario:

1. **A `Workflow` that delegates to itself** (directly, or through a
   cycle of other `Workflow`s) under `hierarchical`/`graph` used to
   crash with an uncatchable `RecursionError` -- now raises a clean,
   immediate `ConfigurationException` instead.
2. **A hallucinated/unknown tool call** used to abort the entire
   `Agent.run()` on the very first bad call, regardless of
   `max_iterations` -- now the model sees the failure and can retry.
3. **One failing tool call in a concurrent batch** (`Agent.arun()`
   executing several tool calls at once) used to leave the other,
   still-succeeding calls silently abandoned -- now every call in the
   batch completes and is reported.
4. **`@tool` on a function with an unresolvable type hint** (e.g. a
   `TYPE_CHECKING`-only import) used to crash with a raw `NameError` --
   now degrades to a permissive schema, as documented.

Run it on its own for a fast, no-API-key sanity check that the release
is behaving as documented:

```bash
.venv\Scripts\python demos\10_recent_fixes.py
```

## Verified

All 9 demos pass end to end against real Gemini output (`python
run_all.py`, full run, 0.30.0). A couple of results worth calling out:

- The AI facade and tool-calling agent in `demos/01_quickstart.py`
  return real Gemini responses, including correct tool selection for a
  prompt that needs two different tools in one turn.
- The persistent-session MCP timing comparison
  (`demos/07_mcp_client.py`) measured a **~1000x speedup** for 10 calls
  (default per-call-reconnect mode vs. `async with client:`), consistent
  with the numbers in
  [ADR-0030](https://github.com/requisite-ai/requisite-ai/blob/main/docs/adr/0030-mcp-persistent-session-mode.md).
- `demos/10_recent_fixes.py`'s 6 structural checks all pass against the
  real installed package, independently of the framework's own test
  suite. This one needs no API key, so it is the fastest way to sanity
  check a fresh install.

## Troubleshooting

- **`ConfigurationException: No provider specified and no default_provider configured`**
  -- `.env` wasn't found or `DEFAULT_PROVIDER`/`GEMINI_API_KEY` aren't
  set. Confirm you're running from this folder (`.env` is read from the
  current working directory) and that you filled in the key.
- **`ConfigurationException: Missing API key for GeminiEmbeddingProvider`**
  even though `.env` has the key -- `GeminiEmbeddingProvider()`'s bare
  constructor reads the *real OS environment* (`os.environ`) directly,
  not `.env` (only `Settings()`, used internally by `Agent`/`AI`, parses
  `.env`). Pass the key explicitly:
  `GeminiEmbeddingProvider(api_key=gemini_api_key())`, using the
  `gemini_api_key()` helper from `demos/_shared.py` (already done in
  `demos/04_memory.py` and `demos/05_rag.py`) -- don't rely on the bare
  constructor if a real shell environment variable isn't also set.
- **429 / rate-limit errors from Gemini** -- see "Rate limiting" above;
  make sure any new `Agent`/`AI` you add goes through
  `demos/_shared.py`'s `make_agent()`/`make_ai()`, not the constructors
  directly.
- **`demos/06_mcp_server.py` "fails" when run directly** -- expected,
  it's a server meant to be spawned as a subprocess by
  `demos/07_mcp_client.py`, not run standalone (`run_all.py` skips it
  for this reason).
- **LangGraph/AutoGen-specific errors** -- confirm the extras installed
  correctly: `.venv\Scripts\python -c "import langgraph, autogen_agentchat"`.

## Links

- Package: <https://pypi.org/project/requisite-ai/>
- Source + docs: <https://github.com/requisite-ai/requisite-ai>
- Full ADR trail (design decisions behind every feature exercised
  above): <https://github.com/requisite-ai/requisite-ai/tree/main/docs/adr>
- Requisite is open source and welcomes contributors: see
  [CONTRIBUTING.md](https://github.com/requisite-ai/requisite-ai/blob/main/CONTRIBUTING.md)
  in the main repo for setup, extension-point walkthroughs, and the PR
  process.
