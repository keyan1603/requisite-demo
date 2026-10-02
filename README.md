# requisite-demo

A standalone project that installs [`requisite-ai`](https://pypi.org/project/requisite-ai/)
fresh from PyPI (not from the framework's own source checkout) and
exercises its features end-to-end, the way an external consumer actually
would. Originally built to verify the 0.30.0 release -- including the
[18 fixes from the ADR-0031 code review pass](https://github.com/requisite-ai/requisite-ai/blob/main/docs/adr/0031-code-review-fixes.md) --
against the real published package, and kept up to date since as a
working "how do I actually use this" reference. Currently verifies
**0.37.0**, including all thirteen multi-agent strategies (`consensus`,
`map_reduce`, `critic`, `debate`, `tree_of_thoughts`, and `planner` were
added to this repo alongside 0.34.0, `reflexion` alongside 0.36.0,
`CostLimiter` alongside 0.37.0, matching what shipped in 0.31.0-0.37.0)
on both the `native` and `langgraph` backends. A `demos/11_adk_orchestrator.py`
demo is also included for the `adk` orchestrator backend shipping in
0.38.0, but is **not yet verified here** -- 0.38.0 isn't published to
PyPI yet, and this repo only installs fresh from PyPI, not from source.

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

Several demos (`demos/01_quickstart.py`, `demos/02_workflows_native.py`,
`demos/05_rag.py`, `demos/09_prompts.py`) include a few checks near the
end that use a scripted provider instead of a live model call and need no
API key at all. Running one of those files end to end still needs a key
though, since the real-call sections run first in the same file; import
the individual function and call it directly if you want a no-key sanity
check on its own. Everything else needs `GEMINI_API_KEY` set.

## What each demo covers

| File | Covers |
|---|---|
| `demos/01_quickstart.py` | `AI` (one-shot chat), `Agent` + `@tool` (tool-calling loop), and how the tool-calling loop handles a hallucinated tool call, a partial failure in a concurrent batch, and a tool with an unresolvable type hint |
| `demos/02_workflows_native.py` | All thirteen multi-agent strategies on the default backend: `sequential`, `parallel`, `consensus`, `map_reduce`, `supervisor`, `planner`, `hierarchical` (nested `Workflow` as a delegate), `reflection`, `critic`, `debate`, `tree_of_thoughts`, `reflexion`, `graph` (developer-declared routing), and how a self-referential hierarchical delegation is handled |
| `demos/03_workflows_langgraph.py` | The same thirteen strategies, switched to the LangGraph execution backend via `.use_langgraph()` -- one-line config change, same `.add()`/`.run()` API |
| `demos/04_memory.py` | `InProcessMemory`, `SQLiteMemory` (persists across restarts), `VectorMemory` (semantic recall via `load_relevant()`) |
| `demos/05_rag.py` | `Retriever` (dense/embedding), `BM25Retriever` (keyword), `HybridRetriever` (fused via RRF), `LLMReranker`, `LLMContextCompressor`, exposing a retriever as an agent capability |
| `demos/06_mcp_server.py` | A self-contained MCP server exposing Requisite tools + an agent (not run directly -- spawned by `07`) |
| `demos/07_mcp_client.py` | `MCPClient` in both default (per-call-reconnect) and opt-in persistent-session mode, plus a timing comparison between the two |
| `demos/08_capabilities.py` | `agent.requires(...)` -- built-in `weather`/`internet_search`/`github` capabilities, and overriding one with a higher-priority provider |
| `demos/09_prompts.py` | `PromptTemplate`/`ChatPromptTemplate`, plus two checks on dotted-field validation and `partial()` injection safety |
| `demos/10_cost_limiter.py` | `CostLimiter`/`cost_per_token()` -- capping real dollar spend on a tiny budget until it genuinely exhausts mid-run, then `reset()` |
| `demos/11_adk_orchestrator.py` | The `adk` orchestrator backend (`workflow.use_adk()`) -- `sequential` and `supervisor` strategies, delegating coordination to Google's Agent Development Kit. Requires `pip install google-adk` and `requisite-ai>=0.38.0`; **not yet verified here** (0.38.0 not yet on PyPI) |

## Verified

All 9 runnable demos except `demos/11_adk_orchestrator.py` pass end to
end against real Gemini output (0.37.0) -- `demos/11_adk_orchestrator.py`
needs `requisite-ai>=0.38.0`, not yet published to PyPI, so it isn't
verified here yet (see the note above). This includes all thirteen
multi-agent strategies (`sequential`, `parallel`,
`consensus`, `map_reduce`, `supervisor`, `planner`, `hierarchical`,
`reflection`, `critic`, `debate`, `tree_of_thoughts`, `reflexion`,
`graph`) on both the `native` and `langgraph` backends -- verified as
individual runs of each demo, not necessarily all 9 back to back in one
`python run_all.py` pass. `demos/04_memory.py`'s `VectorMemory` and
`demos/05_rag.py`'s embedding-backed retrievers share a separate,
noticeably tighter Vertex AI embedding quota
(`aiplatform.googleapis.com/global_embed_content_requests_per_minute_per_base_model`)
from the chat-generation quota `RateLimiter` already paces around --
running both demos back to back (or twice in quick succession) can
exhaust it, even though each one passes cleanly on its own. If you hit
this, space embedding-heavy demos out by a minute or so rather than
running the full suite in one uninterrupted pass. A couple of results
worth calling out:

- The AI facade and tool-calling agent in `demos/01_quickstart.py`
  return real Gemini responses, including correct tool selection for a
  prompt that needs two different tools in one turn.
- `demos/10_cost_limiter.py`'s deliberately tiny `budget_usd` genuinely
  exhausts on the 3rd call (not just asserted) -- the call is blocked
  with a real `CostLimitException` before it ever reaches Gemini.
- The persistent-session MCP timing comparison
  (`demos/07_mcp_client.py`) measured a **~956x speedup** for 10 calls
  (default per-call-reconnect mode vs. `async with client:`), consistent
  with the numbers in
  [ADR-0030](https://github.com/requisite-ai/requisite-ai/blob/main/docs/adr/0030-mcp-persistent-session-mode.md)
  (the exact multiplier varies run to run with local network conditions).
- The scripted, no-key checks in `demos/01_quickstart.py` and
  `demos/02_workflows_native.py` (hallucinated tool recovery, a partial
  failure in a concurrent tool-call batch, an unresolvable type hint on
  `@tool`, and a self-referential hierarchical delegation) all pass
  against the real installed package, independently of the framework's
  own test suite.

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
- **429 specifically from `demos/04_memory.py` or `demos/05_rag.py`,
  mentioning `global_embed_content_requests_per_minute_per_base_model`**
  -- a separate, noticeably tighter Vertex AI quota for embedding calls,
  not the chat-generation quota `RateLimiter` paces around (neither
  `GeminiEmbeddingProvider` nor `Retriever`/`VectorMemory` share the
  shared `RateLimiter` today). Run the two embedding-heavy demos apart
  from each other by a minute or so rather than back to back.
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
