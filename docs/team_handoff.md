# Module Guide

A concise reference for every module in the codebase — what it does, what it owns, and what to look at first when something goes wrong.

---

## `scraper/`

The knowledge base ingestion pipeline. Run once (or on-demand to refresh content).

| File | Purpose |
|---|---|
| `discover.py` | Walks archived category pages to build a list of article URLs; merges `seed_urls.txt` as fallback |
| `wayback.py` | CDX API lookup + `id_` raw HTML fetch + HTML → clean text extraction |
| `ingest.py` | Orchestrates discovery and fetch; writes `articles.jsonl`, `manifest.csv`, `manifest.json`, `robots_check.json` |
| `robots.py` | Checks `robots.txt` before fetching; logs allowed/disallowed decisions |
| `seed_urls.txt` | Hand-curated fallback URL list merged with discovered URLs |

Run: `python -m scraper.ingest --out data/corpus --min-articles 60`

---

## `scripts/`

| File | Purpose |
|---|---|
| `build_kb.py` | Reads `articles.jsonl`, chunks via `app/retrieval/chunking.py`, embeds via `app/retrieval/embeddings.py`, builds FAISS index + metadata JSONL |

Run: `python scripts/build_kb.py`

---

## `app/core/`

| File | Purpose |
|---|---|
| `config.py` | `Settings` class (Pydantic BaseSettings). All configuration is read from `.env` through this object. Call `get_settings()` anywhere to access config. |
| `logging.py` | Structured logging setup (`setup_logging()`). Called once at startup in `backend/main.py` and `frontend/streamlit_app.py`. |

**When to touch:** Adding a new env variable → add a field to `Settings`. Changing log format → `logging.py`.

---

## `app/llm/`

| File | Purpose |
|---|---|
| `client.py` | `LLMClient` wrapper around the OpenAI SDK. Provides `chat()`, `chat_json()`, and `stream()`. Retries on transient errors; skips retry on authentication errors. Get the singleton via `get_llm_client()`. |
| `prompts.py` | All LLM system prompts: `ROUTER_SYSTEM`, `KB_QA_SYSTEM`, `SAFETY_CLASSIFIER_SYSTEM`. Edit prompts here only. |

**When to touch:** LLM behaving incorrectly → `prompts.py`. Endpoint/auth issues → `client.py`.

---

## `app/retrieval/`

| File | Purpose |
|---|---|
| `chunking.py` | Splits article text into heading/paragraph-aware chunks with stable `chunk_id`. Keeps article metadata on every chunk. |
| `embeddings.py` | Calls the remote embedding API; L2-normalises the result. `embed_texts(list[str]) → np.ndarray`. |
| `index_faiss.py` | Builds and saves `faiss.index` + `faiss_meta.jsonl` from a corpus file. Called by `build_kb.py`. |
| `retriever.py` | `HybridRetriever` class. Loads FAISS index and BM25 corpus at init. `retrieve(query, top_k) → list[RetrievedChunk]`. Get the singleton via `get_retriever()`. |

**When to touch:** Retrieval quality issues → `retriever.py` (adjust fusion weights or top-k). Embedding issues → `embeddings.py`. Re-indexing after corpus change → run `build_kb.py`.

---

## `app/agent/`

The core agent logic. Start in `graph.py` for the big picture.

| File | Purpose |
|---|---|
| `graph.py` | LangGraph pipeline definition. Five nodes: `load`, `guard`, `intent`, `dispatch`, `persist`. All routing logic, memory shortcuts, and action orchestration live in `node_dispatch`. |
| `router.py` | `classify_intent(messages, user_text) → IntentResult`. Calls the LLM with `ROUTER_SYSTEM` and parses JSON. Falls back to `AMBIGUOUS` on failure. |
| `guardrails.py` | `run_guardrails(user_text, skip_llm) → GuardResult`. Regex prescreen always runs; LLM classifier runs only when `skip_llm=False`. |
| `qa.py` | `answer_kb(query, chunks, conversation_tail) → KBAnswer`. Generates a grounded answer from retrieved chunks with source attribution. Temperature is 0.0 for determinism. |
| `schemas.py` | Pydantic models and enums: `Intent`, `AgentState`, `AgentResponse`, `IntentResult`, `GuardResult`. |

**When to touch:** Routing failures → `router.py` or `prompts.py`. Safety false positives → `guardrails.py`. Wrong KB answer → `qa.py` or `prompts.py`. Action not triggering/completing → `graph.py` (node_dispatch). State/memory issues → `graph.py` (load_partial_state, scan_history_for_slot).

---

## `app/actions/`

Business logic for each operational action. Each module is independently testable.

| File | Purpose |
|---|---|
| `transaction.py` | `check_transaction(tx_id, asset_type) → dict`. Normalises asset aliases, validates inputs, looks up `mock_transactions`. |
| `ticket.py` | `create_ticket(session_id, issue_type, email, description) → dict`. Normalises issue type (50+ aliases), validates email, persists to `tickets` table. |
| `recovery.py` | `recovery_step(session_id, state, user_reply, router_slots) → dict`. Multi-turn slot filling (email + subtype). Persists partial state to `recovery_cases`. |
| `onboarding.py` | `onboarding_plan(new_to_crypto, goal, region, extra_context) → dict`. RAG + LLM plan generation. Falls back to static checklist on LLM failure. |

**When to touch:** Validation too strict/loose → `ticket.py:normalize_issue_type`, `transaction.py:validate_tx_id`. Recovery not collecting slots → `recovery.py:recovery_step`. Onboarding plan poor quality → `onboarding.py` (adjust evidence length or system prompt).

---

## `app/storage/`

| File | Purpose |
|---|---|
| `sqlite_store.py` | `SqliteStore` class. All database operations: sessions, tickets, recovery cases, transaction lookup. Get the singleton via `get_store()`. UTC timestamps enforced via `_utc_now()`. |

**Tables:** `sessions`, `tickets`, `recovery_cases`, `mock_transactions`

**When to touch:** DB schema changes → add migration in `_init_db()`. New query needed → add method here; never put raw SQL in other modules.

---

## `backend/`

| File | Purpose |
|---|---|
| `main.py` | FastAPI app. Routes: `GET /health`, `POST /v1/chat`, `GET /v1/sessions`, `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}`, `POST /v1/eval/run`. Startup validation checks that FAISS index files exist. |

**When to touch:** Adding new API routes or middleware → `main.py`. Health check failing → verify `data/index/faiss.index` and `data/index/faiss_meta.jsonl` exist.

---

## `frontend/`

| File | Purpose |
|---|---|
| `streamlit_app.py` | Complete Streamlit UI. Login gate (username + password), chat interface, source cards, action result cards, session sidebar with history, new conversation button. Handles in-process agent or FastAPI backend mode. |

Key helpers:
- `require_auth()` — username/password gate, reads credentials from env
- `render_message()` — renders a single chat message with status chip, content, and timestamp
- `render_sources()` — renders citation cards below KB answers
- `_fmt_time()` — converts UTC ISO timestamp to America/Toronto display time
- `_icon_b64()` / `_icon_b64_white()` — Coinbase icon in base64 for CSS injection (blue and white variants)

**When to touch:** UI appearance → CSS block at the top of `streamlit_app.py`. New response types → `render_message()`. Sidebar behaviour → session list section.

---

## `app/eval/`

| File | Purpose |
|---|---|
| `test_cases.json` | 50 test scenarios. Each case defines inputs, expected intent, status, and assertion checks. |
| `runner.py` | Loads test cases, runs each through `run_agent_turn`, evaluates assertions, writes results to `data/eval/`. |
| `retrieval_eval.py` | Retrieval-only smoke test — no LLM needed. Checks FAISS index integrity and retrieval recall on sample queries. |
| `__main__.py` | Entry point for `python -m app.eval`. |

**When to touch:** New agent behaviour → add test cases in `test_cases.json`. New assertion type → add to `_check_expect()` in `runner.py`. Retrieval degraded → `retrieval_eval.py`.

---

## `data/`

| Path | Contents |
|---|---|
| `data/corpus/articles.jsonl` | Cleaned article documents (generated by scraper) |
| `data/corpus/manifest.csv` | URL, title, category, Wayback timestamp per article |
| `data/index/faiss.index` | FAISS index (generated by build_kb.py) |
| `data/index/faiss_meta.jsonl` | Chunk metadata aligned with FAISS index |
| `data/mock/transactions.json` | Seed transaction records loaded into SQLite |
| `data/app.db` | SQLite database (sessions, tickets, recovery, transactions) |
| `data/eval/` | Evaluation outputs (CSV, JSON, chart) |

---

## `deployment/`

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build; installs Python deps, copies app code |
| `docker-compose.yml` | Two services: `api` (FastAPI on 8000) and `ui` (Streamlit on 8501); both mount `data/` as a volume |
| `README.md` | Docker-specific setup instructions |
