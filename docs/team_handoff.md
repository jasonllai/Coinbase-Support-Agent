# Team handoff (module cheat sheet)

## `scraper/`

- **discover.py** — pulls article URLs from archived category pages; merges `seed_urls.txt` if discovery is thin.
- **wayback.py** — CDX lookup + `id_` raw capture fetch + HTML cleanup.
- **ingest.py** — writes `articles.jsonl`, `manifest.csv`, `manifest.json`, `robots_check.json`.

## `app/retrieval/`

- **chunking.py** — heading/paragraph aware chunks with stable `chunk_id`.
- **embeddings.py** — loads `sentence-transformers` model from settings.
- **index_faiss.py** — builds `data/index/faiss.index` + `faiss_meta.jsonl`.
- **retriever.py** — dense + BM25 merge + optional cross-encoder rerank.

## `app/llm/`

- **client.py** — OpenAI-compatible chat + JSON extraction + streaming helper.
- **prompts.py** — Central `ROUTER_SYSTEM`, `KB_QA_SYSTEM`, `SAFETY_CLASSIFIER_SYSTEM` strings.

## `app/agent/`

- **graph.py** — LangGraph pipeline; start here for the “big picture”.
- **router.py** — LLM intent classification to `Intent` enum + slots.
- **guardrails.py** — regex + LLM safety screen.
- **qa.py** — grounded answering over retrieved chunks.

## `app/actions/`

- **transaction.py**, **ticket.py**, **onboarding.py**, **recovery.py** — mock business logic + validation.

## `app/storage/sqlite_store.py`

- Sessions, tickets, recovery cases, mock on-chain DB seeding.

## `backend/main.py`

- `/health`, `/v1/chat`, session list/detail/delete.

## `frontend/streamlit_app.py`

- Password gate, chat UI, sources panel, optional API mode.

## `app/eval/`

- **test_cases.json** + **runner.py** — scenario integration tests → `data/eval/`.
