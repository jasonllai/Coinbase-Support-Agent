# Coinbase Support Agent

An AI-powered customer support assistant that answers questions from the Coinbase Help Center, looks up transaction statuses, manages support tickets, and guides users through account recovery — all in a polished chat interface.

> **Disclaimer:** This is an independent educational project. It is not affiliated with or endorsed by Coinbase, Inc. All knowledge comes from archived public snapshots of `help.coinbase.com`; live policies may differ. Transactions, tickets, and recovery cases are simulated and not connected to real Coinbase systems.

---

## What it does

| Capability | Detail |
|---|---|
| **Help Center Q&A** | Answers questions grounded in retrieved Help Center articles with source citations |
| **Transaction lookup** | Checks status of mock transactions by ID and asset type |
| **Support tickets** | Creates and recalls support tickets across the session |
| **Account recovery** | Multi-turn guided flow collecting email and issue subtype |
| **Onboarding** | Personalised getting-started plan combining RAG evidence and structured output |
| **Conversation memory** | Recalls prior questions, answers, citations, ticket IDs, and case IDs within a session |
| **Guardrails** | Blocks prompt injection, security bypass attempts, investment advice, and off-topic requests |
| **Session persistence** | Full conversation history stored in SQLite and reloadable from the sidebar |

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│   Login gate · Chat interface · Source cards · Session sidebar  │
└────────────────────────────┬────────────────────────────────────┘
                             │ run_agent_turn(session_id, message)
┌────────────────────────────▼────────────────────────────────────┐
│                    LangGraph Agent Pipeline                      │
│  ┌──────┐  ┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  │
│  │ load │→ │ guard  │→ │ intent │→ │ dispatch │→ │ persist │  │
│  └──────┘  └────────┘  └────────┘  └──────────┘  └─────────┘  │
└────────────────────────────┬────────────────────────────────────┘
             ┌───────────────┼────────────────┐
    ┌────────▼──────┐ ┌──────▼─────┐  ┌───────▼────────┐
    │ FAISS + BM25  │ │  Qwen LLM  │  │   SQLite DB    │
    │ Hybrid search │ │  (Qwen3-   │  │ sessions       │
    │ 768-dim embed │ │  30B-A3B)  │  │ tickets        │
    └───────────────┘ └────────────┘  │ recovery_cases │
                                      │ mock_txns      │
                                      └────────────────┘
```

### Agent pipeline nodes

| Node | Responsibility |
|---|---|
| **load** | Hydrate session: prior messages + router trace from SQLite |
| **guard** | Regex prescreen + conditional LLM safety classifier; short-circuits to refusal if blocked |
| **intent** | Structured JSON router → `Intent` enum, slots, optional clarifying question |
| **dispatch** | Execute KB RAG, action flows, memory shortcuts, or refusals based on intent |
| **persist** | Append timestamped messages and trace to SQLite |

---

## Repository layout

```
app/
  agent/          # LangGraph graph, router, guardrails, KB QA
  actions/        # Transaction, ticket, onboarding, recovery logic
  retrieval/      # Chunking, remote embeddings, FAISS index, hybrid retriever
  llm/            # OpenAI-compatible Qwen client + prompt library
  storage/        # SQLite store (sessions, tickets, recovery, mock transactions)
  eval/           # 50-case test suite + runner
  core/           # Settings (Pydantic), logging
backend/          # FastAPI (chat, health, session management)
frontend/         # Streamlit UI
scraper/          # Help Center ingestion via Internet Archive
scripts/          # KB build script (chunk + index)
deployment/       # Dockerfile + Docker Compose
data/
  corpus/         # articles.jsonl + manifest (generated)
  index/          # faiss.index + faiss_meta.jsonl (generated)
  mock/           # Seed transaction data
  eval/           # Eval outputs (CSV, JSON, chart)
docs/             # Architecture, design decisions, demo materials
```

---

## Try it live

A deployed version is publicly accessible — no setup required:

**[https://coinbase-support-agent-jasonllai.streamlit.app/](https://coinbase-support-agent-jasonllai.streamlit.app/)**

| Field | Value |
|---|---|
| Username | `coinbase` |
| Password | `CoinbaseSupport2026!` |

---

## Local setup

To run the agent on your own machine (e.g. to use a different LLM endpoint, rebuild the knowledge base, or run the evaluation suite):

### Requirements

- **Python 3.11 or 3.12**
- Access to an OpenAI-compatible LLM endpoint (Qwen3-30B-A3B-FP8 or equivalent)
- Access to the remote embedding API (`@cf/baai/bge-base-en-v1.5`)
- Internet access for first-time knowledge base ingestion (Internet Archive)

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd Coinbase-Support-Agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```dotenv
LLM_API_KEY=your_api_key_here
```

All other defaults are pre-configured. See the [Environment variables](#environment-variables) section for the full reference.

### 3. Build the knowledge base

Ingest 60+ articles from the Coinbase Help Center via Internet Archive (first run takes a few minutes):

```bash
export PYTHONPATH=.
python -m scraper.ingest --out data/corpus --min-articles 60 --max-discover 200
```

This produces `data/corpus/articles.jsonl` and `data/corpus/manifest.csv`. Re-running the command resumes from where it left off (existing URLs are skipped). Use `--no-resume` for a clean rebuild.

Chunk and index the corpus:

```bash
python scripts/build_kb.py
```

Outputs: `data/index/faiss.index` and `data/index/faiss_meta.jsonl`.

### 4. Run the application

**Option A — Streamlit only (simplest)**

```bash
streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The agent runs in-process; no API server needed.

**Option B — FastAPI + Streamlit**

```bash
# Terminal 1
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run frontend/streamlit_app.py
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
Health check: [http://localhost:8000/health](http://localhost:8000/health)

**Default login credentials** (set in `.env`):

| Field | Default |
|---|---|
| Username | `coinbase` |
| Password | `CoinbaseSupport2026!` |

### 5. Docker

```bash
docker compose -f deployment/docker-compose.yml up --build
```

Mounts `data/` as a volume. Run the ingestion step once inside the container before first use — see [deployment/README.md](deployment/README.md).

---

## Evaluation

Run the full 50-case scenario suite (requires a working LLM endpoint):

```bash
python -m app.eval
```

Results are written to `data/eval/`:

| File | Contents |
|---|---|
| `eval_results.csv` | Per-case pass/fail with intent, status, and message preview |
| `eval_summary.json` | Aggregate metrics (scenario success rate, intent accuracy, guardrail rate, citation rate, action success rate) |
| `failure_analysis.md` | Failures with root causes |
| `eval_metrics_chart.png` | Bar chart of key rates |

Run retrieval quality checks only (no LLM needed):

```bash
python app/eval/retrieval_eval.py --out data/eval/retrieval_eval.csv
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://rsm-8430-finalproject.bjlkeng.io/v1` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | *(required)* | API key for the LLM endpoint |
| `LLM_MODEL` | `qwen3-30b-a3b-fp8` | Model name |
| `LLM_TIMEOUT_S` | `120` | Request timeout in seconds |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature (KB QA uses 0.0 for determinism) |
| `EMBEDDING_BASE_URL` | `https://rsm-8430-a2.bjlkeng.io/v1` | Remote embedding API endpoint |
| `EMBEDDING_MODEL` | `@cf/baai/bge-base-en-v1.5` | Embedding model (768-dim) |
| `EMBEDDING_API_KEY` | *(reuses LLM_API_KEY if blank)* | Key for embedding API |
| `RETRIEVAL_TOP_K` | `8` | Number of candidates from dense + BM25 retrieval |
| `RERANK_TOP_N` | `4` | Top chunks passed to the LLM for answer generation |
| `DEMO_USERNAME` | `coinbase` | Login username |
| `DEMO_PASSWORD` | `CoinbaseSupport2026!` | Login password |
| `SQLITE_PATH` | `data/app.db` | SQLite database path |
| `DATA_DIR` | `data` | Root data directory |
| `API_BASE` | `http://127.0.0.1:8000` | FastAPI base URL (used by Streamlit in API mode) |

---

## Key technical choices

### Retrieval

- **FAISS IndexFlatIP** with L2-normalised 768-dim vectors — inner product equals cosine similarity
- **BM25** lexical search via `rank-bm25` for keyword-heavy queries
- **Hybrid fusion**: min-max normalised scores, weighted 60% dense + 40% BM25
- **Remote embedding API** (`@cf/baai/bge-base-en-v1.5`) — avoids local model loading, consistent embeddings across environments

### LLM

- **Qwen3-30B-A3B-FP8** via OpenAI-compatible endpoint
- Structured JSON output for routing and safety decisions
- Temperature 0.0 for KB QA (deterministic citations), 0.2 for router and actions

### Guardrails

- Two-stage: fast regex prescreen → LLM safety classifier
- LLM classifier skipped for safe input categories (slot-fills, history recall, terse follow-ups, action-recall queries) to avoid false positives
- Intent categories: `SECURITY_SENSITIVE` (bypass/KYC/2FA abuse) and `UNSAFE` (injection, illegal activity, investment advice)

### Conversation memory

- Full session stored in SQLite per turn with UTC timestamps
- Displayed in America/Toronto timezone (EDT/EST)
- Shortcut handlers in `dispatch` node for: citation recall, prior question recall, assistant answer recall, transaction result recall, recovery case recall, ticket lookup

---

## Documentation

| File | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System design, data flow, and technical deep-dives |
| [docs/design_decisions.md](docs/design_decisions.md) | Why specific technologies and approaches were chosen |
| [docs/team_handoff.md](docs/team_handoff.md) | Module-by-module guide for contributors |
| [deployment/README.md](deployment/README.md) | Docker and deployment instructions |

---

## Built with

[FastAPI](https://fastapi.tiangolo.com/) · [Streamlit](https://streamlit.io/) · [LangGraph](https://github.com/langchain-ai/langgraph) · [FAISS](https://github.com/facebookresearch/faiss) · [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [Pydantic](https://docs.pydantic.dev/) · [httpx](https://www.python-httpx.org/) · [Internet Archive Wayback Machine](https://archive.org/)
