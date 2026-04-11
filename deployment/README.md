# Deployment

Instructions for running the Coinbase Support Agent with Docker Compose.

---

## Prerequisites

- Docker and Docker Compose installed
- A `.env` file in the repo root (copy from `.env.example` and fill in `LLM_API_KEY`)
- The knowledge base index built at least once (see below)

---

## Quick start

```bash
# From the repo root
docker compose -f deployment/docker-compose.yml up --build
```

- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** [http://localhost:8000/health](http://localhost:8000/health)

The `data/` directory is mounted as a volume so the knowledge base index and SQLite database persist across container restarts.

---

## First-time setup: build the knowledge base

The index is not included in the repository. Run the ingestion and indexing steps once before starting the application.

**Option A — Run inside the container:**

```bash
docker compose -f deployment/docker-compose.yml run --rm api \
  bash -lc "python -m scraper.ingest --out data/corpus --min-articles 60 && python scripts/build_kb.py"
```

**Option B — Run on the host before compose up:**

```bash
export PYTHONPATH=.
python -m scraper.ingest --out data/corpus --min-articles 60
python scripts/build_kb.py
```

Both options write to `data/`, which is shared with the containers via the volume mount.

---

## Services

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI backend |
| `ui` | 8501 | Streamlit frontend |

The `ui` service sets `API_BASE=http://api:8000` so the frontend calls the backend over the internal Docker network.

---

## Environment variables

All environment variables are read from `../.env` by both services. The minimum required variable is:

```dotenv
LLM_API_KEY=your_key_here
```

See `.env.example` for the full list with defaults.

---

## Login credentials

The Streamlit UI requires a username and password (set in `.env`):

| Variable | Default |
|---|---|
| `DEMO_USERNAME` | `coinbase` |
| `DEMO_PASSWORD` | `CoinbaseSupport2026!` |

For production deployments, change these values and place the application behind HTTPS. Consider adding an OAuth2 proxy or reverse proxy with HTTP basic auth for additional protection.

---

## Stopping and cleanup

```bash
# Stop services
docker compose -f deployment/docker-compose.yml down

# Remove containers and volumes (deletes database and index)
docker compose -f deployment/docker-compose.yml down -v
```

---

## Health check

`GET /health` returns the startup validation state:

```json
{
  "faiss_index": true,
  "faiss_meta": true,
  "corpus": true,
  "ok": true
}
```

If `faiss_index` or `faiss_meta` is `false`, the agent will fail to answer questions. Run the index build step to fix this.
