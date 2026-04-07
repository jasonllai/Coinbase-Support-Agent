# Deployment notes

## Local Docker

1. Copy `../.env.example` to `../.env` and set `LLM_API_KEY` if required by your endpoint.
2. Build and run:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

3. Open Streamlit on port **8501** and FastAPI docs on **8000/docs**.

## Password gate

The Streamlit UI prompts for `DEMO_PASSWORD` (see `.env.example`). For stronger protection, place the app behind **HTTPS basic auth** or an **OAuth2 proxy** at your hosting provider.

## Corpus + index in containers

Mount `data/` as a volume (already configured). Before first run on a fresh volume:

```bash
docker compose -f deployment/docker-compose.yml run --rm api \
  bash -lc "python -m scraper.ingest --out data/corpus --min-articles 60 && python scripts/build_kb.py"
```

This can take a while because it pulls many pages from the Internet Archive.

## Health check

`GET /health` reports whether the FAISS index files exist on disk.
