# Design decisions

## Why Internet Archive instead of live scraping?

`help.coinbase.com` frequently returns **Cloudflare challenges** to automated HTTP clients, which makes polite, reproducible scraping unreliable in CI and for graders. We ingest **archived `id_` captures** from the Internet Archive, log timestamps in the manifest, and still cite **canonical** `https://help.coinbase.com/...` URLs in the product UI so users know where to read the live article.

## Embeddings: `BAAI/bge-small-en-v1.5`

- Strong English dense retrieval for short factual support text.
- Runs **locally** (no paid embedding API, no data leaves the machine).
- Pairs well with **BM25** lexical fallback when users use exact product terms.

## Hybrid retrieval + optional rerank

- **FAISS** inner product on **normalized** vectors ≈ cosine similarity.
- **BM25** catches keyword-heavy queries (e.g., product error strings).
- **Cross-encoder** reranker is optional: if it fails to load, the system degrades to fused dense/BM25 ordering.

## LangGraph over a single “mega-prompt”

Explicit nodes (`load` → `guardrails` → `intent` → `dispatch` → `persist`) make the demo **explainable**: each step maps to a line in `docs/architecture.md` and can be toggled in **debug mode** in the UI.

## SQLite for sessions and mock CRM

- **Sessions**: transcripts + router trace for audit.
- **Tickets / recovery**: durable multi-turn state without standing up Postgres.
- **Mock transactions**: seeded JSON for deterministic eval scenarios.

## `SECURITY_SENSITIVE` vs `UNSAFE`

Course wording distinguishes **security-policy abuse** from generic unsafe content. We expose **`SECURITY_SENSITIVE`** for bypass/KYC/2FA abuse (often from the regex prescreen), and **`UNSAFE`** for injection / illegal / broad threats. Both end in **refusal** paths with appropriate copy.

## Evaluation metrics

Scenario checks are **LLM-dependent**; we report **rates** (intent match, guardrail refusals, citation presence on KB subsets) plus a **smoke** test for on-disk FAISS artifacts so infrastructure issues are separated from model variance.
