# Presentation notes (≈5 slides)

## Slide 1 — Problem & scope

- Retail crypto customers ask repetitive, policy-heavy questions best answered from official help content.
- Scope is **support**, not trading advice or account security bypass.

## Slide 2 — System overview

- **Ingestion:** discover URLs → Internet Archive snapshots → clean text → chunk → FAISS.
- **Agent:** LangGraph with guardrails → intent router → RAG or action handlers.
- **UI/API:** Streamlit + FastAPI, SQLite persistence.

## Slide 3 — Reliability & safety

- Hybrid retrieval + optional cross-encoder rerank.
- Source-attributed answers with URLs.
- Two-layer safety: rules + LLM classifier; logged categories.

## Slide 4 — Evaluation

- 15 scenario tests covering routing, RAG, actions, multi-turn recovery, refusals, malformed inputs.
- Report CSV + `failure_analysis.md` with mitigations.

## Slide 5 — Innovation / polish

- Password-gated Streamlit demo.
- Debug mode exposes intent and citations for graders.
- Docker compose path for repeatable deployment.

### Metrics to show

- Pass rate from `eval_summary.json`.
- Retrieval hit quality table from `retrieval_eval.csv`.
- Example refusal transcripts (injection / bypass / investment advice).
