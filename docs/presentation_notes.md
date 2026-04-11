# Presentation Notes

Talking points and slide structure for a 5–10 minute overview of the Coinbase Support Agent.

---

## Suggested slide structure

### Slide 1 — Problem

**What to say:**
Retail crypto customers ask the same support questions repeatedly — how to set up 2FA, why a withdrawal is delayed, how to start buying crypto safely. These questions are answerable from public documentation, but finding the right article and getting a precise answer is slow.

The goal: an assistant that answers these questions accurately, handles common account operations, and stays firmly within the support scope.

**Key points:**
- Support scope: account access, transactions, security, fees, onboarding, identity verification
- Not a trading advisor — refuses investment recommendations and security bypass requests
- Knowledge grounded in archived public Coinbase Help Center content

---

### Slide 2 — System design

**What to say:**
Every user message passes through five pipeline stages. The guardrail stage runs first so safety is never skipped. The intent router classifies the query into one of nine labels and extracts slots. The dispatch stage executes the right handler — KB retrieval, an action, or a memory shortcut. Everything is saved to SQLite with UTC timestamps.

**Key points:**
- LangGraph pipeline: `load → guard → intent → dispatch → persist`
- Nine intent categories including two safety labels (`UNSAFE`, `SECURITY_SENSITIVE`)
- Hybrid retrieval: FAISS cosine similarity (60%) + BM25 lexical (40%), top-4 chunks
- Remote embedding API (`@cf/baai/bge-base-en-v1.5`, 768-dim) — no local model dependencies
- SQLite for sessions, tickets, recovery cases, mock transactions

**Architecture diagram from README.md works well on this slide.**

---

### Slide 3 — Capabilities

**What to say:**
The agent handles five main workflows. KB Q&A retrieves evidence and cites sources. Transaction lookup searches a mock ledger by ID and asset type. Ticket creation validates inputs and persists to the database. Account recovery is multi-turn — it collects email and issue subtype across separate messages and saves partial state so the session can be resumed. Onboarding combines KB retrieval with LLM-generated structured plans.

**Key points:**
- Source attribution: article title, section, canonical help.coinbase.com URL
- Input validation: email format, issue type normalisation (50+ aliases), transaction ID format
- Memory shortcuts: users can ask "what was my case ID?" or "what did I ask two messages ago?"
- Timestamps stored in UTC, displayed in local time (America/Toronto)

---

### Slide 4 — Safety and reliability

**What to say:**
Safety is two-layer. A regex prescreen runs on every message — it catches obvious injection patterns without an LLM call. If the prescreen passes, an LLM classifier evaluates the message for nuanced policy violations. The classifier is skipped for inputs that are provably safe — slot-fill replies, memory recall queries, short follow-ups — to avoid false positives on legitimate messages like transaction IDs.

KB answers use temperature 0.0, so citations are deterministic across identical queries. The prompt instructs the model to answer only from retrieved evidence and say so clearly when evidence is insufficient.

**Key points:**
- Regex prescreen always runs (fast, deterministic)
- LLM safety classifier skipped for slot-fills, recall queries, and terse follow-ups
- KB QA temperature: 0.0 (consistent, citable answers)
- Out-of-scope refusals redirect to relevant support topics — not generic "I can't help"

---

### Slide 5 — Evaluation

**What to say:**
The evaluation suite has 50 scenario tests covering all major capabilities and edge cases. The runner checks intent routing accuracy, action completion, guardrail effectiveness, citation presence, and status correctness. Results are saved as CSV with a summary JSON and a bar chart.

**Metrics to show** (from `data/eval/eval_summary.json`):
- Overall scenario pass rate
- Intent routing accuracy
- Guardrail success rate (refusals on unsafe inputs)
- Citation presence rate on KB queries
- Action success rate

**Key points:**
- Tests use the weakest assertion that would catch a genuine regression (e.g., `intent_any` for ambiguous queries)
- Failures categorised as agent bugs vs. test over-specification
- Retrieval smoke test runs without an LLM — useful for infrastructure validation

---

## Metrics worth highlighting

| Metric | Where it comes from |
|---|---|
| Scenario pass rate | `eval_summary.json` → `pass_rate` |
| Intent accuracy | `eval_summary.json` → `intent_accuracy` |
| Guardrail success | `eval_summary.json` → `guardrail_rate` |
| Citation presence | `eval_summary.json` → `citation_rate` |
| Action success | `eval_summary.json` → `action_rate` |
| Corpus size | `data/corpus/manifest.csv` — row count |
| Index size | `data/index/faiss_meta.jsonl` — line count |

---

## Technical depth talking points

If asked to go deeper on any component:

**Retrieval:**
> "We run dense and BM25 retrieval independently, min-max normalise the scores, and fuse them at 60/40. This means a transaction ID that's invisible to dense embeddings will still surface from BM25, and a semantic paraphrase that BM25 misses will still rank from FAISS."

**Guardrails:**
> "The regex prescreen catches known-bad patterns in microseconds without an LLM call. The LLM classifier is only invoked when context is needed. For inputs that are structurally safe — a slot-fill like 'BTC' or a recall query — we skip the LLM entirely to avoid false positives."

**Multi-turn state:**
> "Partial recovery state is written to SQLite with a stable case ID after the first clarify turn. If the session is interrupted and resumed, `_load_partial_state` walks the message history to find the most recent incomplete clarify response for that action and reloads it."

**Embeddings:**
> "We switched from local sentence-transformers to a remote API because loading both PyTorch and FAISS in the same process caused an OMP library conflict on macOS. The remote API also gives us a larger model (bge-base vs bge-small) with no local dependency management."
