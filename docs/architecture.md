# Architecture

## System overview

The agent is a retrieval-augmented conversational system with explicit intent routing, multi-turn action flows, and layered safety controls. Every user message passes through a deterministic five-node pipeline before a response is returned.

---

## Data pipeline (ingestion to index)

```
Internet Archive (Wayback)
         │
         ▼
   scraper/discover.py        ← discovers article URLs from category pages
         │
         ▼
   scraper/wayback.py         ← fetches id_ captures, cleans HTML
         │
         ▼
   scraper/ingest.py          ── data/corpus/articles.jsonl
                              ── data/corpus/manifest.csv
                              ── data/corpus/manifest.json
                              ── data/corpus/robots_check.json
         │
         ▼
   app/retrieval/chunking.py  ← heading/paragraph-aware semantic splitting
         │
         ▼
   app/retrieval/index_faiss.py
         │
         ▼
   data/index/faiss.index     (FAISS IndexFlatIP, 768-dim)
   data/index/faiss_meta.jsonl
```

**Why Internet Archive?** The live `help.coinbase.com` returns Cloudflare challenges to automated HTTP clients. Internet Archive `id_` captures provide stable, time-stamped, fully reproducible content. Canonical `help.coinbase.com` URLs are preserved in the manifest so user-facing citations still point to the live site.

---

## Runtime pipeline (LangGraph)

Every call to `run_agent_turn(session_id, message)` traverses five nodes:

```
load ──► guard ──► intent ──► dispatch ──► persist
```

### load

- Calls `SqliteStore.ensure_session()` to create the session row if it does not exist
- Loads prior messages and router trace from `sessions` table
- Passes `AgentState` (TypedDict) downstream

### guard

Runs two safety checks in sequence:

1. **Regex prescreen** — deterministic patterns for prompt injection, security bypass phrases, and critical unsafe content; always runs
2. **LLM safety classifier** — JSON-structured classification into `{blocked, category, reason}`; skipped for known-safe input categories to avoid false positives:
   - Slot-fill replies (last assistant message was a `clarify` with named missing fields)
   - Conversation/action recall queries (`_HIST_PAT`, `_ASST_RECALL_PAT`, `_RECALL_RECOVERY_PAT`, `_RECALL_TX_PAT`)
   - Terse follow-ups (≤6 words when prior conversation exists)

If blocked, the node returns a refusal response immediately without reaching `intent` or `dispatch`.

### intent

- Calls `classify_intent()` which invokes the Qwen LLM with a structured router prompt
- Returns an `Intent` enum value, optional slot dict, and an optional clarifying question
- Falls back to `AMBIGUOUS` on LLM failure (never re-raises)
- Intent categories: `KB_QA`, `ACTION_CHECK_TRANSACTION`, `ACTION_CREATE_TICKET`, `ACTION_ONBOARDING_SUPPORT`, `ACTION_ACCOUNT_RECOVERY`, `AMBIGUOUS`, `OUT_OF_SCOPE`, `UNSAFE`, `SECURITY_SENSITIVE`

### dispatch

The largest node; handles routing to all response types:

**Memory shortcuts** (checked before intent-based routing, regardless of router output):
- Citation recall from the last KB answer
- User question history recall (with ordinal parsing: "second-to-last question")
- Assistant answer recall
- Recovery case recall from session
- Transaction result recall from session
- Ticket/case lookup (current session, then cross-session by email)

**Intent-based routing:**
- `KB_QA` → hybrid retrieval → `answer_kb()` with conversation tail for follow-ups
- `ACTION_CHECK_TRANSACTION` → slot extraction (regex + LLM + history fallback) → `check_transaction()`
- `ACTION_CREATE_TICKET` → slot extraction + validation → `create_ticket()`
- `ACTION_ONBOARDING_SUPPORT` → RAG + LLM plan generation → `onboarding_plan()`
- `ACTION_ACCOUNT_RECOVERY` → multi-turn slot filling → `recovery_step()`
- `AMBIGUOUS` → clarifying question returned
- `OUT_OF_SCOPE` / `UNSAFE` / `SECURITY_SENSITIVE` → warm refusal

**Multi-turn continuation:** On each call, `dispatch` checks whether there is a pending action (last assistant message had `status=clarify` with non-empty `missing` fields). If found and the user is clearly responding to that action, the intent is overridden to continue the action flow rather than treating the reply as a fresh query.

### persist

- Appends user message and assistant message to the session's message list
- Each message carries a UTC ISO timestamp (`ts` field)
- Saves updated messages and router trace to SQLite
- Sets session title from the first user message if not already set

---

## Retrieval design

### Embeddings

- **Model:** `@cf/baai/bge-base-en-v1.5` (768-dim)
- **API:** Remote OpenAI-compatible endpoint — consistent embeddings without local model dependencies
- **Normalisation:** L2-normalised before storage and query so that FAISS `IndexFlatIP` inner product equals cosine similarity

### Hybrid fusion

For every query, both retrievers run independently and their scores are fused:

```
Query
  ├── Dense (FAISS)  → top-8 chunks by cosine similarity
  └── Lexical (BM25) → top-8 chunks by BM25 Okapi score
         │
         ▼
  Min-max normalise each score set to [0, 1]
         │
         ▼
  Merge: score = 0.6 × dense_norm + 0.4 × bm25_norm
  (BM25-only chunks receive 0.4 × norm; dense-only chunks receive 0.6 × norm)
         │
         ▼
  Return top-4 by fused score
```

**Why hybrid?** Dense retrieval misses queries that use exact product terms (e.g., "CB-TX-PENDING01", "SEPA IBAN"). BM25 misses semantic paraphrases. Fusion captures both.

### Citations

Each retrieved chunk contributes an attribution record bundled with the response:

```json
{
  "article_title": "How do I set up 2-step verification?",
  "section_title":  "Overview",
  "url":            "https://help.coinbase.com/...",
  "excerpt":        "To enable 2FA, go to Security Settings..."
}
```

Duplicate URLs are deduplicated at both the QA layer and the frontend, keeping the highest-scoring chunk per article.

---

## Storage design

SQLite database at `data/app.db`:

| Table | Primary key | Purpose |
|---|---|---|
| `sessions` | `session_id` | Full message JSON + router trace JSON + timestamps |
| `tickets` | `ticket_id` (`TCK-…`) | Support tickets: issue type, email, description, session |
| `recovery_cases` | `case_id` (`REC-…`) | Multi-turn recovery state JSON, updated incrementally |
| `mock_transactions` | `(tx_id, asset_type)` | Seeded from `data/mock/transactions.json` |

**Timestamp convention:** All `created_at` / `updated_at` fields and message `ts` fields store UTC ISO 8601 strings. The frontend converts to `America/Toronto` (EDT/EST) for display using Python's `zoneinfo` module.

---

## Safety strategy

```
User message
     │
     ▼
Regex prescreen  ← always runs; catches injection phrases, bypass patterns
     │
     ▼  (if not blocked, and input is not in safe-skip categories)
LLM safety classifier  ← JSON: {blocked, category, reason}
     │
     ▼
Intent router  ← SECURITY_SENSITIVE / UNSAFE returned as explicit intents
     │
     ▼
Refusal copy   ← warm, scope-redirecting; logged with category
```

**Intent safety labels:**
- `SECURITY_SENSITIVE` — KYC bypass, 2FA circumvention, fraud assistance, account manipulation
- `UNSAFE` — prompt injection, illegal financial activity, investment advice, off-platform requests

Both labels produce refusals. The distinction is preserved in the router trace for debugging.

---

## Actions

### Check Transaction Status

- Inputs: `transaction_id`, `asset_type`
- Asset aliases normalised (`bitcoin` → `BTC`, `ethereum` → `ETH`, etc.)
- Looks up `mock_transactions` table; returns status, detail text, and next-step bullets
- Not-found responses use `status=ok` (not `clarify`) so the action is not re-triggered

### Create Support Ticket

- Inputs: `issue_type`, `email`, `problem_description`
- Issue type normalised with 50+ aliases (e.g., `kyc` → `verification`, `restricted` → `account_access`)
- Email validated with regex
- Description must be ≥12 characters; structured slot-filling messages are not accepted as descriptions
- Persisted to `tickets` table with a `TCK-…` identifier

### Onboarding Support

- Inputs: `new_to_crypto`, `goal`, `region` (all optional)
- Retrieves relevant KB chunks, passes evidence to LLM for a structured JSON plan
- Plan sections: `first_steps`, `security_tips`, `verification_guidance`, `next_actions`, `suggested_kb_questions`
- Falls back to a static checklist if the LLM call fails

### Account Recovery (multi-turn)

- Inputs (collected across turns): `email`, `issue_subtype`
- Subtypes: `forgot_password`, `lost_2fa`, `account_locked`, `compromised`
- Free-text subtype detection handles natural phrasing ("I lost my 2FA device")
- Partial state persisted to `recovery_cases` with a stable `REC-…` case ID across turns
- Completes in one turn if all slots are provided upfront

---

## Evaluation

50 scenario test cases covering:

| Category | Count |
|---|---|
| KB Q&A | 12 |
| Transaction action | 8 |
| Ticket action | 6 |
| Onboarding action | 4 |
| Account recovery | 6 |
| Guardrails | 9 |
| Memory / session | 5 |
| Edge cases / routing | 2 |
| Infrastructure smoke | 1 |

Test check types supported: `intent`, `intent_any`, `status`, `status_any`, `status_not`, `substring`, `substring_ci`, `last_substring`, `last_substring_ci`, `last_status`, `citations_nonempty`, `faiss_ready`.
