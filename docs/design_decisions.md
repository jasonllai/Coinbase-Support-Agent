# Design Decisions

Rationale behind the key technical and product choices in this system.

---

## Knowledge base: Internet Archive instead of live scraping

`help.coinbase.com` returns Cloudflare bot challenges to plain HTTP clients. Live scraping is unreliable and breaks in CI environments.

The ingestion pipeline uses [Internet Archive Wayback Machine](https://archive.org/) `id_` raw captures, which return unobfuscated HTML for public pages. Each ingested article records the Wayback timestamp in `manifest.csv` so the knowledge cutoff is explicit. Canonical `help.coinbase.com` URLs are preserved in the metadata and surfaced to users in source cards — so users are always directed to the live authoritative page.

---

## Embeddings: remote API over local model

Earlier versions used a local `sentence-transformers` model (`BAAI/bge-small-en-v1.5`). This was replaced with a remote OpenAI-compatible embedding API (`@cf/baai/bge-base-en-v1.5`, 768-dim) for three reasons:

1. **No OMP library conflict.** Loading both PyTorch (from `sentence-transformers`) and FAISS in the same process on macOS caused a fatal crash (`OMP: Error #15: Initializing libomp.dylib`). Switching to a remote API eliminated the shared-library conflict entirely.
2. **Larger model, better quality.** `bge-base` outperforms `bge-small` on retrieval benchmarks.
3. **Consistent embeddings.** All environments (local, Docker, CI) use the same remote model, so index builds are reproducible without downloading gigabytes of model weights.

L2-normalisation is applied client-side after receiving embeddings so that FAISS `IndexFlatIP` computes cosine similarity via inner product.

---

## Hybrid retrieval: dense + BM25 fusion

Dense retrieval (FAISS) handles semantic paraphrases well but misses queries that use exact Coinbase product strings (transaction IDs, error codes, product names). BM25 handles these precisely but fails on semantic variation.

The pipeline runs both retrievers independently, min-max normalises their scores to `[0, 1]`, and combines them at 60% dense + 40% BM25. This ratio was chosen empirically — dense quality is generally higher, but the BM25 contribution is meaningful enough to warrant a non-trivial weight.

A cross-encoder reranker was implemented but later removed: the PyTorch dependency reintroduced the macOS OMP conflict, and reranking a small top-k candidate set produced minimal quality improvement.

---

## LangGraph: explicit nodes over a monolithic chain

The pipeline is split into five named nodes (`load → guard → intent → dispatch → persist`) instead of a single prompt-and-respond loop. This design choice pays off in three ways:

- **Debuggability.** Each node's input and output is inspectable. The UI's debug mode exposes the router trace per turn.
- **Composability.** The guard node can be extended with new rules without touching the QA logic. The dispatch node handles memory shortcuts independently from action flows.
- **Explainability.** In a demo, each node maps to a sentence: "first it loads history, then it checks safety, then it routes, then it executes, then it saves."

---

## Guardrail skip logic

The LLM safety classifier runs on every user message by default. This caused false positives for legitimate messages that appear suspicious without context:

- `"AB-CD-EFGH01"` — a transaction ID looks like an encoded payload to the classifier
- `"what did I ask before?"` — classified as probing for system internals
- `"BTC"` — a one-word asset type reply to a clarification prompt

The fix: `node_guard` evaluates four conditions before deciding whether to skip the LLM classifier. If any condition is true, only the regex prescreen runs (fast path):

1. Last assistant message was a `clarify` with named missing fields (slot-fill reply)
2. Input matches conversation-history recall patterns
3. Input matches prior-action-result recall patterns (transaction, recovery case)
4. Input is ≤6 words and prior conversation exists (terse follow-up)

Regex prescreen always runs regardless of these conditions, so critical injection patterns are still caught.

---

## Multi-turn state persistence

Recovery (and partially, ticket creation) collect multiple parameters over several turns. Partial state is saved to SQLite at the end of every clarify response with a stable case/ticket ID. This means:

- If the session restarts (e.g., browser refresh), the partial state is reloaded from the database on the next turn
- The `case_id` / `ticket_id` is assigned at the first clarify step, not at completion — so the user always has a reference number
- `_load_partial_state()` in `graph.py` walks the full message history for the most recent clarify response for the named action, only loading state from genuine incomplete turns (requires `status=clarify` AND non-empty `missing` fields)

---

## Issue type normalisation

Early versions of the ticket action rejected LLM-extracted issue types like `"identity verification"` or `"account restriction"` because they didn't match the canonical set. `normalize_issue_type()` in `ticket.py` now:

1. Normalises whitespace/case to `snake_case`
2. Checks exact match against the canonical set
3. Checks a 50+ alias dictionary
4. Falls back to substring matching (e.g., if `"identity"` appears anywhere in the raw string → `"verification"`)

This makes ticket creation resilient to natural language variation without requiring the LLM to return exact canonical values.

---

## `SECURITY_SENSITIVE` vs `UNSAFE`

Both intents return refusals, but the distinction matters for logging and copy:

- **`SECURITY_SENSITIVE`** — the user is trying to abuse a Coinbase-specific security mechanism (bypass 2FA, skip KYC, access another user's account). The refusal redirects to official support channels.
- **`UNSAFE`** — general unsafe content (prompt injection, illegal financial activity, investment predictions, off-platform requests). The refusal is scope-redirecting.

Keeping them separate means logs can distinguish "security probe" events from generic out-of-scope queries, and the refusal copy can be tailored accordingly.

---

## Timestamp handling

All timestamps are stored as UTC ISO 8601 strings in SQLite. Conversion to a display timezone happens exclusively at render time:

- Frontend uses `ZoneInfo("America/Toronto")` to convert stored UTC to EDT/EST
- Agent responses that mention dates (ticket submission time, etc.) also format in Toronto time
- This separation means the database is timezone-agnostic and the display timezone can be changed by editing a single constant

---

## Evaluation test design

Tests use the weakest assertion that would catch a real failure:

- `intent_any` instead of `intent` for queries where two intents both produce correct answers (e.g., "How do I buy Bitcoin for the first time?" can legitimately go to either `KB_QA` or `ACTION_ONBOARDING_SUPPORT`)
- `substring_ci` instead of `substring` for status text that may vary in capitalisation
- `status: clarify` not `status: error` for invalid input — the agent re-prompts rather than hard-failing, which is the correct UX

This philosophy means a test failure always indicates a genuine regression, not a stylistic mismatch.

---

## RAG faithfulness and answer relevancy evaluation

Two LLM-judge metrics were added to assess KB Q&A quality beyond pass/fail scenario tests.

**The core challenge:** standard evals check that a response exists and cites sources — they do not check whether claims are grounded. A model can hallucinate plausible-sounding policies that never appear in retrieved chunks while still passing a citation-presence test.

**Faithfulness judge design:**
- Passes the full retrieved chunk text (not the truncated 400-char UI excerpt) to the judge. Using truncated excerpts produced false hallucination calls when the agent cited text that appeared after character 400 — the judge couldn't see it.
- Honest deferrals score 1.0: if the agent says "I don't have that info" and makes no false claims, that is maximally faithful. Only fabricated specific facts (invented fees, phone numbers, policies) score below 0.7.

**Relevancy judge design:**
- Three tiers for deferrals distinguish quality: partial-info + acknowledged gap → 0.8, topic-named deferral → 0.7, generic "I don't know" → 0.5. This breaks the bimodal "full answer = 1.0, anything else = 0.5" pattern and rewards agents that give useful partial information.
- Routing mismatches (agent routed to an action intent, so no KB context was retrieved) are excluded from the faithfulness aggregate and flagged separately — they represent routing quality failures, not KB QA faithfulness failures.

---

## `_extract_json_object` and Qwen3 `<think>` blocks

Qwen3 produces chain-of-thought reasoning inside `<think>...</think>` tags before every response. The original `_extract_json_object` function used the greedy regex `\{[\s\S]*\}` to find the JSON object. When the think block contained natural-language `{curly braces}`, the greedy match started at the first brace inside the thinking block and ended at the last brace in the actual JSON — producing a span that was not valid JSON.

The fix:
1. Strip `<think>...</think>` blocks with a non-greedy regex before any JSON extraction attempt
2. Try `json.loads` on the cleaned text directly
3. Fall back to finding the last non-nested `{...}` block, then to the greedy match on the already-cleaned text

This fix is applied at the `LLMClient` layer (`client.py`) so all callers benefit automatically. A parallel implementation in `rag_eval.py`'s `_parse_judge_response` handles the same issue for the judge's responses.

---

## KB QA source attribution and deferral quality

Earlier versions of `KB_QA_SYSTEM` instructed the model to "answer from sources only" — necessary but not sufficient. The model would still add plausible-sounding details from training knowledge (authentication methods, document types, contact channels) that did not appear in the retrieved chunks.

The fixes:

1. **Numbered sources.** Retrieved chunks are formatted as `[SOURCE 1]`, `[SOURCE 2]`, etc. The prompt explicitly states: "every factual claim must be traceable to a specific numbered source; if a detail is not in any source, leave it out." This gives the model a concrete audit trail, not just a vague prohibition.

2. **Partial-info synthesis over full deferral.** When sources contain tangential information (e.g., payment method details while asking about fees), the model now shares what IS available and notes the gap, rather than issuing a blanket "I don't have that."

3. **Topic-named deferrals.** Generic "I don't have that info" responses make it unclear what the agent was asked. The prompt now requires the model to name the exact topic: "I don't have specific details about **staking on Coinbase** in my current sources." This improves relevancy scores and user experience simultaneously.

4. **Mirroring the user's terms.** If the user asks about "Bitcoin", the answer says "Bitcoin" — not just "crypto". The prompt includes an explicit instruction to mirror the user's specific asset and feature names throughout the answer.
