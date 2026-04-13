# Evaluation failure analysis

Pass rate: 50/50

## Scenario suite metrics

| Metric | Value |
|---|---|
| Scenario success rate | **100%** (50/50) |
| Intent accuracy | **100%** (24/24) |
| Guardrail refusal rate | **100%** (9/9) |
| KB citation presence rate | **92.3%** (12/13) |
| Action scenario success | **100%** (26/26) |

## RAG quality metrics

Scored on 12 KB Q&A cases using an LLM judge (faithfulness and answer relevancy, 0–1 scale, pass threshold ≥ 0.7).

| Metric | Score | Pass rate (≥ 0.7) |
|---|---|---|
| Avg faithfulness | **0.97** | **100%** |
| Avg answer relevancy | **0.83** | **91.7%** |

Notes:
- 1 routing mismatch excluded from faithfulness (agent routed to ACTION_ACCOUNT_RECOVERY instead of KB_QA for an account-restrictions question; no KB context retrieved).
- Remaining relevancy cases below 0.7: corpus genuinely lacks staking, Advanced Trade, and specific fee articles — agent correctly defers with topic-named messages.

## Failures

None.

## Root causes & mitigations (historical)

- **Router / LLM variance** → added few-shot examples, explicit negative examples distinguishing transaction ID lookups from general delay questions, and no-asset-type transaction ID examples.
- **KB QA hallucination** → numbered source blocks, explicit per-claim traceability requirement, topic-specific deferral instructions, partial-info synthesis preference.
- **JSON extraction failure for Qwen3** → `_extract_json_object` now strips `<think>` blocks before regex matching; `qa.py` falls back to prose extraction before the raw-excerpt fallback.
- **Retrieval thresholds** → `evidence_sufficient` and hybrid fusion weights tuned; cross-encoder reranker removed (OMP conflict on macOS).
- **Safety classifier false positives** → LLM classifier skipped for slot-fill replies, history/action recall patterns, and terse follow-ups; regex prescreen always active.
