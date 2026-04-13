# Evaluation failure analysis

Pass rate: 50/50

## Metrics

- Scenario success rate: **100.00%**
- Intent accuracy (where expected): **1.0**
- Guardrail refusal rate (subset): **1.0**
- KB citation presence rate (subset): **0.9230769230769231**
- Action scenario success (tagged): **1.0**

## Failures


## Root causes & mitigations
- Router / LLM variance → add few-shot examples, lower temperature.
- Retrieval thresholds → tune `evidence_sufficient` and hybrid fusion.
- Safety classifier variance → strengthen regex prescreen list.
