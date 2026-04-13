"""
RAG quality evaluation — Faithfulness and Answer Relevancy for KB Q&A test cases.

Faithfulness  : are all answer claims supported by the retrieved context?
Answer Relevancy : does the answer actually address the question asked?

Both are scored 0.0–1.0 by an LLM judge (same Qwen endpoint, temperature=0.0).
A score >= 0.7 is treated as "passing" for rate calculations.

Run:
    python -m app.eval.rag_eval
    # or
    python app/eval/rag_eval.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import run_agent_turn
from app.llm.client import get_llm_client
from app.retrieval.retriever import get_retriever

# ── Robust response parser for reasoning models ────────────────────────────

def _parse_judge_response(raw: str) -> dict:
    """
    Extract {"score": ..., "reason": ...} from a Qwen3 reasoning model response.

    The model wraps its thinking in <think>...</think> before the actual JSON.
    Standard json extraction is too greedy when the think block contains {curly
    braces} in natural-language sentences. Strategy:
      1. Strip <think>...</think> entirely.
      2. Try json.loads on the remainder.
      3. Fall back to finding the last non-nested {...} block that contains "score".
    """
    import re as _re

    # Step 1: strip reasoning block
    cleaned = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()

    # Step 2: direct parse of remainder
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: find the last {...} block that contains the word "score"
    # Use a non-greedy match on non-nested braces only
    for m in reversed(list(_re.finditer(r"\{[^{}]+\}", cleaned))):
        candidate = m.group(0)
        if '"score"' in candidate or "'score'" in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Step 4: regex-extract score and reason directly as fallback
    score_m  = _re.search(r'"score"\s*:\s*([0-9.]+)', cleaned)
    reason_m = _re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
    if score_m:
        return {
            "score":  float(score_m.group(1)),
            "reason": reason_m.group(1) if reason_m else "",
        }

    raise ValueError(f"no score found in judge response: {raw[:300]}")

# ── LLM judge prompts ──────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
You are a strict RAG quality evaluator. You receive a QUESTION, the CONTEXT
(retrieved source passages the agent used), and the ANSWER the agent produced.

Score FAITHFULNESS: the degree to which every factual claim in the answer is
directly supported by the retrieved context, with nothing hallucinated.

Scoring scale:
  1.0 — Every claim is explicitly supported by the context; OR the answer is an
         honest deferral ("I don't have that info in the Help Center") and the context
         genuinely lacks a direct answer to the question — deferring is faithful.
  0.7 — Most claims supported; a minor detail not verifiable but not contradicted
  0.5 — About half the claims grounded; some appear invented
  0.3 — Most claims not supported or go beyond the context
  0.0 — Answer fabricates specific facts that directly contradict or are absent from
         the context (e.g. invents fees, phone numbers, policies not mentioned)

IMPORTANT: If the answer says "I don't have that info in our Help Center" or similar,
do NOT penalise it for not using the context — score 1.0 if no false claims are made.
Only score below 0.7 when the answer invents specific details not present in the context.

Return ONLY valid JSON — no markdown, no extra text:
{"score": <float 0.0-1.0>, "reason": "<one concise sentence>"}"""

_RELEVANCY_SYSTEM = """\
You are a strict RAG quality evaluator. You receive a QUESTION and the ANSWER
the agent produced.

Score ANSWER RELEVANCY: how directly and completely the answer addresses
what was actually asked.

Scoring scale:
  1.0 — Answer directly and completely addresses the question with specific, accurate details
  0.8 — Answer provides partial relevant information from its sources AND acknowledges what
         it cannot answer — i.e. it gives the user something useful even if incomplete
         (e.g. "Bank accounts are supported for purchases, but I don't have specific fee
         amounts — check help.coinbase.com for current rates")
  0.7 — Answer mostly addresses the question with minor gaps, OR is a topic-specific
         deferral that names the exact topic and redirects (e.g. "I don't have details
         about staking on Coinbase in my sources — check help.coinbase.com")
  0.5 — Answer is a generic deferral that does NOT name the specific topic asked about
  0.3 — Answer barely relates to the question
  0.0 — Answer is entirely unrelated or nonsensical

KEY RULES:
- If the answer includes BOTH partial useful information AND a topic-named deferral for the rest → 0.8
- If the answer only defers but names the exact topic → 0.7
- If the answer defers generically without naming the topic → 0.5
- Never score 0.0 for any deferral — only use 0.0 when entirely off-topic.
- Use the specific asset/feature name from the question when judging how well the answer mirrors it.

Return ONLY valid JSON — no markdown, no extra text:
{"score": <float 0.0-1.0>, "reason": "<one concise sentence>"}"""


# ── Scoring helpers ────────────────────────────────────────────────────────

def _score_faithfulness(question: str, context: str, answer: str) -> tuple[float, str]:
    client = get_llm_client()
    user_msg = (
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER: {answer}"
    )
    try:
        # Use chat() directly so we control JSON extraction ourselves;
        # chat_json() uses a greedy regex that breaks on <think> blocks.
        raw = client.chat(
            [
                {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                {"role": "user",   "content": user_msg[:6000]},
            ],
            temperature=0.0,
        )
        result = _parse_judge_response(raw)
        score  = float(result.get("score", 0.5))
        reason = str(result.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception as exc:
        log.warning("faithfulness judge error: %s", exc)
        return 0.5, f"judge_error: {exc}"


def _score_relevancy(question: str, answer: str) -> tuple[float, str]:
    client = get_llm_client()
    user_msg = f"QUESTION: {question}\n\nANSWER: {answer}"
    try:
        raw = client.chat(
            [
                {"role": "system", "content": _RELEVANCY_SYSTEM},
                {"role": "user",   "content": user_msg[:4000]},
            ],
            temperature=0.0,
        )
        result = _parse_judge_response(raw)
        score  = float(result.get("score", 0.5))
        reason = str(result.get("reason", ""))
        return max(0.0, min(1.0, score)), reason
    except Exception as exc:
        log.warning("relevancy judge error: %s", exc)
        return 0.5, f"judge_error: {exc}"


# ── Case filtering ─────────────────────────────────────────────────────────

def _is_kb_case(case: dict) -> bool:
    """True only for tagged KB cases where KB_QA is an expected/acceptable intent."""
    if "kb" not in (case.get("tags") or []):
        return False
    exp = case.get("expect", {})
    intent     = exp.get("intent", "")
    intent_any = exp.get("intent_any", [])
    return "KB_QA" in (intent, *intent_any)


# ── Context builder ────────────────────────────────────────────────────────

def _build_full_context(question: str) -> str:
    """Re-run retrieval to get the FULL chunk text — same context the agent actually saw.

    The citation excerpts stored in the response are truncated to 400 chars for UI display.
    Using truncated excerpts as the faithfulness judge's context causes false-positive
    'hallucination' calls when the agent cites something present in the full chunk but
    after the 400-char cutoff. Running retrieval fresh gives the judge the complete picture.
    """
    try:
        r = get_retriever()
        hits = r.retrieve(question)
        parts = []
        for i, h in enumerate(hits, 1):
            title   = h.article_title or ""
            section = h.section_title or ""
            text    = (h.text or "").strip()
            header  = f"[{i}] {title}" + (f" — {section}" if section else "")
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts) if parts else "(no context retrieved)"
    except Exception as exc:
        log.warning("retriever error in rag_eval: %s", exc)
        return "(retrieval failed)"


def _build_context(citations: list[dict]) -> str:
    """Fallback context from citation excerpts (used only when retrieval unavailable)."""
    parts = []
    for i, c in enumerate(citations or [], 1):
        title   = c.get("article_title", "")
        section = c.get("section_title", "")
        excerpt = (c.get("excerpt") or c.get("text") or "")[:600]
        header  = f"[{i}] {title}" + (f" — {section}" if section else "")
        parts.append(f"{header}\n{excerpt}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


# ── Main evaluation loop ───────────────────────────────────────────────────

def run_rag_eval(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_path = Path(__file__).with_name("test_cases.json")
    all_cases  = json.loads(cases_path.read_text(encoding="utf-8"))
    kb_cases   = [c for c in all_cases if _is_kb_case(c)]

    print(f"\nRAG eval — {len(kb_cases)} KB Q&A cases\n{'─'*48}")

    rows: list[dict]          = []
    faithfulness_scores: list[float] = []
    relevancy_scores:    list[float] = []

    for case in kb_cases:
        tid      = case["test_id"]
        question = case["turns"][0]["user"]

        print(f"[{tid}]")
        print(f"  Q: {question[:80]}")
        print(f"  → running agent…", end=" ", flush=True)

        sid = str(uuid.uuid4())
        try:
            resp      = run_agent_turn(sid, question)
            answer    = resp.message or ""
            citations = resp.citations or []
        except Exception as exc:
            print(f"AGENT ERROR: {exc}")
            rows.append({
                "test_id":              tid,
                "question":             question[:120],
                "intent":               "",
                "faithfulness_score":   "",
                "faithfulness_reason":  f"agent_error: {exc}",
                "relevancy_score":      "",
                "relevancy_reason":     "",
                "answer_preview":       "",
                "citation_count":       0,
            })
            continue

        # Use full retrieval context for the faithfulness judge, not truncated UI excerpts
        context = _build_full_context(question)

        # ── Routing mismatch guard ─────────────────────────────────────────
        # If the agent routed to an action (not KB_QA) AND retrieved 0 citations,
        # faithfulness is undefined — there is no KB context to be faithful to.
        # Score relevancy only (the answer still may or may not address the question),
        # and exclude faithfulness from the aggregate.
        is_kb_response = (resp.intent == "KB_QA") or (len(citations) > 0)

        if not is_kb_response:
            print(f"ROUTING MISMATCH (intent={resp.intent}, 0 citations) — skipping faithfulness.")
            r_score, r_reason = _score_relevancy(question, answer)
            rows.append({
                "test_id":              tid,
                "question":             question[:120],
                "intent":               resp.intent or "",
                "faithfulness_score":   "N/A (routing mismatch)",
                "faithfulness_reason":  f"Agent routed to {resp.intent} instead of KB_QA; no context retrieved.",
                "relevancy_score":      round(r_score, 3),
                "relevancy_reason":     r_reason,
                "answer_preview":       answer[:220],
                "citation_count":       0,
            })
            relevancy_scores.append(r_score)
            continue

        print(f"got {len(citations)} citations. Scoring…", end=" ", flush=True)

        f_score, f_reason = _score_faithfulness(question, context, answer)
        r_score, r_reason = _score_relevancy(question, answer)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"faithfulness={f_score:.2f}  relevancy={r_score:.2f}")

        rows.append({
            "test_id":              tid,
            "question":             question[:120],
            "intent":               resp.intent or "",
            "faithfulness_score":   round(f_score, 3),
            "faithfulness_reason":  f_reason,
            "relevancy_score":      round(r_score, 3),
            "relevancy_reason":     r_reason,
            "answer_preview":       answer[:220],
            "citation_count":       len(citations),
        })

    # ── Aggregate metrics ──────────────────────────────────────────────────
    routing_mismatches = sum(
        1 for r in rows if str(r.get("faithfulness_score", "")).startswith("N/A")
    )
    nf = len(faithfulness_scores)   # cases with valid faithfulness score
    nr = len(relevancy_scores)      # all cases (including routing mismatches)

    avg_f  = sum(faithfulness_scores) / nf if nf else 0.0
    avg_r  = sum(relevancy_scores) / nr if nr else 0.0
    pass_f = sum(1 for s in faithfulness_scores if s >= 0.7) / nf if nf else 0.0
    pass_r = sum(1 for s in relevancy_scores    if s >= 0.7) / nr if nr else 0.0

    metrics = {
        "cases_evaluated":          len(rows),
        "kb_responses_scored":      nf,
        "routing_mismatches":       routing_mismatches,
        "avg_faithfulness":         round(avg_f,  4),
        "avg_relevancy":            round(avg_r,  4),
        "faithfulness_pass_rate":   round(pass_f, 4),
        "relevancy_pass_rate":      round(pass_r, 4),
        "pass_threshold":           0.7,
        "note_faithfulness":        "Computed only on KB_QA responses with ≥1 citation. Routing mismatches excluded.",
        "note_relevancy":           "Computed on all cases including routing mismatches.",
        "per_case": {
            r["test_id"]: {
                "faithfulness": r.get("faithfulness_score"),
                "relevancy":    r.get("relevancy_score"),
            }
            for r in rows
        },
    }

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = out_dir / "rag_eval_results.csv"
    fieldnames = [
        "test_id", "question", "intent",
        "faithfulness_score", "faithfulness_reason",
        "relevancy_score",    "relevancy_reason",
        "answer_preview",     "citation_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── Write metrics JSON ─────────────────────────────────────────────────
    metrics_path = out_dir / "rag_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ── Chart ──────────────────────────────────────────────────────────────
    _rag_chart(metrics, rows, out_dir / "rag_eval_chart.png")

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'─'*48}")
    print(f"Cases evaluated       : {len(rows)}")
    print(f"  KB responses scored : {nf}  (routing mismatches excluded: {routing_mismatches})")
    print(f"Avg Faithfulness      : {avg_f:.3f}  ({pass_f:.1%} ≥ 0.7)  [KB responses only]")
    print(f"Avg Answer Relevancy  : {avg_r:.3f}  ({pass_r:.1%} ≥ 0.7)  [all cases]")
    print(f"\nArtifacts written to  : {out_dir}")
    print(f"  {csv_path.name}")
    print(f"  {metrics_path.name}")
    print(f"  rag_eval_chart.png")
    return metrics


# ── Chart ──────────────────────────────────────────────────────────────────

def _rag_chart(metrics: dict, rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        def _is_numeric(v) -> bool:
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False

        valid_rows = [
            r for r in rows
            if _is_numeric(r.get("faithfulness_score")) and _is_numeric(r.get("relevancy_score"))
        ]
        if not valid_rows:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        fig.suptitle(
            "RAG Quality — Faithfulness & Answer Relevancy (KB Q&A cases)",
            fontsize=12, fontweight="bold", y=1.01,
        )

        # ── Left: per-case grouped bars ──────────────────────────────────
        short_ids = [r["test_id"].replace("E", "").split("_kb_")[0] + "\n" +
                     r["test_id"].split("_kb_")[-1][:10]
                     for r in valid_rows]
        f_vals = [float(r["faithfulness_score"]) for r in valid_rows]
        r_vals = [float(r["relevancy_score"])    for r in valid_rows]
        x = np.arange(len(short_ids))
        w = 0.38
        ax1.bar(x - w/2, f_vals, w, label="Faithfulness",     color="#4c6ef5", alpha=0.85)
        ax1.bar(x + w/2, r_vals, w, label="Answer Relevancy", color="#51cf66", alpha=0.85)
        ax1.axhline(0.7, color="#fa5252", linewidth=1.3, linestyle="--",
                    label="Pass threshold (0.7)")
        ax1.set_ylim(0, 1.12)
        ax1.set_ylabel("Score (0–1)")
        ax1.set_title("Per-Case Scores")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_ids, fontsize=7.5, rotation=40, ha="right")
        ax1.legend(fontsize=8.5)
        ax1.grid(axis="y", alpha=0.25)

        # ── Right: summary metrics ───────────────────────────────────────
        summary_labels = [
            "Avg\nFaithfulness",
            "Faithfulness\nPass Rate\n(≥0.7)",
            "Avg Answer\nRelevancy",
            "Relevancy\nPass Rate\n(≥0.7)",
        ]
        summary_vals = [
            metrics["avg_faithfulness"],
            metrics["faithfulness_pass_rate"],
            metrics["avg_relevancy"],
            metrics["relevancy_pass_rate"],
        ]
        colours = ["#4c6ef5", "#339af0", "#51cf66", "#20c997"]
        bars = ax2.bar(summary_labels, summary_vals, color=colours, alpha=0.88, width=0.5)
        ax2.axhline(0.7, color="#fa5252", linewidth=1.3, linestyle="--",
                    label="Pass threshold")
        ax2.set_ylim(0, 1.12)
        ax2.set_title("Summary Metrics")
        ax2.set_ylabel("Score / Rate")
        ax2.legend(fontsize=8.5)
        for bar, val in zip(bars, summary_vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.025,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=10.5, fontweight="bold",
            )
        ax2.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  chart skipped: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_rag_eval(ROOT / "data" / "eval")
