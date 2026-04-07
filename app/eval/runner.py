"""Run evaluation scenarios, compute metrics, write CSV/JSON/MD/chart artifacts."""

from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import run_agent_turn
from app.core.config import get_settings


def load_cases() -> list[dict]:
    p = Path(__file__).with_name("test_cases.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_expect(last, exp: dict) -> tuple[bool, list[str]]:
    ok = True
    reasons: list[str] = []
    if "intent" in exp and last.intent != exp["intent"]:
        ok = False
        reasons.append(f"intent want {exp['intent']} got {last.intent}")
    if "status" in exp and last.status != exp["status"]:
        ok = False
        reasons.append(f"status want {exp['status']} got {last.status}")
    if "status_not" in exp and last.status == exp["status_not"]:
        ok = False
        reasons.append(f"status should not be {exp['status_not']}")
    if "substring" in exp and exp["substring"] not in (last.message or ""):
        ok = False
        reasons.append("substring missing in message")
    if "last_substring" in exp and exp["last_substring"] not in (last.message or ""):
        ok = False
        reasons.append("last_substring missing")
    if "last_status" in exp and last.status != exp["last_status"]:
        ok = False
        reasons.append(f"last_status want {exp['last_status']} got {last.status}")
    if exp.get("citations_nonempty") is True and not (last.citations or []):
        ok = False
        reasons.append("expected citations on last turn")
    return ok, reasons


def _run_smoke(case: dict) -> tuple[bool, list[str], dict]:
    exp = case["expect"]
    reasons: list[str] = []
    s = get_settings()
    if exp.get("faiss_ready"):
        ok = s.faiss_index_path.exists() and s.faiss_meta_path.exists()
        if not ok:
            reasons.append("FAISS index or meta missing")
        return ok, reasons, {"faiss_ready": ok}
    return False, ["unknown smoke expect"], {}


def run_all(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    guard_tests = 0
    guard_hits = 0
    citation_tests = 0
    citation_hits = 0
    action_tests = 0
    action_hits = 0

    for case in load_cases():
        kind = case.get("kind", "agent")
        tid = case["test_id"]
        if kind == "smoke":
            ok, reasons, extra = _run_smoke(case)
            rows.append(
                {
                    "test_id": tid,
                    "kind": "smoke",
                    "ok": ok,
                    "reasons": "; ".join(reasons),
                    "intent": "",
                    "status": "",
                    "message_preview": json.dumps(extra)[:200],
                    "citation_count": "",
                },
            )
            continue

        sid = str(uuid.uuid4())
        last = None
        for _t in case["turns"]:
            last = run_agent_turn(sid, _t["user"])
        assert last is not None
        ok, reasons = _check_expect(last, case["expect"])
        tags = case.get("tags") or []

        rows.append(
            {
                "test_id": tid,
                "kind": "agent",
                "ok": ok,
                "reasons": "; ".join(reasons),
                "intent": last.intent,
                "status": last.status,
                "message_preview": (last.message or "")[:240],
                "citation_count": len(last.citations or []),
                "tags": ",".join(tags),
            },
        )

        if "guardrail" in tags or case["test_id"].startswith(("E09_", "E10_", "E11_", "E12_")):
            guard_tests += 1
            if last.status == "refusal":
                guard_hits += 1
        if "kb" in tags or case["test_id"].startswith("E01_") or case["test_id"].startswith("E02_"):
            citation_tests += 1
            if last.citations:
                citation_hits += 1
        if "action" in tags or "ACTION_" in json.dumps(case.get("expect", {})):
            action_tests += 1
            if ok:
                action_hits += 1

    # Fix intent_hits: count rows where expect intent matches
    intent_hits = 0
    intent_tests = 0
    for case in load_cases():
        if case.get("kind") == "smoke":
            continue
        exp = case.get("expect", {})
        if "intent" not in exp:
            continue
        intent_tests += 1
        tid = case["test_id"]
        r = next((x for x in rows if x["test_id"] == tid), None)
        if r and r["intent"] == exp["intent"]:
            intent_hits += 1

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    metrics = {
        "total_cases": total,
        "passed": passed,
        "scenario_success_rate": passed / max(total, 1),
        "intent_accuracy": intent_hits / max(intent_tests, 1) if intent_tests else None,
        "guardrail_refusal_rate": guard_hits / max(guard_tests, 1) if guard_tests else None,
        "kb_citation_rate": citation_hits / max(citation_tests, 1) if citation_tests else None,
        "action_success_rate": action_hits / max(action_tests, 1) if action_tests else None,
        "intent_tests_n": intent_tests,
        "intent_hits": intent_hits,
        "guard_tests_n": guard_tests,
        "guard_hits": guard_hits,
        "citation_tests_n": citation_tests,
        "citation_hits": citation_hits,
        "action_tests_n": action_tests,
        "action_hits": action_hits,
    }

    summary = {**metrics, "rows": len(rows)}
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (out_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["test_id", "kind", "ok", "reasons", "intent", "status", "message_preview", "citation_count"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    failures = [r for r in rows if not r["ok"]]
    (out_dir / "failure_analysis.md").write_text(
        "\n".join(
            [
                "# Evaluation failure analysis",
                "",
                f"Pass rate: {passed}/{total}",
                "",
                "## Metrics",
                "",
                f"- Scenario success rate: **{metrics['scenario_success_rate']:.2%}**",
                f"- Intent accuracy (where expected): **{metrics.get('intent_accuracy')}**",
                f"- Guardrail refusal rate (subset): **{metrics.get('guardrail_refusal_rate')}**",
                f"- KB citation presence rate (subset): **{metrics.get('kb_citation_rate')}**",
                f"- Action scenario success (tagged): **{metrics.get('action_success_rate')}**",
                "",
                "## Failures",
                "",
                *[f"- **{f['test_id']}** ({f.get('kind', '')}): {f['reasons']}" for f in failures],
                "",
                "## Root causes & mitigations",
                "- Router / LLM variance → add few-shot examples, lower temperature.",
                "- Retrieval thresholds → tune `evidence_sufficient` and hybrid fusion.",
                "- Safety classifier variance → strengthen regex prescreen list.",
                "",
            ],
        ),
        encoding="utf-8",
    )

    _try_chart(metrics, out_dir / "eval_metrics_chart.png")
    print(json.dumps(summary, indent=2))
    return summary


def _try_chart(metrics: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = []
        vals = []
        if metrics.get("scenario_success_rate") is not None:
            labels.append("Scenario\nsuccess")
            vals.append(metrics["scenario_success_rate"])
        if metrics.get("intent_accuracy") is not None:
            labels.append("Intent\naccuracy")
            vals.append(metrics["intent_accuracy"])
        if metrics.get("guardrail_refusal_rate") is not None:
            labels.append("Guardrail\nrefusals")
            vals.append(metrics["guardrail_refusal_rate"])
        if metrics.get("kb_citation_rate") is not None:
            labels.append("KB citations")
            vals.append(metrics["kb_citation_rate"])
        if metrics.get("action_success_rate") is not None:
            labels.append("Action\nsuccess")
            vals.append(metrics["action_success_rate"])
        if not vals:
            return
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar(labels, vals, color=["#4c6ef5", "#15aabf", "#fa5252", "#51cf66"][: len(vals)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("Coinbase Support Agent — eval metrics")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    run_all(ROOT / "data" / "eval")
