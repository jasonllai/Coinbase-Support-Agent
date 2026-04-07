"""Lightweight retrieval QA checks (proxy groundedness)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.retriever import get_retriever


DEFAULT_QUERIES = [
    ("q1", "How do I enable two factor authentication on Coinbase?"),
    ("q2", "My identity verification failed what should I do"),
    ("q3", "How are Coinbase fees calculated for buying crypto"),
    ("q4", "How do I withdraw money to my bank account"),
    ("q5", "I think my account was compromised what steps should I take"),
    ("q6", "Payment method verification amounts incorrect"),
    ("q7", "How to contact Coinbase support"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/eval/retrieval_eval.csv"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    r = get_retriever()
    rows = []
    for qid, q in DEFAULT_QUERIES:
        hits = r.retrieve(q)
        rows.append(
            {
                "query_id": qid,
                "query": q,
                "top1_title": hits[0].article_title if hits else "",
                "top1_url": hits[0].canonical_url if hits else "",
                "top1_section": hits[0].section_title if hits else "",
                "top1_score": hits[0].score if hits else 0.0,
                "num_hits": len(hits),
            },
        )
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"wrote": str(args.out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
