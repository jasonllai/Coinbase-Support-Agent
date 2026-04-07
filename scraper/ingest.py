"""CLI entry: discover -> fetch -> write corpus + manifest (resumable)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from scraper.discover import WAYBACK_TS_DEFAULT, discover_urls, load_seed_urls
from scraper.robots import can_fetch
from scraper.wayback import fetch_article, stable_doc_id

log = logging.getLogger(__name__)

_MANIFEST_FIELDS = [
    "doc_id",
    "canonical_url",
    "archive_url",
    "wayback_timestamp",
    "title",
    "category",
    "status",
    "error",
    "char_count",
]


def _load_corpus_urls(corpus_path: Path) -> set[str]:
    if not corpus_path.exists():
        return set()
    out: set[str] = set()
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["canonical_url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _manifest_row_from_doc(doc: dict) -> dict[str, object]:
    return {k: doc.get(k, "") for k in _MANIFEST_FIELDS}


def _rewrite_manifests(corpus_path: Path, manifest_csv: Path, manifest_json: Path) -> None:
    rows: list[dict[str, object]] = []
    if corpus_path.exists():
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                rows.append(_manifest_row_from_doc(doc))
    manifest_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(_MANIFEST_FIELDS))
            w.writeheader()
            w.writerows(rows)


def run_ingest(
    out_dir: Path,
    min_articles: int = 60,
    wayback_ts: str | None = None,
    max_discover: int = 200,
    resume: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "articles.jsonl"
    manifest_csv = out_dir / "manifest.csv"
    manifest_json = out_dir / "manifest.json"
    robots_log = out_dir / "robots_check.json"

    sample_url = "https://help.coinbase.com/en/coinbase"
    decision = can_fetch(sample_url)
    robots_log.write_text(
        json.dumps(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "sample_url": sample_url,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "robots_url": decision.robots_url,
                "note": "Live help.coinbase.com is often Cloudflare-protected; ingestion uses Wayback.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    existing = _load_corpus_urls(corpus_path) if resume else set()
    if resume:
        mode = "a"
    else:
        mode = "w"
        if corpus_path.exists():
            corpus_path.unlink()
        existing = set()

    ts = wayback_ts or WAYBACK_TS_DEFAULT
    urls = discover_urls(ts=ts, max_urls=max_discover)
    merged = list(dict.fromkeys(urls + load_seed_urls()))
    cap = max(max_discover * 2, 400)
    urls = merged[:cap]
    log.info("candidate urls %s (%s already in corpus)", len(urls), len(existing))

    n_ok = len(existing)
    attempted = 0
    with corpus_path.open(mode, encoding="utf-8") as corpus_fp:
        for u in urls:
            if u in existing:
                continue
            attempted += 1
            r = fetch_article(u)
            if r.status == "ok":
                doc_id = stable_doc_id(r.canonical_url)
                row = {
                    "doc_id": doc_id,
                    "canonical_url": r.canonical_url,
                    "archive_url": r.archive_url,
                    "wayback_timestamp": r.timestamp,
                    "title": r.title,
                    "category": r.category,
                    "status": r.status,
                    "error": r.error,
                    "char_count": len(r.text),
                    "breadcrumbs": r.breadcrumbs,
                    "body_text": r.text,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "content_source": "internet_archive_wayback",
                }
                corpus_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                corpus_fp.flush()
                existing.add(u)
                n_ok += 1
                log.info("ingested %s (%s/%s ok)", doc_id, n_ok, min_articles)
            else:
                log.info("skip %s status=%s err=%s", u, r.status, r.error)

            if n_ok >= min_articles:
                break

    _rewrite_manifests(corpus_path, manifest_csv, manifest_json)
    log.info(
        "corpus complete: %s ok articles -> %s (attempted new=%s)",
        n_ok,
        corpus_path,
        attempted,
    )
    if n_ok < min_articles:
        log.warning(
            "fewer than %s successful articles (%s). Re-run with higher --max-discover or check network.",
            min_articles,
            n_ok,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Ingest Coinbase Help Center via Wayback")
    p.add_argument("--out", type=Path, default=Path("data/corpus"))
    p.add_argument("--min-articles", type=int, default=60)
    p.add_argument("--wayback-ts", type=str, default=None)
    p.add_argument("--max-discover", type=int, default=200)
    p.add_argument("--no-resume", action="store_true", help="Start fresh (delete corpus first)")
    args = p.parse_args()
    run_ingest(
        args.out,
        min_articles=args.min_articles,
        wayback_ts=args.wayback_ts,
        max_discover=args.max_discover,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
