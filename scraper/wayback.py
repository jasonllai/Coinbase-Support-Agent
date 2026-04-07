"""Fetch Coinbase Help articles via Internet Archive (bypasses live-site bot challenges)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup

from scraper.robots import USER_AGENT

log = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"


@dataclass
class FetchResult:
    canonical_url: str
    archive_url: str
    timestamp: str
    title: str
    text: str
    breadcrumbs: list[str] = field(default_factory=list)
    category: str = ""
    status: str = "ok"
    error: str = ""


def _host_path(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc}{p.path}".rstrip("/")


def cdx_latest_timestamp(original_url: str, timeout_s: float = 60.0) -> Optional[str]:
    """Return latest Wayback timestamp with HTTP 200 for exact URL."""

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call() -> Optional[str]:
        q = {
            "url": _host_path(original_url),
            "output": "json",
            "fl": "timestamp,statuscode",
            "filter": "statuscode:200",
            "limit": "15",
        }
        with httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(CDX_API, params=q)
            r.raise_for_status()
            rows = r.json()
        if not rows or len(rows) < 2:
            return None
        body = rows[1:]
        return max(str(row[0]) for row in body)

    try:
        return _call()
    except Exception as e:
        log.warning("cdx failed for %s: %s", original_url, e)
        return None


def fetch_wayback_html(original_https_url: str, timestamp: str, timeout_s: float = 60.0) -> str:
    archive = f"https://web.archive.org/web/{timestamp}id_/{original_https_url}"

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _call() -> str:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(archive)
            r.raise_for_status()
            return r.text

    try:
        return _call()
    except RetryError as e:
        raise e.last_attempt.exception() from e


def _strip_boilerplate(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        # Drop very short nav-like lines
        if len(ln.strip()) < 2:
            continue
        out.append(ln.strip())
    # Collapse multiple blanks
    collapsed: list[str] = []
    for ln in out:
        if ln == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(ln)
    return "\n".join(collapsed).strip()


def extract_from_html(html: str, canonical_url: str) -> tuple[str, str, list[str], str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    title = re.sub(r"\s+", " ", title)
    # Wayback titles often prefixed
    title = re.sub(r"^Coinbase Help\s*-\s*", "", title, flags=re.I).strip()

    crumbs: list[str] = []
    for nav in soup.select("nav a, .breadcrumb a"):
        t = nav.get_text(" ", strip=True)
        if t and len(t) < 120:
            crumbs.append(t)
    crumbs = list(dict.fromkeys(crumbs))[:12]

    path = urlparse(canonical_url).path
    parts = [p for p in path.split("/") if p and p not in ("en", "coinbase")]
    category = parts[0].replace("-", " ").title() if parts else ""

    text = ""
    for sel in ("article", "main", "[role=main]", ".article", ".content"):
        node = soup.select_one(sel)
        if node:
            text = node.get_text("\n", strip=True)
            break
    if not text:
        text = soup.get_text("\n", strip=True)
    text = _strip_boilerplate(text)
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    return title, text, crumbs, category


def fetch_article(canonical_url: str, sleep_s: float = 0.08) -> FetchResult:
    ts = cdx_latest_timestamp(canonical_url)
    if not ts:
        return FetchResult(
            canonical_url=canonical_url,
            archive_url="",
            timestamp="",
            title="",
            text="",
            status="skipped",
            error="no_wayback_snapshot",
        )
    archive_url = f"https://web.archive.org/web/{ts}id_/{canonical_url}"
    try:
        try:
            html = fetch_wayback_html(canonical_url, ts)
        except Exception as e:
            return FetchResult(
                canonical_url=canonical_url,
                archive_url=archive_url,
                timestamp=ts,
                title="",
                text="",
                status="error",
                error=f"wayback_fetch:{e}",
            )
        title, text, crumbs, category = extract_from_html(html, canonical_url)
        if len(text) < 80:
            return FetchResult(
                canonical_url=canonical_url,
                archive_url=archive_url,
                timestamp=ts,
                title=title,
                text=text,
                breadcrumbs=crumbs,
                category=category,
                status="skipped",
                error="insufficient_text",
            )
        time.sleep(sleep_s)
        return FetchResult(
            canonical_url=canonical_url,
            archive_url=archive_url,
            timestamp=ts,
            title=title,
            text=text,
            breadcrumbs=crumbs,
            category=category,
            status="ok",
        )
    except Exception as e:
        log.exception("fetch failed %s", canonical_url)
        return FetchResult(
            canonical_url=canonical_url,
            archive_url=archive_url,
            timestamp=ts or "",
            title="",
            text="",
            status="error",
            error=str(e),
        )


def fetch_many(urls: list[str], min_ok: int = 60) -> list[FetchResult]:
    results: list[FetchResult] = []
    for u in urls:
        results.append(fetch_article(u))
        if sum(1 for r in results if r.status == "ok") >= min_ok and len(results) >= min_ok:
            # continue to process remaining? For strict min_ok we can break early if we have enough ok
            pass
    return results


def stable_doc_id(canonical_url: str) -> str:
    h = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"cb_help_{h}"


def results_to_manifest_rows(results: list[FetchResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "doc_id": stable_doc_id(r.canonical_url),
                "canonical_url": r.canonical_url,
                "archive_url": r.archive_url,
                "wayback_timestamp": r.timestamp,
                "title": r.title,
                "category": r.category,
                "status": r.status,
                "error": r.error,
                "char_count": len(r.text),
            },
        )
    return rows
