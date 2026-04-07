"""Discover Help Center article URLs from archived category pages."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from scraper.robots import USER_AGENT

WAYBACK_TS_DEFAULT = "20210228062344"
BASE = "https://help.coinbase.com"

CATEGORY_PATHS: list[str] = [
    "/en/coinbase",
    "/en/coinbase/getting-started",
    "/en/coinbase/managing-my-account",
    "/en/coinbase/trading-and-funding",
    "/en/coinbase/privacy-and-security",
    "/en/coinbase/taxes-reports-and-financial-services",
    "/en/coinbase/other-topics",
]

# Prefer articles aligned with course topics (skip deepest altcoin-only slugs if needed)
PRIORITY_KEYWORDS = (
    "account",
    "verif",
    "2fa",
    "two-factor",
    "security",
    "password",
    "recover",
    "phish",
    "scam",
    "compromise",
    "transaction",
    "send",
    "receive",
    "deposit",
    "withdraw",
    "fee",
    "payment",
    "bank",
    "card",
    "kyc",
    "identity",
    "contact",
    "support",
    "restrict",
    "lock",
    "coinbase",
    "getting-started",
    "wallet",
    "fiat",
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def wayback_fetch(ts: str, path: str, timeout_s: float = 60.0) -> str:
    url = f"https://web.archive.org/web/{ts}id_/{BASE}{path}"
    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text


def extract_coinbase_links(html: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        h = m.group(1)
        if h.startswith("//"):
            h = "https:" + h
        if "help.coinbase.com" in h:
            h = urllib.parse.urljoin(BASE + "/", h)
        elif h.startswith("/en/coinbase"):
            h = BASE + h.split("#")[0]
        else:
            continue
        h = h.split("#")[0]
        if "/en/coinbase" not in h:
            continue
        parts = h.replace(BASE, "").strip("/").split("/")
        # article-like paths are deeper than category roots
        if len(parts) >= 4 and not h.endswith("/en/coinbase"):
            out.add(h)
    return out


def score_url(url: str) -> int:
    u = url.lower()
    return sum(1 for k in PRIORITY_KEYWORDS if k in u)


def load_seed_urls(path: str | None = None) -> list[str]:
    p = Path(__file__).with_name("seed_urls.txt") if path is None else Path(path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln.startswith("http")]


def discover_urls(
    ts: str = WAYBACK_TS_DEFAULT,
    max_urls: int = 120,
    category_paths: Iterable[str] | None = None,
) -> list[str]:
    paths = list(category_paths) if category_paths else CATEGORY_PATHS
    found: set[str] = set()
    for p in paths:
        try:
            html = wayback_fetch(ts, p)
            found |= extract_coinbase_links(html)
        except Exception:
            continue
    ranked = sorted(found, key=lambda u: (-score_url(u), u))
    out = ranked[:max_urls]
    if len(out) < 40:
        out = list(dict.fromkeys(out + load_seed_urls()))
    return out[:max_urls]
