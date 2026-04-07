"""Robots.txt handling with explicit logging when unreachable (e.g. Cloudflare)."""

from __future__ import annotations

import logging
import urllib.robotparser
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

USER_AGENT = (
    "CoinbaseSupportAgentCourseBot/1.0 (+https://github.com/educational-use) "
    "respectful crawl; contact course instructor"
)


@dataclass
class RobotsDecision:
    allowed: bool
    reason: str
    robots_url: str


def fetch_robots_txt(base_url: str, timeout_s: float = 20.0) -> Optional[str]:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(robots_url)
            if r.status_code != 200:
                log.warning("robots.txt non-200: %s status=%s", robots_url, r.status_code)
                return None
            ctype = r.headers.get("content-type", "")
            if "text/plain" not in ctype and "text/html" not in ctype:
                # Cloudflare may return HTML challenge
                if "<html" in r.text[:200].lower():
                    log.warning(
                        "robots.txt appears to be HTML challenge (blocked). "
                        "Skipping automated robots parse; see README for ethics policy.",
                    )
                    return None
            return r.text
    except Exception as e:
        log.warning("robots.txt fetch failed: %s (%s)", robots_url, e)
        return None


def can_fetch(url: str) -> RobotsDecision:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    body = fetch_robots_txt(url)
    if body is None:
        return RobotsDecision(
            allowed=True,
            reason="robots.txt unavailable or blocked; default allow with strict rate limits + Wayback",
            robots_url=robots_url,
        )
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.parse(body.splitlines())
    allowed = rp.can_fetch(USER_AGENT, url)
    return RobotsDecision(
        allowed=allowed,
        reason="parsed robots.txt",
        robots_url=robots_url,
    )


def wayback_raw_url(timestamp: str, original_https_url: str) -> str:
    """Internet Archive raw capture URL (no rewritten JS)."""
    return f"https://web.archive.org/web/{timestamp}id_/{original_https_url}"
