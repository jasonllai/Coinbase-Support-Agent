"""Grounded KB answering over retrieved chunks."""

from __future__ import annotations

import logging

from app.llm.client import get_llm_client
from app.llm.prompts import KB_QA_SYSTEM
from app.retrieval.retriever import RetrievedChunk, get_retriever

log = logging.getLogger(__name__)


def answer_kb(user_text: str) -> tuple[str, str | None, list[dict], float]:
    r = get_retriever()
    hits = r.retrieve(user_text)

    context_blocks = []
    citations = []
    for h in hits:
        context_blocks.append(
            f"SOURCE_TITLE: {h.article_title}\nSECTION: {h.section_title}\nURL: {h.canonical_url}\nTEXT:\n{h.text}\n",
        )
        citations.append(
            {
                "article_title": h.article_title,
                "section_title": h.section_title,
                "url": h.canonical_url,
                "excerpt": h.text[:400],
                "score": h.score,
            },
        )
    context = "\n---\n".join(context_blocks)[:11000]

    client = get_llm_client()
    user = f"Question:\n{user_text}\n\nSources:\n{context}"
    try:
        out = client.chat_json(
            [{"role": "system", "content": KB_QA_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        conf = float(out.get("confidence", 0.5))
        concise = str(out.get("concise_answer", "")).strip()
        details = out.get("details")
        details_str = str(details).strip() if details else None
        return concise, details_str, citations, conf
    except Exception as e:
        log.warning("kb answer failed: %s", e)
        # Build a readable answer directly from the top retrieved excerpts
        if hits:
            top = hits[0]
            excerpt = (top.text or "")[:600].strip()
            section = f" — {top.section_title}" if top.section_title else ""
            fallback = (
                f"Here's what I found in **{top.article_title}**{section}:\n\n"
                f"{excerpt}"
            )
            if len(hits) > 1:
                fallback += f"\n\n*Also relevant: {hits[1].article_title}*"
        else:
            fallback = "I couldn't find a matching Help Center article for that question."
        return fallback, None, citations, 0.2


def evidence_sufficient(hits: list[RetrievedChunk]) -> bool:
    if not hits:
        return False
    return hits[0].score >= 0.08 or len(hits) >= 2
