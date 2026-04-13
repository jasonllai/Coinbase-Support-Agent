"""Grounded KB answering over retrieved chunks, with optional conversation context."""

from __future__ import annotations

import logging
import re

from app.llm.client import get_llm_client
from app.llm.prompts import KB_QA_SYSTEM
from app.retrieval.retriever import RetrievedChunk, get_retriever

log = logging.getLogger(__name__)


def answer_kb(
    user_text: str,
    conversation_tail: str = "",
) -> tuple[str, str | None, list[dict], float]:
    """Retrieve relevant chunks then call the LLM to produce a grounded answer.

    Args:
        user_text: The user's current question.
        conversation_tail: Recent conversation turns (formatted text) for follow-up context.
    Returns:
        (concise_answer, details, citations, confidence)
    """
    r = get_retriever()
    hits = r.retrieve(user_text)

    context_blocks = []
    _seen_urls: dict[str, int] = {}   # url → index of best citation so far
    citations: list[dict] = []

    for idx, h in enumerate(hits, 1):
        context_blocks.append(
            f"[SOURCE {idx}] {h.article_title}\n"
            f"Section: {h.section_title}\n"
            f"URL: {h.canonical_url}\n"
            f"{h.text}\n",
        )
        url = h.canonical_url or ""
        if url and url in _seen_urls:
            # Same article URL already in citations — keep the higher-scoring entry
            prev_idx = _seen_urls[url]
            if h.score > citations[prev_idx]["score"]:
                citations[prev_idx] = {
                    "article_title": h.article_title,
                    "section_title": h.section_title,
                    "url": url,
                    "excerpt": h.text[:400],
                    "score": h.score,
                }
        else:
            _seen_urls[url] = len(citations)
            citations.append(
                {
                    "article_title": h.article_title,
                    "section_title": h.section_title,
                    "url": url,
                    "excerpt": h.text[:400],
                    "score": h.score,
                },
            )

    context = "\n---\n".join(context_blocks)[:11000]

    # Build user prompt — include conversation history for follow-up awareness
    history_block = f"CONVERSATION HISTORY:\n{conversation_tail}\n\n" if conversation_tail else ""
    user_prompt = f"{history_block}Question:\n{user_text}\n\nSources:\n{context}"

    client = get_llm_client()
    try:
        out = client.chat_json(
            [{"role": "system", "content": KB_QA_SYSTEM}, {"role": "user", "content": user_prompt}],
            temperature=0.1,   # deterministic — same question → same answer
        )
        conf = float(out.get("confidence", 0.5))
        concise = str(out.get("concise_answer", "")).strip()
        details = out.get("details")
        details_str = str(details).strip() if details else None
        return concise, details_str, citations, conf
    except Exception as e:
        log.warning("kb answer failed: %s", e)
        # The model sometimes outputs well-formed prose without JSON wrapping.
        # Try a plain-text chat call and use the prose directly if it looks useful.
        try:
            raw = client.chat(
                [{"role": "system", "content": KB_QA_SYSTEM}, {"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
            prose = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            if prose and len(prose) > 30:
                return prose, None, citations, 0.4
        except Exception:
            pass
        # Last resort: pull the most relevant excerpt from the top retrieved chunk
        if hits:
            top = hits[0]
            excerpt = (top.text or "")[:600].strip()
            section = f" — {top.section_title}" if top.section_title else ""
            fallback = f"Here's what I found in **{top.article_title}**{section}:\n\n{excerpt}"
            if len(hits) > 1:
                fallback += f"\n\n*Also relevant: {hits[1].article_title}*"
        else:
            fallback = "I couldn't find a matching Help Center article for that question."
        return fallback, None, citations, 0.2


def evidence_sufficient(hits: list[RetrievedChunk]) -> bool:
    if not hits:
        return False
    return hits[0].score >= 0.08 or len(hits) >= 2
