"""Intent classification with structured output."""

from __future__ import annotations

import logging

from app.agent.schemas import Intent, RouterOutput
from app.llm.client import get_llm_client
from app.llm.prompts import ROUTER_SYSTEM

log = logging.getLogger(__name__)


def _parse_intent(raw) -> Intent:
    if raw is None:
        return Intent.AMBIGUOUS
    if isinstance(raw, Intent):
        return raw
    # Normalise "Intent.KB_QA" (Python-3.12 enum str repr) → "KB_QA"
    s = str(raw).strip()
    if "." in s:
        s = s.split(".")[-1]
    try:
        return Intent(s)
    except ValueError:
        return Intent.AMBIGUOUS


def classify_intent(user_text: str, recent_context: str) -> RouterOutput:
    client = get_llm_client()
    try:
        data = client.chat_json(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Recent context:\n{recent_context}\n\nUser:\n{user_text}"[:12000],
                },
            ],
            temperature=0.0,
        )
        intent = _parse_intent(data.get("intent"))
        return RouterOutput(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "")),
            clarifying_question=data.get("clarifying_question"),
            slots=dict(data.get("slots") or {}),
        )
    except Exception as e:
        log.warning("router failed: %s", e)
        # Fall through to KB_QA so the retriever still attempts an answer.
        # Do NOT set clarifying_question — that would suppress the RAG path.
        return RouterOutput(
            intent=Intent.KB_QA,
            confidence=0.3,
            rationale="router_fallback_llm_unavailable",
            slots={},
        )
