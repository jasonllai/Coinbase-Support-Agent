"""Hybrid guardrails: fast regex/heuristics + LLM classification."""

from __future__ import annotations

import logging
import re

from app.agent.schemas import GuardrailResult, Intent
from app.llm.client import get_llm_client
from app.llm.prompts import SAFETY_CLASSIFIER_SYSTEM

log = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"disregard (the )?system prompt",
    r"reveal (your )?(hidden )?(chain[- ]of[- ]thought|reasoning|system prompt)",
    r"jailbreak",
    r"you are now",
    r"developer mode",
    r"print your instructions",
]

BYPASS_PATTERNS = [
    r"bypass (2fa|two[- ]factor|kyc|verification)",
    r"disable (2fa|two[- ]factor)",
    r"recover (account|wallet) without (email|verification|2fa)",
    r"open (a|my) account without",
    r"fake (id|identity|documents)",
]

ILLEGAL_PATTERNS = [
    r"money laundering",
    r"wash trade",
    r"evade (tax|taxes)",
    r"unregistered securities pump",
]

INVESTMENT_PATTERNS = [
    r"should i buy",
    r"which coin (will|should)",
    r"price prediction",
    r"guaranteed return",
    r"investment advice",
]


def prescreen_rules(user_text: str) -> GuardrailResult | None:
    t = user_text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, t, re.I):
            return GuardrailResult(
                blocked=True,
                category="injection",
                reason=f"matched_pattern:{pat}",
                user_message=(
                    "I can’t follow instructions that try to override my safety rules. "
                    "I’m here to help with Coinbase **customer support** topics only."
                ),
            )
    for pat in BYPASS_PATTERNS:
        if re.search(pat, t, re.I):
            return GuardrailResult(
                blocked=True,
                category="security_bypass",
                reason=f"matched_pattern:{pat}",
                user_message=(
                    "I can’t help bypass security, identity verification, or fraud controls. "
                    "Use the official Coinbase account recovery and support flows in the Help Center."
                ),
            )
    for pat in ILLEGAL_PATTERNS:
        if re.search(pat, t, re.I):
            return GuardrailResult(
                blocked=True,
                category="illegal",
                reason=f"matched_pattern:{pat}",
                user_message="I can’t assist with illegal activity or evasion. Please use legitimate support channels.",
            )
    for pat in INVESTMENT_PATTERNS:
        if re.search(pat, t, re.I):
            return GuardrailResult(
                blocked=True,
                category="investment_advice",
                reason=f"matched_pattern:{pat}",
                user_message=(
                    "I can’t provide personalized investment advice or price predictions. "
                    "I can explain Coinbase **product and support** topics from official help articles."
                ),
            )
    return None


def llm_safety_screen(user_text: str) -> GuardrailResult:
    """Second line when rules pass — catches nuanced policy issues."""
    client = get_llm_client()
    try:
        data = client.chat_json(
            [
                {"role": "system", "content": SAFETY_CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_text[:4000]},
            ],
            temperature=0.0,
        )
        return GuardrailResult(
            blocked=bool(data.get("blocked")),
            category=data.get("category", "ok"),
            reason=str(data.get("reason", "")),
            user_message=str(data.get("user_message", "I can’t help with that request.")),
        )
    except Exception as e:
        log.warning("llm safety screen failed: %s", e)
        return GuardrailResult(
            blocked=False,
            category="ok",
            reason="safety_llm_unavailable",
            user_message="",
        )


def run_guardrails(user_text: str) -> GuardrailResult:
    hit = prescreen_rules(user_text)
    if hit:
        log.info("guardrail_block prescreen %s", hit.category)
        return hit
    llm_hit = llm_safety_screen(user_text)
    if llm_hit.blocked:
        log.info("guardrail_block llm %s", llm_hit.category)
    return llm_hit


def map_guard_to_intent(g: GuardrailResult) -> Intent | None:
    if not g.blocked:
        return None
    if g.category == "security_bypass":
        return Intent.SECURITY_SENSITIVE
    if g.category in ("injection", "illegal"):
        return Intent.UNSAFE
    if g.category == "investment_advice":
        return Intent.OUT_OF_SCOPE
    if g.category == "out_of_scope":
        return Intent.OUT_OF_SCOPE
    return Intent.UNSAFE
