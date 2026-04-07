from __future__ import annotations

from app.llm.client import get_llm_client
from app.retrieval.retriever import get_retriever


def onboarding_plan(
    new_to_crypto: str | None,
    goal: str | None,
    region: str | None,
    extra_context: str,
) -> dict:
    r = get_retriever()
    q_parts = ["Coinbase getting started", goal or "", region or "", extra_context]
    query = " ".join(p for p in q_parts if p).strip()
    hits = r.retrieve(query)
    evidence = "\n\n".join(f"[{h.article_title} — {h.section_title}]\n{h.text[:900]}" for h in hits[:3])

    client = get_llm_client()
    sys = (
        "You create a concise onboarding checklist for a Coinbase retail user. "
        "Use ONLY the provided evidence snippets for factual claims; if evidence is thin, say what is unknown. "
        "Do not promise funds recovery. Return JSON keys: "
        "summary (string), first_steps (array of strings), security_tips (array), verification_guidance (array), "
        "next_actions (array), suggested_kb_questions (array of 3 short follow-ups)."
    )
    user = (
        f"new_to_crypto={new_to_crypto}\n"
        f"goal={goal}\n"
        f"region={region}\n\n"
        f"Evidence:\n{evidence}"
    )
    try:
        plan = client.chat_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user[:12000]}],
            temperature=0.2,
        )
    except Exception:
        plan = {
            "summary": "Here’s a practical starter checklist while we connect more specifics.",
            "first_steps": [
                "Create your Coinbase account with a strong, unique password.",
                "Enable two-factor authentication (2FA).",
                "Complete identity verification when prompted.",
            ],
            "security_tips": [
                "Never share your seed phrase or SMS codes.",
                "Watch for phishing sites that mimic Coinbase.",
            ],
            "verification_guidance": [
                "Use a well-lit photo of an accepted ID and match your legal name.",
            ],
            "next_actions": [
                "Add a payment method from the Payments settings.",
                "Make a small test buy to learn the flow.",
            ],
            "suggested_kb_questions": [
                "How do I enable 2FA?",
                "What documents are required for ID verification?",
                "How do Coinbase fees work?",
            ],
        }
    citations = [
        {
            "article_title": h.article_title,
            "section_title": h.section_title,
            "url": h.canonical_url,
            "excerpt": h.text[:280],
        }
        for h in hits[:4]
    ]
    return {"ok": True, "plan": plan, "citations": citations}
