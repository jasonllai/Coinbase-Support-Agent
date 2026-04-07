from __future__ import annotations

import re

from app.storage.sqlite_store import get_store

ISSUE_TYPES = {
    "account_access",
    "verification",
    "transactions",
    "security",
    "fees",
    "other",
}


def validate_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def normalize_issue_type(raw: str) -> str | None:
    r = raw.strip().lower().replace(" ", "_")
    if r in ISSUE_TYPES:
        return r
    aliases = {
        "login": "account_access",
        "locked": "account_access",
        "2fa": "security",
        "fraud": "security",
        "withdraw": "transactions",
        "deposit": "transactions",
    }
    return aliases.get(r)


def create_ticket(session_id: str, issue_type: str, email: str, problem_description: str) -> dict:
    raw = issue_type.strip().lower().replace(" ", "_")
    it = normalize_issue_type(issue_type) or raw
    if it not in ISSUE_TYPES:
        return {
            "ok": False,
            "error": "invalid_issue_type",
            "message": f"Issue type must be one of: {', '.join(sorted(ISSUE_TYPES))}.",
        }
    if not validate_email(email):
        return {"ok": False, "error": "invalid_email", "message": "Please provide a valid email address."}
    desc = problem_description.strip()
    if len(desc) < 12:
        return {"ok": False, "error": "short_description", "message": "Please describe the issue in at least a sentence."}
    tid = get_store().create_ticket(session_id, it, email.strip(), desc)
    return {"ok": True, "ticket_id": tid, "issue_type": it, "email": email.strip()}
