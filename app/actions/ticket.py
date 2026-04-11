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
    r = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if r in ISSUE_TYPES:
        return r
    aliases: dict[str, str] = {
        # account_access
        "login": "account_access",
        "sign_in": "account_access",
        "signin": "account_access",
        "locked": "account_access",
        "account_locked": "account_access",
        "restricted": "account_access",
        "account_restricted": "account_access",
        "restriction": "account_access",
        "account_restriction": "account_access",
        "suspended": "account_access",
        "banned": "account_access",
        "access": "account_access",
        "can't_log_in": "account_access",
        "cannot_login": "account_access",
        # verification
        "kyc": "verification",
        "identity": "verification",
        "id": "verification",
        "identity_verification": "verification",
        "id_verification": "verification",
        "verify": "verification",
        "document": "verification",
        "photo_id": "verification",
        "passport": "verification",
        "selfie": "verification",
        # transactions
        "transaction": "transactions",
        "transfer": "transactions",
        "withdraw": "transactions",
        "withdrawal": "transactions",
        "deposit": "transactions",
        "send": "transactions",
        "receive": "transactions",
        "payment": "transactions",
        "buy": "transactions",
        "sell": "transactions",
        "purchase": "transactions",
        "crypto": "transactions",
        "pending": "transactions",
        # security
        "2fa": "security",
        "two_factor": "security",
        "two-factor": "security",
        "fraud": "security",
        "hack": "security",
        "hacked": "security",
        "compromised": "security",
        "phishing": "security",
        "suspicious": "security",
        "unauthorised": "security",
        "unauthorized": "security",
        "scam": "security",
        # fees
        "fee": "fees",
        "charge": "fees",
        "billing": "fees",
        "cost": "fees",
        "price": "fees",
        "spread": "fees",
    }
    if r in aliases:
        return aliases[r]
    # Partial / substring match: pick the first alias whose key is a substring of r
    for key, val in aliases.items():
        if key in r:
            return val
    return None


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
