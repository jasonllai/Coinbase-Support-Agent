from __future__ import annotations

import re

from app.storage.sqlite_store import get_store

SUBTYPES = {"forgot_password", "lost_2fa", "account_locked", "compromised"}

SUBTYPE_LABELS = {
    "forgot_password": "Forgot password",
    "lost_2fa":        "Lost 2FA device",
    "account_locked":  "Account locked",
    "compromised":     "Compromised / hacked account",
}


def validate_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def normalize_subtype(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if r in SUBTYPES:
        return r
    # Substring matching for multi-word variants
    if "lost_2fa" in r or "2fa" in r or "two_factor" in r or "authenticator" in r or "lost_device" in r:
        return "lost_2fa"
    if "forgot" in r or "reset_password" in r or "reset password" in r:
        return "forgot_password"
    if "lock" in r:
        return "account_locked"
    if "hack" in r or "compromis" in r or "unauthoriz" in r or "stolen" in r:
        return "compromised"
    aliases = {
        "locked": "account_locked",
        "hacked": "compromised",
        "unauthorized": "compromised",
        "2fa": "lost_2fa",
        "twofactor": "lost_2fa",
        "lost_authenticator": "lost_2fa",
        "lost_phone": "lost_2fa",
        "lost_device": "lost_2fa",
        "password": "forgot_password",
        "forgot": "forgot_password",
        "compromised_account": "compromised",
        "account_hacked": "compromised",
    }
    return aliases.get(r)


def recovery_step(session_id: str, state: dict | None, user_reply: str, router_slots: dict) -> dict:
    """Multi-turn slot filling for account recovery; persists partial state with stable case_id."""
    store = get_store()
    base = dict(state or {})
    base.pop("session_id", None)   # keep stored state_json clean

    # ── Merge router hints (don't overwrite already-collected values) ──
    for k in ("email", "issue_subtype"):
        if router_slots.get(k):
            base.setdefault(k, router_slots[k])

    # ── Parse email from free text if still missing ──
    if not base.get("email"):
        m = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", user_reply)
        if m:
            base["email"] = m.group(0)

    # ── Detect subtype from free text when LLM slot is empty ──
    if not base.get("issue_subtype") and not router_slots.get("issue_subtype"):
        t = user_reply.lower()
        if any(k in t for k in ("lost 2fa", "lost my 2fa", "2fa device", "lost authenticator", "lost my phone")):
            base["issue_subtype"] = "lost_2fa"
        elif any(k in t for k in ("forgot password", "reset password", "forgot my password")):
            base["issue_subtype"] = "forgot_password"
        elif any(k in t for k in ("account locked", "locked out", "my account is locked")):
            base["issue_subtype"] = "account_locked"
        elif any(k in t for k in ("compromised", "hacked", "unauthorized access", "stolen")):
            base["issue_subtype"] = "compromised"

    subtype = normalize_subtype(base.get("issue_subtype")) or normalize_subtype(router_slots.get("issue_subtype"))
    if subtype:
        base["issue_subtype"] = subtype

    # ── Validate and build targeted error messages ──
    validation_errors: list[str] = []
    missing: list[str] = []

    # Email validation
    raw_email = base.get("email", "")
    if raw_email:
        if not validate_email(str(raw_email)):
            validation_errors.append(
                f"The email **{raw_email}** doesn't look valid "
                "(it needs an @ and a proper domain, e.g. john@example.com). Please correct it."
            )
            base.pop("email", None)   # clear so user must re-enter
    if not base.get("email"):
        missing.append("email")

    # Subtype validation — if user provided something unrecognized, tell them
    raw_sub = base.get("issue_subtype", "")
    if raw_sub and raw_sub not in SUBTYPES:
        # Already tried normalize above — if still not in SUBTYPES, it's unrecognized
        validation_errors.append(
            f"I didn't recognise **{raw_sub}** as an issue category. "
            "Please choose one of: **forgot password**, **lost 2FA device**, "
            "**account locked**, or **compromised account**."
        )
        base.pop("issue_subtype", None)
    if not base.get("issue_subtype"):
        missing.append("issue_subtype")

    if validation_errors or missing:
        # Build a clear, friendly prompt
        parts: list[str] = list(validation_errors)

        prompts = {
            "email": "What email address is associated with your Coinbase account?",
            "issue_subtype": (
                "Which of these best describes your situation?\n"
                "- **Forgot password**\n"
                "- **Lost 2FA device** (lost access to your authenticator app or phone)\n"
                "- **Account locked**\n"
                "- **Compromised / hacked account**"
            ),
        }
        for field in missing:
            if field in prompts:
                parts.append(prompts[field])

        cid = store.upsert_recovery(base.get("case_id"), session_id, base)
        base["case_id"] = cid
        return {
            "ok": True,
            "complete": False,
            "state": base,
            "assistant": "\n\n".join(parts) if parts else "Could you share the missing details?",
            "missing": missing,
        }

    # ── All slots collected — create/update the recovery case ──
    guidance = {
        "forgot_password": (
            "Visit the Coinbase sign-in page and click **Forgot password**. "
            "Check your inbox and spam folder for the reset link."
        ),
        "lost_2fa": (
            "Use the 2FA reset flow in the Coinbase Help Center. "
            "Be prepared for identity verification — Coinbase will not bypass security steps."
        ),
        "account_locked": (
            "Follow the prompts in the Coinbase app and check your email for security notices. "
            "Avoid repeated failed login attempts, which can extend the lock period."
        ),
        "compromised": (
            "**First**, secure your email account immediately. "
            "Then follow Coinbase's account security guidance and contact official Coinbase support."
        ),
    }

    case_id = store.upsert_recovery(base.get("case_id"), session_id, base)
    subtype_label = SUBTYPE_LABELS.get(base["issue_subtype"], base["issue_subtype"].replace("_", " ").title())
    msg = (
        f"Recovery case **`{case_id}`** has been opened for **{subtype_label}**.\n\n"
        f"{guidance.get(base['issue_subtype'], guidance['account_locked'])}\n\n"
        "For your security, I cannot recover accounts or bypass any protections. "
        "Please use Coinbase's official in-product recovery flows."
    )
    return {
        "ok": True,
        "complete": True,
        "state": {**base, "case_id": case_id},
        "assistant": msg,
        "missing": [],
    }
