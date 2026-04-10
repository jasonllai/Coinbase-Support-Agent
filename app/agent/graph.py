"""LangGraph orchestration for the support agent."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.actions.onboarding import onboarding_plan
from app.actions.recovery import recovery_step
from app.actions.ticket import create_ticket
from app.actions.transaction import check_transaction
from app.agent.guardrails import map_guard_to_intent, run_guardrails
from app.agent.qa import answer_kb, evidence_sufficient
from app.retrieval.retriever import get_retriever
from app.agent.router import classify_intent
from app.agent.schemas import AgentResponse, Intent
from app.storage.sqlite_store import get_store

log = logging.getLogger(__name__)

# ── Regex-based slot extraction ────────────────────────────────────────────
_TX_PATTERN    = re.compile(r"\b(CB-[A-Z0-9][\w\-]{2,40})\b", re.I)
_ASSET_PATTERN = re.compile(
    r"\b(BTC|ETH|USDC|SOL|LTC|DOGE|XRP|MATIC|ADA|DOT|AVAX|"
    r"bitcoin|ethereum|solana|litecoin|dogecoin|ripple|cardano|polkadot|avalanche)\b",
    re.I,
)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

# Detects when the user is asking about their own conversation / question history
_HIST_PAT = re.compile(
    # "what did I ask / say / write"
    r"\bwhat\s+did\s+i\s+(ask|say|write|mention)\b"
    # "what was my question / message / last"
    r"|\bwhat\s+was\s+my\s+(question|message|last|previous)\b"
    # "what is the question I asked …"
    r"|\bwhat\s+is\s+the\s+question\s+i\s+(asked?|said?|mentioned?)\b"
    # "what have I asked / said / talked about"
    r"|\bwhat\s+(have\s+i|i\s+have)\s+(asked?|said?|talked?|mentioned?)\b"
    # "my last / previous / prior question"
    r"|\bmy\s+(last|previous|prior)\s+(question|message|ask)\b"
    # "bring back / recall / remind me of the conversation"
    r"|\b(bring\s+back|recall|remind\s+me(\s+of)?|show\s+me)\b.{0,40}\b(conversation|question|asked?|what\s+i)\b"
    # "what have we talked about / discussed"
    r"|\bwhat\s+(have\s+)?we\s+(talked?|discussed?|covered|talked\s+about)\b"
    # "conversation history / log"
    r"|\bconversation\s+(history|log|we\s+had|just\s+now)\b"
    # "the question I asked N turns / conversations before/ago"
    r"|\bthe\s+question\s+i\s+(asked?|said?).{0,40}\b(ago|before|earlier|back)\b"
    # generic: N conversations / turns / messages ago  (extended to 35 chars)
    r"|\b(question|ask|message|conversation)\b.{0,35}\b(ago|before|earlier|back)\b"
    r"|\b(ago|before|earlier)\b.{0,35}\b(question|ask|message|conversation)\b",
    re.I,
)

# Detects when the user is asking about the ASSISTANT's previous reply
_ASST_RECALL_PAT = re.compile(
    # "what did you say / reply / replied / answer / tell / respond"
    r"\bwhat\s+did\s+you\s+(say|reply|replied|answer|answered|respond|responded|tell|told|write|mention)\b"
    # "what was your reply / answer / response"
    r"|\bwhat\s+was\s+your\s+(reply|answer|response|last|previous)\b"
    # "your last / previous reply / answer"
    r"|\byour\s+(last|previous|prior)\s+(reply|answer|response|message)\b"
    # "what you said / replied / answered / told me"
    r"|\bwhat\s+you\s+(said|replied|answered?|told\s+me|wrote)\b"
    # "repeat / show your last answer"
    r"|\b(repeat|show)\s+(your\s+)?(last\s+)?(reply|answer|response|message)\b",
    re.I,
)

# Detects queries about existing tickets OR cases (same object, just different user terminology)
_TICKET_LOOKUP_PAT = re.compile(
    r"\b(my|show|check|find|what|view|see)\b.{0,40}\b(ticket|case)s?\b"
    r"|\b(ticket|case)s?\b.{0,40}\b(status|id|number|created?|did\s+i|i\s+(created?|made|opened?|submitted?))\b"
    r"|\b(what|which)\s+(ticket|case)\b"
    r"|\b(ticket|case)\s+i\s+(created?|submitted?|opened?|made)\b",
    re.IGNORECASE,
)


def _scan_history_for_slot(msgs: list[dict[str, Any]], pattern: re.Pattern) -> str:
    """Scan user messages (most-recent-first) for the first regex match."""
    for m in reversed(msgs):
        if m.get("role") == "user":
            hit = pattern.search(m.get("content", ""))
            if hit:
                return hit.group(1) if hit.lastindex else hit.group(0)
    return ""


def _load_partial_state(msgs: list[dict[str, Any]], action_name: str) -> dict[str, Any]:
    """Return partial slot state saved in the most-recent clarify response for action_name.

    Walks the ENTIRE history (not just the last message) so state survives
    multiple clarify rounds without being lost.

    Only loads state from genuine "needs more info" clarify turns, never from terminal
    statuses like "not_found", "ok", or bare "error" with no missing fields.
    """
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        meta = m.get("meta") or {}
        action = meta.get("action") or {}
        status = meta.get("status", "")
        # Only restore state from turns that were genuinely mid-collection
        if action.get("name") == action_name and status == "clarify" and action.get("missing"):
            return dict(action.get("state") or {})
    return {}


def _extract_tx_slots(
    text: str,
    slots: dict[str, Any],
    msgs: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Extract transaction_id + asset_type from current text, LLM slots, and message history."""
    tx    = str(slots.get("transaction_id") or "").strip()
    asset = str(slots.get("asset_type")     or "").strip()

    if not tx:
        m = _TX_PATTERN.search(text)
        if m:
            tx = m.group(1).upper()
    if not asset:
        m = _ASSET_PATTERN.search(text)
        if m:
            asset = m.group(1).upper()

    # Fall back to history scan
    if msgs:
        if not tx:
            h = _scan_history_for_slot(msgs, _TX_PATTERN)
            if h:
                tx = h.upper()
        if not asset:
            h = _scan_history_for_slot(msgs, _ASSET_PATTERN)
            if h:
                asset = h.upper()

    return tx, asset


def _extract_ticket_slots(
    text: str,
    slots: dict[str, Any],
    msgs: list[dict[str, Any]] | None = None,
) -> tuple[str, str, str]:
    """Extract issue_type, email, problem_description from current text, LLM slots, and history."""
    it    = str(slots.get("issue_type")           or "").strip()
    email = str(slots.get("email")                or "").strip()
    desc  = str(slots.get("problem_description")  or "").strip()

    if not email:
        m = _EMAIL_PATTERN.search(text)
        if m:
            email = m.group(0)

    # Fall back to history scan for email
    if msgs and not email:
        h = _scan_history_for_slot(msgs, _EMAIL_PATTERN)
        if h:
            email = h

    return it, email, desc


# ── Context builders ────────────────────────────────────────────────────────

def _build_router_context(messages: list[dict[str, Any]], max_turns: int = 10) -> str:
    """Build structured context for the intent router.

    Shows the last max_turns messages with role labels + structured assistant metadata
    (intent / status / action / missing) so the router can handle continuations correctly.
    """
    tail = messages[-max_turns:]
    lines: list[str] = []

    for m in tail:
        role = m.get("role", "user")
        content = (m.get("content") or "")[:600]
        meta = m.get("meta") or {}

        if role == "assistant":
            intent   = meta.get("intent", "")
            status   = meta.get("status", "")
            action   = meta.get("action") or {}
            act_name = action.get("name", "")
            missing  = action.get("missing") or []

            label = f"ASSISTANT [intent={intent}, status={status}"
            if act_name:
                label += f", action={act_name}"
            if missing:
                label += f", missing={missing}"
            label += "]"
            lines.append(f"{label}: {content}")
        else:
            lines.append(f"USER: {content}")

    return "\n".join(lines)


def _conversation_tail_for_qa(messages: list[dict[str, Any]], max_turns: int = 6) -> str:
    """Plain-text conversation tail for KB QA follow-up context (last 6 turns = 12 messages)."""
    tail = messages[-max_turns * 2:]
    lines = []
    for m in tail:
        role    = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "")[:800]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)



class AgentState(TypedDict, total=False):
    session_id: str
    user_input: str
    messages: list[dict[str, Any]]
    router_trace: list[dict[str, Any]]
    response: dict[str, Any]
    _route: dict[str, Any]


def node_load(state: AgentState) -> AgentState:
    store = get_store()
    sid = store.ensure_session(state.get("session_id"))
    rec = store.load_session(sid)
    return {"session_id": sid, "messages": rec.messages, "router_trace": rec.router_trace}


def node_guard(state: AgentState) -> AgentState:
    user_text = state["user_input"]
    msgs      = state.get("messages", [])

    # Determine whether to skip the LLM safety classifier.
    # The classifier has NO conversation context; it sees a single message in
    # isolation and can wrongly block perfectly normal follow-up replies.
    # We still always run the fast regex prescreen (injection / bypass patterns).

    # 1. Slot-fill reply: user is answering a clarifying question (tx ID, email …)
    last_meta = _last_assistant_meta(msgs)
    is_slot_fill = bool(
        last_meta.get("status") == "clarify"
        and (last_meta.get("action") or {}).get("name")
        and (last_meta.get("action") or {}).get("missing")
    )

    # 2. Conversation history recall: "what did I ask before?", "what did you reply?" …
    is_history_recall = bool(_HIST_PAT.search(user_text.lower()) or _ASST_RECALL_PAT.search(user_text.lower()))

    # 3. Very short reply (≤ 6 words) right after ANY assistant message — usually
    #    a terse follow-up ("yes", "ok", "why", "BTC", "tell me more") that the
    #    classifier frequently mis-classifies.
    word_count  = len(user_text.split())
    has_prior   = bool(msgs)
    is_terse_followup = has_prior and word_count <= 6

    skip_llm = is_slot_fill or is_history_recall or is_terse_followup

    g = run_guardrails(user_text, skip_llm=skip_llm)
    trace = list(state.get("router_trace", []))
    trace.append({"node": "guardrails", "blocked": g.blocked, "category": g.category, "reason": g.reason})
    if g.blocked:
        intent = map_guard_to_intent(g) or Intent.UNSAFE
        resp = AgentResponse(
            session_id=state["session_id"],
            message=g.user_message,
            intent=intent.value,
            status="refusal",
            trace={"guardrails": g.model_dump()},
        )
        return {"router_trace": trace, "response": resp.model_dump()}
    return {"router_trace": trace}


def node_route_after_guard(state: AgentState) -> Literal["intent", "end"]:
    if state.get("response"):
        return "end"
    return "intent"


def node_intent(state: AgentState) -> AgentState:
    # Build rich structured context so the router understands continuations
    ctx = _build_router_context(state.get("messages", []))
    route = classify_intent(state["user_input"], ctx)
    trace = list(state.get("router_trace", []))
    route_dict = route.model_dump(mode="json")  # mode=json serialises enum → str
    trace.append({"node": "intent", "output": route_dict})
    return {"router_trace": trace, "_route": route_dict}  # type: ignore[typeddict-item]


def _last_citations_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            cits = (m.get("meta") or {}).get("citations")
            if cits:
                return list(cits)
    return []


def _last_assistant_meta(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return meta dict from the most recent assistant message, or {}."""
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            return m.get("meta") or {}
    return {}


def _pending_action(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Return (action_name, missing_fields) if the last assistant turn is awaiting user input.

    Only returns a non-empty name when:
    - status is "clarify"  (asking for missing info)
    - AND there are actual missing fields listed

    Terminal statuses like "not_found", "ok", "error" (with no missing list) are ignored
    so the agent never gets stuck re-running the same action after a conclusive result.
    """
    meta = _last_assistant_meta(messages)
    if meta.get("status") != "clarify":
        return "", []
    action = meta.get("action") or {}
    name = action.get("name", "")
    missing = list(action.get("missing") or [])
    # No missing fields = this clarify was a "soft" message (e.g., not_found with status=clarify);
    # treat it as terminal so the next user turn is routed freely.
    if not missing:
        return "", []
    return name, missing


def node_dispatch(state: AgentState) -> AgentState:
    sid = state["session_id"]
    user = state["user_input"]
    ul = user.lower()
    msgs = state.get("messages", [])

    if any(
        p in ul
        for p in (
            "article you mentioned",
            "article you cited",
            "sources you used",
            "source you linked",
            "link you shared earlier",
            "help article from before",
        )
    ):
        cits = _last_citations_from_messages(msgs)
        if cits:
            lines = [f"- **{c.get('article_title', 'Article')}** — _{c.get('section_title', '')}_" for c in cits[:6]]
            links = "\n".join(f"  - [{c.get('article_title', 'Link')}]({c.get('url', '')})" for c in cits[:6] if c.get("url"))
            resp = AgentResponse(
                session_id=sid,
                message="Here are the Help Center sources from my last answer:\n\n" + "\n".join(lines) + "\n\n" + links,
                intent=Intent.KB_QA.value,
                status="ok",
                citations=cits,
            )
            return {"response": resp.model_dump()}
        resp = AgentResponse(
            session_id=sid,
            message="I don’t have a prior article citation in this session yet. Ask a Coinbase support question and I’ll cite sources.",
            intent=Intent.KB_QA.value,
            status="clarify",
        )
        return {"response": resp.model_dump()}

    # ── Conversation history recall ───────────────────────────────────────────
    # Handles three modes:
    #   1. User asks about their own prior question(s)  → _HIST_PAT
    #   2. User asks about the assistant's last reply   → _ASST_RECALL_PAT
    #   3. User asks about BOTH ("my question and your answer") → both patterns match
    _ORDINALS = {
        "last": 1, "previous": 1, "prior": 1,
        "one": 1, "first": 1,
        "two": 2, "second": 2,
        "three": 3, "third": 3,
        "four": 4, "fourth": 4,
        "five": 5, "fifth": 5,
    }

    wants_user_q = bool(_HIST_PAT.search(ul))
    wants_asst_a = bool(_ASST_RECALL_PAT.search(ul))

    if wants_user_q or wants_asst_a:
        past_user_msgs = [m["content"] for m in msgs if m.get("role") == "user" and m.get("content")]
        past_asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant" and m.get("content")]

        if not past_user_msgs and not past_asst_msgs:
            resp = AgentResponse(
                session_id=sid,
                message=(
                    "This is actually the start of our conversation — nothing to look back on yet! "
                    "Feel free to ask anything about Coinbase."
                ),
                intent=Intent.KB_QA.value,
                status="ok",
            )
            return {"response": resp.model_dump()}

        # Resolve ordinal offset (1 = most recent, 2 = second-to-last, …)
        target_idx: int | None = None
        for word, offset in _ORDINALS.items():
            if re.search(rf"\b{word}\b", ul):
                target_idx = offset
                break
        num_m = re.search(r"\b(\d+)\b", ul)
        if num_m:
            target_idx = int(num_m.group(1))

        parts: list[str] = []

        # --- User question part ---
        if wants_user_q and past_user_msgs:
            if target_idx is not None and 1 <= target_idx <= len(past_user_msgs):
                q = past_user_msgs[-target_idx]
                label = {1: "last", 2: "second-to-last", 3: "third-to-last"}.get(
                    target_idx, f"{target_idx} turns ago"
                )
                parts.append(f"**Your {label} question:**\n> {q}")
            elif not wants_asst_a:
                # No ordinal + only asking about questions → show full list
                recent = list(reversed(past_user_msgs[-8:]))
                lines = [f"{i + 1}. {q[:120]}{'…' if len(q) > 120 else ''}" for i, q in enumerate(recent)]
                parts.append(
                    "Here are your questions from this session (most recent first):\n\n"
                    + "\n".join(lines)
                )
            else:
                # Combined "question + answer" request with no ordinal → use most recent pair
                parts.append(f"**Your last question:**\n> {past_user_msgs[-1]}")

        # --- Assistant answer part ---
        if wants_asst_a and past_asst_msgs:
            if target_idx is not None and 1 <= target_idx <= len(past_asst_msgs):
                a = past_asst_msgs[-target_idx]
            else:
                a = past_asst_msgs[-1]
            # Trim very long answers to keep the recall response readable
            a_display = a[:600].strip() + ("…" if len(a) > 600 else "")
            parts.append(f"**My last answer:**\n{a_display}")

        msg_text = "\n\n".join(parts) if parts else (
            "I don't have a prior exchange to show — this is the start of our session."
        )

        resp = AgentResponse(
            session_id=sid,
            message=msg_text,
            intent=Intent.KB_QA.value,
            status="ok",
        )
        return {"response": resp.model_dump()}

    # ── Prior action-result recall ────────────────────────────────────────────
    # "what was my recovery case?" / "show me my transaction result" / "what happened with my tx?"
    _RECALL_RECOVERY = re.compile(
        r"\b(my|the)\s+(recovery\s+(case|request)|account\s+recovery)\b"
        r"|\b(recovery|case)\s+(id|number|status|result)\b"
        r"|\bwhat.{0,20}(recovery|case)\b",
        re.I,
    )
    _RECALL_TX = re.compile(
        r"\b(what|show|tell).{0,30}(transaction|tx|transfer)\b"
        r"|\b(transaction|tx|transfer).{0,20}(result|status|detail|i checked)\b",
        re.I,
    )
    if _RECALL_RECOVERY.search(ul):
        st = get_store().load_recovery_for_session(sid)
        if st:
            cid  = st.get("case_id", "—")
            sub  = (st.get("issue_subtype") or "unknown").replace("_", " ")
            em   = st.get("email", "")
            resp = AgentResponse(
                session_id=sid,
                message=(
                    f"Here's your account recovery request from this session:\n\n"
                    f"- **Case ID:** `{cid}`\n"
                    f"- **Issue:** {sub.title()}\n"
                    + (f"- **Email:** {em}\n" if em else "")
                    + "\nLet me know if you need anything else!"
                ),
                intent=Intent.ACTION_ACCOUNT_RECOVERY.value,
                status="ok",
                action={"name": "account_recovery_recall", "payload": st},
            )
            return {"response": resp.model_dump()}

    if _RECALL_TX.search(ul):
        # Find the most recent successful transaction check in message history
        for m in reversed(msgs):
            if m.get("role") != "assistant":
                continue
            act = (m.get("meta") or {}).get("action") or {}
            payload = act.get("payload") or {}
            if act.get("name") == "check_transaction" and payload.get("found"):
                tx_r  = payload.get("transaction_id", "—")
                ast_r = payload.get("asset_type", "—")
                st_r  = payload.get("status", "—").title()
                dtl   = payload.get("detail", "")
                resp = AgentResponse(
                    session_id=sid,
                    message=(
                        f"Here's the transaction I looked up earlier:\n\n"
                        f"- **ID:** `{tx_r}`\n"
                        f"- **Asset:** {ast_r}\n"
                        f"- **Status:** {st_r}\n"
                        + (f"\n{dtl}" if dtl else "")
                    ),
                    intent=Intent.ACTION_CHECK_TRANSACTION.value,
                    status="ok",
                    action={"name": "check_transaction", "payload": payload},
                )
                return {"response": resp.model_dump()}

    if "continue" in ul and "recover" in ul:
        st = get_store().load_recovery_for_session(sid)
        if st:
            subtype = st.get("issue_subtype", "unknown")
            email = st.get("email", "")
            cid = st.get("case_id", "")
            msg = (
                f"Resuming recovery context: **{subtype}** for `{email}`.\n\n"
                f"Saved case id: `{cid}`.\n\n"
                "Tell me if anything changed, or say **complete** if you finished the official recovery steps."
            )
            resp = AgentResponse(
                session_id=sid,
                message=msg,
                intent=Intent.ACTION_ACCOUNT_RECOVERY.value,
                status="ok",
                action={"name": "account_recovery_resume", "payload": st},
            )
            return {"response": resp.model_dump()}

    # ── Ticket / case lookup ────────────────────────────────────────
    # "ticket" and "case" are the same object — always display as "support ticket".
    if _TICKET_LOOKUP_PAT.search(ul):
        store = get_store()
        tickets = store.recent_tickets(sid, limit=5)
        # If no tickets in this session, try email-based cross-session lookup
        if not tickets:
            email_hit = _scan_history_for_slot(msgs, _EMAIL_PATTERN)
            if not email_hit:
                em = _EMAIL_PATTERN.search(ul)
                email_hit = em.group(0) if em else ""
            if email_hit:
                tickets = store.search_tickets_by_email(email_hit, limit=5)
        if tickets:
            from zoneinfo import ZoneInfo as _ZI
            from datetime import datetime as _dt, timezone as _tz
            _TOR = _ZI("America/Toronto")

            def _fmt_ticket_ts(iso: str) -> str:
                try:
                    d = _dt.fromisoformat(iso.replace("Z", "+00:00")).astimezone(_TOR)
                    return d.strftime("%b %d %Y, %I:%M %p %Z")
                except Exception:
                    return iso[:10]

            _lines = []
            for t in tickets:
                it_disp = t.get("issue_type", "").replace("_", " ").title()
                date    = _fmt_ticket_ts(t.get("created_at") or "")
                email   = t.get("email", "")
                _lines.append(
                    f"- **`{t['ticket_id']}`** · {it_disp}"
                    + (f" · {email}" if email else "")
                    + (f" · submitted {date}" if date else "")
                )
            msg = (
                "Here are your support tickets:\n\n"
                + "\n".join(_lines)
                + "\n\nLet me know if you need anything else!"
            )
            resp = AgentResponse(
                session_id=sid,
                message=msg,
                intent=Intent.ACTION_CREATE_TICKET.value,
                status="ok",
                action={"name": "ticket_recall", "payload": tickets},
            )
            return {"response": resp.model_dump()}
        resp = AgentResponse(
            session_id=sid,
            message=(
                "I don't see any support tickets linked to this session. "
                "If you'd like to open one, just share your issue type, email address, "
                "and a brief description — I'll take care of the rest!"
            ),
            intent=Intent.ACTION_CREATE_TICKET.value,
            status="clarify",
        )
        return {"response": resp.model_dump()}

    # ── Continuation: detect if last turn was mid-action and router lost track ──
    pending_act, _pending_missing = _pending_action(msgs)

    route_data = state.get("_route") or {}
    intent_val = route_data.get("intent")
    if isinstance(intent_val, Intent):
        intent = intent_val
    else:
        # Normalise "Intent.KB_QA" (Python-3.12 str repr) → "KB_QA"
        raw_s = str(intent_val or "").strip()
        if "." in raw_s:
            raw_s = raw_s.split(".")[-1]
        try:
            intent = Intent(raw_s)
        except ValueError:
            intent = Intent.AMBIGUOUS
    slots = dict(route_data.get("slots") or {})

    # Override router if it mis-classified a follow-up to an in-progress action.
    # CRITICAL: only override when there are actual MISSING fields (_pending_missing is non-empty).
    # If _pending_missing is empty the previous turn was a terminal result (not_found, error, ok)
    # and the user's next message should be routed freely by the LLM router.
    _safe_intents = {Intent.UNSAFE, Intent.SECURITY_SENSITIVE, Intent.OUT_OF_SCOPE}
    if pending_act == "check_transaction" and _pending_missing and intent not in _safe_intents:
        tx_try, asset_try = _extract_tx_slots(user, slots, msgs)
        if tx_try or asset_try:
            intent = Intent.ACTION_CHECK_TRANSACTION
    elif pending_act == "create_ticket" and _pending_missing and intent not in _safe_intents:
        _, email_try, _ = _extract_ticket_slots(user, slots, msgs)
        if email_try or (intent not in (Intent.KB_QA,) and len(user.strip()) > 3):
            intent = Intent.ACTION_CREATE_TICKET
    elif pending_act == "account_recovery" and _pending_missing and intent not in _safe_intents:
        if intent != Intent.KB_QA:
            intent = Intent.ACTION_ACCOUNT_RECOVERY

    if intent == Intent.SECURITY_SENSITIVE:
        msg = (
            "That's a sensitive area — for everyone's protection I'm not able to help with requests "
            "that involve bypassing security steps, identity verification, or fraud controls. "
            "These safeguards exist to keep your account safe.\n\n"
            "If you're locked out or having trouble with 2FA, I'm happy to walk you through "
            "Coinbase's official account recovery process. Just say **\"I need account recovery help\"**."
        )
        resp = AgentResponse(session_id=sid, message=msg, intent=intent.value, status="refusal")
        return {"response": resp.model_dump()}

    if intent == Intent.UNSAFE:
        msg = (
            "I'm not able to help with that one — it falls outside what I'm here for as a "
            "Coinbase support assistant. If there's something about your Coinbase account, "
            "transactions, or security I can assist with, I'm all ears!"
        )
        resp = AgentResponse(session_id=sid, message=msg, intent=intent.value, status="refusal")
        return {"response": resp.model_dump()}

    if intent == Intent.OUT_OF_SCOPE:
        # Rotate across a few warm, natural-sounding redirects based on session turn count
        _oos_msgs = [
            (
                "Hmm, that one's a bit outside my lane! I'm set up specifically to help with "
                "Coinbase support — things like account access, transactions, fees, security, "
                "and getting started with crypto. Can I help you with any of those?"
            ),
            (
                "Good question, but not one I can answer well from here — my expertise is "
                "Coinbase customer support. I can help with things like checking a transaction, "
                "recovering account access, or explaining Coinbase fees. What would you like to know?"
            ),
            (
                "I wish I could help, but that's a bit beyond my scope as a Coinbase support "
                "assistant. I'm best at questions about your account, security settings, "
                "crypto transactions, and Coinbase products. Happy to dig into any of those!"
            ),
        ]
        turn = len(msgs) // 2  # rough turn count
        msg = _oos_msgs[turn % len(_oos_msgs)]
        resp = AgentResponse(session_id=sid, message=msg, intent=intent.value, status="refusal")
        return {"response": resp.model_dump()}

    # Only ask for clarification when intent IS ambiguous — never gate valid intents on this
    if intent == Intent.AMBIGUOUS:
        q = route_data.get("clarifying_question") or "Could you clarify what you need help with on Coinbase?"
        resp = AgentResponse(session_id=sid, message=q, intent=intent.value, status="clarify")
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_CHECK_TRANSACTION:
        # Load partial state from previous clarify (survives multiple rounds)
        partial_tx = _load_partial_state(msgs, "check_transaction")
        merged_slots = {**partial_tx, **slots}
        tx, asset = _extract_tx_slots(user, merged_slots, msgs)

        if not tx or not asset:
            missing = [k for k, v in {"transaction_id": tx, "asset_type": asset}.items() if not v]
            hints = []
            if "transaction_id" in missing:
                hints.append("your **transaction ID** (e.g. CB-TX-PENDING01)")
            if "asset_type" in missing:
                hints.append("the **asset type** (e.g. BTC, ETH, USDC)")
            resp = AgentResponse(
                session_id=sid,
                message="To look up the transaction, please provide " + " and ".join(hints) + ".",
                intent=intent.value,
                status="clarify",
                action={
                    "name": "check_transaction",
                    "missing": missing,
                    "state": {"transaction_id": tx, "asset_type": asset},
                },
            )
            return {"response": resp.model_dump()}

        result = check_transaction(tx, asset)
        if not result.get("ok"):
            err = result.get("error", "")
            if err == "invalid_transaction_id":
                msg = (
                    f"\u26a0\ufe0f `{tx}` doesn't look like a valid transaction ID "
                    "(expected format: CB-TX-XXXXXXXX). Please double-check and try again."
                )
                bad_tx, bad_asset = True, False
            elif err == "invalid_asset":
                msg = (
                    f"\u26a0\ufe0f `{asset}` isn't a recognised asset symbol. "
                    "Please use a standard ticker like BTC, ETH, USDC, or SOL."
                )
                bad_tx, bad_asset = False, True
            else:
                msg = result.get("message", "Invalid transaction details provided.")
                bad_tx, bad_asset = True, True
            resp = AgentResponse(
                session_id=sid,
                message=msg,
                intent=intent.value,
                status="clarify",
                action={
                    "name": "check_transaction",
                    "missing": (["transaction_id"] if bad_tx else []) + (["asset_type"] if bad_asset else []),
                    "state": {
                        "transaction_id": "" if bad_tx else tx,
                        "asset_type": "" if bad_asset else asset,
                    },
                },
            )
            return {"response": resp.model_dump()}

        if not result.get("found"):
            resp = AgentResponse(
                session_id=sid,
                message=(
                    f"Sorry, I couldn't find a transaction matching **{tx}** ({asset}) in our records. "
                    "Please double-check the ID and asset type.\n\n"
                    "If you think this is an error, I can open a support ticket for you — "
                    "just say **\"create a ticket\"** and I'll get that started."
                ),
                intent=intent.value,
                status="ok",   # lookup completed (no match) — NOT "clarify", prevents continuation loop
                action={"name": "check_transaction", "payload": {"found": False, "transaction_id": tx, "asset_type": asset}},
            )
            return {"response": resp.model_dump()}

        tx_id_disp  = result["transaction_id"]
        asset_disp  = result["asset_type"]
        status_disp = result["status"].title()
        detail_disp = result["detail"]
        steps = "\n".join(f"- {s}" for s in result.get("next_steps", []))
        msg = (
            f"**Transaction:** `{tx_id_disp}` ({asset_disp})\n\n"
            f"**Status:** {status_disp}\n\n"
            f"{detail_disp}\n\n{steps}"
        )
        resp = AgentResponse(
            session_id=sid,
            message=msg,
            intent=intent.value,
            status="ok",
            action={"name": "check_transaction", "payload": result},
        )
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_CREATE_TICKET:
        from app.actions.ticket import validate_email as _ve, normalize_issue_type as _ni

        # Load partial state so slot values survive across multiple clarify rounds
        partial_tk = _load_partial_state(msgs, "create_ticket")
        it, email, desc = _extract_ticket_slots(user, slots, msgs)

        # Merge new values over saved partial state
        it    = it    or partial_tk.get("issue_type", "")
        email = email or partial_tk.get("email", "")
        desc  = desc  or partial_tk.get("problem_description", "")
        if not desc or len(desc.strip()) < 12:
            desc = user  # fall back to full user message as description

        validation_errors: list[str] = []
        missing_fields: list[str] = []

        # --- Email validation ---
        if email:
            if not _ve(email):
                validation_errors.append(
                    f"The email **{email}** doesn't look valid "
                    "(it needs an @ and a domain, e.g. john@example.com). Please correct it."
                )
                email = ""  # force re-entry
        if not email:
            missing_fields.append("email address")

        # --- Issue type validation ---
        if it:
            norm = _ni(it)
            if norm:
                it = norm
            else:
                validation_errors.append(
                    f"**{it}** isn't a recognised issue type. "
                    "Please choose one of: Account Access, Verification, Transactions, Security, Fees, or Other."
                )
                it = ""  # force re-entry
        if not it:
            missing_fields.append("issue type")

        if len(desc.strip()) < 12:
            missing_fields.append("problem description (at least one sentence)")

        if validation_errors or missing_fields:
            parts: list[str] = list(validation_errors)
            if missing_fields:
                missing_str = ", ".join(f"**{f}**" for f in missing_fields)
                parts.append(f"Still needed: {missing_str}.")
                if "issue type" in missing_fields:
                    parts.append(
                        "Issue types: **Account Access**, **Verification**, **Transactions**, "
                        "**Security**, **Fees**, or **Other**."
                    )
            resp = AgentResponse(
                session_id=sid,
                message="\n\n".join(parts),
                intent=intent.value,
                status="clarify",
                action={
                    "name": "create_ticket",
                    "missing": missing_fields,
                    "state": {
                        "issue_type": it,
                        "email": email,
                        "problem_description": desc if len(desc.strip()) >= 12 else "",
                    },
                },
            )
            return {"response": resp.model_dump()}

        result = create_ticket(sid, it, email, desc)
        if not result.get("ok"):
            # Unexpected fallback errors
            err = result.get("error", "")
            if err == "invalid_email":
                msg = (
                    f"The email **{email}** still doesn't pass validation. "
                    "Please provide a full email address (e.g. john@example.com)."
                )
                email = ""
            elif err == "invalid_issue_type":
                msg = (
                    f"**{it}** isn't a valid issue type. "
                    "Choose from: Account Access, Verification, Transactions, Security, Fees, or Other."
                )
                it = ""
            else:
                msg = result.get("message", "Unable to create ticket — please try again.")
            resp = AgentResponse(
                session_id=sid,
                message=msg,
                intent=intent.value,
                status="clarify",
                action={
                    "name": "create_ticket",
                    "missing": ["email address" if not email else "issue type" if not it else "details"],
                    "state": {"issue_type": it, "email": email, "problem_description": desc},
                },
            )
            return {"response": resp.model_dump()}

        ticket_id  = result["ticket_id"]
        issue_disp = result.get("issue_type", it).replace("_", " ").title()
        email_disp = result.get("email", email)
        resp = AgentResponse(
            session_id=sid,
            message=(
                f"Your support ticket **`{ticket_id}`** has been created. \u2713\n\n"
                f"**Issue:** {issue_disp}\n"
                f"**Email:** {email_disp}\n\n"
                "We'll follow up at the email you provided. "
                "You can ask me anytime: \"What ticket did I create?\""
            ),
            intent=intent.value,
            status="ok",
            action={"name": "create_ticket", "payload": result},
        )
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_ONBOARDING_SUPPORT:
        # Validate + normalise the goal slot
        _VALID_GOALS = {"buy_crypto", "security", "verification", "deposit", "sell_crypto", "general"}
        raw_goal = str(slots.get("goal") or "").strip().lower().replace(" ", "_")
        goal_val = raw_goal if raw_goal in _VALID_GOALS else (slots.get("goal") or None)

        new_to_crypto = str(slots.get("new_to_crypto") or "unknown").strip().lower()
        if new_to_crypto not in ("yes", "no", "unknown"):
            new_to_crypto = "unknown"

        try:
            plan = onboarding_plan(new_to_crypto, goal_val, slots.get("region"), user)
        except Exception:
            log.exception("onboarding_plan failed")
            plan = {
                "ok": True,
                "plan": {
                    "summary": (
                        "Happy to help you get started! Here's a quick checklist to kick things off safely."
                    ),
                    "first_steps": [
                        "Create your Coinbase account with a strong, unique password.",
                        "Enable two-factor authentication (2FA) right away.",
                        "Complete identity verification when prompted.",
                    ],
                    "security_tips": [
                        "Never share your seed phrase or one-time SMS codes with anyone.",
                        "Watch for phishing sites that look like Coinbase — always check the URL.",
                    ],
                    "verification_guidance": [
                        "Use a well-lit photo of a government-issued ID and make sure your legal name matches.",
                    ],
                    "next_actions": [
                        "Add a payment method in the Payments settings.",
                        "Make a small test purchase to get comfortable with the flow.",
                    ],
                    "suggested_kb_questions": [
                        "How do I enable 2FA on Coinbase?",
                        "What documents are needed for ID verification?",
                        "How do Coinbase fees work?",
                    ],
                },
                "citations": [],
            }

        p = plan["plan"]
        summary = str(p.get("summary", "Here's your personalised onboarding plan!"))

        # Build details only from non-empty sections
        detail_parts: list[str] = []
        if p.get("first_steps"):
            detail_parts += ["**First steps**", *[f"- {x}" for x in p["first_steps"]], ""]
        if p.get("security_tips"):
            detail_parts += ["**Security tips**", *[f"- {x}" for x in p["security_tips"]], ""]
        if p.get("verification_guidance"):
            detail_parts += ["**Verification**", *[f"- {x}" for x in p["verification_guidance"]], ""]
        if p.get("next_actions"):
            detail_parts += ["**Next actions**", *[f"- {x}" for x in p["next_actions"]], ""]
        if p.get("suggested_kb_questions"):
            detail_parts += [
                "**You might also want to know:**",
                *[f"- {q}" for q in p["suggested_kb_questions"]],
            ]

        resp = AgentResponse(
            session_id=sid,
            message=summary,
            details="\n".join(detail_parts).strip() or None,
            intent=intent.value,
            status="ok",
            citations=plan.get("citations", []),
            action={"name": "onboarding", "payload": p},
        )
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_ACCOUNT_RECOVERY:
        store = get_store()
        existing = store.load_recovery_for_session(sid) or {}
        st = {**existing, "session_id": sid}
        out = recovery_step(sid, st, user, slots)
        if not out["complete"]:
            resp = AgentResponse(
                session_id=sid,
                message=out["assistant"],
                intent=intent.value,
                status="clarify",
                action={"name": "account_recovery", "state": out["state"], "missing": out["missing"]},
            )
            return {"response": resp.model_dump()}
        resp = AgentResponse(
            session_id=sid,
            message=out["assistant"],
            intent=intent.value,
            status="ok",
            action={"name": "account_recovery", "payload": out["state"]},
        )
        return {"response": resp.model_dump()}

    # KB_QA default
    hits = get_retriever().retrieve(user)
    if not evidence_sufficient(hits):
        resp = AgentResponse(
            session_id=sid,
            message=(
                "I couldn’t find strong matching Help Center guidance for that question. "
                "Try rephrasing with a Coinbase product keyword (2FA, verification, fees, withdraw), "
                "or tell me if you want to open a support ticket."
            ),
            intent=Intent.KB_QA.value,
            status="clarify",
            citations=[],
        )
        return {"response": resp.model_dump()}

    conv_tail = _conversation_tail_for_qa(msgs)
    concise, details, citations, conf = answer_kb(user, conversation_tail=conv_tail)
    if conf < 0.25:
        concise = concise + "\n\n_(Low confidence — please verify in the linked articles.)_"
    resp = AgentResponse(
        session_id=sid,
        message=concise,
        details=details,
        intent=Intent.KB_QA.value,
        status="ok",
        citations=citations,
        trace={"confidence": conf},
    )
    return {"response": resp.model_dump()}


def node_persist(state: AgentState) -> AgentState:
    from datetime import datetime, timezone as _tz  # local import avoids circular risk
    _now_utc = datetime.now(_tz.utc).isoformat()

    store = get_store()
    sid = state["session_id"]
    resp = state.get("response") or {}
    user_msg = {"role": "user", "content": state["user_input"], "meta": {}, "ts": _now_utc}
    asst_msg = {
        "role": "assistant",
        "content": resp.get("message", ""),
        "ts": _now_utc,
        "meta": {
            "intent":    resp.get("intent"),
            "status":    resp.get("status"),   # ← needed for continuation detection
            "citations": resp.get("citations"),
            "action":    resp.get("action"),
        },
    }
    messages = list(state.get("messages", []))
    messages.extend([user_msg, asst_msg])
    rec = store.load_session(sid)
    new_title = None
    u0 = state["user_input"].strip()
    if rec.title is None and u0:
        new_title = u0[:72] + ("…" if len(u0) > 72 else "")
    store.save_session(sid, messages, state.get("router_trace", []), title=new_title)
    return {"messages": messages}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load", node_load)
    g.add_node("guard", node_guard)
    g.add_node("intent", node_intent)
    g.add_node("dispatch", node_dispatch)
    g.add_node("persist", node_persist)

    g.set_entry_point("load")
    g.add_edge("load", "guard")
    g.add_conditional_edges("guard", node_route_after_guard, {"intent": "intent", "end": "persist"})
    g.add_edge("intent", "dispatch")
    g.add_edge("dispatch", "persist")
    g.add_edge("persist", END)
    return g.compile()


GRAPH = build_graph()


def run_agent_turn(session_id: str | None, user_input: str) -> AgentResponse:
    try:
        out = GRAPH.invoke({"session_id": session_id or "", "user_input": user_input})
        return AgentResponse.model_validate(out["response"])
    except Exception as e:
        log.exception("graph invoke failed")
        sid = session_id or ""
        try:
            store = get_store()
            sid = store.ensure_session(sid or None)
        except Exception:
            pass
        return AgentResponse(
            session_id=sid,
            message=(
                "Sorry, something went wrong on my end — please try again in a moment. "
                "If the issue keeps happening, feel free to open a support ticket and our team will help."
            ),
            intent=Intent.AMBIGUOUS.value,
            status="error",
            trace={"error": str(e)},
        )
