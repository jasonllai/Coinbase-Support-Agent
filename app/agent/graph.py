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

# ── Regex-based slot extraction fallbacks ─────────────────────────────────
_TX_PATTERN    = re.compile(r"\b(CB-[A-Z0-9][\w\-]{2,40})\b", re.I)
_ASSET_PATTERN = re.compile(
    r"\b(BTC|ETH|USDC|SOL|LTC|DOGE|XRP|MATIC|ADA|DOT|AVAX|"
    r"bitcoin|ethereum|solana|litecoin|dogecoin|ripple|cardano|polkadot|avalanche)\b",
    re.I,
)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _extract_tx_slots(text: str, slots: dict[str, Any]) -> tuple[str, str]:
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
    return tx, asset


def _extract_ticket_slots(text: str, slots: dict[str, Any]) -> tuple[str, str, str]:
    it    = str(slots.get("issue_type")           or "").strip()
    email = str(slots.get("email")                or "").strip()
    desc  = str(slots.get("problem_description")  or "").strip()
    if not email:
        m = _EMAIL_PATTERN.search(text)
        if m:
            email = m.group(0)
    return it, email, desc


def _build_router_context(messages: list[dict[str, Any]], max_turns: int = 6) -> str:
    """Build structured context for the intent router.

    Includes the last N turns with role labels, plus a structured summary of
    the most recent assistant message (intent / status / action) so the router
    can handle continuations correctly.
    """
    tail = messages[-max_turns:]
    lines: list[str] = []

    for m in tail:
        role = m.get("role", "user")
        content = (m.get("content") or "")[:500]
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


def _conversation_tail_for_qa(messages: list[dict[str, Any]], max_turns: int = 4) -> str:
    """Plain-text conversation tail for KB QA follow-up context."""
    tail = messages[-max_turns * 2:]
    lines = []
    for m in tail:
        role    = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "")[:600]
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
    g = run_guardrails(state["user_input"])
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
    """Return (action_name, missing_fields) if the last assistant turn was a clarify for an action."""
    meta = _last_assistant_meta(messages)
    if meta.get("status") != "clarify":
        return "", []
    action = meta.get("action") or {}
    name = action.get("name", "")
    missing = list(action.get("missing") or [])
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

    if any(p in ul for p in ("what issue type", "which issue type", "issue type did i")):
        tickets = get_store().recent_tickets(sid, limit=3)
        if tickets:
            lines = [f"- `{t['ticket_id']}` → **{t['issue_type']}**" for t in tickets]
            resp = AgentResponse(
                session_id=sid,
                message="Here’s what we have on file for tickets in this session:\n\n" + "\n".join(lines),
                intent=Intent.ACTION_CREATE_TICKET.value,
                status="ok",
                action={"name": "ticket_issue_recall", "payload": tickets},
            )
            return {"response": resp.model_dump()}

    if any(p in ul for p in ("ticket did i", "ticket i created", "my support ticket", "last ticket")):
        tickets = get_store().recent_tickets(sid, limit=5)
        if tickets:
            lines = [f"- `{t['ticket_id']}` — {t['issue_type']} — {t['created_at']}" for t in tickets]
            resp = AgentResponse(
                session_id=sid,
                message="Here are your recent tickets in this session:\n\n" + "\n".join(lines),
                intent=Intent.ACTION_CREATE_TICKET.value,
                status="ok",
                action={"name": "ticket_recall", "payload": tickets},
            )
            return {"response": resp.model_dump()}
        resp = AgentResponse(
            session_id=sid,
            message="I don’t see a support ticket created in this session yet. Tell me your issue and email if you’d like to open one.",
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

    # Override router if it mis-classified a follow-up to an in-progress action
    _safe_intents = {Intent.UNSAFE, Intent.SECURITY_SENSITIVE, Intent.OUT_OF_SCOPE}
    if pending_act == "check_transaction" and intent not in _safe_intents:
        tx_try, asset_try = _extract_tx_slots(user, slots)
        if tx_try or asset_try:
            intent = Intent.ACTION_CHECK_TRANSACTION
    elif pending_act == "create_ticket" and intent not in _safe_intents:
        _, email_try, _ = _extract_ticket_slots(user, slots)
        if email_try or (intent not in (Intent.KB_QA,) and len(user.strip()) > 3):
            intent = Intent.ACTION_CREATE_TICKET
    elif pending_act == "account_recovery" and intent not in _safe_intents:
        if intent != Intent.KB_QA:
            intent = Intent.ACTION_ACCOUNT_RECOVERY

    if intent in (Intent.UNSAFE, Intent.SECURITY_SENSITIVE):
        msg = (
            "I can't help bypass account security, KYC, or fraud protections."
            if intent == Intent.SECURITY_SENSITIVE
            else "I can't help with that request."
        )
        resp = AgentResponse(
            session_id=sid,
            message=msg,
            intent=intent.value,
            status="refusal",
        )
        return {"response": resp.model_dump()}

    if intent == Intent.OUT_OF_SCOPE:
        resp = AgentResponse(
            session_id=sid,
            message=(
                "I'm focused on **Coinbase customer support** topics (account access, security, "
                "transactions, fees, and getting started). Try asking about one of those areas."
            ),
            intent=intent.value,
            status="refusal",
        )
        return {"response": resp.model_dump()}

    # Only ask for clarification when intent IS ambiguous — never gate valid intents on this
    if intent == Intent.AMBIGUOUS:
        q = route_data.get("clarifying_question") or "Could you clarify what you need help with on Coinbase?"
        resp = AgentResponse(session_id=sid, message=q, intent=intent.value, status="clarify")
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_CHECK_TRANSACTION:
        tx, asset = _extract_tx_slots(user, slots)
        if not tx or not asset:
            missing = [k for k, v in {"transaction_id": tx, "asset_type": asset}.items() if not v]
            hints = []
            if "transaction_id" in missing:
                hints.append("your **transaction ID** (e.g. CB-TX-PENDING01)")
            if "asset_type" in missing:
                hints.append("the **asset** (e.g. BTC, ETH, USDC)")
            resp = AgentResponse(
                session_id=sid,
                message="To look up the transaction, please provide " + " and ".join(hints) + ".",
                intent=intent.value,
                status="clarify",
                action={"name": "check_transaction", "missing": missing},
            )
            return {"response": resp.model_dump()}
        result = check_transaction(tx, asset)
        if not result.get("ok"):
            resp = AgentResponse(
                session_id=sid,
                message=result.get("message", "Invalid transaction details provided."),
                intent=intent.value,
                status="error",
                action={"name": "check_transaction", "payload": result},
            )
            return {"response": resp.model_dump()}
        if not result.get("found"):
            resp = AgentResponse(
                session_id=sid,
                message=result.get("message", "Transaction not found."),
                intent=intent.value,
                status="clarify",
                action={"name": "check_transaction", "payload": result},
            )
            return {"response": resp.model_dump()}
        msg = (
            f"**Status:** {result['status'].title()}\n\n"
            f"{result['detail']}\n\n"
            + "\n".join(f"{s}" for s in result.get("next_steps", []))
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
        it, email, desc = _extract_ticket_slots(user, slots)
        if not desc:
            desc = user
        missing_fields = []
        if not it:
            missing_fields.append("issue type")
        if not email:
            missing_fields.append("email address")
        if len(desc) < 12:
            missing_fields.append("problem description")
        if missing_fields:
            missing_str = ", ".join(f"**{f}**" for f in missing_fields)
            resp = AgentResponse(
                session_id=sid,
                message=(
                    f"To open a support ticket, please provide: {missing_str}.\n\n"
                    "Issue types: Account Access, Verification, Transactions, Security, Fees, or Other."
                ),
                intent=intent.value,
                status="clarify",
                action={"name": "create_ticket", "missing": missing_fields},
            )
            return {"response": resp.model_dump()}
        result = create_ticket(sid, it, email, desc)
        if not result.get("ok"):
            resp = AgentResponse(
                session_id=sid,
                message=result.get("message", "Unable to create ticket."),
                intent=intent.value,
                status="error",
                action=result,
            )
            return {"response": resp.model_dump()}
        resp = AgentResponse(
            session_id=sid,
            message=(
                f"Your support ticket **`{result['ticket_id']}`** has been created.\n\n"
                f"Issue: {result.get('issue_type', it).replace('_', ' ').title()}\n"
                "We'll send updates to the email you provided."
            ),
            intent=intent.value,
            status="ok",
            action={"name": "create_ticket", "payload": result},
        )
        return {"response": resp.model_dump()}

    if intent == Intent.ACTION_ONBOARDING_SUPPORT:
        plan = onboarding_plan(
            slots.get("new_to_crypto"),
            slots.get("goal"),
            slots.get("region"),
            user,
        )
        p = plan["plan"]
        summary = str(p.get("summary", ""))
        details = "\n".join(
            [
                "**First steps**",
                *[f"- {x}" for x in p.get("first_steps", [])],
                "",
                "**Security**",
                *[f"- {x}" for x in p.get("security_tips", [])],
                "",
                "**Verification**",
                *[f"- {x}" for x in p.get("verification_guidance", [])],
            ]
        )
        resp = AgentResponse(
            session_id=sid,
            message=summary,
            details=details,
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
    store = get_store()
    sid = state["session_id"]
    resp = state.get("response") or {}
    user_msg = {"role": "user", "content": state["user_input"], "meta": {}}
    asst_msg = {
        "role": "assistant",
        "content": resp.get("message", ""),
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
                "Something went wrong while processing your request. Please try again in a moment. "
                "If the problem persists, check that the LLM endpoint is reachable and the KB index is built."
            ),
            intent=Intent.AMBIGUOUS.value,
            status="error",
            trace={"error": str(e)},
        )
