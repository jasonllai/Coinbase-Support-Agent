"""Coinbase Support Agent — production-quality client-facing Streamlit UI."""

from __future__ import annotations

import os

# Must be set before any FAISS / PyTorch import to prevent macOS libomp conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import html as _html_lib

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import run_agent_turn
from app.core.config import get_settings
from app.storage.sqlite_store import get_store

# ─── Brand icon ─────────────────────────────────────────────────────────────
_ICON_PATH = ROOT / "coinbase_icon.png"

def _icon_b64(size: int = 40) -> str:
    """Return an <img> tag embedding the Coinbase icon at `size` px."""
    try:
        img = Image.open(_ICON_PATH).convert("RGBA").resize((size, size), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return (
            f'<img src="data:image/png;base64,{data}" '
            f'width="{size}" height="{size}" style="display:block;" />'
        )
    except Exception:
        return "🔵"

def _icon_pil(size: int = 64) -> Image.Image | None:
    """Return a PIL Image for use in Streamlit widget avatar/page_icon params."""
    try:
        return Image.open(_ICON_PATH).convert("RGBA").resize((size, size), Image.LANCZOS)
    except Exception:
        return None

def _icon_b64_white(size: int = 34) -> str:
    """White silhouette of the Coinbase icon — for display on the blue header bar.
    Detects logo pixels (non-white) via grayscale threshold and recolours them white
    on a transparent background, so it sits cleanly on any coloured surface."""
    try:
        import io as _io
        img = Image.open(_ICON_PATH).convert("RGBA")
        gray = img.convert("L")
        # Logo pixels are the darker (non-white) ones; background is near-white (>220)
        logo_mask = gray.point(lambda v: 255 if v < 220 else 0, "L")
        white_layer = Image.new("RGBA", img.size, (255, 255, 255, 255))
        transparent  = Image.new("RGBA", img.size, (0, 0, 0, 0))
        result = Image.composite(white_layer, transparent, logo_mask)
        result = result.resize((size, size), Image.LANCZOS)
        buf = _io.BytesIO()
        result.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return (
            f'<img src="data:image/png;base64,{data}" '
            f'width="{size}" height="{size}" style="display:block;" />'
        )
    except Exception:
        return "🔵"


# Pre-load a 48 px avatar used for every assistant chat bubble
_ASST_AVATAR: Image.Image | str = _icon_pil(48) or "🔵"

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coinbase Support",
    page_icon=_icon_pil(64) or "🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Brand colours ─────────────────────────────────────────────────────────
_BLUE       = "#0052FF"
_BLUE_DARK  = "#003EBF"
_BLUE_LIGHT = "#EBF0FF"

# ─── CSS (inlined, no external font CDN that can fail) ─────────────────────
CSS = f"""
<style>
/* hide Streamlit chrome */
#MainMenu, footer {{ visibility:hidden; }}
.stDeployButton {{ display:none; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}

/* tighten main container */
.block-container {{ padding-top:0 !important; max-width:980px; }}

/* sidebar */
[data-testid="stSidebar"] {{ border-right:1px solid #E2E8F0; }}

/* ── Chips ── explicit colours, work in both themes ── */
.cb-chip {{
    display:inline-flex; align-items:center; gap:5px;
    padding:3px 11px; border-radius:20px;
    font-size:11.5px; font-weight:600; letter-spacing:.02em;
    margin-bottom:6px;
}}
.cb-chip-knowledge {{ background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }}
.cb-chip-action    {{ background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }}
.cb-chip-clarify   {{ background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; }}
.cb-chip-refusal   {{ background:#FFF1F2; color:#BE123C; border:1px solid #FECDD3; }}
.cb-chip-error     {{ background:#FFF7ED; color:#C2410C; border:1px solid #FED7AA; }}

/* ── Source cards ── */
.cb-src {{
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:10px; padding:12px 14px; margin:6px 0;
}}
.cb-src:hover {{ border-color:{_BLUE}; background:{_BLUE_LIGHT}; }}
.cb-src-title   {{ font-size:13px; font-weight:600; color:#1E293B; margin:0 0 2px 0; }}
.cb-src-section {{ font-size:12px; color:#64748B; margin:0 0 5px 0; }}
.cb-src-excerpt {{ font-size:12px; color:#475569; line-height:1.5; margin:0 0 6px 0; }}
.cb-src-link    {{ font-size:12px; color:{_BLUE}; font-weight:500; text-decoration:none; }}
.cb-src-link:hover {{ text-decoration:underline; }}

/* ── Action cards ── */
.cb-act        {{ background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:12px 14px; margin:8px 0; }}
.cb-act.err    {{ background:#FFF1F2; border-color:#FECDD3; }}
.cb-act.clarify {{ background:#FFFBEB; border-color:#FDE68A; }}
.cb-act-label  {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#15803D; margin-bottom:6px; }}
.cb-act.err    .cb-act-label {{ color:#BE123C; }}
.cb-act.clarify .cb-act-label {{ color:#B45309; }}
.cb-act-row    {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:4px 0; }}
.cb-act-key    {{ font-size:12px; color:#64748B; font-weight:500; min-width:110px; }}
.cb-act-val    {{ font-size:13px; color:#1E293B; font-weight:600; }}
.cb-tx-status  {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }}
.cb-tx-completed {{ background:#DCFCE7; color:#15803D; }}
.cb-tx-pending   {{ background:#FEF9C3; color:#854D0E; }}
.cb-tx-failed    {{ background:#FFE4E6; color:#9F1239; }}
.cb-tx-review    {{ background:#FEF3C7; color:#92400E; }}

/* ── Header ── */
.cb-hdr      {{ background:{_BLUE}; padding:14px 24px; display:flex; align-items:center;
               gap:12px; margin:0 -1rem 20px -1rem; border-radius:0 0 12px 12px; }}
.cb-hdr-t    {{ color:white; font-size:18px; font-weight:700; margin:0; }}
.cb-hdr-s    {{ color:rgba(255,255,255,.8); font-size:13px; margin:0; }}

/* ── Welcome ── */
.cb-welcome      {{ background:white; border:1px solid #E2E8F0; border-radius:16px;
                    padding:36px 32px; text-align:center; max-width:540px; margin:40px auto; }}
.cb-welcome h2   {{ color:#1E293B; font-size:22px; font-weight:700; margin-bottom:10px; }}
.cb-welcome p    {{ color:#64748B; font-size:14px; line-height:1.65; margin-bottom:16px; }}
.cb-pill         {{ display:inline-block; background:{_BLUE_LIGHT}; color:{_BLUE};
                    border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; margin:3px; }}

/* ── Sidebar helpers ── */
.cb-sec  {{ font-size:11px; font-weight:700; text-transform:uppercase;
            letter-spacing:.08em; color:#94A3B8; margin:16px 0 8px 0; }}
.cb-hr   {{ border:none; border-top:1px solid #E2E8F0; margin:10px 0; }}
.cb-badge {{ display:inline-block; background:#EFF6FF; color:#1D4ED8;
             border-radius:6px; padding:2px 8px; font-size:11px; font-weight:600; font-family:monospace; }}

/* ── Warning banner ── */
.cb-warn {{ background:#FEF3C7; border:1px solid #FDE68A; border-radius:8px;
            padding:10px 14px; font-size:13px; color:#92400E; margin:0 0 12px 0; }}

/* ── Conversation layout: left = agent, right = user ── */

/* User bubble — rendered as custom HTML, right-aligned */
.cb-user-bubble-wrap {{
    display: flex;
    justify-content: flex-end;
    margin: 10px 0 2px 0;
}}
.cb-user-bubble {{
    background: {_BLUE};
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px 10px 16px;
    max-width: 68%;
    word-break: break-word;
    box-shadow: 0 2px 8px rgba(0,82,255,0.15);
    font-size: 14.5px;
    line-height: 1.55;
}}
.cb-user-bubble p {{ margin: 0; }}
.cb-user-ts {{
    font-size: 10.5px;
    color: rgba(255,255,255,0.62);
    margin-top: 5px;
    text-align: right;
}}

/* Assistant messages — keep st.chat_message on left, constrain width */
[data-testid="stChatMessage"] {{
    max-width: 82%;
    margin-right: auto;
    padding: 4px 0;
}}
</style>
"""


# ─── Helpers ───────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _llm_configured() -> bool:
    try:
        key = get_settings().llm_api_key or os.getenv("LLM_API_KEY", "")
        return bool(key and key.strip() and key.lower().strip() not in ("dummy", ""))
    except Exception:
        return False


_TZ_TORONTO = ZoneInfo("America/Toronto")


def _fmt_time(iso: str, include_tz: bool = True) -> str:
    """Parse a UTC ISO timestamp and render it in Toronto local time."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        dt_local = dt.astimezone(_TZ_TORONTO)
        # %Z gives "EST" / "EDT" automatically
        fmt = "%b %d, %I:%M %p %Z" if include_tz else "%b %d, %I:%M %p"
        return dt_local.strftime(fmt)
    except Exception:
        return iso[:16]


def _status_chip(intent: str, status: str) -> str:
    if status in ("refusal",):
        return '<span class="cb-chip cb-chip-refusal">🚫 Not supported</span>'
    if status == "error":
        return '<span class="cb-chip cb-chip-error">⚠️ Error</span>'
    if status == "clarify":
        return '<span class="cb-chip cb-chip-clarify">💬 More info needed</span>'
    if intent == "KB_QA":
        return '<span class="cb-chip cb-chip-knowledge">📚 Help Center</span>'
    if intent and intent.startswith("ACTION_"):
        labels = {
            "ACTION_CHECK_TRANSACTION":  "Transaction Status",
            "ACTION_CREATE_TICKET":      "Support Ticket",
            "ACTION_ONBOARDING_SUPPORT": "Getting Started",
            "ACTION_ACCOUNT_RECOVERY":   "Account Recovery",
        }
        return f'<span class="cb-chip cb-chip-action">⚡ {labels.get(intent, "Action")}</span>'
    return '<span class="cb-chip cb-chip-clarify">💬 Reply</span>'


# ─── Auth gate ─────────────────────────────────────────────────────────────

def require_auth() -> None:
    if st.session_state.get("_auth_ok"):
        return
    inject_css()

    # Build the icon img tag once (56 px for login card)
    icon_html = _icon_b64(56)

    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:72vh;text-align:center;">
          <div style="background:white;border:1px solid #E2E8F0;border-radius:20px;
                      padding:48px 40px;max-width:400px;width:100%;
                      box-shadow:0 4px 24px rgba(0,0,0,0.07);">
            <div style="margin:0 auto 18px;width:56px;height:56px;">
              {icon_html}
            </div>
            <h2 style="color:#1E293B;font-size:22px;font-weight:700;margin:0 0 8px;">
              Coinbase Support</h2>
            <p style="color:#64748B;font-size:14px;margin:0 0 24px;line-height:1.6;">
              AI-powered support assistant.<br>Sign in to continue.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c, _ = st.columns([1, 2, 1])
    with c:
        username = st.text_input(
            "Username", label_visibility="visible",
            placeholder="Enter your username…"
        )
        pw = st.text_input(
            "Password", type="password", label_visibility="visible",
            placeholder="Enter your password…"
        )
        if st.button("Sign in →", use_container_width=True, type="primary"):
            cfg = get_settings()
            expected_user = os.getenv("DEMO_USERNAME", cfg.demo_username)
            expected_pw   = os.getenv("DEMO_PASSWORD", cfg.demo_password)
            if not username.strip():
                st.error("Please enter your username.")
            elif username.strip() != expected_user:
                st.error("Username not recognised. Please try again.")
            elif pw != expected_pw:
                st.error("Incorrect password. Please try again.")
            else:
                st.session_state["_auth_ok"]   = True
                st.session_state["_auth_user"]  = username.strip()
                st.rerun()
    st.stop()


# ─── Source cards ──────────────────────────────────────────────────────────

def render_source_cards(cits: list[dict[str, Any]]) -> None:
    if not cits:
        st.markdown(
            '<p style="font-size:13px;color:#94A3B8;font-style:italic;">'
            "Sources will appear here after a knowledge question."
            "</p>",
            unsafe_allow_html=True,
        )
        return
    # Deduplicate by URL — keep the highest-scoring entry per unique URL
    seen_urls: dict[str, dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    for cit in cits:
        url = cit.get("url") or cit.get("canonical_url") or ""
        if url and url in seen_urls:
            if (cit.get("score") or 0) > (seen_urls[url].get("score") or 0):
                seen_urls[url].update(cit)   # replace in-place to preserve order
        elif url:
            seen_urls[url] = cit
            deduped.append(cit)
        else:
            deduped.append(cit)   # no URL — always include
    for cit in deduped:
        title   = cit.get("article_title") or "Coinbase Help Article"
        section = cit.get("section_title") or ""
        raw     = cit.get("excerpt") or cit.get("text") or ""
        excerpt = raw[:220].replace("\n", " ") + ("…" if len(raw) > 220 else "")
        url     = cit.get("url") or cit.get("canonical_url") or ""
        link    = f'<a class="cb-src-link" href="{url}" target="_blank">Open in Help Center ↗</a>' if url else ""
        sec_html = f'<p class="cb-src-section">Section: {section}</p>' if section else ""
        exc_html = f'<p class="cb-src-excerpt">{excerpt}</p>' if excerpt else ""
        st.markdown(
            f'<div class="cb-src"><p class="cb-src-title">{title}</p>'
            f'{sec_html}{exc_html}{link}</div>',
            unsafe_allow_html=True,
        )


# ─── Action result card ────────────────────────────────────────────────────

def render_action_card(action: dict[str, Any], status: str) -> None:
    name    = action.get("name", "")
    payload = action.get("payload") or {}
    css     = "err" if status == "error" else ("clarify" if status == "clarify" else "")

    if name == "check_transaction" and payload.get("found"):
        tx_s = payload.get("status", "unknown").lower()
        badge = {"completed":"cb-tx-completed","pending":"cb-tx-pending",
                 "failed":"cb-tx-failed","delayed review":"cb-tx-review"}.get(tx_s, "")
        st.markdown(
            f'<div class="cb-act {css}"><div class="cb-act-label">Transaction Details</div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Transaction ID</span>'
            f'<span class="cb-act-val" style="font-family:monospace">{payload.get("transaction_id","—")}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Asset</span>'
            f'<span class="cb-act-val">{payload.get("asset_type","—")}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Status</span>'
            f'<span class="cb-tx-status {badge}">{tx_s.title()}</span></div></div>',
            unsafe_allow_html=True,
        )
    elif name == "create_ticket" and payload.get("ticket_id"):
        st.markdown(
            f'<div class="cb-act {css}"><div class="cb-act-label">Ticket Confirmed</div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Ticket ID</span>'
            f'<span class="cb-act-val" style="font-family:monospace">{payload.get("ticket_id","—")}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Issue Type</span>'
            f'<span class="cb-act-val">{(payload.get("issue_type") or "").replace("_"," ").title()}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Email</span>'
            f'<span class="cb-act-val">{payload.get("email","—")}</span></div></div>',
            unsafe_allow_html=True,
        )
    elif name == "account_recovery" and payload.get("case_id"):
        subtype = (payload.get("issue_subtype") or "").replace("_"," ").title()
        st.markdown(
            f'<div class="cb-act {css}"><div class="cb-act-label">Recovery Case Opened</div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Case ID</span>'
            f'<span class="cb-act-val" style="font-family:monospace">{payload.get("case_id","—")}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Issue</span>'
            f'<span class="cb-act-val">{subtype}</span></div>'
            f'<div class="cb-act-row"><span class="cb-act-key">Email</span>'
            f'<span class="cb-act-val">{payload.get("email","—")}</span></div></div>',
            unsafe_allow_html=True,
        )


# ─── Render a single message ───────────────────────────────────────────────

def render_user_msg(content: str, ts: str = "") -> None:
    """Render the user message as a right-aligned blue bubble (custom HTML)."""
    safe = _html_lib.escape(content).replace("\n", "<br>")
    ts_html = (
        f'<div class="cb-user-ts">{_fmt_time(ts)}</div>'
        if ts else ""
    )
    st.markdown(
        f'<div class="cb-user-bubble-wrap">'
        f'  <div class="cb-user-bubble">'
        f'    <p>{safe}</p>'
        f'    {ts_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_assistant_body(data: dict[str, Any], debug: bool = False, ts: str = "") -> None:
    """Render the interior of an assistant message bubble."""
    intent  = data.get("intent", "")
    status  = data.get("status", "ok")
    message = data.get("message", "")
    cits    = data.get("citations") or []
    action  = data.get("action") or {}
    details = data.get("details") or ""

    st.markdown(_status_chip(intent, status), unsafe_allow_html=True)
    st.markdown(message)

    if details:
        with st.expander("View full details"):
            st.markdown(details)

    if action and action.get("name"):
        render_action_card(action, status)

    if cits:
        with st.expander(f"📎 {len(cits)} Help Center source{'s' if len(cits)!=1 else ''}", expanded=False):
            render_source_cards(cits)

    if ts:
        st.caption(_fmt_time(ts))

    if debug:
        with st.expander("🛠 Debug trace", expanded=False):
            st.json({k: v for k, v in data.items() if k not in ("message", "details")})


def render_assistant_msg(data: dict[str, Any], debug: bool = False, _inside_bubble: bool = False, ts: str = "") -> None:
    """Render an assistant message, optionally wrapping in a chat_message bubble."""
    if _inside_bubble:
        _render_assistant_body(data, debug, ts=ts)
    else:
        with st.chat_message("assistant", avatar=_ASST_AVATAR):
            _render_assistant_body(data, debug, ts=ts)


# ─── Welcome screen ────────────────────────────────────────────────────────

def render_welcome() -> None:
    st.markdown(
        """
        <div class="cb-welcome">
          <h2>How can we help you today?</h2>
          <p>
            I'm your Coinbase support assistant. Ask me anything about your account,
            transactions, security settings, or getting started with crypto.
          </p>
          <div>
            <span class="cb-pill">Account Access</span>
            <span class="cb-pill">2FA &amp; Security</span>
            <span class="cb-pill">Transactions</span>
            <span class="cb-pill">Fees</span>
            <span class="cb-pill">Identity Verification</span>
            <span class="cb-pill">Getting Started</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar(store) -> None:
    with st.sidebar:
        # Logo + title
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0 16px;">
              <div style="width:36px;height:36px;flex-shrink:0;">
                {_icon_b64(36)}
              </div>
              <div>
                <div style="font-size:15px;font-weight:700;color:#1E293B;">Coinbase Support</div>
                <div style="font-size:12px;color:#64748B;">AI-powered assistant</div>
              </div>
            </div>
            <hr class="cb-hr">
            """,
            unsafe_allow_html=True,
        )

        if st.button("✚  New conversation", use_container_width=True, type="primary"):
            st.session_state.update({
                "session_id": None, "messages": [],
                "_loaded_sid": None, "last_citations": [],
            })
            st.rerun()

        # What I can help with
        st.markdown('<p class="cb-sec">What I can help with</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:13px;color:#475569;line-height:2.0;">
            📋&nbsp;&nbsp;Coinbase Help Center questions<br>
            🔍&nbsp;&nbsp;Transaction status lookup<br>
            🎫&nbsp;&nbsp;Open a support ticket<br>
            🚀&nbsp;&nbsp;Getting-started guide<br>
            🔐&nbsp;&nbsp;Account recovery assistance
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Example prompts
        st.markdown('<hr class="cb-hr"><p class="cb-sec">Try asking</p>', unsafe_allow_html=True)
        examples = [
            "How do I set up two-factor authentication?",
            "Check transaction CB-TX-PENDING01 for ETH",
            "I'm new — how do I start buying Bitcoin safely?",
            "I lost my 2FA device, help me recover my account",
            "Why was my withdrawal delayed?",
            "What are Coinbase's fees for buying crypto?",
        ]
        for i, ex in enumerate(examples):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["_inject"] = ex
                st.rerun()

        # Sources panel
        cits = st.session_state.get("last_citations") or []
        if cits:
            st.markdown(
                '<hr class="cb-hr"><p class="cb-sec">Latest sources</p>',
                unsafe_allow_html=True,
            )
            render_source_cards(cits)

        # Session history
        sessions = store.list_sessions(20)
        if sessions:
            st.markdown(
                '<hr class="cb-hr"><p class="cb-sec">Recent conversations</p>',
                unsafe_allow_html=True,
            )
            for sess in sessions[:6]:
                sid    = sess["session_id"]
                label  = (sess.get("title") or "Untitled chat")[:32]
                ts     = _fmt_time(sess.get("updated_at", ""))
                active = st.session_state.get("session_id") == sid
                btn_lbl = f"{'▶ ' if active else ''}{label}"
                if st.button(btn_lbl, key=f"sess_{sid}", use_container_width=True):
                    if not active:
                        rec = store.load_session(sid)
                        st.session_state.update({
                            "session_id": sid, "messages": rec.messages,
                            "_loaded_sid": sid, "last_citations": [],
                        })
                        st.rerun()
                st.caption(ts)

        # Export + debug
        st.markdown('<hr class="cb-hr">', unsafe_allow_html=True)
        if st.session_state.get("messages"):
            export = json.dumps(
                [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.messages],
                indent=2, ensure_ascii=False,
            )
            st.download_button(
                "⬇ Export conversation", data=export,
                file_name="coinbase_chat.json", mime="application/json",
                use_container_width=True,
            )

        st.session_state["_debug"] = st.toggle("Debug mode", value=False)
        st.markdown(
            '<p style="font-size:11px;color:#94A3B8;margin-top:10px;">'
            "Educational demo · Not affiliated with Coinbase, Inc.</p>",
            unsafe_allow_html=True,
        )

        # Current session badge
        sid = st.session_state.get("session_id")
        if sid:
            st.markdown(
                f'<p style="font-size:11px;color:#94A3B8;margin:6px 0 2px;">Session</p>'
                f'<span class="cb-badge">{sid[:20]}…</span>',
                unsafe_allow_html=True,
            )


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    require_auth()
    inject_css()

    # ── Init state ──
    for k, v in [("messages",[]), ("session_id",None),
                 ("last_citations",[]), ("_debug",False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    store = get_store()
    render_sidebar(store)

    # ── API key warning ──
    if not _llm_configured():
        st.markdown(
            '<div class="cb-warn">⚠️ <strong>LLM not configured.</strong> '
            'Set <code>LLM_API_KEY=your_student_id</code> in <code>.env</code>. '
            "Answers use retrieval-only mode until then.</div>",
            unsafe_allow_html=True,
        )

    # ── Header ──
    st.markdown(
        f"""
        <div class="cb-hdr">
          <div style="width:34px;height:34px;flex-shrink:0;">
            {_icon_b64_white(34)}
          </div>
          <div>
            <p class="cb-hdr-t">Coinbase Support Agent</p>
            <p class="cb-hdr-s">Ask me about transactions, account access, security, fees &amp; more</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CHAT INPUT at page level → Streamlit pins it to viewport bottom ──
    prompt = st.chat_input("Ask about your Coinbase account…")

    # Inject example prompt (from sidebar buttons)
    if st.session_state.get("_inject"):
        prompt = st.session_state.pop("_inject")

    # ── 1. Render stored history first ──
    if not st.session_state.messages and not prompt:
        render_welcome()
    else:
        for m in st.session_state.messages:
            msg_ts = m.get("ts", "")
            if m["role"] == "user":
                render_user_msg(m["content"], ts=msg_ts)
            else:
                meta = m.get("meta") or {}
                render_assistant_msg(
                    {
                        "intent":    meta.get("intent", ""),
                        "status":    meta.get("status", "ok"),
                        "message":   m.get("content", ""),
                        "citations": meta.get("citations") or [],
                        "action":    meta.get("action"),
                        "details":   meta.get("details"),
                    },
                    debug=st.session_state.get("_debug", False),
                    ts=msg_ts,
                )

    # ── 2. Handle new prompt — show user bubble IMMEDIATELY, then process ──
    if prompt:
        # Show user message right away (before any processing)
        render_user_msg(prompt)

        # Process inside an assistant bubble; spinner shows while waiting
        with st.chat_message("assistant", avatar=_ASST_AVATAR):
            with st.spinner("Searching Help Center…"):
                try:
                    resp = run_agent_turn(st.session_state.session_id, prompt)
                    data = resp.model_dump()
                except Exception as exc:
                    data = {
                        "session_id": st.session_state.session_id or "",
                        "intent": "AMBIGUOUS", "status": "error",
                        "message": f"Something went wrong: {exc}",
                        "citations": [],
                    }

            st.session_state["session_id"] = data.get("session_id")
            cits = data.get("citations") or []
            if cits:
                st.session_state["last_citations"] = cits

            _render_assistant_body(data, debug=st.session_state.get("_debug", False))

        # Persist both messages to session state for next render
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("message", ""),
            "meta": {
                "intent":    data.get("intent"),
                "status":    data.get("status"),
                "citations": cits,
                "action":    data.get("action"),
                "details":   data.get("details"),
            },
        })


if __name__ == "__main__":
    main()
