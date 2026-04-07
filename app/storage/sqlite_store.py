"""SQLite persistence for sessions, tickets, recovery flows, and mock transactions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    title: str | None
    messages: list[dict[str, Any]]
    summary: str | None
    router_trace: list[dict[str, Any]]
    updated_at: str


class SqliteStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_settings().sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate_transactions_pk(self, c: sqlite3.Connection) -> None:
        """One-time migration: drop old single-column PK and recreate with composite PK."""
        try:
            info = c.execute("PRAGMA table_info(mock_transactions)").fetchall()
            if not info:
                return  # table doesn't exist yet
            # Check if current PK is composite (tx_id + asset_type)
            pk_cols = [r["name"] for r in info if r["pk"] > 0]
            if len(pk_cols) < 2:
                c.executescript(
                    """
                    ALTER TABLE mock_transactions RENAME TO _mock_transactions_old;
                    CREATE TABLE mock_transactions (
                        tx_id TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        detail TEXT NOT NULL,
                        PRIMARY KEY (tx_id, asset_type)
                    );
                    INSERT OR REPLACE INTO mock_transactions SELECT tx_id, asset_type, status, detail
                        FROM _mock_transactions_old;
                    DROP TABLE _mock_transactions_old;
                    """
                )
        except Exception:
            pass

    def _init_db(self) -> None:
        with self._conn() as c:
            self._migrate_transactions_pk(c)
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    messages_json TEXT NOT NULL,
                    summary TEXT,
                    router_trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    issue_type TEXT NOT NULL,
                    email TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recovery_cases (
                    case_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    email TEXT,
                    issue_subtype TEXT,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mock_transactions (
                    tx_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    PRIMARY KEY (tx_id, asset_type)
                );
                """,
            )

    def ensure_session(self, session_id: str | None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._conn() as c:
            row = c.execute("SELECT session_id FROM sessions WHERE session_id=?", (sid,)).fetchone()
            if row:
                return sid
            now = _utc_now()
            c.execute(
                "INSERT INTO sessions (session_id, title, messages_json, summary, router_trace_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (sid, None, "[]", None, "[]", now, now),
            )
        return sid

    def load_session(self, session_id: str) -> SessionRecord:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row:
                self.ensure_session(session_id)
                return self.load_session(session_id)
            return SessionRecord(
                session_id=row["session_id"],
                title=row["title"],
                messages=json.loads(row["messages_json"]),
                summary=row["summary"],
                router_trace=json.loads(row["router_trace_json"]),
                updated_at=row["updated_at"],
            )

    def save_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        router_trace: list[dict[str, Any]],
        summary: str | None = None,
        title: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._conn() as c:
            c.execute(
                """UPDATE sessions SET messages_json=?, router_trace_json=?, summary=COALESCE(?, summary),
                title=COALESCE(?, title), updated_at=? WHERE session_id=?""",
                (
                    json.dumps(messages),
                    json.dumps(router_trace),
                    summary,
                    title,
                    now,
                    session_id,
                ),
            )

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT session_id, title, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_ticket(
        self,
        session_id: str,
        issue_type: str,
        email: str,
        description: str,
    ) -> str:
        tid = f"TCK-{uuid.uuid4().hex[:10].upper()}"
        with self._conn() as c:
            c.execute(
                "INSERT INTO tickets (ticket_id, session_id, issue_type, email, description, created_at) VALUES (?,?,?,?,?,?)",
                (tid, session_id, issue_type, email, description, _utc_now()),
            )
        return tid

    def recent_tickets(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tickets WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_recovery(
        self,
        case_id: str | None,
        session_id: str,
        state: dict[str, Any],
    ) -> str:
        cid = case_id or f"REC-{uuid.uuid4().hex[:10].upper()}"
        now = _utc_now()
        with self._conn() as c:
            row = c.execute("SELECT case_id FROM recovery_cases WHERE case_id=?", (cid,)).fetchone()
            payload = json.dumps(state)
            if row:
                c.execute(
                    "UPDATE recovery_cases SET state_json=?, updated_at=?, session_id=? WHERE case_id=?",
                    (payload, now, session_id, cid),
                )
            else:
                c.execute(
                    """INSERT INTO recovery_cases (case_id, session_id, email, issue_subtype, state_json, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)""",
                    (
                        cid,
                        session_id,
                        state.get("email"),
                        state.get("issue_subtype"),
                        payload,
                        now,
                        now,
                    ),
                )
        return cid

    def load_recovery_for_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM recovery_cases WHERE session_id=? ORDER BY updated_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"case_id": row["case_id"], **json.loads(row["state_json"])}

    def seed_mock_transactions(self, path: Path | None = None) -> None:
        data_path = path or Path(__file__).resolve().parents[2] / "data" / "mock" / "transactions.json"
        if not data_path.exists():
            return
        txs = json.loads(data_path.read_text(encoding="utf-8"))
        with self._conn() as c:
            for t in txs:
                c.execute(
                    """INSERT OR REPLACE INTO mock_transactions (tx_id, asset_type, status, detail) VALUES (?,?,?,?)""",
                    (t["tx_id"], t["asset_type"], t["status"], t["detail"]),
                )

    def lookup_transaction(self, tx_id: str, asset_type: str) -> dict[str, Any] | None:
        with self._conn() as c:
            # Exact match first (case-insensitive on both fields)
            row = c.execute(
                "SELECT * FROM mock_transactions WHERE lower(tx_id)=lower(?) AND lower(asset_type)=lower(?)",
                (tx_id.strip(), asset_type.strip()),
            ).fetchone()
            if row:
                return dict(row)
            # Fallback: match by tx_id alone (in case asset alias not normalised)
            rows = c.execute(
                "SELECT * FROM mock_transactions WHERE lower(tx_id)=lower(?)",
                (tx_id.strip(),),
            ).fetchall()
            if len(rows) == 1:
                return dict(rows[0])
        return None


_store: SqliteStore | None = None


def get_store() -> SqliteStore:
    global _store
    if _store is None:
        _store = SqliteStore()
        _store.seed_mock_transactions()
    return _store
