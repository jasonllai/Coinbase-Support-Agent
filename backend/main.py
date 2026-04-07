"""FastAPI backend for the Coinbase Support Agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.graph import run_agent_turn
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.storage.sqlite_store import get_store

setup_logging()
log = logging.getLogger(__name__)


def _validate_startup() -> dict[str, Any]:
    s = get_settings()
    checks = {
        "faiss_index": s.faiss_index_path.exists(),
        "faiss_meta": s.faiss_meta_path.exists(),
        "corpus": s.corpus_path.exists(),
    }
    checks["ok"] = checks["faiss_index"] and checks["faiss_meta"]
    if not checks["ok"]:
        log.warning("startup_validation degraded: %s", checks)
    else:
        log.info("startup_validation ok")
    return checks


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_checks = _validate_startup()
    yield


app = FastAPI(title="Coinbase Support Agent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    data: dict[str, Any]


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    ok_index = s.faiss_index_path.exists() and s.faiss_meta_path.exists()
    checks = getattr(app.state, "startup_checks", None)
    return {
        "status": "ok" if ok_index else "degraded",
        "faiss_ready": ok_index,
        "environment": s.environment,
        "startup_checks": checks,
    }


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        resp = run_agent_turn(req.session_id, req.message)
        return ChatResponse(data=resp.model_dump())
    except Exception as e:
        log.exception("chat error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/sessions")
def sessions() -> dict[str, Any]:
    rows = get_store().list_sessions(100)
    return {"sessions": rows}


@app.get("/v1/sessions/{session_id}")
def session_detail(session_id: str) -> dict[str, Any]:
    rec = get_store().load_session(session_id)
    return {
        "session_id": rec.session_id,
        "title": rec.title,
        "messages": rec.messages,
        "router_trace": rec.router_trace,
        "updated_at": rec.updated_at,
    }


@app.delete("/v1/sessions/{session_id}")
def session_delete(session_id: str) -> dict[str, Any]:
    store = get_store()
    store.ensure_session(session_id)
    store.save_session(session_id, [], [])
    return {"ok": True}


class EvalRunRequest(BaseModel):
    dry_run: bool = False


@app.post("/v1/eval/run")
def eval_run(req: EvalRunRequest) -> dict[str, Any]:
    """Runs the full eval suite (LLM + retrieval). Can take several minutes."""
    if req.dry_run:
        from app.eval.runner import load_cases

        return {"dry_run": True, "case_count": len(load_cases())}
    from app.eval.runner import run_all

    out = Path(__file__).resolve().parents[1] / "data" / "eval"
    summary = run_all(out)
    summary["artifacts_dir"] = str(out)
    return summary
