from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(str, Enum):
    KB_QA = "KB_QA"
    ACTION_CHECK_TRANSACTION = "ACTION_CHECK_TRANSACTION"
    ACTION_CREATE_TICKET = "ACTION_CREATE_TICKET"
    ACTION_ONBOARDING_SUPPORT = "ACTION_ONBOARDING_SUPPORT"
    ACTION_ACCOUNT_RECOVERY = "ACTION_ACCOUNT_RECOVERY"
    AMBIGUOUS = "AMBIGUOUS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNSAFE = "UNSAFE"
    # Policy / security-abuse requests (distinct from generic UNSAFE for grading & analytics)
    SECURITY_SENSITIVE = "SECURITY_SENSITIVE"


class GuardrailResult(BaseModel):
    blocked: bool
    category: str = "ok"
    reason: str = ""
    user_message: str = ""


class RouterOutput(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    clarifying_question: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    session_id: str
    message: str
    details: str | None = None
    intent: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    action: dict[str, Any] | None = None
    status: Literal["ok", "clarify", "refusal", "error"] = "ok"
    trace: dict[str, Any] = Field(default_factory=dict)
