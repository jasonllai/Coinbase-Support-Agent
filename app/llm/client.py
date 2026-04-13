"""OpenAI-compatible client for the course Qwen endpoint."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from openai import AuthenticationError, OpenAI
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings

log = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output, handling Qwen3 <think> blocks.

    The model wraps reasoning in <think>...</think> before the actual output. Those
    blocks often contain {curly braces} in natural language, which make a greedy
    regex match the wrong span.  We strip the think block first, then parse.
    """
    # 1. Strip reasoning blocks
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # 2. Direct parse of whatever remains
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Find the last non-nested {...} block that contains "concise_answer" or any key
    for m in reversed(list(re.finditer(r"\{[^{}]+\}", cleaned))):
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue

    # 4. Greedy match on cleaned text (think block already stripped)
    m2 = re.search(r"\{[\s\S]*\}\s*$", cleaned)
    if not m2:
        raise ValueError("no json object in model output")
    return json.loads(m2.group(0))


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self._model = s.llm_model
        self._client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "dummy", timeout=s.llm_timeout_s)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_not_exception_type(AuthenticationError),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        s = get_settings()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else s.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else s.llm_max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("llm chat failed: %s", e)
            raise

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Request JSON; fall back to parsing if API ignores response_format."""
        try:
            content = self.chat(
                messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return json.loads(content)
        except Exception:
            content = self.chat(messages, temperature=temperature)
            return _extract_json_object(content)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        s = get_settings()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature if temperature is not None else s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
