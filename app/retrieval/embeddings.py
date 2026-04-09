"""Embedding client — calls the remote OpenAI-compatible embedding endpoint.

Endpoint: https://rsm-8430-a2.bjlkeng.io/v1/embeddings
Model:    @cf/baai/bge-base-en-v1.5  (768-dim, normalised)
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from openai import OpenAI

from app.core.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> tuple[OpenAI, str]:
    """Return (openai_client, model_name) — cached for the process lifetime."""
    s = get_settings()
    api_key = s.embedding_api_key or s.llm_api_key or "no-key"
    client = OpenAI(base_url=s.embedding_base_url, api_key=api_key)
    return client, s.embedding_model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts and return an (N, 768) float32 array."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    client, model = _client()
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            # Sort by index to guarantee order matches input
            batch_vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_vecs.extend(batch_vecs)
        except Exception:
            log.exception("Embedding API call failed for batch %d-%d", i, i + len(batch))
            raise

    arr = np.array(all_vecs, dtype=np.float32)

    # L2-normalise so cosine similarity == inner product (matches IndexFlatIP)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-9)
    return arr
