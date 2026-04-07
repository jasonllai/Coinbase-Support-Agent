from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    s = get_settings()
    return SentenceTransformer(s.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    m = _model()
    vecs = m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)
