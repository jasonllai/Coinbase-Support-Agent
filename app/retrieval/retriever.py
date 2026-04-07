"""Dense retrieval + optional cross-encoder rerank + BM25 fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.retrieval.embeddings import embed_texts


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    article_title: str
    section_title: str
    canonical_url: str
    category: str
    text: str
    score: float


class HybridRetriever:
    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._index = faiss.read_index(str(s.faiss_index_path))
        self._metas: list[dict[str, Any]] = []
        with s.faiss_meta_path.open(encoding="utf-8") as f:
            for line in f:
                self._metas.append(json.loads(line))
        corpus_tokens = [_tokenize(m["text"]) for m in self._metas]
        self._bm25 = BM25Okapi(corpus_tokens)
        self._reranker = None

    def _load_reranker(self):
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._reranker = False  # type: ignore[assignment]
        return self._reranker

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        s = self._s
        k = top_k or s.retrieval_top_k
        qv = embed_texts([query])
        scores, idxs = self._index.search(qv, k)

        dense: list[RetrievedChunk] = []
        for sc, i in zip(scores[0], idxs[0], strict=False):
            if i < 0 or i >= len(self._metas):
                continue
            m = self._metas[i]
            dense.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    doc_id=m["doc_id"],
                    article_title=m["article_title"],
                    section_title=m["section_title"],
                    canonical_url=m["canonical_url"],
                    category=m.get("category", ""),
                    text=m["text"],
                    score=float(sc),
                ),
            )

        bm = self._bm25.get_scores(_tokenize(query))
        bm_idx = np.argsort(-bm)[:k]
        bm_chunks: list[RetrievedChunk] = []
        for i in bm_idx:
            m = self._metas[int(i)]
            bm_chunks.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    doc_id=m["doc_id"],
                    article_title=m["article_title"],
                    section_title=m["section_title"],
                    canonical_url=m["canonical_url"],
                    category=m.get("category", ""),
                    text=m["text"],
                    score=float(bm[int(i)]),
                ),
            )

        # Normalise dense scores to [0, 1] range for rank fusion
        dense_scores = [c.score for c in dense]
        d_max = max(dense_scores) if dense_scores else 1.0
        d_min = min(dense_scores) if dense_scores else 0.0
        d_range = max(d_max - d_min, 1e-9)

        bm_scores = [c.score for c in bm_chunks]
        b_max = max(bm_scores) if bm_scores else 1.0
        b_min = min(bm_scores) if bm_scores else 0.0
        b_range = max(b_max - b_min, 1e-9)

        merged: dict[str, RetrievedChunk] = {}
        for c in dense:
            c.score = (c.score - d_min) / d_range
            merged[c.chunk_id] = c
        for c in bm_chunks:
            norm_score = (c.score - b_min) / b_range
            if c.chunk_id in merged:
                # Combine scores: give dense 60%, BM25 40%
                merged[c.chunk_id].score = 0.6 * merged[c.chunk_id].score + 0.4 * norm_score
            else:
                c.score = 0.4 * norm_score  # BM25-only chunk gets partial credit
                merged[c.chunk_id] = c

        candidates = list(merged.values())
        reranker = self._load_reranker()
        if reranker and reranker is not False:
            pairs = [[query, c.text] for c in candidates]
            ce_scores = reranker.predict(pairs)
            for c, sc in zip(candidates, ce_scores, strict=False):
                c.score = float(sc)
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[: s.rerank_top_n]

        # Without reranker: sort by fused normalised score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[: s.rerank_top_n]


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
