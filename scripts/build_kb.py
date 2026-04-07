#!/usr/bin/env python3
"""Chunk corpus and build FAISS index (run after scraper/ingest)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.retrieval.chunking import build_chunks_jsonl
from app.retrieval.index_faiss import build_index


def main() -> None:
    s = get_settings()
    n = build_chunks_jsonl(s.corpus_path, s.chunks_path)
    print(f"Wrote {n} chunks -> {s.chunks_path}")
    build_index(s.chunks_path, s.faiss_index_path, s.faiss_meta_path)
    print(f"FAISS index -> {s.faiss_index_path}")


if __name__ == "__main__":
    main()
