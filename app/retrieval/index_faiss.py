"""Build FAISS index + sidecar metadata JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.retrieval.embeddings import embed_texts


def build_index(chunks_path: Path, index_path: Path, meta_path: Path) -> None:
    settings = get_settings()
    texts: list[str] = []
    metas: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append(obj)

    if not texts:
        raise RuntimeError(f"No chunks in {chunks_path}; run scraper + chunking first.")

    vecs = embed_texts(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main() -> None:
    s = get_settings()
    build_index(s.chunks_path, s.faiss_index_path, s.faiss_meta_path)


if __name__ == "__main__":
    main()
