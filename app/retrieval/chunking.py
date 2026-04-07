"""Semantic-ish chunking: split on headings and paragraph boundaries."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator


def _slug(s: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:max_len] or "section"


def chunk_article(doc: dict[str, Any], max_chars: int = 1200, overlap: int = 120) -> Iterator[dict[str, Any]]:
    doc_id = doc["doc_id"]
    url = doc["canonical_url"]
    title = doc.get("title") or "Untitled"
    category = doc.get("category") or ""
    body = doc.get("body_text") or ""

    # Split on lines that look like headings (ALL CAPS short lines) or numbered sections
    raw_blocks = re.split(r"\n{2,}", body)
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    buf: list[str] = []
    for block in raw_blocks:
        b = block.strip()
        if not b:
            continue
        is_heading = (
            len(b) < 80
            and "\n" not in b
            and (b.isupper() or (b.endswith(":") and len(b) < 60))
        )
        if is_heading and len(b.split()) <= 12:
            if buf:
                sections.append((current_heading, "\n\n".join(buf)))
                buf = []
            current_heading = b.rstrip(":")
            continue
        buf.append(b)
    if buf:
        sections.append((current_heading, "\n\n".join(buf)))

    if not sections:
        sections = [("Overview", body)]

    chunk_idx = 0
    for sec_title, sec_text in sections:
        if not sec_text.strip():
            continue
        # Sub-chunk long sections
        start = 0
        while start < len(sec_text):
            end = min(len(sec_text), start + max_chars)
            piece = sec_text[start:end]
            if end < len(sec_text):
                # backtrack to sentence boundary
                cut = piece.rfind(". ")
                if cut > max_chars * 0.55:
                    piece = piece[: cut + 1]
                    end = start + len(piece)
            stable = f"{doc_id}|{sec_title}|{chunk_idx}|{piece[:64]}"
            chunk_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
            yield {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "article_title": title,
                "section_title": sec_title,
                "canonical_url": url,
                "category": category,
                "chunk_index": chunk_idx,
                "text": piece.strip(),
                "chunk_slug": _slug(f"{title}-{sec_title}-{chunk_idx}"),
            }
            chunk_idx += 1
            start = max(end - overlap, end)


def build_chunks_jsonl(corpus_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with corpus_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for ch in chunk_article(doc):
                fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                n += 1
    return n
