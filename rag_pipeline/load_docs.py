#!/usr/bin/env python3
"""
Local-only document loading and chunking utilities for the secure VibeCode RAG.
- Supports .txt, .md, .pdf by default (safe, no external calls)
- Validates file names, extensions, and sizes to prevent abuse (OWASP A03/A05)
- Produces small chunks to reduce prompt injection risk (LLM01)
"""

import os
import io
import re
import hashlib
from pathlib import Path
from typing import List, Dict

import pdfplumber

# Security controls
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_TEXT_CHARS = 200_000  # cap after extraction to avoid DoS


def _safe_filename(name: str) -> str:
    """Return a sanitized filename (no path traversal, limited charset)."""
    name = os.path.basename(name)
    # Remove characters other than alnum, dash, underscore, dot, and space
    name = re.sub(r"[^a-zA-Z0-9._\- ]+", "_", name)
    return name[:255]


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Simple, deterministic character-based chunker with overlap.
    Keeps chunks small to limit model context poisoning and costs.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def _read_txt(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        return data[:MAX_TEXT_CHARS]


def _read_md(path: Path) -> str:
    # Treat like text; rendering is not needed for RAG context
    return _read_txt(path)


def _read_pdf(path: Path) -> str:
    # Use pdfplumber locally; no external services
    text_parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                text_parts.append(txt)
            if sum(len(t) for t in text_parts) > MAX_TEXT_CHARS:
                break
    return "\n".join(text_parts)[:MAX_TEXT_CHARS]


def load_and_split(file_path: str) -> List[Dict]:
    """
    Load one file, validate, extract text, and return chunk dicts:
    [{"text": str, "metadata": {...}}]
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE_BYTES})")

    # Read bytes for hashing
    raw_bytes = path.read_bytes()
    file_hash = _sha256_bytes(raw_bytes)

    # Extract text per type
    if ext == ".txt":
        text = _read_txt(path)
    elif ext == ".md":
        text = _read_md(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    safe_name = _safe_filename(path.name)
    chunks = _chunk_text(text)

    out: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        meta = {
            "source": safe_name,
            "source_path": str(path.resolve()),
            "sha256": file_hash,
            "chunk_index": idx,
            "ext": ext,
        }
        out.append({"text": chunk, "metadata": meta})
    return out


def load_all_from_dir(data_dir: str) -> List[Dict]:
    """Load and split all allowed files from directory recursively."""
    base = Path(data_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    results: List[Dict] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            results.extend(load_and_split(str(p)))
        except Exception:
            # Skip problematic files but continue processing
            continue
    return results
