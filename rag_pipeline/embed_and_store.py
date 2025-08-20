#!/usr/bin/env python3
"""
Embedding and vector store utilities for local, secure RAG.
- Uses sentence-transformers locally (no external API)
- Stores embeddings in local Chroma with persistence
- Security: telemetry disabled, input validation, capped batch sizes
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Constants / defaults
DEFAULT_COLLECTION = "rag_documents"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
MAX_BATCH = 256  # Prevent large memory spikes / DoS

logger = logging.getLogger(__name__)

# Lazy-loaded global model to avoid repeated loads
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # All-MiniLM-L6-v2 balances speed/quality and is local
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_chroma_collection(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = DEFAULT_PERSIST_DIR,
):
    """Return an existing or new Chroma collection with secure settings."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    # Use new Chroma client API to avoid deprecation. Keep telemetry disabled.
    # Note: PersistentClient creates/opens a local DB at `path`.
    client = PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,  # Security: no phone-home
            allow_reset=False,
        ),
    )
    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(
            name=collection_name, metadata={"created_at": time.time(), "version": "1.0"}
        )


def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    # Note: model.encode returns numpy arrays; convert to lists for JSON/chromadb
    # Security: limit batch size
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), MAX_BATCH):
        batch = texts[i : i + MAX_BATCH]
        vecs = model.encode(batch, normalize_embeddings=True)
        embeddings.extend(v.tolist() for v in vecs)
    return embeddings


def create_vectorstore(
    chunks: List[Dict],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Tuple[int, int]:
    """
    Upsert chunks into Chroma. Each chunk is a dict: {"text": str, "metadata": dict}
    Returns (added_count, total_count)
    """
    if not chunks:
        return (0, 0)

    # Prepare inputs with basic validation
    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for c in chunks:
        text = (c.get("text") or "").strip()
        meta = c.get("metadata") or {}
        if not text:
            continue
        # Security: cap length per doc to avoid bloat
        if len(text) > 5000:
            text = text[:5000]
        documents.append(text)
        metadatas.append(meta)
        # Build a deterministic ID from source+chunk_index if present
        src = str(meta.get("sha256", ""))
        idx = str(meta.get("chunk_index", "0"))
        ids.append(f"{src}-{idx}")

    if not documents:
        return (0, 0)

    collection = get_chroma_collection(collection_name=collection_name, persist_dir=persist_dir)

    # Embed and upsert
    vectors = _embed_texts(documents)
    start = time.time()
    collection.upsert(documents=documents, metadatas=metadatas, embeddings=vectors, ids=ids)
    total = collection.count()
    duration_ms = int((time.time() - start) * 1000)

    # Basic local audit log
    try:
        log_dir = Path("./logs"); log_dir.mkdir(exist_ok=True)
        with (log_dir / "audit.log").open("a", encoding="utf-8") as lf:
            lf.write(
                f"{time.time()}|operation=upsert|added={len(documents)}|total={total}|ms={duration_ms}\n"
            )
    except Exception:
        pass

    return (len(documents), total)
