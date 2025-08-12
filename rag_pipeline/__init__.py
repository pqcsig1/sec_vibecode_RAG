"""
RAG pipeline package for local, secure vibe coding assistant.
Provides document loading/chunking, embedding+Chroma persistence, and query engine.
"""

from .load_docs import load_and_split, load_all_from_dir
from .embed_and_store import create_vectorstore, get_chroma_collection
from .query_engine import run_query
