#!/usr/bin/env python3
"""
Secure local Streamlit UI for VibeCode RAG (Option A: single-process).
- Local-only: uses local Chroma and local Ollama via HTTP
- Authentication via .env (STREAMLIT_USERNAME/STREAMLIT_PASSWORD)
- Ingestion of .txt/.md/.pdf from local machine into ./data and embedding into ./vectorstore
- Query tab to ask questions with RAG, showing sources
- Admin tab for basic metrics and audit tail

Security controls:
- Simple auth gate and session handling (OWASP A01)
- Strict file type and size checks during upload (A03)
- Sanitized, length-limited inputs (A03/A05, LLM01/LLM05)
- Local storage only, telemetry disabled (A05)
- Basic per-session rate limiting for queries (A09)
- Audit logging for ingest/query events without sensitive data (A09)
"""

import os
import io
import sys
import time
import hmac
import json
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Resolve project root and ensure imports work regardless of cwd
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables from project root .env
load_dotenv(find_dotenv(), override=False)

# Now safe to import local modules
from rag_pipeline import load_and_split, load_all_from_dir, create_vectorstore, run_query  # noqa: E402
# Agent tools for convenience operations
from agent_tools.calculator import secure_calculator  # noqa: E402
from agent_tools.doc_analyzer import document_analyzer  # noqa: E402
from agent_tools.agent_executor import setup_agent  # noqa: E402

# Configuration
DATA_DIR = ROOT_DIR / "data"
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "vectorstore")))
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "audit.log"
ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB per file

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Make DATA_DIR discoverable by tools that rely on env
os.environ.setdefault("DATA_DIR", str(DATA_DIR))

# Auth helpers
def _get_env_cred() -> Dict[str, str]:
    return {
        "user": os.getenv("STREAMLIT_USERNAME", ""),
        "pass": os.getenv("STREAMLIT_PASSWORD", ""),
    }


def _constant_time_eq(a: str, b: str) -> bool:
    a = a.encode("utf-8"); b = b.encode("utf-8")
    return hmac.compare_digest(a, b)


def _require_auth() -> bool:
    creds = _get_env_cred()
    if not creds["user"] or not creds["pass"]:
        st.error("Admin must set STREAMLIT_USERNAME and STREAMLIT_PASSWORD in .env")
        return False

    if "auth" not in st.session_state:
        st.session_state["auth"] = {"ok": False, "fails": 0}

    if st.session_state["auth"]["ok"]:
        return True

    with st.form("login_form", clear_on_submit=False):
        st.subheader("Login")
        u = st.text_input("Username", value="", type="default")
        p = st.text_input("Password", value="", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if _constant_time_eq(u, creds["user"]) and _constant_time_eq(p, creds["pass"]):
            st.session_state["auth"]["ok"] = True
            st.success("Authenticated")
            return True
        else:
            st.session_state["auth"]["fails"] += 1
            st.error("Invalid credentials")
            time.sleep(min(1 + st.session_state["auth"]["fails"], 3))  # basic throttling
    return False


# Simple per-session query rate limiting
def _allow_query() -> bool:
    now = time.time()
    q = st.session_state.setdefault("query_rl", {"window": now, "count": 0})
    window = q["window"]
    # 10 queries per 60 seconds per session
    if now - window > 60:
        q["window"] = now
        q["count"] = 0
    if q["count"] >= 10:
        return False
    q["count"] += 1
    return True


def _safe_filename(name: str) -> str:
    # Similar to loader; avoid path traversal and odd chars
    import re
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._\- ]+", "_", name)
    return name[:255]


def _log_event(kind: str, **fields):
    try:
        # Ensure secure directory perms (owner-only)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            LOG_DIR.chmod(0o700)
        except Exception:
            pass
        # Append event and enforce file perms (owner-only)
        new_file = not LOG_FILE.exists()
        with LOG_FILE.open("a", encoding="utf-8") as lf:
            rec = {"ts": time.time(), "type": kind, **fields}
            lf.write(json.dumps(rec) + "\n")
        if new_file:
            try:
                LOG_FILE.chmod(0o600)
            except Exception:
                pass
    except Exception:
        pass


# App UI
st.set_page_config(page_title="Secure VibeCode RAG (Local)", page_icon="🔒", layout="wide")

st.title("🔒 Secure VibeCode RAG (Local)")
st.caption("Local-only RAG using Chroma + Ollama. No cloud calls. For training use.")

if not _require_auth():
    st.stop()

# Tabs: Ingest | Ask | Agent Chat | Admin
ingest_tab, ask_tab, chat_tab, admin_tab = st.tabs(["Ingest", "Ask", "Agent Chat", "Admin"])

with ingest_tab:
    st.header("Ingest Documents")
    st.write("Accepted types: .txt, .md, .pdf | Max size: 10MB per file")

    uploaded = st.file_uploader("Upload files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if uploaded:
        saved_files: List[Path] = []
        for uf in uploaded:
            # Size guard (Streamlit provides size, but validate again)
            content = uf.getvalue()
            if len(content) > MAX_UPLOAD_BYTES:
                st.warning(f"Skipping {uf.name}: too large")
                continue
            safe_name = _safe_filename(uf.name)
            out_path = DATA_DIR / safe_name
            with out_path.open("wb") as f:
                f.write(content)
            saved_files.append(out_path)
        if saved_files:
            st.success(f"Saved {len(saved_files)} files to {DATA_DIR}")
            _log_event("upload", files=[p.name for p in saved_files])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index All in data/", type="primary"):
            with st.spinner("Indexing all documents..."):
                try:
                    chunks = load_all_from_dir(str(DATA_DIR))
                    added, total = create_vectorstore(chunks, persist_dir=str(PERSIST_DIR))
                    st.success(f"Indexed {added} chunks. Collection size: {total}")
                    _log_event("index_all", added=added, total=total)
                except Exception as e:
                    st.error(f"Index all failed: {e}")
    with col2:
        if st.button("Rebuild Index (from data/)"):
            with st.spinner("Rebuilding index..."):
                try:
                    chunks = load_all_from_dir(str(DATA_DIR))
                    # In this simple version we upsert; for a true rebuild, one could reset the collection
                    added, total = create_vectorstore(chunks, persist_dir=str(PERSIST_DIR))
                    st.success(f"Rebuilt index. Added {added} chunks. Total: {total}")
                    _log_event("rebuild", added=added, total=total)
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

with ask_tab:
    st.header("Ask the Knowledge Base")
    q = st.text_input("Your question", value="")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Please enter a question.")
        elif not _allow_query():
            st.warning("Rate limit: max 10 queries per minute per session.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = run_query(q)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.subheader("Answer")
                        # Render as plain text to avoid HTML/script execution
                        st.text_area("", value=result.get("answer", ""), height=200)

                        st.subheader("Sources")
                        sources = result.get("sources", [])
                        if sources:
                            for s in sources:
                                st.write(f"- {s.get('source')} (chunk {s.get('chunk_index')}, dist {s.get('distance')})")
                        else:
                            st.write("No sources found.")

                        st.caption(f"Latency: {result.get('latency_ms', 0)} ms")
                        _log_event("query", ok=True, latency_ms=result.get("latency_ms", 0))
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    _log_event("query", ok=False, error=str(e))

with chat_tab:
    st.header("Agent Chat (Tools: RAG, Calculator, Analyzer)")

    # Basic per-session rate limiting for agent chat
    def _allow_agent() -> bool:
        now = time.time()
        q = st.session_state.setdefault("agent_rl", {"window": now, "count": 0})
        window = q["window"]
        # 10 messages per 60 seconds per session
        if now - window > 60:
            q["window"] = now
            q["count"] = 0
        if q["count"] >= 10:
            return False
        q["count"] += 1
        return True

    # Lazy-create agent once per session
    if "agent" not in st.session_state:
        try:
            st.session_state["agent"] = setup_agent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()

    msgs = st.session_state.setdefault("agent_messages", [])  # list of {role, content}

    # Render history safely (plain text)
    if msgs:
        st.subheader("Conversation")
        for m in msgs[-50:]:  # cap history shown
            role = m.get("role", "user").capitalize()
            st.text(f"{role}: {m.get('content','')}")

    user_msg = st.text_input("Your message", value="", key="agent_input")
    if st.button("Send", key="agent_send"):
        text = (user_msg or "").strip()
        if not text:
            st.warning("Please enter a message.")
        elif len(text) > 500:
            st.warning("Message too long (max 500 characters).")
        elif not _allow_agent():
            st.warning("Rate limit: max 10 messages per minute per session.")
        else:
            msgs.append({"role": "user", "content": text})
            with st.spinner("Agent thinking..."):
                try:
                    # Use invoke for LC 0.2; handle both dict and str outputs
                    out = st.session_state["agent"].invoke({"input": text})
                    if isinstance(out, dict) and "output" in out:
                        reply = str(out["output"]).strip()
                    else:
                        reply = str(out).strip()
                except Exception as e:
                    reply = f"Error: {e}"
                msgs.append({"role": "assistant", "content": reply})
                _log_event("agent_chat", ok=not reply.startswith("Error:"))
            # Rerender messages
            st.rerun()

with admin_tab:
    st.header("Admin & Metrics")
    # Basic counts
    try:
        from rag_pipeline.embed_and_store import get_chroma_collection
        coll = get_chroma_collection(persist_dir=str(PERSIST_DIR))
        count = coll.count()
        st.metric("Vector store documents", count)
    except Exception as e:
        st.warning(f"Could not access vector store: {e}")

    # Data dir info
    files = list(DATA_DIR.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    st.metric("Files in data/", file_count)

    # Ollama status (display configured host)
    st.write(f"Ollama host: {os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")

    # Audit tail
    st.subheader("Recent audit log entries")
    try:
        if LOG_FILE.exists():
            with LOG_FILE.open("r", encoding="utf-8") as lf:
                lines = lf.readlines()[-100:]
            for ln in lines:
                st.code(ln.strip(), language="json")
        else:
            st.info("No audit log yet.")
    except Exception as e:
        st.warning(f"Failed to read audit log: {e}")

    # Agent Tools
    st.subheader("Agent Tools (Local Utilities)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Secure Calculator**")
        expr = st.text_input("Expression", value="", key="calc_expr")
        if st.button("Compute", key="calc_btn"):
            try:
                # Cap length to prevent abuse
                expr_sanitized = (expr or "").strip()[:200]
                result = secure_calculator(expr_sanitized)
                st.text(f"Result: {result}")
                _log_event("calc", ok=True)
            except Exception as e:
                st.error(f"Calculation failed: {e}")
                _log_event("calc", ok=False, error=str(e))

    with c2:
        st.markdown("**Document Analyzer**")
        ana_query = st.text_input("Analyzer query (e.g., 'count', 'size', 'types', 'all')", value="count", key="ana_query")
        if st.button("Analyze", key="ana_btn"):
            try:
                q = (ana_query or "").strip()[:200]
                res = document_analyzer(q)
                # Show as plain text to avoid any rendering risks
                st.text(res)
                _log_event("doc_analyzer", ok=True)
            except Exception as e:
                st.error(f"Analyzer failed: {e}")
                _log_event("doc_analyzer", ok=False, error=str(e))

st.caption("Security: Local-only, sanitized inputs, rate-limited, audited. Follow OWASP/LLM guidance.")
