# Secure VibeCode RAG (Local-Only)

Local, secure Retrieval Augmented Generation (RAG) with Streamlit UI, Ollama, and ChromaDB. Runs entirely on your machine. No cloud calls.

- Local-only operation (privacy by default)
- Secure ingestion of .pdf/.md/.txt
- Vector search using ChromaDB (telemetry disabled)
- Tool-augmented agent (calculator, document analyzer)
- Env-based auth, session rate limiting, input/output sanitization, audit logging

> Important: This project is designed for local, single-user use. Do not expose it to the public internet without hardening and a proper auth gateway.

## Features
- Local LLM via Ollama (default: `qwen3:8b`; supports `qwen3:1.7b` for CPU)
- Deterministic chunking with overlap and metadata for citations
- Secure prompt construction with strict instruction filtering
- Admin tab for vector store metrics and audit log tail
- Audit logs written locally with restrictive permissions (0700 dir, 0600 file)

## Architecture
```
Streamlit UI → Ingest/Chunk → Embed → Chroma (persistent)
User Query → Query Embed → Vector Search → Prompt → Ollama → Answer
```

Key modules:
- `streamlit_ui/app.py` – UI, auth, rate limiting, safe file uploads, audit logging
- `rag_pipeline/load_docs.py` – secure loaders, chunking, metadata
- `rag_pipeline/embed_and_store.py` – local embeddings + Chroma persistence
- `rag_pipeline/query_engine.py` – retrieval + secure prompt + LLM call
- `agent_tools/` – calculator, document analyzer, agent executor

## Requirements
- Python 3.10+
- Ollama installed and running locally
- RAM: 8GB+ recommended for `qwen3:8b` (use `qwen3:1.7b` on lower-end machines)

## Quick Start
```bash
# 0) Create and activate a virtualenv
python3 -m venv .venv
. .venv/bin/activate

# 1) Install dependencies
pip install -r requirements.txt

# 2) Create local env
cp .env.example .env
# Edit .env to set STREAMLIT_USERNAME and STREAMLIT_PASSWORD
# Optionally set OLLAMA_MODEL=qwen3:8b (or qwen3:1.7b)

# 3) Pull a model
ollama pull qwen3:8b
# or
ollama pull qwen3:1.7b

# 4) Run the app
streamlit run streamlit_ui/app.py
```

## Configuration (.env)
Required:
- `STREAMLIT_USERNAME` – login user for UI
- `STREAMLIT_PASSWORD` – login password for UI

Optional (defaults shown in code):
- `OLLAMA_HOST=http://127.0.0.1:11434`
- `OLLAMA_MODEL=qwen3:8b` (use `qwen3:1.7b` for CPU)
- `CHROMA_PERSIST_DIR=./vectorstore`
- `DATA_DIR=./data`
- `RAG_TOP_K=4`

Notes:
- `.env` lines must be KEY=VALUE only (no quotes, comments on the same line) to avoid parse errors.
- `.gitignore` excludes `.env`, `logs/`, `vectorstore/`, `data/`, `.venv/`.

## Security
- Env-based authentication with constant-time compare (no JWT in this version)
- Per-session rate limiting for Ask and Agent Chat
- Input validation on uploads (type/size) and queries (length caps)
- Plain-text rendering only; no `unsafe_allow_html`
- Audit logging to `logs/audit.log` with restricted filesystem permissions
- Chroma anonymized telemetry disabled; local persistence only

OWASP alignment (high-level): A01 Broken Access Control (local auth), A03 Injection (sanitized inputs/prompts), A05 Security Misconfiguration (safe defaults), A09 Logging/Monitoring (audit log). See code comments for inline controls and assumptions.

## Usage
- Ingest tab: upload `.pdf/.md/.txt`, then “Index All” to embed/store
- Ask tab: query the KB; answers include cited sources and latency
- Agent Chat tab: tool-augmented chat using calculator and document analyzer
- Admin tab: vector store metrics, audit log tail, tool utilities

## Model switching
- Default: `qwen3:8b`
- CPU-friendly: `qwen3:1.7b`
Update `.env` and restart the app after switching models.

## Roadmap
- Optional Hugging Face Spaces support (provider switch or proxy)
- Log rotation and permission hardening scripts
- Unit tests (chunking determinism, deterministic IDs)
- CLI helpers for headless indexing and querying
- Retrieval tuning surfaced via env (chunk sizes, top-k bounds)

## Contributing
PRs welcome. Please follow these guidelines:

- **Scope & branches**
  - Fork the repo and create feature branches: `feature/<short-title>` or `fix/<short-title>`.
  - Keep PRs small, focused, and easy to review.

- **Setup**
  - Use a virtualenv: `python3 -m venv .venv && . .venv/bin/activate`.
  - Install deps: `pip install -r requirements.txt`.

- **Security-first changes**
  - Do not introduce `exec`, `os.system`, or `subprocess` with `shell=True`.
  - Do not use unsafe deserialization (`pickle`, `yaml.load` without a safe loader).
  - Sanitize all user inputs and render outputs as plain text (no `unsafe_allow_html`).
  - Follow OWASP guidance; add inline comments highlighting critical security controls or assumptions.

- **Dependencies**
  - Pin versions in `requirements.txt`. Avoid uncommon/low-reputation packages.
  - Do not add networked/cloud dependencies by default (project is local-first).

- **Tests & docs**
  - Add tests when feasible (e.g., chunking determinism, retrieval bounds).
  - Update README or docstrings for new features/config.

- **Secrets & artifacts**
  - Never commit secrets. `.env`, `logs/`, `vectorstore/`, `data/`, `.venv/` are git-ignored—keep it that way.
  - Ensure example configs remain safe (`.env.example` only).

- **PR checklist**
  - [ ] Security review (inputs validated, no dangerous APIs)
  - [ ] Dependencies pinned/justified
  - [ ] README/docs updated
  - [ ] Tests added/updated or manual validation steps included

By contributing, you agree your contributions are licensed under the repository’s Apache-2.0 license.

## License
Apache-2.0. See `LICENSE`.

## Security Policy
Please report vulnerabilities privately via issues marked as security or by contacting the maintainers. Do not disclose publicly before a fix is available.
