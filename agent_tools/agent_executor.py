

#!/usr/bin/env python3
"""
Secure agent executor for vibe coding RAG knowledge system

"""

from langchain.agents import initialize_agent, Tool, AgentExecutor, load_agent
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Ollama
from rag_pipeline.query_engine import run_query
from .calculator import secure_calculator
from .doc_analyzer import document_analyzer
import os

def setup_agent():
    """Setup vibe coding agent with professional tools"""
    def kb_query_text(q: str) -> str:
        """Wrapper around run_query to return a human-readable string.
        Avoids leaking internal structures while keeping citations."""
        try:
            res = run_query(q)
            if isinstance(res, dict) and res.get("error"):
                return f"Error: {res['error']}"
            if not isinstance(res, dict):
                return str(res)
            ans = str(res.get("answer", "")).strip()
            sources = res.get("sources", [])
            if sources:
                src_lines = [
                    f"- {s.get('source', 'unknown')} (chunk {s.get('chunk_index')}, dist {s.get('distance')})"
                    for s in sources
                ]
                return ans + "\n\nSources:\n" + "\n".join(src_lines)
            return ans or "No answer."
        except Exception as e:
            return f"Error: {str(e)}"
    tools = [
        Tool(
            name="Secure_Calculator",
            func=secure_calculator,
            description="Perform secure mathematical calculations. Input must be a valid arithmetic expression with basic operators (+, -, *, /, ^, sqrt, log, sin, cos, tan, abs, pow). No code execution risks."
        ),
        Tool(
            name="Knowledge_Base_Query",
            func=kb_query_text,
            description="Search and retrieve information from the local knowledge base (RAG). Returns an answer with cited sources."
        ),
        Tool(
            name="Document_Analyzer",
            func=document_analyzer,
            description="Analyze documents in the knowledge base. Provides metadata, statistics, document counts, file types, and content insights. Use for questions about document inventory and analysis."
        )
    ]
    # Support comma-separated model preference list, try in order
    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    models_env = os.getenv("OLLAMA_MODEL", "qwen3:1.7b, qwen3:8b")
    candidates = [m.strip() for m in models_env.split(",") if m.strip()]
    llm = None
    last_err = None
    for model_name in candidates:
        try:
            llm = Ollama(model=model_name, base_url=base_url)
            # quick no-op check by formatting a trivial prompt (lazy init)
            _ = str(llm)  # ensures object constructed
            break
        except Exception as e:
            last_err = e
            continue
    if llm is None:
        raise RuntimeError(f"Failed to initialize Ollama with models {candidates}: {last_err}")

    # agent = load_agent(
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )

    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent,
    #     tools=tools,
    #     max_iterations=1,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True 
    )
    return agent
    #return agent_executor
