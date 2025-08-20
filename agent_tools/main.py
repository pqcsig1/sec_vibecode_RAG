

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Ollama
from rag_pipeline.query_engine import run_query
from .calculator import calculator
import os

def setup_agent():
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for math problems. Input must be a valid arithmetic expression."
        ),
        Tool(
            name="RAG Retriever",
            func=lambda q: run_query(q),
            description="Useful when answering question about ingress controller ."
        )
    ]
    llm = Ollama(model="qwen3:8b", base_url=os.getenv("OLLAMA_HOST"))
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True 
    )
    return agent
