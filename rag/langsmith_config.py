"""
rag/langsmith_config.py

LangSmith observability configuration.

What this file does:
1. Loads environment variables for LangSmith
2. Creates a LangSmith client for manual logging
3. Provides decorators to trace custom functions
4. Provides evaluation utilities

Why centralized:
- One place to enable/disable tracing (change .env only)
- All LangSmith imports in one file
- Easy to test without LangSmith (set LANGCHAIN_TRACING_V2=false)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Check if tracing is enabled ───────────────────────────────────────────────
# Reads from environment — no hardcoding
TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-retail-intelligence")


def get_langsmith_client():
    """
    Returns a LangSmith client instance.
    
    Why use the client directly (not just env vars):
    - Auto-tracing via env vars covers LangChain calls only
    - For your FastAPI endpoints and custom analytics functions,
      you need the client to log manually
    - Client lets you create feedback, run evaluations, log metrics
    """
    if not TRACING_ENABLED:
        return None
    
    try:
        from langsmith import Client
        client = Client()
        log.info("LangSmith client connected | project=%s", LANGSMITH_PROJECT)
        return client
    except Exception as e:
        log.warning("LangSmith client failed to initialize: %s", str(e))
        return None


def trace_function(name: str, run_type: str = "chain"):
    """
    Decorator that wraps any Python function in a LangSmith trace span.
    
    Use this on functions NOT already traced by LangChain automatically
    (i.e. your custom Python functions, not LangChain chains).
    
    Why: LangChain auto-traces its own objects (ChatOllama, retriever chains)
    but NOT your custom functions like route_question(), format_documents(),
    check_grounding(). This decorator fills that gap.
    
    run_type options:
    - "chain"     → a processing step
    - "retriever" → a retrieval step
    - "llm"       → a language model call
    - "tool"      → a tool/function call
    """
    def decorator(func):
        if not TRACING_ENABLED:
            return func  # no-op if tracing disabled
        
        try:
            from langsmith import traceable
            # @traceable automatically creates a span in the current trace
            # with the given name and run_type
            return traceable(name=name, run_type=run_type)(func)
        except ImportError:
            return func  # graceful fallback if langsmith not installed
    
    return decorator


def get_tracer_callbacks():
    """
    Returns LangSmith callback handlers for LangChain calls.
    
    Why callbacks:
    When you call chain.invoke() with callbacks=[handler],
    LangSmith records that specific call with full detail —
    inputs, outputs, token usage, latency.
    
    Used in RAGPipeline.answer() to trace the LLM generation step.
    """
    if not TRACING_ENABLED:
        return []
    
    try:
        from langsmith.run_helpers import get_current_run_tree
        from langchain_core.tracers import LangChainTracer
        
        tracer = LangChainTracer(project_name=LANGSMITH_PROJECT)
        return [tracer]
    except Exception as e:
        log.warning("Could not create LangSmith callbacks: %s", str(e))
        return []