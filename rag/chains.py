"""
Day 11: LangChain Expression Language (LCEL) Chains

Chains connect: prompt template → Ollama LLM → output parser

Each chain is a function that returns a runnable.
'Runnable' means it has .invoke() and .ainvoke() methods.
.invoke()  = synchronous  (use in scripts/tests)
.ainvoke() = asynchronous (use in FastAPI endpoints)
"""

import logging
import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.prompts import (
    get_base_prompt,
    get_analytics_prompt,
    get_inventory_prompt,
    get_rag_prompt,
)

log = logging.getLogger(__name__)

# ── Ollama LLM Configuration ──────────────────────────────────────────────────
def get_llm(
    model       : str   = "phi3",
    temperature : float = 0.1,
    max_tokens  : int   = 512,
) -> ChatOllama:
    """
    Creates a ChatOllama instance pointing to your local Ollama server.
    
    temperature=0.1:
    - 0.0 = completely deterministic (same input → same output always)
    - 1.0 = very creative/random
    - 0.1 = mostly deterministic with slight variation
    - For retail analytics: LOW temperature is correct.
      You want "Tapal Tea stock is at 12 units" not creative variations.
    
    max_tokens=512:
    - Limits response length
    - Prevents Ollama from writing essays when the owner wants 2 sentences
    - Keeps API response times fast
    
    base_url: where your local Ollama server runs
    """
    return ChatOllama(
        model       = model,
        temperature = temperature,
        num_predict = max_tokens,
        base_url    = "http://localhost:11434",
    )


# ── Output Parser ─────────────────────────────────────────────────────────────
# StrOutputParser extracts the plain text string from the model's response object
# Without it, you'd get a complex AIMessage object instead of a clean string
output_parser = StrOutputParser()


# ── Chain 1: Base Chain ───────────────────────────────────────────────────────
def get_base_chain():
    """
    Simplest chain: question → Ollama model → answer string
    
    The | operator connects components left to right:
    prompt  receives {"question": "..."}
    llm     receives the formatted prompt text
    parser  receives AIMessage, returns plain string
    
    Use for: general store questions with no specific data context
    """
    llm = get_llm()
    return get_base_prompt() | llm | output_parser


# ── Chain 2: Analytics Chain ──────────────────────────────────────────────────
def get_analytics_chain():
    """
    Analytics chain: injects live KPI data into the prompt.
    
    RunnablePassthrough() passes the input dict through unchanged
    to the next step. The prompt template then fills:
    - {store_context} from the input dict
    - {question} from the input dict
    
    Usage:
    chain.invoke({
        "store_context": json.dumps(get_store_kpis()),
        "question": "What is my profit margin?"
    })
    """
    llm = get_llm()
    return get_analytics_prompt() | llm | output_parser


# ── Chain 3: Inventory Chain ──────────────────────────────────────────────────
def get_inventory_chain():
    """
    Inventory chain: injects inventory health + alerts into prompt.
    
    Usage:
    chain.invoke({
        "inventory_context": json.dumps(get_inventory_health_summary()),
        "alerts_context": json.dumps(get_low_stock_alerts().to_dict()),
        "question": "What should I restock tomorrow?"
    })
    """
    llm = get_llm()
    return get_inventory_prompt() | llm | output_parser


# ── Chain 4: RAG Chain (skeleton — completed Day 16) ─────────────────────────
def get_rag_chain(retriever=None):
    """
    Full RAG chain — completed Day 16 when FAISS retriever exists.
    Today: build the structure, test with a mock retriever.
    
    In full RAG:
    1. retriever fetches relevant docs from FAISS for the question
    2. docs are formatted as context string
    3. prompt fills {context} + {question}
    4. llm generates grounded answer
    5. parser extracts string
    
    RunnablePassthrough.assign() runs the retriever in parallel
    with passing the question through — both happen simultaneously.
    """
    llm = get_llm()

    if retriever is None:
        # Day 11 skeleton — retriever not ready yet
        # Returns the chain structure for testing prompt + LLM only
        def mock_retriever(inputs):
            return "No context available yet. RAG pipeline activates Day 16."

        chain = (
            {
                "context" : mock_retriever,
                "question": RunnablePassthrough(),
            }
            | get_rag_prompt()
            | llm
            | output_parser
        )
    else:
        # Full RAG chain (Day 16)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context" : retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | get_rag_prompt()
            | llm
            | output_parser
        )

    return chain


# ── Smart Router: picks the right chain based on question type ────────────────
def route_question(question: str) -> str:
    """
    Classifies the question to pick the right chain.
    
    Why: You have 3 specialized chains. You could make the user
    specify which one, but routing automatically is better UX.
    
    Simple keyword routing is enough for Day 11.
    Day 16 upgrades this to semantic routing via embeddings.
    """
    q = question.lower()

    inventory_keywords = [
        "restock", "re-stock", "reorder", "re-order", "stock", "inventory",
        "dead", "alert", "running out", "order", "buy", "purchase", "supplier",
        "units left", "low", "shortage", "refill", "what to order", "need to buy"
    ]
    analytics_keywords = [
        "revenue", "profit", "margin", "sales", "kpi", "sell", "selling",
        "sold", "best seller", "best selling", "best-selling", "bestseller",
        "top seller", "top product", "top products", "most popular", "worst",
        "slowest", "fastest", "earning", "income", "category", "performance",
        "money", "pkr"
    ]

    if any(kw in q for kw in inventory_keywords):
        return "inventory"
    elif any(kw in q for kw in analytics_keywords):
        return "analytics"
    else:
        return "base"


def get_routed_chain(question: str):
    """
    Returns the correct chain based on question classification.
    
    This is the single entry point your FastAPI /rag/ask
    endpoint will call — it doesn't need to know which
    chain handles the question.
    """
    route = route_question(question)
    log.info("Question routed to: %s chain", route)

    if route == "inventory":
        return get_inventory_chain(), route
    elif route == "analytics":
        return get_analytics_chain(), route
    else:
        return get_base_chain(), route