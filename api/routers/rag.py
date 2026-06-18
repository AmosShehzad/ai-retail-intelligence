"""
Day 11 update: RAG router now calls real LangChain chains.
No more placeholder response.
"""

import logging
import json
from fastapi import APIRouter, HTTPException
from api.schemas.rag import RAGQueryRequest, RAGQueryResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG Assistant"])


@router.post("/ask", response_model=RAGQueryResponse)
async def ask_question(body: RAGQueryRequest):
    """
    Routes question to correct LangChain chain.
    Injects relevant store data as context.
    Returns AI-generated answer grounded in your data.
    """
    try:
        from rag.chains import get_routed_chain, route_question
        from analytics.engines import get_store_kpis, get_category_margins
        from analytics.inventory import (
            get_inventory_health_summary,
            get_low_stock_alerts,
        )

        question = body.question
        route    = route_question(question)
        chain, _ = get_routed_chain(question)

        # Build context based on route
        # Why: inject only relevant data to keep prompt focused
        if route == "analytics":
            store_context = json.dumps(get_store_kpis(), indent=2)
            answer = await chain.ainvoke({
                "store_context": store_context,
                "question"     : question,
            })

        elif route == "inventory":
            inventory_context = json.dumps(
                get_inventory_health_summary(), indent=2
            )
            alerts_df         = get_low_stock_alerts()
            alerts_context    = alerts_df.head(10).to_string(index=False) \
                                if not alerts_df.empty \
                                else "No low stock alerts currently."

            answer = await chain.ainvoke({
                "inventory_context": inventory_context,
                "alerts_context"   : alerts_context,
                "question"         : question,
            })

        else:
            answer = await chain.ainvoke({"question": question})

        # Log query to database for audit
        from database.queries import log_query
        log_query(question, answer)

        return RAGQueryResponse(
            success  = True,
            question = question,
            answer   = answer,
            sources  = [],          # FAISS sources added Day 16
            model    = "phi3",
        )

    except Exception as e:
        log.exception("RAG chain failed for question: %s", body.question)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def rag_status():
    """Checks if Ollama is reachable."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            phi3_ready = any("phi3" in m for m in models)

        return {
            "status"       : "online" if phi3_ready else "model_missing",
            "ollama"       : "connected",
            "phi3_ready" : phi3_ready,
            "available"    : phi3_ready,
            "models"       : models,
        }
    except Exception as e:
        return {
            "status"   : "offline",
            "ollama"   : "not reachable",
            "available": False,
            "error"    : str(e),
        }