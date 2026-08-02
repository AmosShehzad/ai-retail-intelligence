"""
Day 16 update: RAG router now uses the COMPLETE pipeline.
This replaces the Day 11 placeholder logic entirely.
"""

import logging
from fastapi import APIRouter, HTTPException
from api.schemas.rag import RAGQueryRequest, RAGQueryResponse, SourceDocument

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG Assistant"])


@router.post("/ask", response_model=RAGQueryResponse)
async def ask_question(body: RAGQueryRequest):
    """
    THE MAIN RAG ENDPOINT — full pipeline from Day 16.

    Flow: question → retriever → context → prompt → Ollama →
    hallucination check → structured answer with sources.
    """
    try:
        from rag.pipeline import get_rag_pipeline

        pipeline = get_rag_pipeline()
        result   = await pipeline.aanswer(body.question)

        if not result.success:
            raise HTTPException(status_code=503, detail=result.error)

        # Log query to database for audit (from Day 2's schema)
        from database.queries import log_query
        log_query(body.question, result.answer)

        return RAGQueryResponse(
            success  = True,
            question = result.question,
            answer   = result.answer,
            sources  = [
                SourceDocument(
                    content     = s["preview"],
                    score       = s["score"],
                    product_name= s.get("product_name"),
                    category    = s.get("category"),
                )
                for s in result.sources
            ],
            model    = pipeline.llama.provider,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("RAG endpoint failed for question: %s", body.question)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def rag_status():
    """Reports full pipeline readiness — retriever AND LLM."""
    from rag.pipeline import get_rag_pipeline

    pipeline = get_rag_pipeline()

    return {
        "status"           : "online" if pipeline.is_ready() else "offline",
        "available"        : pipeline.is_ready(),
        "documents_indexed": len(pipeline.retriever.documents),
        "llama_ready"      : pipeline.llama.is_ready(),
    }