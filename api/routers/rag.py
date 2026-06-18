"""
Day 10 update: RAG router uses Pydantic request/response schemas.
"""

import logging
from fastapi import APIRouter, HTTPException
from api.schemas.rag import RAGQueryRequest, RAGQueryResponse

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG Assistant"])


@router.post("/ask", response_model=RAGQueryResponse)
async def ask_question(body: RAGQueryRequest):
    """
    FastAPI automatically validates body against RAGQueryRequest.
    Empty question → 422 error before this function even runs.
    question > 500 chars → 422 error automatically.
    No manual validation code needed here at all.
    """
    return RAGQueryResponse(
        success  = True,
        question = body.question,
        answer   = "RAG pipeline not yet implemented. Coming Day 11.",
        sources  = [],
    )


@router.get("/status")
async def rag_status():
    return {"status": "placeholder", "available": False}