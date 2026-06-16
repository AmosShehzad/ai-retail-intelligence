"""
RAG AI Assistant Endpoints — fully implemented Day 11+.
Placeholder endpoint added now so the router is registered
and testable from Day 8 onwards.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG Assistant"])


class QueryRequest(BaseModel):
    question: str
    top_k   : int = 5


@router.post("/ask")
async def ask_question(body: QueryRequest):
    """
    RAG-powered natural language query endpoint.
    Placeholder until Day 11 — returns a holding message.
    """
    return {
        "success" : True,
        "question": body.question,
        "answer"  : "RAG pipeline not yet implemented. Coming Day 11.",
        "sources" : [],
    }


@router.get("/status")
async def rag_status():
    return {"status": "placeholder", "available": False}