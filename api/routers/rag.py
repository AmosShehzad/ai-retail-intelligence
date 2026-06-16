"""
RAG AI assistant endpoints — filled on Day 11+.
Natural language queries answered by LangChain + Ollama.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=["RAG Assistant"])


@router.get("/health-check")
async def rag_health():
    return {"status": "rag router online"}