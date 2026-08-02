"""
Pydantic schemas for RAG assistant endpoints.
Validates natural language queries and structured AI responses.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RAGQueryRequest(BaseModel):
    """
    What the client sends when asking the AI a question.
    
    question: the natural language query
    top_k: how many context documents to retrieve (more = richer context,
           but slower response). Capped at 20 to prevent overloading Ollama.
    
    @field_validator ensures empty questions are rejected immediately
    before they waste RAG pipeline resources.
    """
    question : str = Field(
        ...,
        min_length = 3,
        max_length = 500,
        description="Natural language question about your store"
    )
    top_k    : int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context documents to retrieve"
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        return v.strip()


class SourceDocument(BaseModel):
    """
    One retrieved context document used to generate the answer.
    Showing sources lets the user verify the AI's answer.
    """
    content      : str
    score        : Optional[float] = None
    product_name : Optional[str]   = None
    category     : Optional[str]   = None


class RAGQueryResponse(BaseModel):
    """
    What the AI returns after answering a question.
    
    sources: the actual DB records the AI used to form its answer.
    This is what prevents hallucination — the answer is grounded
    in these specific retrieved documents.
    """
    success  : bool
    question : str
    answer   : str
    sources  : List[SourceDocument] = []
    model    : str = "llm"

    class Config:
        from_attributes = True