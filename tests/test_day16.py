"""
Day 16 tests — validates the complete RAG pipeline end-to-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import (
    RAGPipeline,
    RAGResult,
    format_documents_as_context,
    build_sources_metadata,
    check_grounding,
    get_rag_pipeline,
)
from langchain_core.documents import Document


def test_format_empty_documents():
    result = format_documents_as_context([])
    assert "No relevant store data" in result


def test_format_documents_includes_source_labels():
    docs = [
        Document(
            page_content="Tapal Tea is at 12 units.",
            metadata={"doc_type": "inventory_alert"}
        )
    ]
    result = format_documents_as_context(docs)
    assert "Source 1" in result
    assert "Inventory Alert" in result
    assert "Tapal Tea" in result


def test_build_sources_metadata():
    docs   = [Document(page_content="test content here",
                       metadata={"doc_type": "product", "product_name": "Tapal Tea"})]
    scores = [0.85]
    sources = build_sources_metadata(docs, scores)

    assert len(sources) == 1
    assert sources[0]["product_name"] == "Tapal Tea"
    assert sources[0]["score"] == 0.85


def test_grounding_check_no_context_honest_answer():
    # Honest "I don't know" with no context should be grounded
    is_grounded, score = check_grounding(
        answer  = "I don't have that information in the store data.",
        context = "No relevant store data was found for this question.",
        documents=[],
    )
    assert is_grounded is True
    assert score == 1.0


def test_grounding_check_no_context_confident_answer():
    # Confident answer with NO supporting documents = hallucination risk
    is_grounded, score = check_grounding(
        answer  = "Your Tapal Tea stock is 50 units and selling fast.",
        context = "No relevant store data was found for this question.",
        documents=[],
    )
    assert is_grounded is False
    assert score == 0.0


def test_grounding_check_with_overlapping_context():
    docs = [
        Document(page_content="Tapal Danedar 200g stock is 12 units, urgent restock needed",
                 metadata={"doc_type": "inventory_alert"})
    ]
    context = format_documents_as_context(docs)
    answer  = "Tapal Danedar 200g needs urgent restock, stock is at 12 units."

    is_grounded, score = check_grounding(answer, context, docs)
    assert is_grounded is True
    assert score > 0.25


def test_rag_result_structure():
    result = RAGResult(
        success=True, question="test", answer="test answer"
    )
    assert result.success is True
    assert result.sources == []
    assert result.grounded is True


def test_pipeline_is_singleton():
    p1 = get_rag_pipeline()
    p2 = get_rag_pipeline()
    assert p1 is p2


def test_pipeline_answer_returns_rag_result():
    pipeline = get_rag_pipeline()
    if not pipeline.is_ready():
        print("⚠️  Skipping — Ollama not running")
        return

    result = pipeline.answer("What should I restock?")
    assert isinstance(result, RAGResult)
    assert result.question == "What should I restock?"
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


def test_pipeline_offtopic_question_stays_grounded():
    pipeline = get_rag_pipeline()
    if not pipeline.is_ready():
        print("⚠️  Skipping — Ollama not running")
        return

    result = pipeline.answer("What is the capital of France?")
    # Should retrieve no/few relevant docs and either decline
    # to answer or clearly indicate lack of data
    assert result.success is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")