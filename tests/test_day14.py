"""
Day 14 tests — validates calibrated retrieval logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import (
    RetailRetriever,
    create_retriever,
    get_retrieval_config,
    filter_documents_by_type,
    RETRIEVAL_CONFIG,
    DEFAULT_CONFIG,
)
from langchain_core.documents import Document


def test_retrieval_config_exists_for_all_routes():
    # Every route from Day 11's router must have a calibration entry
    assert "inventory" in RETRIEVAL_CONFIG
    assert "analytics" in RETRIEVAL_CONFIG
    assert "base"      in RETRIEVAL_CONFIG


def test_inventory_config_has_lower_threshold():
    # Inventory should be MORE permissive (catch more alerts)
    # than base queries (which should be strict/precise)
    inv_config  = get_retrieval_config("inventory")
    base_config = get_retrieval_config("base")
    assert inv_config["score_threshold"] < base_config["score_threshold"]


def test_inventory_config_has_higher_top_k():
    # Inventory should retrieve MORE documents than base
    inv_config  = get_retrieval_config("inventory")
    base_config = get_retrieval_config("base")
    assert inv_config["top_k"] > base_config["top_k"]


def test_unknown_route_falls_back_to_default():
    config = get_retrieval_config("nonexistent_route")
    assert config == DEFAULT_CONFIG


def test_filter_documents_by_type():
    docs = [
        (Document(page_content="a", metadata={"doc_type": "product"}), 0.9),
        (Document(page_content="b", metadata={"doc_type": "inventory_alert"}), 0.8),
        (Document(page_content="c", metadata={"doc_type": "inventory_alert"}), 0.7),
    ]
    filtered = filter_documents_by_type(docs, "inventory_alert")
    assert len(filtered) == 2
    assert all(doc.metadata["doc_type"] == "inventory_alert" for doc, _ in filtered)


def test_filter_with_none_returns_unchanged():
    docs = [
        (Document(page_content="a", metadata={"doc_type": "product"}), 0.9),
    ]
    filtered = filter_documents_by_type(docs, None)
    assert len(filtered) == len(docs)


def test_retriever_is_langchain_compatible():
    # Confirms the retriever can be used with .invoke() —
    # the standard LangChain interface required for LCEL chains
    retriever = create_retriever(force_rebuild=False)
    docs = retriever.invoke("What should I restock?")
    assert isinstance(docs, list)
    if docs:
        assert isinstance(docs[0], Document)


def test_inventory_query_returns_inventory_docs_only():
    retriever = create_retriever(force_rebuild=False)
    docs = retriever.invoke("What products need urgent restocking?")
    # Every returned doc should be an inventory_alert
    # (because of the doc_type_filter calibration)
    for doc in docs:
        assert doc.metadata.get("doc_type") == "inventory_alert"


def test_scores_returned_correctly():
    retriever = create_retriever(force_rebuild=False)
    results   = retriever.get_relevant_documents_with_scores(
        "What is my profit margin?"
    )
    for doc, score in results:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")