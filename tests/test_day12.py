import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from rag.document_synthesizer import (
    synthesize_product_documents,
    synthesize_category_documents,
    synthesize_inventory_documents,
    synthesize_analytics_documents,
    run_document_synthesis,
    load_documents_from_disk,
)
from langchain_core.documents import Document


def test_product_documents_structure():
    docs = synthesize_product_documents()
    assert len(docs) > 0
    doc = docs[0]
    assert isinstance(doc, Document)
    assert len(doc.page_content) > 50
    assert "product_id"   in doc.metadata
    assert "product_name" in doc.metadata
    assert "category"     in doc.metadata
    assert "doc_type"     in doc.metadata
    assert doc.metadata["doc_type"] == "product"


def test_product_documents_have_price_info():
    docs = synthesize_product_documents()
    for doc in docs[:5]:
        assert "Rs." in doc.page_content
        assert "margin" in doc.page_content.lower()


def test_category_documents_structure():
    docs = synthesize_category_documents()
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["doc_type"] == "category"
        assert "category" in doc.metadata
        assert "Revenue" in doc.page_content


def test_inventory_documents_have_urgency():
    docs = synthesize_inventory_documents()
    low_stock = [d for d in docs
                 if d.metadata.get("alert_type") == "low_stock"]
    if low_stock:
        for doc in low_stock[:3]:
            assert "urgency" in doc.metadata
            assert doc.metadata["urgency"] in ("CRITICAL", "HIGH", "MEDIUM")


def test_analytics_documents_structure():
    docs = synthesize_analytics_documents()
    assert len(docs) > 0
    types = [d.metadata.get("analytics_type") for d in docs]
    assert "kpi_overview" in types


def test_full_synthesis_and_save():
    docs = run_document_synthesis(save_to_disk=True)
    assert len(docs) > 0

    import os
    assert os.path.exists("data/processed/synthesized_documents.json")


def test_load_from_disk():
    run_document_synthesis(save_to_disk=True)
    docs = load_documents_from_disk()
    assert len(docs) > 0
    assert isinstance(docs[0], Document)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")