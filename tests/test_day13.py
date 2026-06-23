"""
Day 13 tests — validates FAISS index builds and searches correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
from rag.embedder import (
    load_embedding_model,
    embed_texts,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    search_index,
    build_vector_store,
    FAISS_INDEX_PATH,
    FAISS_DOCS_PATH,
    EMBEDDING_DIM,
)
from langchain_core.documents import Document


def test_embedding_model_loads():
    model = load_embedding_model()
    assert model is not None


def test_embed_texts_shape():
    # Embedding 3 texts should return shape (3, 384)
    model  = load_embedding_model()
    texts  = ["Tapal Tea is low on stock",
               "Surf Excel detergent",
               "Revenue this week is Rs. 50,000"]
    result = embed_texts(texts, model, show_progress=False)

    assert result.shape    == (3, EMBEDDING_DIM), \
        f"Expected (3, {EMBEDDING_DIM}), got {result.shape}"
    assert result.dtype    == np.float32


def test_embeddings_are_normalized():
    # Normalized vectors should have length ~1.0
    model      = load_embedding_model()
    texts      = ["test sentence one", "test sentence two"]
    embeddings = embed_texts(texts, model, show_progress=False)

    for vec in embeddings:
        length = np.linalg.norm(vec)
        assert abs(length - 1.0) < 0.001, \
            f"Vector not normalized: length={length}"


def test_faiss_index_builds():
    model      = load_embedding_model()
    texts      = [f"Document number {i}" for i in range(10)]
    embeddings = embed_texts(texts, model, show_progress=False)
    index      = build_faiss_index(embeddings)

    # Index should contain exactly 10 vectors
    assert index.ntotal == 10


def test_save_and_load_index():
    """Uses temp paths so it never overwrites the real FAISS index."""
    import tempfile
    import os
    import pickle
    import faiss as faiss_lib

    model      = load_embedding_model()
    docs       = [
        Document(page_content=f"Test document {i}",
                 metadata={"doc_type": "test", "id": i})
        for i in range(5)
    ]
    texts      = [d.page_content for d in docs]
    embeddings = embed_texts(texts, model, show_progress=False)
    index      = build_faiss_index(embeddings)

    # Use temp files — NEVER touch the real index
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_index = os.path.join(tmpdir, "test.index")
        tmp_docs  = os.path.join(tmpdir, "test.pkl")

        faiss_lib.write_index(index, tmp_index)
        with open(tmp_docs, "wb") as f:
            pickle.dump(docs, f)

        loaded_index = faiss_lib.read_index(tmp_index)
        with open(tmp_docs, "rb") as f:
            loaded_docs = pickle.load(f)

    assert loaded_index.ntotal == 5
    assert len(loaded_docs)    == 5
    print("✅ test_save_and_load_index passed (used temp path, real index untouched)")

def test_search_returns_relevant_result():
    model = load_embedding_model()
    docs  = [
        Document(page_content="Tapal Tea is running low on stock needs reorder",
                 metadata={"doc_type": "inventory_alert", "product_name": "Tapal Tea"}),
        Document(page_content="The weather in Lahore is very hot today",
                 metadata={"doc_type": "unrelated"}),
        Document(page_content="Surf Excel detergent has high profit margin",
                 metadata={"doc_type": "product", "product_name": "Surf Excel"}),
    ]
    texts      = [d.page_content for d in docs]
    embeddings = embed_texts(texts, model, show_progress=False)
    index      = build_faiss_index(embeddings)

    results = search_index(
        query          = "Which product needs restocking?",
        index          = index,
        documents      = docs,
        model          = model,
        top_k          = 2,
        score_threshold= 0.1,
    )

    assert len(results) > 0
    # Top result should be Tapal Tea (restock context), not weather
    top_doc, top_score = results[0]
    assert "Tapal" in top_doc.page_content or "reorder" in top_doc.page_content
    assert top_score > 0.1


def test_build_vector_store_full():
    # Full integration test
    index, documents, model = build_vector_store(
        force_rebuild   = False,
        synthesize_fresh= False,
    )
    assert index.ntotal    > 0
    assert len(documents)  > 0
    assert index.ntotal   == len(documents)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name} passed")