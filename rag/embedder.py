"""
Day 13: Local Embedding Generation & FAISS Storage

What this file does:
1. Loads your Day 12 synthesized documents
2. Converts each document's text into a vector using
   HuggingFace Sentence Transformers (runs 100% locally)
3. Stores all vectors in a FAISS index
4. Saves the FAISS index to disk so Day 14 can load and search it

Key concepts:
- Embedding model: converts text → list of numbers (vector)
- FAISS index: stores vectors, finds nearest neighbors fast
- Persistence: saving index to disk means you don't re-embed
  every time the server starts (embedding takes minutes)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import logging
import pickle
import time
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from rag.document_synthesizer import (
    run_document_synthesis,
    load_documents_from_disk,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Configuration constants ───────────────────────────────────────────────────

# The embedding model we use locally (no API key needed)
# "all-MiniLM-L6-v2" is a small, fast model that:
# - Produces 384-dimensional vectors
# - Downloads once (~90MB), then runs fully offline
# - Excellent for semantic similarity tasks
# - Perfect for portfolio projects (fast on CPU)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where FAISS index files are saved on disk
# Two files are saved:
# - faiss_index/retail.index  → the actual FAISS vector index
# - faiss_index/retail.pkl    → the document texts + metadata
#   (FAISS stores only vectors, not the original text,
#    so we save documents separately in a pickle file)
FAISS_INDEX_PATH = "faiss_index/retail.index"
FAISS_DOCS_PATH  = "faiss_index/retail.pkl"

# Embedding dimension (must match what all-MiniLM-L6-v2 produces)
EMBEDDING_DIM = 384


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Loads the HuggingFace Sentence Transformer model.

    First run: downloads model from HuggingFace Hub (~90MB).
    Subsequent runs: loads from local cache (fast, no internet).

    SentenceTransformer is a pre-trained model that understands
    English (and some Urdu/multilingual) sentence meaning.
    It was trained on millions of sentence pairs so it already
    understands that "restock" and "reorder" are similar concepts.
    We do NOT need to train it — just use it as-is.
    """
    log.info("Loading embedding model: %s", model_name)
    start = time.time()

    # This downloads the model on first run, then caches it locally
    model = SentenceTransformer(model_name)

    duration = round(time.time() - start, 2)
    log.info("Model loaded in %.2fs", duration)

    return model


def embed_texts(
    texts     : List[str],
    model     : SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Converts a list of text strings into a 2D numpy array of vectors.

    Input:  ["Tapal Tea is low on stock", "Surf Excel 500g..."]
            → list of N strings

    Output: numpy array of shape (N, 384)
            → each row is one document's vector

    batch_size=32:
        Processes 32 documents at a time instead of all at once.
        Why: prevents memory overflow on large document sets.
        Your 294 documents fit easily, but this is good practice.

    normalize_embeddings=True:
        Makes all vectors unit length (length = 1.0).
        Why: enables cosine similarity via dot product, which is
        faster than computing actual cosine similarity separately.
        FAISS's IndexFlatIP (Inner Product) index then gives
        cosine similarity scores directly.
    """
    log.info("Embedding %d texts with batch_size=%d...", len(texts), batch_size)
    start = time.time()

    # encode() is the main SentenceTransformer method
    # It returns a numpy array automatically
    embeddings = model.encode(
        texts,
        batch_size          = batch_size,
        show_progress_bar   = show_progress,
        normalize_embeddings= True,   # unit normalize for cosine similarity
        convert_to_numpy    = True,   # return numpy array (required by FAISS)
    )

    duration = round(time.time() - start, 2)
    log.info(
        "Embedding complete: %d vectors, shape=%s, time=%.2fs",
        len(embeddings), embeddings.shape, duration
    )

    return embeddings  # shape: (num_documents, 384)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from the embedding vectors.

    FAISS index types (we use IndexFlatIP):
    - IndexFlatL2  : exact search using Euclidean distance
    - IndexFlatIP  : exact search using Inner Product (cosine sim
                     when vectors are normalized — our case)
    - IndexIVFFlat : approximate search, faster for 1M+ vectors
    - IndexHNSW    : approximate search, very fast but memory heavy

    We use IndexFlatIP because:
    - Our corpus is ~300 documents (small)
    - Exact search on 300 vectors is near-instant
    - No approximation errors
    - Simple and reliable for portfolio use

    For 1M+ documents, switch to IndexIVFFlat.

    Steps:
    1. Create empty index with correct dimension (384)
    2. Convert embeddings to float32 (FAISS requirement)
    3. Add all vectors to the index
    """
    log.info("Building FAISS IndexFlatIP with dim=%d...", EMBEDDING_DIM)

    # Step 1: Create the index
    # faiss.IndexFlatIP = Flat (no compression) + Inner Product similarity
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Step 2: FAISS requires float32 specifically
    # numpy default is float64, so we must convert
    embeddings_float32 = embeddings.astype(np.float32)

    # Step 3: Add all vectors to the index
    # After this, the index contains all 294 document vectors
    index.add(embeddings_float32)

    log.info(
        "FAISS index built: %d vectors stored, index type=%s",
        index.ntotal,               # total vectors in index
        type(index).__name__        # IndexFlatIP
    )

    return index


def save_faiss_index(
    index    : faiss.Index,
    documents: List[Document],
) -> None:
    """
    Saves FAISS index + documents to disk.

    Why two separate files:
    - FAISS can only store vectors (numbers), not text
    - We need the original text to return readable results
    - So we save vectors in .index file (FAISS format)
    - And save documents in .pkl file (Python pickle format)
    - On retrieval: FAISS finds vector indices → we look up
      the same indices in our document list

    File structure after saving:
    faiss_index/
    ├── retail.index  ← binary FAISS index (fast vector search)
    └── retail.pkl    ← Python list of Document objects
    """
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Save FAISS index
    # faiss.write_index saves in FAISS's own binary format
    faiss.write_index(index, FAISS_INDEX_PATH)
    log.info("FAISS index saved → %s", FAISS_INDEX_PATH)

    # Save documents as pickle
    # We save the full Document objects (page_content + metadata)
    # so retrieval can return both the text and structured metadata
    with open(FAISS_DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    log.info("Documents saved → %s", FAISS_DOCS_PATH)

    # Log file sizes for visibility
    index_size = os.path.getsize(FAISS_INDEX_PATH) / 1024
    docs_size  = os.path.getsize(FAISS_DOCS_PATH) / 1024
    log.info(
        "File sizes: index=%.1fKB, docs=%.1fKB",
        index_size, docs_size
    )


def load_faiss_index() -> Tuple[faiss.Index, List[Document]]:
    """
    Loads FAISS index + documents from disk.

    Called by:
    - Day 14 retriever (every time a query comes in)
    - Day 16 RAG pipeline (at server startup)

    Returns tuple: (faiss_index, documents_list)
    The index and documents are always in sync because
    documents[i] corresponds to the vector at index position i.
    This positional alignment is maintained by always adding
    vectors and documents in the same order.

    Raises FileNotFoundError if index hasn't been built yet.
    (Solution: run this file first to build the index)
    """
    # Check both files exist before trying to load
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run rag/embedder.py first to build the index."
        )

    if not os.path.exists(FAISS_DOCS_PATH):
        raise FileNotFoundError(
            f"Documents file not found at {FAISS_DOCS_PATH}. "
            "Run rag/embedder.py first to build the index."
        )

    # Load FAISS index from binary file
    index = faiss.read_index(FAISS_INDEX_PATH)
    log.info(
        "FAISS index loaded: %d vectors, dim=%d",
        index.ntotal, EMBEDDING_DIM
    )

    # Load documents from pickle file
    with open(FAISS_DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    log.info("Documents loaded: %d documents", len(documents))

    # Sanity check: index vector count must match document count
    # If they differ, the index and documents are out of sync
    # (this shouldn't happen if you always use save_faiss_index())
    if index.ntotal != len(documents):
        raise ValueError(
            f"Index/document mismatch: "
            f"{index.ntotal} vectors vs {len(documents)} documents. "
            "Rebuild the index by running rag/embedder.py."
        )

    return index, documents


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: SIMILARITY SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_index(
    query     : str,
    index     : faiss.Index,
    documents : List[Document],
    model     : SentenceTransformer,
    top_k     : int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[Document, float]]:
    """
    Searches the FAISS index for documents most similar to the query.

    Process:
    1. Convert query text to a vector using the same embedding model
    2. Ask FAISS to find the top_k closest vectors in the index
    3. Return the corresponding Document objects + their scores

    score_threshold=0.3:
        Only return documents with similarity score >= 0.3.
        Why: prevents returning irrelevant documents when no
        good match exists. A score of 0.3 means "at least
        somewhat related". Score of 1.0 = identical.
        Scores below 0.3 = probably unrelated to the query.

    top_k=5:
        Return the 5 most similar documents.
        More context = better LLM answers, but slower.
        5 is the sweet spot for your corpus size.

    Returns:
        List of (Document, score) tuples, sorted by score descending
        Example: [(doc_about_tapal_tea, 0.89), (doc_about_inventory, 0.72), ...]
    """
    # Step 1: Embed the query using the same model used for documents
    # Must use same model — different models produce incomparable vectors
    query_vector = model.encode(
        [query],                    # encode() expects a list
        normalize_embeddings=True,  # must match how documents were embedded
        convert_to_numpy=True,
    ).astype(np.float32)            # FAISS requires float32

    # Step 2: Search FAISS
    # index.search() returns:
    # - scores: numpy array of similarity scores shape (1, top_k)
    # - indices: numpy array of document indices shape (1, top_k)
    # The [0] indexing flattens from 2D to 1D (we only have 1 query)
    scores, indices = index.search(query_vector, top_k)
    scores  = scores[0]    # shape: (top_k,)  e.g. [0.89, 0.72, 0.65, ...]
    indices = indices[0]   # shape: (top_k,)  e.g. [42, 7, 183, ...]

    # Step 3: Build results list
    results = []

    for score, idx in zip(scores, indices):
        # idx=-1 means FAISS couldn't find enough results
        # (happens when top_k > number of documents in index)
        if idx == -1:
            continue

        # Filter by score threshold
        # float(score) converts numpy float32 to Python float
        if float(score) < score_threshold:
            continue

        results.append((
            documents[idx],   # the actual Document object
            float(score),     # similarity score 0.0 to 1.0
        ))

    return results  # sorted by score descending (FAISS returns in order)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: MASTER BUILD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_vector_store(
    force_rebuild   : bool = False,
    synthesize_fresh: bool = False,
) -> Tuple[faiss.Index, List[Document], SentenceTransformer]:
    """
    Master function: builds or loads the complete vector store.

    Decision logic:
    1. If index exists on disk AND force_rebuild=False → just load it
       (fast: ~1 second)
    2. If index doesn't exist OR force_rebuild=True → build from scratch
       (slow: ~2-5 minutes first time due to model download + embedding)

    force_rebuild=True use cases:
    - New products added to DB
    - Daily refresh via scheduler
    - After re-running document synthesis

    synthesize_fresh=True:
    - Re-runs document synthesis before embedding
    - Use when DB data has changed significantly
    - Otherwise, loads existing synthesized_documents.json

    Returns: (faiss_index, documents, embedding_model)
    All three are needed for search operations.
    """
    index_exists = (
        os.path.exists(FAISS_INDEX_PATH) and
        os.path.exists(FAISS_DOCS_PATH)
    )

    # ── Fast path: load existing index ────────────────────────────────────
    if index_exists and not force_rebuild:
        log.info("Loading existing FAISS index from disk...")
        model             = load_embedding_model()
        index, documents  = load_faiss_index()
        return index, documents, model

    # ── Slow path: build fresh index ──────────────────────────────────────
    log.info("=" * 55)
    log.info("BUILDING VECTOR STORE FROM SCRATCH")
    log.info("=" * 55)

    start_total = time.time()

    # Step 1: Load or synthesize documents
    if synthesize_fresh:
        log.info("Step 1/4: Running fresh document synthesis...")
        documents = run_document_synthesis(save_to_disk=True)
    else:
        log.info("Step 1/4: Loading existing synthesized documents...")
        documents = load_documents_from_disk()

    log.info("  → %d documents ready for embedding", len(documents))

    # Step 2: Load embedding model
    log.info("Step 2/4: Loading embedding model...")
    model = load_embedding_model()

    # Step 3: Embed all document texts
    log.info("Step 3/4: Generating embeddings...")

    # Extract just the text content from Document objects
    # model.encode() needs plain strings, not Document objects
    texts = [doc.page_content for doc in documents]

    embeddings = embed_texts(texts, model, batch_size=32)

    # Step 4: Build and save FAISS index
    log.info("Step 4/4: Building and saving FAISS index...")
    index = build_faiss_index(embeddings)
    save_faiss_index(index, documents)

    total_time = round(time.time() - start_total, 2)
    log.info("=" * 55)
    log.info(
        "VECTOR STORE BUILT: %d vectors in %.2fs",
        index.ntotal, total_time
    )
    log.info("=" * 55)

    return index, documents, model


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: INSPECTION & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_vector_store(
    index    : faiss.Index,
    documents: List[Document],
    model    : SentenceTransformer,
) -> None:
    """
    Runs test searches to validate the vector store works correctly.

    Runs 5 real questions and prints the top retrieved document.
    This is your quality check before Day 14 builds the retriever.

    What to look for:
    - "restock Tapal Tea" → should retrieve Tapal Tea inventory alert
    - "profit margin" → should retrieve analytics or product docs
    - "dead stock" → should retrieve dead stock alert documents
    If irrelevant documents appear, adjust score_threshold or
    re-check document synthesis quality.
    """
    TEST_QUERIES = [
        "Which products should I restock urgently?",
        "What is the profit margin for tea products?",
        "Which products are dead stock wasting shelf space?",
        "How much revenue did the store make this week?",
        "What is the best selling product in the store?",
    ]

    print("\n" + "=" * 60)
    print("VECTOR STORE VALIDATION — Test Searches")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\nQUERY: {query}")
        print("─" * 50)

        results = search_index(
            query          = query,
            index          = index,
            documents      = documents,
            model          = model,
            top_k          = 3,
            score_threshold= 0.2,  # lower threshold for validation
        )

        if not results:
            print("  No results above threshold.")
            continue

        for i, (doc, score) in enumerate(results, 1):
            doc_type     = doc.metadata.get("doc_type", "unknown")
            product_name = doc.metadata.get("product_name", "")
            category     = doc.metadata.get("category", "")

            # Show first 150 chars of content
            content_preview = doc.page_content[:150].replace("\n", " ")

            print(f"  Result {i} | Score: {score:.4f} | "
                  f"Type: {doc_type} | {product_name or category}")
            print(f"  Preview: {content_preview}...")

    print("\n" + "=" * 60)


def print_index_stats(
    index    : faiss.Index,
    documents: List[Document],
) -> None:
    """
    Prints statistics about the built vector store.
    Useful for README documentation and interviews.
    """
    # Count documents by type
    type_counts = {}
    for doc in documents:
        t = doc.metadata.get("doc_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n" + "=" * 55)
    print("VECTOR STORE STATISTICS")
    print("=" * 55)
    print(f"  Total vectors in FAISS : {index.ntotal}")
    print(f"  Vector dimensions      : {EMBEDDING_DIM}")
    print(f"  Index type             : {type(index).__name__}")
    print(f"  Embedding model        : {EMBEDDING_MODEL_NAME}")
    print(f"\n  Documents by type:")
    for dtype, count in sorted(type_counts.items()):
        print(f"    {dtype:30s}: {count}")
    print(f"\n  Index file  : {FAISS_INDEX_PATH}")
    print(f"  Docs file   : {FAISS_DOCS_PATH}")

    # File sizes
    if os.path.exists(FAISS_INDEX_PATH):
        size_kb = os.path.getsize(FAISS_INDEX_PATH) / 1024
        print(f"  Index size  : {size_kb:.1f} KB")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Running this file directly:
    1. Builds the complete vector store (or loads existing)
    2. Validates it with test searches
    3. Prints statistics

    First run: ~2-5 minutes (model download + embedding)
    Subsequent runs: ~30 seconds (model load + embedding)
    If index exists: ~5 seconds (just load from disk)
    """
    # Build vector store
    # force_rebuild=True ensures fresh build even if index exists
    # synthesize_fresh=True re-runs document synthesis
    index, documents, model = build_vector_store(
        force_rebuild   = True,
        synthesize_fresh= True,
    )

    # Print statistics
    print_index_stats(index, documents)

    # Validate with test searches
    validate_vector_store(index, documents, model)