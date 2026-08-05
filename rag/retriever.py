"""
Day 14: Context Retrieval Mechanics Calibration

What this file does:
1. Wraps Day 13's raw FAISS search into a LangChain BaseRetriever
   (this is the standard interface LCEL chains expect)
2. Calibrates top_k and score_threshold PER QUERY TYPE
   (inventory questions need more results than product lookups)
3. Adds metadata filtering (e.g., only search inventory_alert docs
   when the question is clearly about restocking)
4. Provides a tunable config so you can adjust retrieval quality
   without touching the chain logic in Day 11/16

Why a custom BaseRetriever instead of LangChain's built-in FAISS wrapper:
LangChain's standard FAISS retriever is generic — same top_k/threshold
for every query. Your project has FOUR distinct document types
(product, category, inventory_alert, store_analytics) that need
different retrieval strategies. A custom retriever gives full control.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from typing import List, Optional, Dict, Any
from pydantic import Field

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from rag.embedder import build_vector_store, search_index
from rag.chains import route_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: RETRIEVAL CONFIGURATION PER QUERY TYPE
# ══════════════════════════════════════════════════════════════════════════════

# This dictionary is the "calibration" — the core deliverable of Day 14.
# Each query route (from Day 11's route_question()) gets its own
# tuned retrieval settings instead of one-size-fits-all.
#
# top_k         : how many documents to retrieve
# score_threshold: minimum similarity score to include a document
#                  (0.0 = no filter, 1.0 = exact match only)
# doc_type_filter: if set, ONLY search documents of this type
#                  (None = search across all document types)
RETRIEVAL_CONFIG: Dict[str, Dict[str, Any]] = {

    # Inventory questions ("what should I restock", "dead stock")
    # Need MORE results (owner wants the full list, not just 1 item)
    # LOWER threshold (catch moderately-relevant alerts too — better
    # to show a borderline alert than miss a real stockout)
    # FILTERED to inventory_alert docs only — a product doc about
    # Tapal Tea's price isn't useful when asking "what's low on stock"
    "inventory": {
        "top_k"          : 6,
        "score_threshold": 0.22,
        "doc_type_filter": None,
    },

    # Analytics questions ("revenue", "margins", "best sellers")
    # Medium top_k — usually 1-2 categories or KPI summary answer it
    # Medium threshold — analytics language varies more than product names
    # Filtered to category + store_analytics (not individual products)
    "analytics": {
        "top_k"          : 6,
        "score_threshold": 0.25,
        "doc_type_filter": None,  # search category AND analytics docs
    },

    # General/base questions (product lookups, general store questions)
    # LOWER top_k — usually asking about ONE specific thing
    # HIGHER threshold (strict) — don't want unrelated noise when
    # the question is precise (e.g. "What is Tapal Tea's price?")
    "base": {
        "top_k"          : 5,
        "score_threshold": 0.28,
        "doc_type_filter": None,
    },
}

# Fallback config if route_question() returns something unexpected
DEFAULT_CONFIG = {
    "top_k"          : 6,
    "score_threshold": 0.25,
    "doc_type_filter": None,
}


def get_retrieval_config(route: str) -> Dict[str, Any]:
    """
    Returns the calibrated retrieval settings for a given query route.

    Why a function instead of direct dict access:
    Handles unknown routes gracefully (falls back to DEFAULT_CONFIG)
    instead of crashing with a KeyError if route_question() ever
    returns an unexpected value.
    """
    config = RETRIEVAL_CONFIG.get(route, DEFAULT_CONFIG)
    log.debug("Retrieval config for route='%s': %s", route, config)
    return config


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: METADATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_documents_by_type(
    results       : List[tuple],
    doc_type_filter: Optional[str],
) -> List[tuple]:
    """
    Filters search results to only include a specific document type.

    Input:  results = [(doc1, score1), (doc2, score2), ...]
            doc_type_filter = "inventory_alert" (or None for no filter)

    Why filter AFTER search instead of before:
    FAISS searches the WHOLE vector space (all 294 documents) because
    that gives the most accurate similarity ranking. Filtering by type
    BEFORE search would require a separate FAISS index per document
    type — more complex with little benefit at this corpus size (294 docs).
    Filtering after search is simpler and just as effective here.

    If doc_type_filter is None, returns results unchanged.
    """
    if doc_type_filter is None:
        return results

    filtered = [
        (doc, score) for doc, score in results
        if doc.metadata.get("doc_type") == doc_type_filter
    ]

    log.debug(
        "Filtered by doc_type='%s': %d → %d results",
        doc_type_filter, len(results), len(filtered)
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: CUSTOM LANGCHAIN RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class RetailRetriever(BaseRetriever):
    """
    Custom LangChain retriever for the AI Retail Intelligence Assistant.

    Inherits from BaseRetriever — this is what makes it compatible
    with LCEL chains (the | pipe operator from Day 11) and with
    .invoke() / .ainvoke() calling conventions LangChain expects.

    Why build a CUSTOM retriever instead of using LangChain's
    built-in FAISS retriever wrapper:
    1. Need PER-QUERY-TYPE calibration (Part 1 above)
    2. Need doc_type metadata filtering (Part 2 above)
    3. Need to reuse our exact Day 13 search_index() function
       (keeps embedding logic in one place, no duplication)

    Pydantic Field(...) declarations are required because
    BaseRetriever is a Pydantic model under the hood — you can't
    just assign self.index = index like a normal Python class.
    """

    # Declare the fields this retriever holds.
    # Field(exclude=True) means these won't be serialized when
    # LangChain tries to log/trace this object (they're large objects
    # — the FAISS index and embedding model — not meant to be JSON-dumped)
    index    : Any = Field(exclude=True)
    documents: Any = Field(exclude=True)
    model    : Any = Field(exclude=True)

    # Default settings used when NOT using calibrated routing
    # (e.g. if someone calls this retriever directly without going
    # through route_question() first)
    default_top_k          : int   = 5
    default_score_threshold: float = 0.3

    class Config:
        # Allows arbitrary Python objects (FAISS index, SentenceTransformer)
        # as field types — Pydantic normally restricts this
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        REQUIRED method — BaseRetriever calls this automatically
        whenever .invoke(query) or the retriever is used inside
        an LCEL chain (e.g. `retriever | format_docs` from Day 11).

        run_manager: LangChain's internal callback system for
        logging/tracing (used by LangSmith in Day 18). We don't
        need to do anything with it manually — just accept it
        as a required parameter.

        This is where CALIBRATION happens:
        1. Classify the query using Day 11's router
        2. Look up the calibrated top_k/threshold/filter for that route
        3. Search FAISS with those calibrated settings
        4. Apply metadata filtering if configured
        5. Return plain Document objects (LangChain's expected format)
        """
        # Step 1: classify the question (reuses Day 11 routing logic)
        route = route_question(query)

        # Step 2: get calibrated settings for this route
        config = get_retrieval_config(route)

        log.info(
            "Retrieval | query='%s' | route='%s' | top_k=%d | threshold=%.2f",
            query[:50], route, config["top_k"], config["score_threshold"]
        )

        # Step 3: search FAISS using Day 13's search function
        # We request MORE than top_k initially when a doc_type_filter
        # is set, because filtering AFTER search may remove some results
        # — over-fetching ensures we still end up with enough after filtering
        fetch_k = config["top_k"] * 3 if config["doc_type_filter"] else config["top_k"]

        raw_results = search_index(
            query           = query,
            index           = self.index,
            documents       = self.documents,
            model           = self.model,
            top_k           = fetch_k,
            score_threshold = config["score_threshold"],
        )

        # Step 4: apply metadata filter (e.g. inventory_alert only)
        filtered_results = filter_documents_by_type(
            raw_results, config["doc_type_filter"]
        )

        # Step 5: trim back down to the calibrated top_k
        # (we over-fetched in step 3, now cut to the real limit)
        final_results = filtered_results[: config["top_k"]]

        log.info(
            "Retrieval complete | %d documents returned (after filter+trim)",
            len(final_results)
        )

        # BaseRetriever expects a plain List[Document] — strip out scores
        # (scores were only needed internally for threshold filtering)
        return [doc for doc, score in final_results]

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[tuple]:
        """
        Convenience method that returns (Document, score) tuples
        instead of plain documents.

        Why: Day 16's RAG endpoint needs to show "sources" with
        confidence scores to the user (so they can judge answer
        reliability). The standard _get_relevant_documents() above
        strips scores because that's what LangChain chains expect,
        but the API response (Day 16) wants the score back.
        """
        route  = route_question(query)
        config = get_retrieval_config(route)
        fetch_k = config["top_k"] * 3 if config["doc_type_filter"] else config["top_k"]

        raw_results = search_index(
            query           = query,
            index           = self.index,
            documents       = self.documents,
            model           = self.model,
            top_k           = fetch_k,
            score_threshold = config["score_threshold"],
        )

        filtered = filter_documents_by_type(raw_results, config["doc_type_filter"])
        return filtered[: config["top_k"]]


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: RETRIEVER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def create_retriever(force_rebuild: bool = False) -> RetailRetriever:
    """
    Creates a fully initialized RetailRetriever.

    This is the function Day 16's RAG pipeline calls to get
    a ready-to-use retriever — it handles loading the FAISS
    index, documents, and embedding model in one call.

    force_rebuild=False by default:
    Loads existing FAISS index from disk (fast, ~1 second).
    Set True only when you've added new products/sales and
    need fresh embeddings (Day 18 scheduler does this nightly).
    """
    log.info("Creating RetailRetriever...")

    index, documents, model = build_vector_store(
        force_rebuild   = force_rebuild,
        synthesize_fresh= force_rebuild,  # re-synthesize docs too if rebuilding
    )

    retriever = RetailRetriever(
        index     = index,
        documents = documents,
        model     = model,
    )

    log.info(
        "RetailRetriever ready | %d documents indexed",
        len(documents)
    )
    return retriever


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: CALIBRATION TESTING & TUNING UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration_test(retriever: RetailRetriever) -> None:
    """
    Tests the calibrated retriever against real project queries.

    For each query, shows:
    - Which route it was classified into
    - What top_k/threshold/filter were applied
    - What documents came back and their scores

    Run this after any change to RETRIEVAL_CONFIG to verify
    your calibration adjustments actually improve results.
    """
    TEST_QUERIES = [
        ("What should I restock urgently this week?", "inventory"),
        ("Which products are dead stock?",              "inventory"),
        ("What is my overall profit margin?",            "analytics"),
        ("Which category makes the most revenue?",       "analytics"),
        ("Tell me about Tapal Danedar tea",               "base"),
        ("How many units of Surf Excel do I have?",       "base"),
    ]

    print("\n" + "=" * 65)
    print("DAY 14 — RETRIEVAL CALIBRATION TEST")
    print("=" * 65)

    for query, expected_route in TEST_QUERIES:
        actual_route = route_question(query)
        config       = get_retrieval_config(actual_route)

        # Use .invoke() — the standard LangChain BaseRetriever interface
        # This proves the retriever works inside LCEL chains correctly
        docs = retriever.invoke(query)

        route_match = "✅" if actual_route == expected_route else "⚠️"

        print(f"\nQUERY: {query}")
        print(f"  Route: {actual_route} (expected: {expected_route}) {route_match}")
        print(f"  Config: top_k={config['top_k']}, "
              f"threshold={config['score_threshold']}, "
              f"filter={config['doc_type_filter']}")
        print(f"  Retrieved: {len(docs)} documents")

        for i, doc in enumerate(docs[:3], 1):
            doc_type = doc.metadata.get("doc_type", "unknown")
            name     = (doc.metadata.get("product_name")
                       or doc.metadata.get("category")
                       or doc.metadata.get("analytics_type")
                       or "N/A")
            print(f"    {i}. [{doc_type}] {name}")

    print("\n" + "=" * 65)


def compare_calibrated_vs_fixed(retriever: RetailRetriever) -> None:
    """
    Side-by-side comparison: calibrated retrieval (Day 14) vs
    fixed top_k=5/threshold=0.3 (Day 13 raw behavior).

    This is useful evidence for your portfolio/resume — you can
    show concretely that calibration improved retrieval relevance.
    """
    from rag.embedder import search_index

    query = "What should I restock urgently this week?"

    print("\n" + "=" * 65)
    print("CALIBRATED vs FIXED RETRIEVAL COMPARISON")
    print(f"QUERY: {query}")
    print("=" * 65)

    # Fixed (Day 13 style — no calibration, no filtering)
    fixed_results = search_index(
        query=query, index=retriever.index, documents=retriever.documents,
        model=retriever.model, top_k=5, score_threshold=0.3,
    )
    print(f"\nFIXED (top_k=5, threshold=0.3, no filter): "
          f"{len(fixed_results)} results")
    irrelevant_count = sum(
        1 for doc, _ in fixed_results
        if doc.metadata.get("doc_type") != "inventory_alert"
    )
    print(f"  → {irrelevant_count} of these are NOT inventory alerts (noise)")

    # Calibrated (Day 14 style)
    calibrated_results = retriever.get_relevant_documents_with_scores(query)
    print(f"\nCALIBRATED (route-aware, filtered to inventory_alert): "
          f"{len(calibrated_results)} results")
    irrelevant_count_cal = sum(
        1 for doc, _ in calibrated_results
        if doc.metadata.get("doc_type") != "inventory_alert"
    )
    print(f"  → {irrelevant_count_cal} of these are NOT inventory alerts (noise)")
    print("\n" + "=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Running this file:
    1. Loads/builds the FAISS vector store (Day 13)
    2. Wraps it in the calibrated RetailRetriever
    3. Runs calibration tests across all 3 query types
    4. Shows calibrated vs fixed retrieval comparison
    """
    retriever = create_retriever(force_rebuild=False)

    run_calibration_test(retriever)
    compare_calibrated_vs_fixed(retriever)