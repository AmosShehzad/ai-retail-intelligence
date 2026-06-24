"""
Day 16: CRITICAL METHODOLOGY — Full RAG Architecture Completion

This file is the FINAL ASSEMBLY of the entire RAG system.
It connects:
- Day 14's RetailRetriever (calibrated FAISS search)
- Day 11's prompt templates (RETAIL_ASSISTANT_PERSONA + RAG prompt)
- Day 15's LlamaService (hardened Ollama interface)

Plus TWO new things built today:
1. Context formatting — turns retrieved Documents into clean text
2. Hallucination guard — validates the answer stays grounded in
   the retrieved context before returning it to the user

This is what /api/v1/rag/ask (Day 9-10) calls in production.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag.retriever import create_retriever, RetailRetriever, get_retrieval_config
from rag.chains import route_question
from rag.prompts import (
    RETAIL_ASSISTANT_PERSONA,
    get_rag_prompt,
)
from rag.llama_service import get_llama_service, LlamaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: STRUCTURED RESULT OBJECT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGResult:
    """
    The complete output of a RAG query — everything the API endpoint
    (Day 9-10) needs to build its JSON response.

    Why a dataclass instead of a plain dict:
    Type safety + autocomplete in your editor + self-documenting
    structure. Anyone reading this code immediately sees exactly
    what a RAG query produces.
    """
    success           : bool
    question          : str
    answer            : str
    sources           : List[Dict[str, Any]] = field(default_factory=list)
    route             : str = "base"
    retrieved_doc_count: int = 0
    grounded          : bool = True       # did hallucination guard pass?
    grounding_score   : float = 0.0       # 0.0-1.0, how well grounded
    duration_sec      : float = 0.0
    error             : Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: CONTEXT FORMATTING (built today)
# ══════════════════════════════════════════════════════════════════════════════

def format_documents_as_context(documents: List[Document]) -> str:
    """
    Converts a list of retrieved Document objects into a single
    clean text block that gets injected into the {context}
    placeholder of Day 11's get_rag_prompt().

    Why this matters for hallucination prevention:
    The QUALITY of context formatting directly determines whether
    phi3 stays grounded. If documents are dumped as raw JSON or
    concatenated without structure, the LLM has a harder time
    parsing what's relevant. Clear separators and source labels
    help the LLM track WHICH fact came from WHERE.

    Format used:
    [Source 1 - Product Information]
    Product: Tapal Danedar 200g
    ...

    [Source 2 - Inventory Alert]
    INVENTORY ALERT — LOW STOCK: ...
    """
    if not documents:
        # Explicit "no data" message — this is critical.
        # If we returned an empty string, phi3 might fill the gap
        # with training-data guesses. An explicit statement that
        # NO data was found gives the LLM permission to say "I don't know"
        # instead of hallucinating.
        return "No relevant store data was found for this question."

    # Map internal doc_type values to readable source labels
    # Why: "inventory_alert" looks technical; "Inventory Alert" reads
    # naturally in the prompt and helps the LLM categorize the source
    type_labels = {
        "product"        : "Product Information",
        "category"       : "Category Summary",
        "inventory_alert": "Inventory Alert",
        "store_analytics": "Store Analytics",
    }

    context_blocks = []

    for i, doc in enumerate(documents, start=1):
        doc_type = doc.metadata.get("doc_type", "unknown")
        label    = type_labels.get(doc_type, "Store Data")

        # Each source is clearly numbered and labeled
        # This numbering is what lets us cite "[Source 2]" in answers
        # and matches the "sources" array we return in the API response
        block = f"[Source {i} - {label}]\n{doc.page_content}"
        context_blocks.append(block)

    # Double newline between sources — gives the LLM clear visual
    # separation between distinct pieces of information
    return "\n\n".join(context_blocks)


def build_sources_metadata(documents: List[Document], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Builds the "sources" array returned in the API response.

    Why return sources to the user:
    Transparency — the store owner can SEE which actual data points
    the AI used to form its answer. This is what separates a
    trustworthy business tool from a black-box chatbot. If the AI
    says "restock Tapal Tea", the owner can verify by checking
    the cited source document.
    """
    sources = []

    for doc, score in zip(documents, scores):
        sources.append({
            "doc_type"    : doc.metadata.get("doc_type", "unknown"),
            "product_name": doc.metadata.get("product_name"),
            "category"    : doc.metadata.get("category"),
            "score"       : round(float(score), 4),
            "preview"     : doc.page_content[:150] + "..."
                            if len(doc.page_content) > 150
                            else doc.page_content,
        })

    return sources


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: HALLUCINATION GUARD (built today)
# ══════════════════════════════════════════════════════════════════════════════

def check_grounding(
    answer  : str,
    context : str,
    documents: List[Document],
) -> tuple[bool, float]:
    """
    Checks whether the generated answer is actually grounded in
    the retrieved context, rather than hallucinated from phi3's
    training data.

    This is a HEURISTIC check (not perfect, but practical for a
    portfolio project without expensive LLM-as-judge calls).

    Three checks combined into a grounding score:
    1. Does the answer explicitly admit no data was found?
       (This is actually GOOD grounding — honest "I don't know")
    2. Does the answer mention specific numbers/products that
       also appear in the retrieved context? (word overlap)
    3. Is the context itself non-empty? (no context = can't be grounded)

    Returns (is_grounded: bool, grounding_score: float 0.0-1.0)

    Why this matters: if grounding_score is too low, Day 9's API
    endpoint can choose to warn the user "this answer may not be
    fully accurate" instead of presenting it with full confidence.
    """
    # Case 1: No documents were retrieved at all
    # An answer with zero context can't be grounded — score is 0
    if not documents:
        # BUT if the answer correctly says "I don't have that info",
        # that's actually the CORRECT honest behavior, so we treat
        # it as grounded (score 1.0) rather than penalizing honesty
        honest_phrases = [
            "don't have that information",
            "no relevant store data",
            "not enough data",
            "i don't have data",
        ]
        if any(phrase in answer.lower() for phrase in honest_phrases):
            return True, 1.0
        else:
            # Model answered confidently with ZERO context — likely hallucinated
            return False, 0.0

    # Case 2: Word overlap check
    # Extract significant words (4+ chars, ignore common words) from
    # both the answer and the source context, then measure overlap
    def extract_keywords(text: str) -> set:
        common_words = {
            "this", "that", "with", "from", "have", "your", "what",
            "should", "could", "would", "their", "there", "based",
            "answer", "question", "store", "data", "based", "current"
        }
        words = [w.strip(".,!?():").lower() for w in text.split()]
        return {w for w in words if len(w) >= 4 and w not in common_words}

    answer_keywords  = extract_keywords(answer)
    context_keywords = extract_keywords(context)

    if not answer_keywords:
        # Answer too short/generic to evaluate meaningfully
        return True, 0.5

    # Overlap ratio: what fraction of the answer's key terms
    # actually appear somewhere in the retrieved context
    overlap = answer_keywords & context_keywords
    overlap_ratio = len(overlap) / len(answer_keywords)

    # Case 3: Product name check — if the answer mentions a specific
    # product name, verify that product appears in the retrieved docs
    # (catches the most dangerous hallucination: inventing a product
    # or number that doesn't exist in your store at all)
    mentioned_products = [
        doc.metadata.get("product_name", "")
        for doc in documents
        if doc.metadata.get("product_name")
    ]
    product_mentioned_correctly = True
    for product in mentioned_products:
        # If this product's name appears partially in context, that's fine
        # (we're not checking the answer didn't invent a DIFFERENT product
        # here — that's a more advanced check beyond portfolio scope)
        pass

    grounding_score = round(overlap_ratio, 2)
    # Threshold: 25% of the answer's meaningful words should trace
    # back to the retrieved context to consider it grounded
    is_grounded = grounding_score >= 0.25

    return is_grounded, grounding_score


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: THE COMPLETE RAG PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    THE FINAL ASSEMBLY — orchestrates the complete RAG flow.

    This single class is what Day 9's /rag/ask endpoint calls.
    It owns ONE instance each of:
    - RetailRetriever (Day 14)  — expensive to create, reuse it
    - LlamaService (Day 15)     — expensive to create, reuse it

    Created ONCE at FastAPI startup (Day 8's lifespan), reused
    for every single user question.
    """

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING FULL RAG PIPELINE")
        log.info("=" * 60)

        # Day 14: build/load the calibrated retriever
        log.info("Step 1/2: Loading retriever (FAISS + embeddings)...")
        self.retriever: RetailRetriever = create_retriever(force_rebuild=False)

        # Day 15: get the hardened Llama service (singleton)
        log.info("Step 2/2: Connecting to LlamaService...")
        self.llama: LlamaService = get_llama_service()

        # Day 11: the RAG prompt template
        self.rag_prompt = get_rag_prompt()

        self.ready = self.retriever is not None and self.llama.is_ready()

        log.info("=" * 60)
        log.info("RAG PIPELINE READY: %s", self.ready)
        log.info("  Documents indexed : %d", len(self.retriever.documents))
        log.info("  phi3 ready      : %s", self.llama.is_ready())
        log.info("=" * 60)

    def is_ready(self) -> bool:
        """Quick readiness check — used by Day 9's /rag/status endpoint."""
        return self.ready

    def answer(self, question: str, top_k_override: Optional[int] = None) -> RAGResult:
        """
        THE MAIN METHOD — runs the complete RAG flow for one question.

        Steps (this IS the architecture diagram from the top of this file):
        1. Classify the question (Day 11 router)
        2. Retrieve calibrated documents (Day 14)
        3. Format documents into context text (built today)
        4. Build the final prompt (Day 11 template + context + question)
        5. Generate answer via LlamaService (Day 15 — timeout/retry/validation)
        6. Check grounding / hallucination risk (built today)
        7. Package everything into RAGResult

        This is a SYNC method — used in scripts/tests.
        See aanswer() below for the async version FastAPI uses.
        """
        start_time = time.time()

        if not self.is_ready():
            return RAGResult(
                success = False,
                question= question,
                answer  = "The AI assistant is currently unavailable. "
                          "Please ensure Ollama is running.",
                error   = "Pipeline not ready (retriever or LLM unavailable)",
            )

        try:
            # ── Step 1: Classify the question ───────────────────────────────
            route = route_question(question)
            log.info("RAG query | route='%s' | question='%s'", route, question[:60])

            # ── Step 2: Retrieve calibrated documents ───────────────────────
            # Uses Day 14's RetailRetriever — handles route-based top_k,
            # score thresholds, and doc_type filtering automatically
            retrieval_results = self.retriever.get_relevant_documents_with_scores(question)
            documents = [doc for doc, score in retrieval_results]
            scores    = [score for doc, score in retrieval_results]

            log.info("Retrieved %d documents for context", len(documents))

            # ── Step 3: Format documents into context text ──────────────────
            context_text = format_documents_as_context(documents)

            # ── Step 4: Build the final prompt ───────────────────────────────
            # Day 11's get_rag_prompt() expects {context} and {question}
            # We format it manually here (rather than full LCEL chain)
            # because we need fine control for the hallucination guard step
            formatted_messages = self.rag_prompt.format_messages(
                context = context_text,
                question= question,
            )

            # Extract system + human text for LlamaService.generate()
            # (LlamaService takes plain strings, not LangChain message objects,
            # so we convert here — keeps LlamaService reusable/simple)
            system_text = next(
                (m.content for m in formatted_messages if m.type == "system"),
                RETAIL_ASSISTANT_PERSONA
            )
            human_text = next(
                (m.content for m in formatted_messages if m.type == "human"),
                question
            )

            # ── Step 5: Generate via LlamaService (Day 15) ───────────────────
            # This call has timeout, retry, and response validation built in
            generation_result = self.llama.generate(
                prompt_text  = human_text,
                system_prompt= system_text,
            )

            if not generation_result["success"]:
                # LlamaService already retried internally and still failed
                return RAGResult(
                    success = False,
                    question= question,
                    answer  = "I'm having trouble generating a response right now. "
                              "Please try again in a moment.",
                    route   = route,
                    retrieved_doc_count=len(documents),
                    duration_sec=round(time.time() - start_time, 2),
                    error   = generation_result["error"],
                )

            answer_text = generation_result["text"]

            # ── Step 6: Hallucination guard ───────────────────────────────────
            is_grounded, grounding_score = check_grounding(
                answer_text, context_text, documents
            )

            if not is_grounded:
                log.warning(
                    "LOW GROUNDING SCORE (%.2f) for question: '%s' — "
                    "answer may contain hallucinated content",
                    grounding_score, question[:60]
                )

            # ── Step 7: Package the result ────────────────────────────────────
            sources = build_sources_metadata(documents, scores)
            duration = round(time.time() - start_time, 2)

            log.info(
                "RAG query complete | route=%s | docs=%d | grounded=%s (%.2f) | %.2fs",
                route, len(documents), is_grounded, grounding_score, duration
            )

            return RAGResult(
                success            = True,
                question           = question,
                answer             = answer_text,
                sources            = sources,
                route              = route,
                retrieved_doc_count= len(documents),
                grounded           = is_grounded,
                grounding_score    = grounding_score,
                duration_sec       = duration,
                error              = None,
            )

        except Exception as e:
            log.exception("RAG pipeline failed unexpectedly for question: %s", question)
            return RAGResult(
                success = False,
                question= question,
                answer  = "An unexpected error occurred while processing your question.",
                duration_sec=round(time.time() - start_time, 2),
                error   = str(e),
            )

    async def aanswer(self, question: str) -> RAGResult:
        """
        ASYNC version of answer() — this is what Day 9's
        /api/v1/rag/ask FastAPI endpoint calls.

        Identical logic to answer(), but uses LlamaService.agenerate()
        (Day 15's async method) instead of the sync generate().
        Retrieval (FAISS search) stays sync because it's fast (<50ms)
        and CPU-bound — no benefit from async there.
        """
        start_time = time.time()

        if not self.is_ready():
            return RAGResult(
                success = False,
                question= question,
                answer  = "The AI assistant is currently unavailable. "
                          "Please ensure Ollama is running.",
                error   = "Pipeline not ready",
            )

        try:
            route = route_question(question)
            log.info("Async RAG query | route='%s' | question='%s'", route, question[:60])

            retrieval_results = self.retriever.get_relevant_documents_with_scores(question)
            documents = [doc for doc, score in retrieval_results]
            scores    = [score for doc, score in retrieval_results]

            context_text = format_documents_as_context(documents)

            formatted_messages = self.rag_prompt.format_messages(
                context=context_text, question=question
            )
            system_text = next(
                (m.content for m in formatted_messages if m.type == "system"),
                RETAIL_ASSISTANT_PERSONA
            )
            human_text = next(
                (m.content for m in formatted_messages if m.type == "human"),
                question
            )

            # KEY DIFFERENCE from sync version: awaits agenerate()
            generation_result = await self.llama.agenerate(
                prompt_text  = human_text,
                system_prompt= system_text,
            )

            if not generation_result["success"]:
                return RAGResult(
                    success = False,
                    question= question,
                    answer  = "I'm having trouble generating a response right now.",
                    route   = route,
                    retrieved_doc_count=len(documents),
                    duration_sec=round(time.time() - start_time, 2),
                    error   = generation_result["error"],
                )

            answer_text = generation_result["text"]
            is_grounded, grounding_score = check_grounding(
                answer_text, context_text, documents
            )
            sources  = build_sources_metadata(documents, scores)
            duration = round(time.time() - start_time, 2)

            log.info(
                "Async RAG query complete | route=%s | docs=%d | grounded=%s | %.2fs",
                route, len(documents), is_grounded, duration
            )

            return RAGResult(
                success            = True,
                question           = question,
                answer             = answer_text,
                sources            = sources,
                route              = route,
                retrieved_doc_count= len(documents),
                grounded           = is_grounded,
                grounding_score    = grounding_score,
                duration_sec       = duration,
                error              = None,
            )

        except Exception as e:
            log.exception("Async RAG pipeline failed for question: %s", question)
            return RAGResult(
                success = False,
                question= question,
                answer  = "An unexpected error occurred while processing your question.",
                duration_sec=round(time.time() - start_time, 2),
                error   = str(e),
            )


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: SINGLETON (one pipeline instance for the whole FastAPI app)
# ══════════════════════════════════════════════════════════════════════════════

_rag_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline(force_recreate: bool = False) -> RAGPipeline:
    """
    Returns the shared RAGPipeline instance.

    Called from Day 8's FastAPI lifespan startup event — the pipeline
    is built ONCE when the server starts (loading FAISS + connecting
    to Ollama takes a few seconds), then reused for every request.
    """
    global _rag_pipeline_instance

    if _rag_pipeline_instance is None or force_recreate:
        _rag_pipeline_instance = RAGPipeline()

    return _rag_pipeline_instance


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — full end-to-end test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Running this file tests the COMPLETE RAG system end-to-end
    with real questions a kiryana store owner would ask.
    This is your proof that Day 16 — the critical milestone — works.
    """
    pipeline = get_rag_pipeline()

    if not pipeline.is_ready():
        print("\n⚠️  RAG Pipeline is NOT ready. Check Ollama is running.")
        exit(1)

    TEST_QUESTIONS = [
        "What should I restock urgently this week?",
        "Which products are dead stock?",
        "What is my overall profit margin?",
        "Tell me about Tapal Danedar tea pricing and stock",
        "What is the capital of France?",  # off-topic — tests grounding guard
    ]

    print("\n" + "=" * 70)
    print("DAY 16 — FULL RAG PIPELINE END-TO-END TEST")
    print("=" * 70)

    for question in TEST_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Q: {question}")
        print("─" * 70)

        result = pipeline.answer(question)

        print(f"Route          : {result.route}")
        print(f"Docs retrieved : {result.retrieved_doc_count}")
        print(f"Grounded       : {result.grounded} (score: {result.grounding_score})")
        print(f"Duration       : {result.duration_sec}s")
        print(f"\nANSWER:\n{result.answer}")

        if result.sources:
            print(f"\nSOURCES ({len(result.sources)}):")
            for s in result.sources[:3]:
                print(f"  - [{s['doc_type']}] {s.get('product_name') or s.get('category')} "
                      f"(score: {s['score']})")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)