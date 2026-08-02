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

from rag.langsmith_config import (
    trace_function,
    get_tracer_callbacks,
    TRACING_ENABLED,
    LANGSMITH_PROJECT,
)

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
# PART 2: CONTEXT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_documents_as_context(documents: List[Document]) -> str:
    """
    Converts a list of retrieved Document objects into a single
    clean text block that gets injected into the {context}.
    """
    if not documents:
        return "No relevant store data was found for this question."

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
        block = f"[Source {i} - {label}]\n{doc.page_content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_sources_metadata(documents: List[Document], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Builds the "sources" array returned in the API response.
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
# PART 3: HALLUCINATION GUARD
# ══════════════════════════════════════════════════════════════════════════════

def check_grounding(
    answer  : str,
    context : str,
    documents: List[Document],
) -> tuple[bool, float]:
    """
    Checks whether the generated answer is actually grounded in
    the retrieved context.
    """
    if not documents:
        honest_phrases = [
            "don't have that information",
            "no relevant store data",
            "not enough data",
            "i don't have data",
        ]
        if any(phrase in answer.lower() for phrase in honest_phrases):
            return True, 1.0
        else:
            return False, 0.0

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
        return True, 0.5

    overlap = answer_keywords & context_keywords
    overlap_ratio = len(overlap) / len(answer_keywords)

    grounding_score = round(overlap_ratio, 2)
    is_grounded = grounding_score >= 0.25

    return is_grounded, grounding_score


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: LANGSMITH TRACED HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@trace_function(name="route_question", run_type="chain")
def _traced_route_question(question: str) -> str:
    """Traced version of route_question — appears as a span in LangSmith."""
    return route_question(question)

@trace_function(name="grounding_check", run_type="chain")
def _traced_grounding_check(answer, context, documents):
    """Traced version of check_grounding — appears as a span in LangSmith."""
    return check_grounding(answer, context, documents)

def _log_rag_result_to_langsmith(
    question: str,
    answer: str,
    route: str,
    doc_count: int,
    grounding_score: float,
    duration: float,
) -> None:
    """Logs structured metadata about each RAG result to LangSmith."""
    try:
        from langsmith import Client
        
        if not TRACING_ENABLED:
            return

        client = Client()
        client.create_run(
            name        = "rag_query_metadata",
            run_type    = "chain",
            project_name= LANGSMITH_PROJECT,
            inputs      = {"question": question, "route": route},
            outputs     = {
                "answer"         : answer[:500],
                "grounding_score": grounding_score,
                "doc_count"      : doc_count,
                "duration_sec"   : duration,
                "grounded"       : grounding_score >= 0.25,
            },
        )
    except Exception as e:
        log.debug("LangSmith metadata log failed (non-critical): %s", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: THE COMPLETE RAG PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """THE FINAL ASSEMBLY — orchestrates the complete RAG flow."""

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING FULL RAG PIPELINE")
        log.info("=" * 60)

        log.info("Step 1/2: Loading retriever (FAISS + embeddings)...")
        self.retriever: RetailRetriever = create_retriever(force_rebuild=False)

        log.info("Step 2/2: Connecting to LlamaService...")
        self.llama: LlamaService = get_llama_service()

        self.rag_prompt = get_rag_prompt()

        self.ready = self.retriever is not None and self.llama.is_ready()

        log.info("=" * 60)
        log.info("RAG PIPELINE READY: %s", self.ready)
        log.info("  Documents indexed : %d", len(self.retriever.documents))
        log.info("  %s ready : %s", self.llama.provider, self.llama.is_ready())
        log.info("=" * 60)

    def is_ready(self) -> bool:
        return self.ready

    def answer(self, question: str, top_k_override: Optional[int] = None) -> RAGResult:
        """THE MAIN METHOD — runs complete RAG flow."""
        start_time = time.time()

        if not self.is_ready():
            return RAGResult(
                success=False,
                question=question,
                answer="The AI assistant is currently unavailable.",
                error="Pipeline not ready",
            )

        try:
            route = _traced_route_question(question)
            log.info("RAG query | route='%s' | question='%s'", route, question[:60])

            retrieval_results = self.retriever.get_relevant_documents_with_scores(question)
            documents = [doc for doc, score in retrieval_results]
            scores    = [score for doc, score in retrieval_results]
            log.info("Retrieved %d documents", len(documents))

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

            callbacks = get_tracer_callbacks()
            generation_result = self.llama.generate(
                prompt_text   = human_text,
                system_prompt = system_text,
            )

            if not generation_result["success"]:
                return RAGResult(
                    success=False,
                    question=question,
                    answer="I'm having trouble generating a response right now.",
                    route=route,
                    retrieved_doc_count=len(documents),
                    duration_sec=round(time.time() - start_time, 2),
                    error=generation_result["error"],
                )

            answer_text = generation_result["text"]

            is_grounded, grounding_score = _traced_grounding_check(
                answer_text, context_text, documents
            )

            if not is_grounded:
                log.warning("LOW GROUNDING (%.2f): %s", grounding_score, question[:50])

            sources  = build_sources_metadata(documents, scores)
            duration = round(time.time() - start_time, 2)

            _log_rag_result_to_langsmith(
                question=question,
                answer=answer_text,
                route=route,
                doc_count=len(documents),
                grounding_score=grounding_score,
                duration=duration,
            )

            log.info(
                "RAG complete | route=%s | docs=%d | grounded=%s (%.2f) | %.2fs",
                route, len(documents), is_grounded, grounding_score, duration
            )

            return RAGResult(
                success=True,
                question=question,
                answer=answer_text,
                sources=sources,
                route=route,
                retrieved_doc_count=len(documents),
                grounded=is_grounded,
                grounding_score=grounding_score,
                duration_sec=duration,
                error=None,
            )

        except Exception as e:
            log.exception("RAG pipeline failed: %s", question)
            return RAGResult(
                success=False,
                question=question,
                answer="An unexpected error occurred.",
                duration_sec=round(time.time() - start_time, 2),
                error=str(e),
            )

    async def aanswer(self, question: str) -> RAGResult:
        """ASYNC version of answer()"""
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
# PART 6: SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_rag_pipeline_instance: Optional[RAGPipeline] = None

def get_rag_pipeline(force_recreate: bool = False) -> RAGPipeline:
    global _rag_pipeline_instance

    if _rag_pipeline_instance is None or force_recreate:
        _rag_pipeline_instance = RAGPipeline()

    return _rag_pipeline_instance


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = get_rag_pipeline()

    if not pipeline.is_ready():
        print("\n⚠️  RAG Pipeline is NOT ready. Check Ollama is running.")
        exit(1)

    TEST_QUESTIONS = [
        "What should I restock urgently this week?",
        "Which products are dead stock?",
        "What is my overall profit margin?",
        "Tell me about Tapal Danedar tea pricing and stock",
        "What is the capital of France?",
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