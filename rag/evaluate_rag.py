"""
rag/evaluate_rag.py

Day 18: RAG Evaluation using LangSmith (Shortened to 5 test queries).

What this does:
- Runs 5 real kiryana store questions through your RAG pipeline
- Records every trace to LangSmith
- Measures: grounding score, response time, retrieval quality
- Prints a report you can screenshot for your portfolio

Run: python rag/evaluate_rag.py
"""

import sys
from pathlib import Path
# Force append parent directory to prevent absolute module path errors
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── 5 test questions covering all routes ─────────────────────────────────────
# Format: (question, expected_route, what_a_good_answer_contains)
EVAL_QUESTIONS = [
    (
        "What products should I restock urgently this week?",
        "inventory",
        ["restock", "stock", "units"]
    ),
    (
        "Which products are dead stock wasting my shelf space?",
        "inventory",
        ["dead", "stock", "capital"]
    ),
    (
        "What is my overall profit margin?",
        "analytics",
        ["margin", "profit", "%"]
    ),
    (
        "Show me sales trends for Surf Excel over the last 30 days.",
        "analytics",
        ["sales", "Surf Excel", "trend"]
    ),
    (
        "How do I add a new product to my inventory system manually?",
        "base",
        ["dashboard", "add", "product"]
    )
]


def run_evaluation():
    from rag.pipeline import get_rag_pipeline
    from rag.langsmith_config import TRACING_ENABLED, LANGSMITH_PROJECT

    log.info("=" * 60)
    log.info("RAG EVALUATION STARTED (5-Query Test Run)")
    log.info(f"LangSmith tracing: {'ENABLED' if TRACING_ENABLED else 'DISABLED'}")
    log.info(f"Project: {LANGSMITH_PROJECT}")
    log.info("=" * 60)

    pipeline = get_rag_pipeline()
    if not pipeline.is_ready():
        log.error("RAG pipeline failed to initialize. Aborting evaluation.")
        return

    results = []
    correct_routes = 0

    for i, (question, expected_route, keywords) in enumerate(EVAL_QUESTIONS, 1):
        log.info(f"\n[{i}/{len(EVAL_QUESTIONS)}] Processing: '{question}'")
        
        start_time = time.time()
        # Execute query through the full RAG pipeline
        response = pipeline.answer(question)
        duration = time.time() - start_time

        # Metric 1: Route Accuracy
        actual_route = response.route
        is_route_correct = (actual_route == expected_route)
        if is_route_correct:
            correct_routes += 1

        # Metric 2: Context Retrieval Quality (Keyword match fallback)
        matched_keywords = sum(1 for kw in keywords if kw.lower() in response.answer.lower())
        quality_score = matched_keywords / len(keywords) if keywords else 1.0

        results.append({
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": is_route_correct,
            "grounding_score": response.grounding_score,
            "quality_score": quality_score,
            "duration": duration,
            "answer": response.answer
        })

        log.info(f"Done in {duration:.2f}s | Route: {actual_route} (Expected: {expected_route}) | Grounding: {response.grounding_score:.2f}")

    # ── Final Metrics Breakdown ───────────────────────────────────────────────
    total = len(EVAL_QUESTIONS)
    route_accuracy = correct_routes / total
    avg_grounding = sum(r["grounding_score"] for r in results) / total
    avg_quality = sum(r["quality_score"] for r in results) / total
    avg_duration = sum(r["duration"] for r in results) / total

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY REPORT               ")
    print("=" * 60)
    print(f"  Total Questions   : {total}")
    print(f"  Route Accuracy    : {route_accuracy:.0%}")
    print(f"  Avg Grounding Score: {avg_grounding:.0%}")
    print(f"  Avg Quality Score : {avg_quality:.0%}")
    print(f"  Avg response time : {avg_duration:.1f}s")
    print()

    if TRACING_ENABLED:
        print(f"  All traces sent to LangSmith ✅")
        print(f"  View at: https://smith.langchain.com/projects/{LANGSMITH_PROJECT}")
    else:
        print("  Tracing disabled. Set LANGCHAIN_TRACING_V2=true in .env to enable.")

    print("=" * 60)

    # Save results to JSON for reference
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/eval_results.json", "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "summary": {
                "route_accuracy": route_accuracy,
                "avg_grounding": avg_grounding,
                "avg_quality": avg_quality,
                "avg_duration": avg_duration
            },
            "runs": results
        }, f, indent=4)


if __name__ == "__main__":
    run_evaluation()