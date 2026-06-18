"""
Day 11: Baseline LangChain Evaluation

Tests the prompt + Ollama chain with real store questions
BEFORE adding RAG complexity.

Why run this:
- Confirms Ollama is running and responding
- Confirms your prompts produce sensible outputs
- Gives you a quality baseline to compare against
  after you add RAG (Day 16). If RAG answers are worse
  than baseline, something is wrong with retrieval.

Run: python rag/evaluate_baseline.py
"""

import json
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.engines import get_store_kpis, get_category_margins
from analytics.inventory import get_inventory_health_summary, get_low_stock_alerts
from rag.chains import get_base_chain, get_analytics_chain, get_inventory_chain

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# Test questions — real queries a kiryana owner would ask
TEST_QUESTIONS = [
    {
        "id"      : 1,
        "chain"   : "base",
        "question": "What are the most common reasons a kiryana store loses customers?",
        "context" : {},
    },
    {
        "id"      : 2,
        "chain"   : "analytics",
        "question": "What is my overall profit margin and is it healthy for a retail store?",
        "context" : None,  # will be filled with live data below
    },
    {
        "id"      : 3,
        "chain"   : "analytics",
        "question": "Which product category is making me the most money?",
        "context" : None,
    },
    {
        "id"      : 4,
        "chain"   : "inventory",
        "question": "What products should I urgently restock this week?",
        "context" : None,
    },
    {
        "id"      : 5,
        "chain"   : "inventory",
        "question": "Which items are dead stock and wasting my shelf space?",
        "context" : None,
    },
]


def run_baseline_evaluation():
    log.info("=" * 60)
    log.info("BASELINE EVALUATION — Day 11 LangChain Setup")
    log.info("=" * 60)

    # Load live store data once
    store_kpis       = get_store_kpis()
    inventory_health = get_inventory_health_summary()
    alerts_df        = get_low_stock_alerts()
    alerts_str       = alerts_df.head(10).to_string(index=False) \
                       if not alerts_df.empty else "No alerts."

    results = []

    for test in TEST_QUESTIONS:
        print(f"\n{'─'*60}")
        print(f"TEST {test['id']}: {test['question']}")
        print(f"Chain: {test['chain'].upper()}")
        print("─" * 60)

        start = time.time()

        try:
            if test["chain"] == "base":
                chain  = get_base_chain()
                answer = chain.invoke({"question": test["question"]})

            elif test["chain"] == "analytics":
                chain  = get_analytics_chain()
                answer = chain.invoke({
                    "store_context": json.dumps(store_kpis, indent=2),
                    "question"     : test["question"],
                })

            elif test["chain"] == "inventory":
                chain  = get_inventory_chain()
                answer = chain.invoke({
                    "inventory_context": json.dumps(inventory_health, indent=2),
                    "alerts_context"   : alerts_str,
                    "question"         : test["question"],
                })

            duration = round(time.time() - start, 2)

            print(f"ANSWER ({duration}s):")
            print(answer)

            results.append({
                "id"      : test["id"],
                "chain"   : test["chain"],
                "question": test["question"],
                "answer"  : answer[:200],   # truncate for summary
                "duration": duration,
                "status"  : "PASS",
            })

        except Exception as e:
            duration = round(time.time() - start, 2)
            log.error("Test %d FAILED: %s", test["id"], str(e))
            results.append({
                "id"    : test["id"],
                "status": "FAIL",
                "error" : str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"  Passed : {len(passed)}/{len(TEST_QUESTIONS)}")
    print(f"  Failed : {len(failed)}/{len(TEST_QUESTIONS)}")

    if passed:
        avg_time = round(sum(r["duration"] for r in passed) / len(passed), 2)
        print(f"  Avg response time: {avg_time}s")

    if failed:
        print("\nFailed tests:")
        for f in failed:
            print(f"  Test {f['id']}: {f.get('error', 'unknown error')}")

    print("=" * 60)


if __name__ == "__main__":
    run_baseline_evaluation()