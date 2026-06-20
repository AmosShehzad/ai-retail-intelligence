"""
Day 12: Automated Document Synthesis Routines

Converts structured DB data → human-readable text documents
ready for vector embedding in Day 13.

Four document types:
1. Product documents      (one per product)
2. Category summaries     (one per category)
3. Inventory alerts       (one per flagged product)
4. Store analytics        (daily/weekly/monthly summaries)

Each document is a LangChain Document object:
- page_content : the text FAISS will embed and search
- metadata     : structured data for filtering/display
                 (not embedded, but returned with search results)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from langchain_core.documents import Document

from database.db_manager import get_connection
from analytics.engines import (
    get_store_kpis,
    get_revenue_by_period,
    get_category_margins,
    get_product_velocity,
)
from analytics.inventory import (
    get_inventory_health_summary,
    get_dead_stock,
    get_low_stock_alerts,
    get_restock_recommendations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# Where synthesized documents are saved as JSON
DOCS_OUTPUT_PATH = "data/processed/synthesized_documents.json"


# ── Helper: safe float formatter ──────────────────────────────────────────────
def _fmt_price(value) -> str:
    """
    Formats a number as PKR currency string.
    Handles None, NaN, and non-numeric values safely.
    
    Why: DB can return None for optional fields.
    String formatting crashes on None without this guard.
    """
    try:
        return f"Rs. {float(value):,.0f}"
    except (TypeError, ValueError):
        return "Rs. N/A"


def _fmt_float(value, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZER 1 — Product Documents
# ══════════════════════════════════════════════════════════════════════════════
def synthesize_product_documents() -> List[Document]:
    """
    Creates one text document per product in the database.

    Why one-per-product:
    When the user asks "Tell me about Tapal Tea", FAISS retrieves
    the Tapal Tea document specifically. If all products were in
    one giant document, the retriever couldn't isolate one product.

    Text format is conversational prose, not JSON or CSV.
    Why: Sentence transformers (Day 13) were trained on natural
    language. Prose embeddings are more semantically rich than
    "product_name: Tapal Tea, price: 220".
    """
    log.info("Synthesizing product documents...")

    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            p.cost_price,
            p.selling_price,
            p.stock,
            p.supplier,
            COALESCE(SUM(s.quantity), 0) AS total_units_sold,
            COALESCE(COUNT(DISTINCT s.sale_date), 0) AS days_with_sales
        FROM products p
        LEFT JOIN sales s ON p.product_id = s.product_id
        GROUP BY p.product_id
    """, conn)
    conn.close()

    documents = []

    for _, row in df.iterrows():
        # Calculate margin percentage
        try:
            margin_pct = ((row["selling_price"] - row["cost_price"])
                         / row["selling_price"] * 100)
        except (ZeroDivisionError, TypeError):
            margin_pct = 0.0

        # Calculate daily velocity
        try:
            daily_velocity = (row["total_units_sold"] / 30
                             if row["total_units_sold"] > 0 else 0)
        except (ZeroDivisionError, TypeError):
            daily_velocity = 0.0

        # Determine stock status
        if row["stock"] == 0:
            stock_status = "OUT OF STOCK — immediate reorder required"
        elif row["stock"] <= 5:
            stock_status = f"CRITICALLY LOW — only {row['stock']} units remaining"
        elif row["stock"] <= 15:
            stock_status = f"LOW — {row['stock']} units remaining, consider restocking"
        else:
            stock_status = f"adequate at {row['stock']} units"

        # Determine velocity description
        if daily_velocity == 0:
            velocity_desc = "not selling — potential dead stock"
        elif daily_velocity < 0.5:
            velocity_desc = f"slow mover at {_fmt_float(daily_velocity)} units/day"
        elif daily_velocity < 2:
            velocity_desc = f"moderate seller at {_fmt_float(daily_velocity)} units/day"
        else:
            velocity_desc = f"fast mover at {_fmt_float(daily_velocity)} units/day"

        # Build the text document
        # Every sentence adds searchable semantic meaning
        text = f"""Product: {row['product_name']}
Category: {row['category']}
Supplier: {row['supplier'] or 'Unknown'}

Pricing: Cost price is {_fmt_price(row['cost_price'])}, \
selling price is {_fmt_price(row['selling_price'])}, \
giving a profit margin of {_fmt_float(margin_pct)}%.

Stock Status: Current inventory is {stock_status}.

Sales Performance: This product is a {velocity_desc}. \
Total units sold in the last 30 days: {int(row['total_units_sold'])}.

Business Insight: {_generate_product_insight(
    margin_pct, daily_velocity, row['stock'], row['product_name']
)}"""

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_type"     : "product",
                "product_id"   : int(row["product_id"]),
                "product_name" : row["product_name"],
                "category"     : row["category"],
                "stock"        : int(row["stock"]),
                "selling_price": float(row["selling_price"]),
                "margin_pct"   : round(margin_pct, 2),
                "daily_velocity": round(daily_velocity, 4),
                "supplier"     : row["supplier"] or "Unknown",
            }
        ))

    log.info("Generated %d product documents.", len(documents))
    return documents


def _generate_product_insight(
    margin_pct: float,
    daily_velocity: float,
    stock: int,
    product_name: str
) -> str:
    """
    Generates a one-sentence business insight per product.
    
    Why: This is the most semantically valuable part of the document.
    When the RAG system retrieves context for "what should I focus on?",
    these insight sentences directly answer the question.
    The LLM doesn't have to infer — the insight is pre-written.
    """
    if daily_velocity == 0 and stock > 10:
        return (f"{product_name} is dead stock — it has not sold recently "
                f"and is locking up capital in unsold inventory. "
                f"Consider discounting or returning to supplier.")

    if stock == 0:
        return (f"{product_name} is out of stock and needs immediate reordering "
                f"to avoid lost sales.")

    if daily_velocity > 2 and stock < 15:
        days_left = round(stock / daily_velocity, 1) if daily_velocity > 0 else 999
        return (f"{product_name} is a high-velocity product that will run out "
                f"in approximately {days_left} days at current sales rate. "
                f"Restock urgently.")

    if margin_pct > 30:
        return (f"{product_name} is a high-margin product ({_fmt_float(margin_pct)}% margin) "
                f"— prioritize keeping it in stock to maximize profitability.")

    if margin_pct < 10:
        return (f"{product_name} has a low margin ({_fmt_float(margin_pct)}%) "
                f"— review pricing or negotiate better cost with supplier.")

    return (f"{product_name} is performing normally with a {_fmt_float(margin_pct)}% "
            f"margin and stable sales velocity.")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZER 2 — Category Summary Documents
# ══════════════════════════════════════════════════════════════════════════════
def synthesize_category_documents() -> List[Document]:
    """
    Creates one summary document per product category.

    Why: Category-level questions like "Which category makes the
    most money?" or "How is Tea performing?" need aggregated data.
    Individual product documents don't answer these well — the
    category document gives the aggregated picture.
    """
    log.info("Synthesizing category documents...")

    df_margins  = get_category_margins()
    df_velocity = get_product_velocity(top_n=100)

    # Build per-category top seller lookup
    top_seller_by_category = {}
    if not df_velocity["top_sellers"].empty:
        for _, row in df_velocity["top_sellers"].iterrows():
            cat = row.get("category", "Unknown")
            if cat not in top_seller_by_category:
                top_seller_by_category[cat] = row.get("product_name", "Unknown")

    documents = []

    for _, row in df_margins.iterrows():
        category = row["category"]
        top_seller = top_seller_by_category.get(category, "data not available")

        text = f"""Category Overview: {category}

Revenue Performance: The {category} category generated total revenue of \
{_fmt_price(row['total_revenue'])} with a profit of \
{_fmt_price(row['total_profit'])}.

Profitability: Average profit margin for {category} products is \
{_fmt_float(row['margin_pct'])}%. This category contributes \
{_fmt_float(row['revenue_share_pct'])}% of total store revenue.

Sales Volume: A total of {int(row['units_sold'])} units were sold \
across all {category} products.

Top Performer: The best selling product in {category} is {top_seller}.

Business Insight: {_generate_category_insight(
    category, row['margin_pct'],
    row['revenue_share_pct'], row['total_revenue']
)}"""

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_type"         : "category",
                "category"         : category,
                "total_revenue"    : float(row["total_revenue"]),
                "margin_pct"       : float(row["margin_pct"]),
                "revenue_share_pct": float(row["revenue_share_pct"]),
                "units_sold"       : int(row["units_sold"]),
            }
        ))

    log.info("Generated %d category documents.", len(documents))
    return documents


def _generate_category_insight(
    category: str,
    margin_pct: float,
    revenue_share_pct: float,
    total_revenue: float
) -> str:
    if revenue_share_pct > 25:
        return (f"{category} is the dominant revenue category, contributing over "
                f"25% of store income. Ensure it is always well-stocked.")
    if margin_pct > 28:
        return (f"{category} is a high-margin category. Expanding the product "
                f"range here would directly increase overall store profitability.")
    if margin_pct < 15:
        return (f"{category} has thin margins below 15%. Review supplier pricing "
                f"or consider reducing SKUs in this category.")
    return (f"{category} is performing steadily. Monitor for seasonal demand "
            f"shifts common in Pakistani retail markets.")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZER 3 — Inventory Alert Documents
# ══════════════════════════════════════════════════════════════════════════════
def synthesize_inventory_documents() -> List[Document]:
    """
    Creates text documents for dead stock and low stock alerts.

    Why separate from product documents:
    Inventory questions are urgent and action-oriented.
    A dedicated document type means FAISS retrieves these
    specifically when the question is about restocking,
    not general product information.
    These documents have explicit action language
    ("ORDER NOW", "DEAD STOCK") that makes them highly
    retrievable for restock queries.
    """
    log.info("Synthesizing inventory alert documents...")

    documents = []

    # ── Dead stock documents ───────────────────────────────────────────────
    dead_df = get_dead_stock()

    for _, row in dead_df.iterrows():
        text = f"""INVENTORY ALERT — DEAD STOCK: {row['product_name']}

Status: DEAD STOCK — this product has not been selling.
Category: {row['category']}
Supplier: {row['supplier'] or 'Unknown'}

Stock Details: {int(row['current_stock'])} units are sitting unsold \
in inventory. Only {int(row['units_sold'])} units were sold in the last 30 days. \
Daily sales velocity is {_fmt_float(row['daily_velocity'])} units per day.

Financial Impact: Capital locked in this dead stock is \
{_fmt_price(row['capital_locked_pkr'])}. This money could be used \
for faster-moving products.

Recommended Action: Stop reordering {row['product_name']}. Consider \
discounting to clear stock, or return unsold units to {row['supplier'] or 'the supplier'} \
if possible."""

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_type"         : "inventory_alert",
                "alert_type"       : "dead_stock",
                "product_name"     : row["product_name"],
                "category"         : row["category"],
                "current_stock"    : int(row["current_stock"]),
                "units_sold"       : int(row["units_sold"]),
                "capital_locked"   : float(row["capital_locked_pkr"]),
                "supplier"         : row["supplier"] or "Unknown",
            }
        ))

    # ── Low stock alert documents ──────────────────────────────────────────
    alerts_df = get_low_stock_alerts()

    for _, row in alerts_df.iterrows():
        urgency_text = {
            "CRITICAL": "CRITICAL — ORDER IMMEDIATELY",
            "HIGH"    : "HIGH PRIORITY — order within 2 days",
            "MEDIUM"  : "MEDIUM — plan reorder this week",
        }.get(row["urgency"], row["urgency"])

        text = f"""INVENTORY ALERT — LOW STOCK: {row['product_name']}

Urgency: {urgency_text}
Category: {row['category']}
Supplier: {row['supplier'] or 'Unknown'}

Stock Details: Only {int(row['current_stock'])} units remaining. \
At current sales rate of {_fmt_float(row['daily_velocity'])} units per day, \
stock will run out in {_fmt_float(row['days_of_stock_remaining'])} days.

Reorder Information: Suggested reorder quantity is \
{int(row['suggested_reorder_qty'])} units. \
Estimated reorder cost: {_fmt_price(row['estimated_reorder_cost_pkr'])}.

Action: Contact {row['supplier'] or 'supplier'} to order \
{int(row['suggested_reorder_qty'])} units of {row['product_name']} \
to maintain 30 days of stock."""

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_type"                  : "inventory_alert",
                "alert_type"                : "low_stock",
                "urgency"                   : row["urgency"],
                "product_name"              : row["product_name"],
                "category"                  : row["category"],
                "current_stock"             : int(row["current_stock"]),
                "days_of_stock_remaining"   : float(row["days_of_stock_remaining"]),
                "suggested_reorder_qty"     : int(row["suggested_reorder_qty"]),
                "estimated_reorder_cost_pkr": float(row["estimated_reorder_cost_pkr"]),
                "supplier"                  : row["supplier"] or "Unknown",
            }
        ))

    log.info("Generated %d inventory alert documents "
             "(%d dead stock, %d low stock).",
             len(documents), len(dead_df), len(alerts_df))
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZER 4 — Store Analytics Documents
# ══════════════════════════════════════════════════════════════════════════════
def synthesize_analytics_documents() -> List[Document]:
    """
    Creates documents from store-level analytics summaries.

    Three analytics documents:
    1. Overall store KPI summary
    2. Weekly revenue trend
    3. Monthly performance overview

    Why: When the owner asks "How is my business doing this month?"
    the RAG system retrieves these documents. They contain
    aggregated business performance language that product-level
    documents don't have.
    """
    log.info("Synthesizing analytics documents...")

    documents = []
    kpis = get_store_kpis()

    # ── Document 1: Store KPI Overview ────────────────────────────────────
    kpi_text = f"""Store Performance Overview — AI Retail Intelligence Summary

Overall Financials: The store has generated total revenue of \
{_fmt_price(kpis['total_revenue'])} with a net profit of \
{_fmt_price(kpis['total_profit'])}. Overall profit margin is \
{_fmt_float(kpis['overall_margin_pct'])}%.

Sales Activity: Total units sold: {kpis['total_units_sold']:,}. \
Average order value: {_fmt_price(kpis['avg_order_value'])}.

Inventory Overview: The store carries {kpis['total_products']} products \
with {kpis['total_stock_units']:,} total units in stock. \
Current inventory is valued at {_fmt_price(kpis['current_inventory_value'])}.

Business Health: {_generate_store_health_insight(kpis)}"""

    documents.append(Document(
        page_content=kpi_text,
        metadata={
            "doc_type"            : "store_analytics",
            "analytics_type"      : "kpi_overview",
            "total_revenue"       : float(kpis["total_revenue"]),
            "total_profit"        : float(kpis["total_profit"]),
            "overall_margin_pct"  : float(kpis["overall_margin_pct"]),
            "generated_at"        : datetime.now().isoformat(),
        }
    ))

    # ── Document 2: Weekly Revenue Trend ──────────────────────────────────
    weekly_df = get_revenue_by_period("W")

    if not weekly_df.empty:
        recent_weeks = weekly_df.tail(4)
        weeks_text   = []

        for _, row in recent_weeks.iterrows():
            weeks_text.append(
                f"Week of {row.get('sale_date', 'N/A')}: "
                f"Revenue {_fmt_price(row['total_revenue'])}, "
                f"Profit {_fmt_price(row['total_profit'])}, "
                f"Margin {_fmt_float(row['profit_margin_pct'])}%"
            )

        # Trend direction
        if len(recent_weeks) >= 2:
            latest   = float(recent_weeks.iloc[-1]["total_revenue"])
            previous = float(recent_weeks.iloc[-2]["total_revenue"])
            trend    = "increasing" if latest > previous else "decreasing"
            trend_pct = abs(round((latest - previous) / previous * 100, 1)) \
                        if previous > 0 else 0
            trend_sentence = (f"Revenue is {trend} — "
                              f"{trend_pct}% compared to previous week.")
        else:
            trend_sentence = "Insufficient data for trend analysis."

        weekly_text = f"""Weekly Revenue Performance — Last 4 Weeks

{chr(10).join(weeks_text)}

Trend Analysis: {trend_sentence}

This data covers the most recent 4 weeks of store operations \
and reflects actual transaction records."""

        documents.append(Document(
            page_content=weekly_text,
            metadata={
                "doc_type"      : "store_analytics",
                "analytics_type": "weekly_revenue",
                "weeks_covered" : len(recent_weeks),
                "generated_at"  : datetime.now().isoformat(),
            }
        ))

    # ── Document 3: Category Performance Summary ──────────────────────────
    cat_df = get_category_margins()

    if not cat_df.empty:
        top_cat    = cat_df.iloc[0]
        bottom_cat = cat_df.iloc[-1]

        cat_lines = []
        for _, row in cat_df.iterrows():
            cat_lines.append(
                f"{row['category']}: "
                f"Revenue {_fmt_price(row['total_revenue'])}, "
                f"Margin {_fmt_float(row['margin_pct'])}%, "
                f"Share {_fmt_float(row['revenue_share_pct'])}% of total"
            )

        cat_text = f"""Category Performance Summary

Revenue by Category:
{chr(10).join(cat_lines)}

Top Category: {top_cat['category']} leads with \
{_fmt_price(top_cat['total_revenue'])} revenue and \
{_fmt_float(top_cat['margin_pct'])}% margin.

Lowest Performer: {bottom_cat['category']} contributes only \
{_fmt_float(bottom_cat['revenue_share_pct'])}% of total revenue.

Strategic Insight: Focus on expanding {top_cat['category']} products \
as they drive the most store income."""

        documents.append(Document(
            page_content=cat_text,
            metadata={
                "doc_type"          : "store_analytics",
                "analytics_type"    : "category_performance",
                "top_category"      : top_cat["category"],
                "categories_covered": len(cat_df),
                "generated_at"      : datetime.now().isoformat(),
            }
        ))

    log.info("Generated %d analytics documents.", len(documents))
    return documents


def _generate_store_health_insight(kpis: dict) -> str:
    margin = kpis.get("overall_margin_pct", 0)
    if margin >= 25:
        return ("The store is operating with a healthy profit margin above 25%. "
                "Focus on maintaining stock levels of top sellers "
                "and reducing dead stock to further improve profitability.")
    elif margin >= 15:
        return ("The store has an acceptable profit margin. "
                "Review high-margin categories and consider expanding them "
                "to push overall margins above 25%.")
    else:
        return ("The store profit margin is below 15% — below healthy retail levels. "
                "Urgently review product pricing, supplier costs, "
                "and eliminate dead stock.")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER SYNTHESIZER — runs all four synthesizers
# ══════════════════════════════════════════════════════════════════════════════
def run_document_synthesis(save_to_disk: bool = True) -> List[Document]:
    """
    Runs all four synthesizers and combines output.
    
    save_to_disk=True:
    Saves documents as JSON so you don't have to re-synthesize
    every time FAISS needs them. Day 13 loads from this file.
    Re-synthesize when new sales data arrives (daily via scheduler).
    
    Returns list of LangChain Document objects ready for embedding.
    """
    log.info("=" * 55)
    log.info("DOCUMENT SYNTHESIS PIPELINE STARTED")
    log.info("=" * 55)

    all_documents = []

    product_docs   = synthesize_product_documents()
    category_docs  = synthesize_category_documents()
    inventory_docs = synthesize_inventory_documents()
    analytics_docs = synthesize_analytics_documents()

    all_documents.extend(product_docs)
    all_documents.extend(category_docs)
    all_documents.extend(inventory_docs)
    all_documents.extend(analytics_docs)

    log.info("-" * 55)
    log.info("SYNTHESIS SUMMARY")
    log.info("  Product documents   : %d", len(product_docs))
    log.info("  Category documents  : %d", len(category_docs))
    log.info("  Inventory documents : %d", len(inventory_docs))
    log.info("  Analytics documents : %d", len(analytics_docs))
    log.info("  TOTAL               : %d", len(all_documents))
    log.info("-" * 55)

    if save_to_disk:
        _save_documents(all_documents)

    log.info("DOCUMENT SYNTHESIS COMPLETE ✅")
    return all_documents


def _save_documents(documents: List[Document]) -> None:
    """
    Saves documents to JSON for inspection and reuse.
    
    Why save to disk:
    - Day 13 embedder loads from here (no re-synthesis needed)
    - You can inspect what documents look like before embedding
    - Scheduler re-runs synthesis daily, overwrites this file
      with fresh data
    """
    os.makedirs(os.path.dirname(DOCS_OUTPUT_PATH), exist_ok=True)

    serialized = [
        {
            "page_content": doc.page_content,
            "metadata"    : doc.metadata,
        }
        for doc in documents
    ]

    with open(DOCS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False, default=str)

    log.info("Documents saved → %s", DOCS_OUTPUT_PATH)


def load_documents_from_disk() -> List[Document]:
    """
    Loads previously synthesized documents from JSON.
    Used by Day 13 embedder to avoid re-synthesizing every run.
    """
    if not os.path.exists(DOCS_OUTPUT_PATH):
        log.warning("No saved documents found. Running synthesis...")
        return run_document_synthesis()

    with open(DOCS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in data
    ]

    log.info("Loaded %d documents from disk.", len(documents))
    return documents


# ── Quick inspection utility ──────────────────────────────────────────────────
def inspect_documents(documents: List[Document], sample: int = 2) -> None:
    """
    Prints sample documents so you can verify quality before embedding.
    Always run this after synthesis to check output looks correct.
    """
    print(f"\n{'='*60}")
    print(f"DOCUMENT INSPECTION — {len(documents)} total documents")
    print("=" * 60)

    doc_types = {}
    for doc in documents:
        t = doc.metadata.get("doc_type", "unknown")
        doc_types[t] = doc_types.get(t, 0) + 1

    print("\nDocument type breakdown:")
    for dtype, count in doc_types.items():
        print(f"  {dtype:25s}: {count}")

    print(f"\nShowing {sample} sample documents per type:\n")

    shown = {}
    for doc in documents:
        dtype = doc.metadata.get("doc_type", "unknown")
        if shown.get(dtype, 0) < sample:
            print(f"{'─'*60}")
            print(f"TYPE: {dtype.upper()}")
            print(f"METADATA: {json.dumps(doc.metadata, indent=2, default=str)}")
            print(f"CONTENT PREVIEW:\n{doc.page_content[:400]}...")
            print()
            shown[dtype] = shown.get(dtype, 0) + 1


if __name__ == "__main__":
    docs = run_document_synthesis(save_to_disk=True)
    inspect_documents(docs, sample=1)