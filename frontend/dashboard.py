"""
Day 17: Interactive Streamlit UI Dashboard

Three pages:
1. Store Overview    — KPI cards + revenue chart + category breakdown
2. Inventory Intel   — dead stock + low stock alerts + restock list
3. AI Assistant      — RAG-powered chat with source citations

Connects to:
- FastAPI backend (Days 8-10) for analytics/inventory data
- RAG pipeline (Day 16) directly for AI chat
"""

import sys
from pathlib import Path

# Add project root to path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

# ── Page configuration — MUST be the first Streamlit command ──────────────────
# This sets the browser tab title, icon, and layout
st.set_page_config(
    page_title = "AI Retail Intelligence",
    page_icon  = "🏪",
    layout     = "wide",           # use full browser width
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# STYLING — Custom CSS injected into the page
# ══════════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    """
    Injects custom CSS to make the dashboard look professional.
    st.markdown with unsafe_allow_html=True lets us write real CSS.
    """
    st.markdown("""
    <style>
    /* ── Global font and background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Hide Streamlit default header/footer ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ── Main container padding ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* ── KPI Card styling ── */
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }
    .kpi-delta {
        font-size: 0.82rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .kpi-positive { color: #4ade80; }
    .kpi-negative { color: #f87171; }
    .kpi-neutral  { color: #94a3b8; }
    
    /* ── Alert badges ── */
    .alert-critical {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #fca5a5;
        font-weight: 600;
    }
    .alert-high {
        background: rgba(249,115,22,0.15);
        border: 1px solid rgba(249,115,22,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #fdba74;
        font-weight: 600;
    }
    .alert-medium {
        background: rgba(234,179,8,0.15);
        border: 1px solid rgba(234,179,8,0.4);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        color: #fde047;
        font-weight: 600;
    }
    
    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    /* ── Chat message styling ── */
    .chat-user {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        border-radius: 16px 16px 4px 16px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        color: white;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.92rem;
    }
    .chat-assistant {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px 16px 16px 4px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
        max-width: 92%;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .chat-source {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 8px;
        padding: 0.4rem 0.7rem;
        font-size: 0.75rem;
        color: #a5b4fc;
        margin-top: 0.4rem;
        display: inline-block;
    }
    
    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* ── Streamlit metric override ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* ── DataFrame table styling ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — fetch from your analytics/inventory engines directly
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_kpis():
    """
    @st.cache_data(ttl=300):
    Caches the result for 300 seconds (5 minutes).
    Why: KPI calculations hit the DB and run analytics.
    Without caching, every page interaction (button click,
    slider move) re-runs the full analytics query.
    ttl=300 means data refreshes every 5 minutes automatically.
    """
    try:
        from analytics.engines import get_store_kpis
        return get_store_kpis()
    except Exception as e:
        st.error(f"Could not load KPIs: {e}")
        return {}


@st.cache_data(ttl=300)
def load_revenue(period: str = "D"):
    """Loads revenue data for the given period (D/W/M)."""
    try:
        from analytics.engines import get_revenue_by_period
        df = get_revenue_by_period(period)
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_category_margins():
    """Loads category-level margin and revenue data."""
    try:
        from analytics.engines import get_category_margins
        return get_category_margins()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_velocity(top_n: int = 10):
    """Loads product velocity (best sellers + slow movers)."""
    try:
        from analytics.engines import get_product_velocity
        return get_product_velocity(top_n)
    except Exception as e:
        return {"top_sellers": pd.DataFrame(), "slow_movers": pd.DataFrame()}


@st.cache_data(ttl=300)
def load_inventory_health():
    """Loads full inventory health summary."""
    try:
        from analytics.inventory import get_inventory_health_summary
        return get_inventory_health_summary()
    except Exception as e:
        return {}


@st.cache_data(ttl=300)
def load_dead_stock():
    """Loads dead stock products."""
    try:
        from analytics.inventory import get_dead_stock
        return get_dead_stock()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_alerts():
    """Loads low stock alerts."""
    try:
        from analytics.inventory import get_low_stock_alerts
        return get_low_stock_alerts()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_restock():
    """Loads restock recommendations."""
    try:
        from analytics.inventory import get_restock_recommendations
        return get_restock_recommendations()
    except Exception as e:
        return pd.DataFrame()


def ask_rag(question: str) -> dict:
    """
    Calls the RAG pipeline directly (not via FastAPI).
    Returns the RAGResult as a dict for display.
    No caching — every question must be freshly answered.
    """
    try:
        from rag.pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        result   = pipeline.answer(question)
        return {
            "success"      : result.success,
            "answer"       : result.answer,
            "sources"      : result.sources,
            "route"        : result.route,
            "grounded"     : result.grounded,
            "grounding_score": result.grounding_score,
            "duration_sec" : result.duration_sec,
        }
    except Exception as e:
        return {
            "success": False,
            "answer" : f"Error: {str(e)}",
            "sources": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: KPI CARD HTML
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral"):
    """
    Renders a styled KPI card using custom HTML.
    delta_type: "positive" (green), "negative" (red), "neutral" (grey)
    """
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta kpi-{delta_type}">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def fmt_pkr(value) -> str:
    """Formats a number as PKR currency string."""
    try:
        v = float(value)
        if v >= 1_000_000:
            return f"Rs. {v/1_000_000:.1f}M"
        elif v >= 1_000:
            return f"Rs. {v/1_000:.0f}K"
        else:
            return f"Rs. {v:,.0f}"
    except:
        return "Rs. N/A"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: STORE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_store_overview():
    """
    Renders the Store Overview page:
    - Row 1: 4 KPI cards (revenue, profit, margin, units sold)
    - Row 2: 4 KPI cards (products, stock units, inventory value, avg order)
    - Row 3: Revenue trend chart (daily/weekly/monthly toggle)
    - Row 4: Category breakdown (pie chart + bar chart side by side)
    - Row 5: Best sellers + slow movers tables
    """
    st.markdown('<div class="section-header">📊 Store Performance Overview</div>',
                unsafe_allow_html=True)

    kpis = load_kpis()

    if not kpis:
        st.warning("Could not load store KPIs. Check your database connection.")
        return

    # ── Row 1: Financial KPIs ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Total Revenue",
            fmt_pkr(kpis.get("total_revenue", 0)),
            delta_type="positive"
        )
    with c2:
        kpi_card(
            "Total Profit",
            fmt_pkr(kpis.get("total_profit", 0)),
            delta_type="positive"
        )
    with c3:
        margin = kpis.get("overall_margin_pct", 0)
        kpi_card(
            "Profit Margin",
            f"{margin:.1f}%",
            delta="Healthy" if margin >= 20 else "Low",
            delta_type="positive" if margin >= 20 else "negative"
        )
    with c4:
        kpi_card(
            "Units Sold",
            f"{kpis.get('total_units_sold', 0):,}",
            delta_type="neutral"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Inventory KPIs ─────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        kpi_card("Total Products",
                 f"{kpis.get('total_products', 0):,}")
    with c6:
        kpi_card("Stock Units",
                 f"{kpis.get('total_stock_units', 0):,}")
    with c7:
        kpi_card("Inventory Value",
                 fmt_pkr(kpis.get("current_inventory_value", 0)))
    with c8:
        kpi_card("Avg Order Value",
                 fmt_pkr(kpis.get("avg_order_value", 0)))

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Revenue Trend Chart ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Revenue Trend</div>',
                unsafe_allow_html=True)

    # Period selector — radio buttons for D/W/M
    period = st.radio(
        "View by:",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
        label_visibility="collapsed",
        key="revenue_period"
    )
    period_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    df_rev = load_revenue(period_map[period])

    if not df_rev.empty:
        # Plotly line chart — interactive, hover shows values
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x    = df_rev["sale_date"],
            y    = df_rev["total_revenue"],
            name = "Revenue",
            line = dict(color="#6366f1", width=2.5),
            fill = "tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="<b>%{x}</b><br>Revenue: Rs. %{y:,.0f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x    = df_rev["sale_date"],
            y    = df_rev["total_profit"],
            name = "Profit",
            line = dict(color="#4ade80", width=2),
            hovertemplate="<b>%{x}</b><br>Profit: Rs. %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="#94a3b8", family="Inter"),
            legend        = dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis = dict(
                showgrid    = True,
                gridcolor   = "rgba(255,255,255,0.05)",
                showline    = False,
                tickfont    = dict(size=11),
            ),
            yaxis = dict(
                showgrid    = True,
                gridcolor   = "rgba(255,255,255,0.05)",
                showline    = False,
                tickprefix  = "Rs. ",
                tickformat  = ",.0f",
            ),
            margin  = dict(l=0, r=0, t=10, b=0),
            height  = 320,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue data available.")

    st.divider()

    # ── Category Breakdown ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏷️ Category Performance</div>',
                unsafe_allow_html=True)

    df_cat = load_category_margins()

    if not df_cat.empty:
        col_pie, col_bar = st.columns([1, 1])

        with col_pie:
            # Donut chart — revenue share per category
            fig_pie = px.pie(
                df_cat,
                names  = "category",
                values = "total_revenue",
                hole   = 0.55,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title  = "Revenue Share by Category",
            )
            fig_pie.update_traces(
                textposition="outside",
                textinfo="label+percent",
                textfont_size=10,
            )
            fig_pie.update_layout(
                plot_bgcolor  = "rgba(0,0,0,0)",
                paper_bgcolor = "rgba(0,0,0,0)",
                font          = dict(color="#94a3b8", family="Inter"),
                legend        = dict(font=dict(size=10)),
                title_font    = dict(color="#e2e8f0", size=13),
                margin        = dict(l=0, r=0, t=40, b=0),
                height        = 350,
                showlegend    = False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            # Horizontal bar — margin % per category
            df_sorted = df_cat.sort_values("margin_pct", ascending=True)
            fig_bar = px.bar(
                df_sorted,
                x     = "margin_pct",
                y     = "category",
                orientation = "h",
                color = "margin_pct",
                color_continuous_scale="Viridis",
                title = "Profit Margin % by Category",
                text  = "margin_pct",
            )
            fig_bar.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )
            fig_bar.update_layout(
                plot_bgcolor  = "rgba(0,0,0,0)",
                paper_bgcolor = "rgba(0,0,0,0)",
                font          = dict(color="#94a3b8", family="Inter"),
                title_font    = dict(color="#e2e8f0", size=13),
                coloraxis_showscale=False,
                xaxis = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis = dict(showgrid=False),
                margin= dict(l=0, r=60, t=40, b=0),
                height= 350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Best Sellers & Slow Movers ─────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Product Velocity</div>',
                unsafe_allow_html=True)

    top_n = st.slider("Number of products to show", 5, 20, 10, key="velocity_n")
    velocity = load_velocity(top_n)

    col_top, col_slow = st.columns(2)

    with col_top:
        st.markdown("**🔥 Top Sellers**")
        df_top = velocity.get("top_sellers", pd.DataFrame())
        if not df_top.empty:
            display_cols = ["product_name", "category", "units_per_day", "total_revenue"]
            available    = [c for c in display_cols if c in df_top.columns]
            df_display   = df_top[available].copy()
            if "total_revenue" in df_display.columns:
                df_display["total_revenue"] = df_display["total_revenue"].apply(
                    lambda x: f"Rs. {x:,.0f}"
                )
            if "units_per_day" in df_display.columns:
                df_display["units_per_day"] = df_display["units_per_day"].apply(
                    lambda x: f"{x:.2f}"
                )
            df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    with col_slow:
        st.markdown("**🐌 Slow Movers**")
        df_slow = velocity.get("slow_movers", pd.DataFrame())
        if not df_slow.empty:
            display_cols = ["product_name", "category", "units_per_day", "total_revenue"]
            available    = [c for c in display_cols if c in df_slow.columns]
            df_display   = df_slow[available].copy()
            if "total_revenue" in df_display.columns:
                df_display["total_revenue"] = df_display["total_revenue"].apply(
                    lambda x: f"Rs. {x:,.0f}"
                )
            if "units_per_day" in df_display.columns:
                df_display["units_per_day"] = df_display["units_per_day"].apply(
                    lambda x: f"{x:.4f}"
                )
            df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
            st.dataframe(df_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: INVENTORY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_inventory_intelligence():
    """
    Renders the Inventory Intelligence page:
    - Row 1: 4 inventory health KPI cards
    - Row 2: Low stock alerts with urgency color coding
    - Row 3: Dead stock table with capital locked column
    - Row 4: Restock recommendation list
    """
    st.markdown('<div class="section-header">📦 Inventory Intelligence</div>',
                unsafe_allow_html=True)

    health = load_inventory_health()

    if health:
        # ── Inventory KPI cards ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Total Products",
                f"{health.get('total_products', 0):,}",
            )
        with c2:
            dead_count = health.get("dead_stock_products", 0)
            kpi_card(
                "Dead Stock Items",
                str(dead_count),
                delta="⚠️ Requires Action" if dead_count > 0 else "✅ None",
                delta_type="negative" if dead_count > 0 else "positive",
            )
        with c3:
            critical = health.get("low_stock_critical", 0)
            kpi_card(
                "Critical Alerts",
                str(critical),
                delta="🔴 Order Now" if critical > 0 else "✅ All Good",
                delta_type="negative" if critical > 0 else "positive",
            )
        with c4:
            kpi_card(
                "Restock Cost Est.",
                fmt_pkr(health.get("estimated_restock_cost_pkr", 0)),
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Capital locked in dead stock — prominent warning
        dead_value = health.get("dead_stock_value_pkr", 0)
        if dead_value > 0:
            st.markdown(f"""
            <div class="alert-critical">
                💰 Capital Locked in Dead Stock: {fmt_pkr(dead_value)} 
                — These products are not selling and tying up your working capital.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # ── Low Stock Alerts ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Low Stock Alerts</div>',
                unsafe_allow_html=True)

    df_alerts = load_alerts()

    if df_alerts.empty:
        st.success("✅ All products are adequately stocked.")
    else:
        # Show urgency summary badges
        critical_count = len(df_alerts[df_alerts["urgency"] == "CRITICAL"])
        high_count     = len(df_alerts[df_alerts["urgency"] == "HIGH"])
        medium_count   = len(df_alerts[df_alerts["urgency"] == "MEDIUM"])

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(
                f'<div class="alert-critical">🔴 CRITICAL: {critical_count}</div>',
                unsafe_allow_html=True
            )
        with b2:
            st.markdown(
                f'<div class="alert-high">🟠 HIGH: {high_count}</div>',
                unsafe_allow_html=True
            )
        with b3:
            st.markdown(
                f'<div class="alert-medium">🟡 MEDIUM: {medium_count}</div>',
                unsafe_allow_html=True
            )
        with b4:
            st.markdown(
                f'<div class="alert-critical">📋 Total: {len(df_alerts)}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter by urgency
        urgency_filter = st.selectbox(
            "Filter by urgency:",
            ["All", "CRITICAL", "HIGH", "MEDIUM"],
            key="alert_filter"
        )

        df_filtered = df_alerts if urgency_filter == "All" else \
                      df_alerts[df_alerts["urgency"] == urgency_filter]

        # Display table with color-coded urgency
        display_cols = [
            "product_name", "category", "current_stock",
            "days_of_stock_remaining", "urgency",
            "suggested_reorder_qty", "estimated_reorder_cost_pkr", "supplier"
        ]
        available = [c for c in display_cols if c in df_filtered.columns]
        df_show   = df_filtered[available].copy()

        # Format columns for readability
        if "estimated_reorder_cost_pkr" in df_show.columns:
            df_show["estimated_reorder_cost_pkr"] = df_show["estimated_reorder_cost_pkr"].apply(
                lambda x: f"Rs. {x:,.0f}"
            )
        if "days_of_stock_remaining" in df_show.columns:
            df_show["days_of_stock_remaining"] = df_show["days_of_stock_remaining"].apply(
                lambda x: f"{x:.1f} days"
            )

        df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()

    # ── Dead Stock Table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">💀 Dead Stock Analysis</div>',
                unsafe_allow_html=True)

    df_dead = load_dead_stock()

    if df_dead.empty:
        st.success("✅ No dead stock detected.")
    else:
        col_info, col_chart = st.columns([1.2, 1])

        with col_info:
            # Top dead stock by capital locked
            display_cols = [
                "product_name", "category", "current_stock",
                "units_sold", "capital_locked_pkr", "supplier"
            ]
            available = [c for c in display_cols if c in df_dead.columns]
            df_show   = df_dead[available].head(15).copy()

            if "capital_locked_pkr" in df_show.columns:
                df_show["capital_locked_pkr"] = df_show["capital_locked_pkr"].apply(
                    lambda x: f"Rs. {x:,.0f}"
                )

            df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        with col_chart:
            # Bar chart: top 10 dead stock by capital locked
            if "capital_locked_pkr" in df_dead.columns:
                df_top10 = df_dead.nlargest(10, "capital_locked_pkr")
                fig = px.bar(
                    df_top10,
                    x     = "capital_locked_pkr",
                    y     = "product_name",
                    orientation="h",
                    color = "capital_locked_pkr",
                    color_continuous_scale="Reds",
                    title = "Top 10 Dead Stock by Capital Locked",
                )
                fig.update_layout(
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    font          = dict(color="#94a3b8", family="Inter"),
                    title_font    = dict(color="#e2e8f0", size=12),
                    coloraxis_showscale=False,
                    xaxis = dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                        tickprefix="Rs. ", tickformat=",.0f"
                    ),
                    yaxis = dict(showgrid=False, tickfont=dict(size=9)),
                    margin= dict(l=0, r=10, t=40, b=0),
                    height= 380,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Restock Recommendations ───────────────────────────────────────────
    st.markdown('<div class="section-header">🛒 Restock Recommendations</div>',
                unsafe_allow_html=True)

    df_restock = load_restock()

    if df_restock.empty:
        st.success("✅ No restocking needed at this time.")
    else:
        total_cost = df_restock["estimated_reorder_cost_pkr"].sum() \
                     if "estimated_reorder_cost_pkr" in df_restock.columns else 0

        st.info(f"📋 **{len(df_restock)} items** need restocking | "
                f"Estimated total cost: **{fmt_pkr(total_cost)}**")

        display_cols = [
            "product_name", "category", "supplier",
            "current_stock", "suggested_reorder_qty",
            "estimated_reorder_cost_pkr", "urgency"
        ]
        available = [c for c in display_cols if c in df_restock.columns]
        df_show   = df_restock[available].copy()

        if "estimated_reorder_cost_pkr" in df_show.columns:
            df_show["estimated_reorder_cost_pkr"] = df_show["estimated_reorder_cost_pkr"].apply(
                lambda x: f"Rs. {x:,.0f}"
            )

        df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: AI ASSISTANT (RAG CHAT)
# ══════════════════════════════════════════════════════════════════════════════

def page_ai_assistant():
    """
    Renders the AI Assistant page — a full chat interface.
    
    Uses st.session_state to persist chat history across interactions.
    session_state is Streamlit's way of storing data that survives
    page rerenders (which happen on every user interaction).
    
    Without session_state, chat history would reset every time
    the user types a new message.
    """
    st.markdown('<div class="section-header">🤖 AI Retail Assistant</div>',
                unsafe_allow_html=True)

    # Subtitle
    st.markdown("""
    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem;">
    Ask questions about your store in plain English. 
    The assistant uses your actual store data to answer.
    </p>
    """, unsafe_allow_html=True)

    # ── Initialize session state ──────────────────────────────────────────
    # st.session_state persists across rerenders within a session
    # "messages" stores the full chat history as a list of dicts
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_ready" not in st.session_state:
        # Check if RAG pipeline is ready (Ollama running, FAISS loaded)
        try:
            from rag.pipeline import get_rag_pipeline
            pipeline = get_rag_pipeline()
            st.session_state.rag_ready = pipeline.is_ready()
        except Exception:
            st.session_state.rag_ready = False

    # ── RAG status indicator ──────────────────────────────────────────────
    if st.session_state.rag_ready:
        st.markdown(
            '<span style="color:#4ade80;font-size:0.8rem;">● AI Assistant Online (phi3)</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span style="color:#f87171;font-size:0.8rem;">● AI Assistant Offline — Start Ollama</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Suggested questions ───────────────────────────────────────────────
    # Quick-action buttons let users click instead of type
    st.markdown("**Quick Questions:**")
    q_cols = st.columns(3)

    suggested = [
        "What should I restock this week?",
        "Which products are dead stock?",
        "What is my overall profit margin?",
        "Which category makes the most revenue?",
        "Tell me about Tapal Danedar tea",
        "How is my business performing?",
    ]

    for i, question in enumerate(suggested):
        with q_cols[i % 3]:
            # Each button adds the question to the input
            if st.button(f"💬 {question}", key=f"suggested_{i}",
                         use_container_width=True):
                st.session_state.pending_question = question

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Chat history display ──────────────────────────────────────────────
    # Render all previous messages from session state
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # Assistant message with metadata
            st.markdown(
                f'<div class="chat-assistant">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True
            )

            # Show sources if available
            if msg.get("sources"):
                with st.expander(
                    f"📚 Sources ({len(msg['sources'])} documents used)",
                    expanded=False
                ):
                    for s in msg["sources"][:4]:
                        doc_type = s.get("doc_type", "unknown")
                        name     = s.get("product_name") or s.get("category") or "Store Data"
                        score    = s.get("score", 0)
                        preview  = s.get("preview", "")[:200]
                        st.markdown(f"""
                        <div class="chat-source">
                            [{doc_type.upper()}] {name} — Relevance: {score:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(preview)

            # Show grounding score if available
            if "grounding_score" in msg and msg["grounding_score"] is not None:
                score = msg["grounding_score"]
                color = "#4ade80" if score >= 0.4 else "#fde047" if score >= 0.2 else "#f87171"
                st.markdown(
                    f'<span style="color:{color};font-size:0.72rem;">'
                    f'Grounding: {score:.0%} | Route: {msg.get("route","?")} | '
                    f'Time: {msg.get("duration","?")}</span>',
                    unsafe_allow_html=True
                )

    # ── Handle pending question from suggested buttons ────────────────────
    if "pending_question" in st.session_state:
        pending = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": pending})
        st.rerun()

    # ── Chat input box ────────────────────────────────────────────────────
    # st.chat_input renders the sticky input bar at the bottom
    user_input = st.chat_input(
        "Ask about your store inventory, sales, or get recommendations...",
        key="chat_input"
    )

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role"   : "user",
            "content": user_input,
        })

        # Show spinner while RAG pipeline generates answer
        # phi3 on CPU takes 60-180 seconds — spinner prevents
        # the user thinking the page is frozen
        with st.spinner("🤔 Analyzing your store data..."):
            if not st.session_state.rag_ready:
                response = {
                    "success": False,
                    "answer" : "The AI assistant is offline. Please start Ollama with 'ollama serve'.",
                    "sources": [],
                }
            else:
                response = ask_rag(user_input)

        # Build assistant message with metadata
        assistant_msg = {
            "role"           : "assistant",
            "content"        : response.get("answer", "Sorry, I could not generate a response."),
            "sources"        : response.get("sources", []),
            "grounding_score": response.get("grounding_score"),
            "route"          : response.get("route", "?"),
            "duration"       : f"{response.get('duration_sec', 0):.1f}s",
        }
        st.session_state.messages.append(assistant_msg)

        # Rerun rerenders the page with new messages displayed
        st.rerun()

    # ── Clear chat button ─────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """
    Renders the sidebar with navigation and store metadata.
    Returns the selected page name.
    """
    with st.sidebar:
        # Logo / brand
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-size:2.2rem;">🏪</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-top:0.4rem;">
                AI Retail Intelligence
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:0.2rem;">
                Pakistani Kiryana Store BI
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            options=["📊 Store Overview", "📦 Inventory Intel", "🤖 AI Assistant"],
            label_visibility="collapsed",
            key="nav"
        )

        st.divider()

        # Store status summary
        st.markdown("**System Status**")

        kpis   = load_kpis()
        health = load_inventory_health()

        # Database status
        db_ok = bool(kpis)
        st.markdown(
            f'<span style="color:{"#4ade80" if db_ok else "#f87171"};font-size:0.82rem;">'
            f'{"●" if db_ok else "○"} Database: {"Connected" if db_ok else "Error"}</span>',
            unsafe_allow_html=True
        )

        # RAG status
        try:
            from rag.pipeline import get_rag_pipeline
            rag_ok = get_rag_pipeline().is_ready()
        except:
            rag_ok = False

        st.markdown(
            f'<span style="color:{"#4ade80" if rag_ok else "#f87171"};font-size:0.82rem;">'
            f'{"●" if rag_ok else "○"} AI (phi3): {"Online" if rag_ok else "Offline"}</span>',
            unsafe_allow_html=True
        )

        st.divider()

        # Quick stats
        if kpis:
            st.markdown("**Quick Stats**")
            st.markdown(
                f'<div style="font-size:0.8rem;color:#94a3b8;">'
                f'Products: {kpis.get("total_products",0):,}<br>'
                f'Revenue: {fmt_pkr(kpis.get("total_revenue",0))}<br>'
                f'Margin: {kpis.get("overall_margin_pct",0):.1f}%</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            # Clears all @st.cache_data caches
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()

        # Timestamp
        st.markdown(
            f'<div style="font-size:0.7rem;color:#334155;margin-top:1rem;text-align:center;">'
            f'Last loaded: {datetime.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True
        )

    return page


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main function — called when Streamlit runs this file.
    1. Inject CSS
    2. Render sidebar (returns selected page)
    3. Render selected page
    """
    inject_custom_css()
    page = render_sidebar()

    if "Store Overview" in page:
        page_store_overview()
    elif "Inventory Intel" in page:
        page_inventory_intelligence()
    elif "AI Assistant" in page:
        page_ai_assistant()


# Streamlit runs main() automatically when you run:
# streamlit run frontend/dashboard.py
if __name__ == "__main__":
    main()