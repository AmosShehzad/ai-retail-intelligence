"""
Day 17: Interactive Streamlit UI Dashboard - Enterprise Edition

Pages:
1. Store Overview    — KPI cards + revenue chart + category breakdown
2. Inventory Intel   — dead stock + low stock alerts + restock list
3. AI Assistant      — RAG-powered chat with source citations
4. Manage Store      — Search, filter, add products, adjust stock, manage purchase orders
"""

import os
import sys
from pathlib import Path
# Force append parent directory to prevent absolute module path errors
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.classes.__path__ = []  # <-- Add this line first to stop the log spam

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AI Retail Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# LUCIDE SVG ICON LIBRARY
# Inline SVGs — zero external dependency, crisp at any DPI.
# ══════════════════════════════════════════════════════════════════════════════

ICONS = {
    # ── Navigation ──
    "overview":   '<path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/>',
    "inventory":  '<path d="M21 8l-9-5-9 5v8l9 5 9-5z"/><path d="M3.3 8.3L12 13l8.7-4.7M12 13v8"/>',
    "ai":         '<rect x="5" y="9" width="14" height="10" rx="2"/><circle cx="9.5" cy="14" r="1.3" fill="currentColor" stroke="none"/><circle cx="14.5" cy="14" r="1.3" fill="currentColor" stroke="none"/><path d="M12 9V5m-3 0h6"/>',
    "manage":     '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.13.31.43.66 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "store":      '<path d="M3 9l1.5-5h15L21 9"/><path d="M3 9v10a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V9M3 9h18"/><path d="M9 13a2 2 0 1 1-4 0"/><path d="M19 13a2 2 0 1 1-4 0"/>',
    "sync":       '<path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    # ── KPI cards ──
    "revenue":    '<path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trending":   '<path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/>',
    "scale":      '<path d="M12 3v18M5 7h14M5 7l-3 7a3.5 3.5 0 0 0 7 0zM19 7l-3 7a3.5 3.5 0 0 0 7 0z"/>',
    "box":        '<path d="M21 8l-9-5-9 5v8l9 5 9-5z"/><path d="M3.3 8.3L12 13l8.7-4.7M12 13v8"/>',
    "tag":        '<path d="M20.6 13.4 13 21l-9-9 7.6-7.6c.4-.4.9-.6 1.4-.6H19a2 2 0 0 1 2 2v5.2c0 .5-.2 1-.6 1.4z"/><circle cx="14.5" cy="9.5" r="1.5"/>',
    "building":   '<rect x="4" y="3" width="16" height="18" rx="1"/><path d="M9 7h1M14 7h1M9 11h1M14 11h1M9 15h1M14 15h1"/><path d="M9 21v-3a3 3 0 0 1 6 0v3"/>',
    "wallet":     '<rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="12" cy="12" r="3"/><path d="M6 10v.01M18 14v.01"/>',
    "cart":       '<circle cx="9" cy="20" r="1.3"/><circle cx="18" cy="20" r="1.3"/><path d="M3 4h2l2.6 12.2A2 2 0 0 0 9.5 18h8.1a2 2 0 0 0 1.95-1.57L21.5 9H6"/>',
    "skull":      '<circle cx="12" cy="11" r="7"/><circle cx="9" cy="11" r="1.1" fill="currentColor" stroke="none"/><circle cx="15" cy="11" r="1.1" fill="currentColor" stroke="none"/><path d="M9 16l1-2h4l1 2M10 18h4"/>',
    "bell":       '<path d="M12 2a6 6 0 0 0-6 6v6H4v2h16v-2h-2V8a6 6 0 0 0-6-6z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
    "card":       '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20M6 15h4"/>',
    "chart":      '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/>',
    "zap":        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "db":         '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 5v4c0 1.66-4.03 3-9 3S3 10.66 3 9V5"/><path d="M3 9v6c0 1.66 4.03 3 9 3s9-1.34 9-3V9"/>',
    "cpu":        '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "arrow_up":   '<path d="M17 7l-10 10M17 7H8M17 7v9"/>',
    "arrow_down": '<path d="M7 17l10-10M7 17h9M7 17V8"/>',
    "check":      '<polyline points="20 6 9 17 4 12"/>',
    "plus":       '<path d="M12 5v14M5 12h14"/>',
    "trash":      '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    "msg":        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "book":       '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "list":       '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
}


def svg(key: str, size: int = 18, color: str = "currentColor", stroke: float = 1.8) -> str:
    """Return a self-contained Lucide SVG string."""
    path = ICONS.get(key, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Variables ─────────────────────────────────────────────────────────── */
:root {
    --bg:          #080F1E;
    --bg-2:        #0D1628;
    --card:        #111827;
    --card-hover:  #161F32;
    --border:      #1E2D45;
    --border-hi:   #2A3F5F;
    --text:        #F1F5F9;
    --muted:       #64748B;
    --muted-hi:    #94A3B8;
    --blue:        #3B82F6;
    --blue-dim:    rgba(59,130,246,0.12);
    --blue-glow:   rgba(59,130,246,0.35);
    --green:       #10B981;
    --green-dim:   rgba(16,185,129,0.12);
    --red:         #EF4444;
    --red-dim:     rgba(239,68,68,0.12);
    --amber:       #F59E0B;
    --amber-dim:   rgba(245,158,11,0.12);
    --indigo:      #6366F1;
    --indigo-dim:  rgba(99,102,241,0.12);
    --r-sm:  6px;
    --r-md:  10px;
    --r-lg:  14px;
    --r-xl:  18px;
}

/* ─── Reset & Font ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header { visibility: hidden; }

/* ─── App Background ────────────────────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
.main  { background: var(--bg) !important; }
.main .block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1480px;
}

/* ─── Animations ────────────────────────────────────────────────────────── */
@keyframes fadeUp   { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes glow-in  { from { opacity:0; transform:scale(0.94); }     to { opacity:1; transform:scale(1); } }
@keyframes pulse-dot{ 0%,100%{ transform:scale(1); opacity:1; }      50%{ transform:scale(1.4); opacity:.6; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* ─── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* ─── Custom Nav Items ──────────────────────────────────────────────────── */
.nav-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 1.25rem;
    margin: 1.5rem 0 0.5rem;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 1.25rem;
    margin: 0.1rem 0.75rem;
    border-radius: var(--r-md);
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    border: 1px solid transparent;
    text-decoration: none !important;
    position: relative;
}
.nav-item:hover {
    background: rgba(59,130,246,0.07);
    border-color: var(--border);
}
.nav-item.active {
    background: var(--blue-dim);
    border-color: rgba(59,130,246,0.25);
}
.nav-item.active::before {
    content: '';
    position: absolute;
    left: -0.75rem; top: 20%; bottom: 20%;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--blue);
    box-shadow: 0 0 10px var(--blue-glow);
}
.nav-icon {
    width: 34px; height: 34px;
    border-radius: var(--r-sm);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: transform 0.2s ease;
}
.nav-item:hover .nav-icon { transform: scale(1.08); }
.nav-item.active .nav-icon { transform: scale(1.05); }
.nav-label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--muted-hi);
    transition: color 0.15s;
}
.nav-item.active .nav-label { color: var(--text) !important; font-weight: 600; }
.nav-item:hover .nav-label  { color: var(--text) !important; }

/* ─── Sidebar Brand ─────────────────────────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.875rem;
    padding: 1.5rem 1.25rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.brand-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
    border-radius: var(--r-md);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(59,130,246,0.35), 0 0 0 1px rgba(255,255,255,0.06) inset;
}
.brand-name  { font-size: 1rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.brand-sub   { font-size: 0.72rem; color: var(--muted); margin-top: 1px; }

/* ─── Sidebar Footer ────────────────────────────────────────────────────── */
.sidebar-footer {
    padding: 1rem 1.25rem;
    border-top: 1px solid var(--border);
    margin-top: auto;
}
.status-row {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.8rem; color: var(--muted-hi);
    margin-bottom: 0.4rem;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.status-dot.on  { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse-dot 2.5s ease infinite; }
.status-dot.off { background: var(--red); }
.sidebar-meta {
    display: flex; justify-content: space-between;
    font-size: 0.68rem; color: var(--muted);
    margin-top: 0.75rem; padding-top: 0.75rem;
    border-top: 1px solid var(--border);
}
.sidebar-metrics {
    padding: 0 1.25rem;
    margin-bottom: 1rem;
}
.sidebar-metric-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.8rem; padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
}
.sidebar-metric-row:last-child { border-bottom: none; }
.sidebar-metric-label { color: var(--muted); }
.sidebar-metric-val   { color: var(--text); font-weight: 600; font-size: 0.82rem; }

/* ─── Page Header ───────────────────────────────────────────────────────── */
.page-header {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 2rem;
    animation: fadeUp 0.3s ease both;
}
.page-header-icon {
    width: 46px; height: 46px;
    border-radius: var(--r-md);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    border: 1px solid var(--border-hi);
}
.page-title {
    font-size: 1.5rem !important; font-weight: 800 !important;
    color: var(--text) !important; margin: 0 !important;
    letter-spacing: -0.03em; line-height: 1.2;
}
.page-sub { font-size: 0.875rem; color: var(--muted); margin-top: 0.2rem; }

/* ─── Section Header ────────────────────────────────────────────────────── */
.section-head {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1rem; font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-head svg { color: var(--blue); }
.section-head:first-child { margin-top: 0; }

/* ─── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-xl);
    padding: 1.4rem 1.5rem;
    position: relative; overflow: hidden;
    display: flex; flex-direction: column; gap: 0.6rem;
    transition: transform 0.2s cubic-bezier(.16,1,.3,1), box-shadow 0.2s ease, border-color 0.2s ease;
    animation: fadeUp 0.35s ease both;
    min-height: 140px;
}
.kpi-card::after {
    content: '';
    position: absolute;
    inset: 0; border-radius: inherit;
    background: radial-gradient(circle at top right, var(--accent-glow, rgba(59,130,246,0.06)) 0%, transparent 60%);
    pointer-events: none;
}
.kpi-card:hover {
    transform: translateY(-5px);
    border-color: var(--border-hi);
    box-shadow: 0 20px 40px -12px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.02) inset;
}
.kpi-card:hover .kpi-icon { transform: scale(1.1) rotate(-6deg); }
.kpi-top {
    display: flex; justify-content: space-between; align-items: flex-start;
}
.kpi-label {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted);
}
.kpi-icon {
    width: 36px; height: 36px;
    border-radius: var(--r-sm);
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.2s cubic-bezier(.16,1,.3,1);
    flex-shrink: 0;
}
.kpi-value {
    font-size: 2.1rem; font-weight: 800;
    color: var(--text);
    letter-spacing: -0.04em; line-height: 1;
    margin-top: 0.25rem;
}
.kpi-foot {
    margin-top: auto;
    display: flex; align-items: center; gap: 0.4rem;
    min-height: 1.5rem;
}
.kpi-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.72rem; font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
}
.kpi-badge svg { flex-shrink: 0; }
.badge-up   { background: var(--green-dim); color: var(--green); }
.badge-down { background: var(--red-dim);   color: var(--red); }
.badge-flat { background: rgba(100,116,139,0.15); color: var(--muted-hi); }

/* ─── Alert Pill ────────────────────────────────────────────────────────── */
.alert-pill {
    padding: 0.85rem 1.1rem;
    border-radius: var(--r-md);
    display: flex; align-items: center; gap: 0.75rem;
    font-size: 0.875rem; font-weight: 500;
    margin-bottom: 1rem;
    border: 1px solid transparent;
}
.alert-pill svg { flex-shrink: 0; }
.ap-red    { background: var(--red-dim);   border-color: rgba(239,68,68,0.25);   color: #FCA5A5; }
.ap-amber  { background: var(--amber-dim); border-color: rgba(245,158,11,0.25);  color: #FCD34D; }
.ap-blue   { background: var(--blue-dim);  border-color: rgba(59,130,246,0.25);  color: #93C5FD; }
.ap-subtle { background: var(--card);      border-color: var(--border);          color: var(--text); }

/* ─── Stat Pill row (alert counts) ─────────────────────────────────────── */
.stat-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 0.75rem 1rem;
    font-size: 0.875rem; font-weight: 600; color: var(--text);
}
.stat-pill svg { flex-shrink: 0; }

/* ─── Chat Bubbles ──────────────────────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: #fff;
    border-radius: 14px 14px 3px 14px;
    padding: 0.9rem 1.15rem;
    margin: 0.65rem 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.9rem;
    line-height: 1.55;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3);
}
.chat-ai {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 14px 14px 14px 3px;
    padding: 0.9rem 1.15rem;
    margin: 0.65rem 0;
    max-width: 84%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-ai-header {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.78rem; font-weight: 700;
    color: var(--blue); margin-bottom: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.chat-source {
    background: var(--indigo-dim);
    border: 1px solid rgba(99,102,241,0.2);
    color: #A5B4FC;
    border-radius: var(--r-sm);
    padding: 0.45rem 0.7rem;
    font-size: 0.75rem; font-weight: 500;
    margin-top: 0.5rem;
    display: inline-block;
}
.chat-meta {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.72rem; color: var(--muted);
    margin: 0.3rem 0 0 0.1rem;
}

/* ─── Streamlit Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.4) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary buttons */
[data-btn="secondary"] .stButton > button {
    background: transparent !important;
    color: var(--muted-hi) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
[data-btn="secondary"] .stButton > button:hover {
    border-color: var(--blue) !important;
    color: var(--text) !important;
    background: var(--blue-dim) !important;
}
/* Danger buttons */
[data-btn="danger"] .stButton > button {
    background: transparent !important;
    color: var(--red) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    box-shadow: none !important;
}
[data-btn="danger"] .stButton > button:hover {
    background: var(--red-dim) !important;
    border-color: var(--red) !important;
}

/* ─── Inputs ────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-md) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-md) !important;
}

/* ─── Radio (period toggle only — segmented pill) ───────────────────────── */
.stRadio [role="radiogroup"] { gap: 0.3rem; }
.stRadio [role="radiogroup"] label {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.4rem 0.85rem !important;
    transition: all 0.15s ease;
    cursor: pointer;
}
.stRadio [role="radiogroup"] label:hover { border-color: var(--blue); }
.stRadio [role="radiogroup"] label:has(input:checked) {
    background: var(--blue-dim);
    border-color: rgba(59,130,246,0.4);
}

/* ─── Slider ────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] > div > div { background: var(--blue) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #fff !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.25) !important;
}

/* ─── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important; font-weight: 500;
    padding: 0.85rem 0.25rem;
    border-bottom: 2px solid transparent;
    transition: color 0.15s;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
}

/* ─── DataFrames ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    /* removed overflow: hidden; to restore horizontal scrolling */
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #0A1220 !important;
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] [role="row"]:nth-of-type(even) {
    background: rgba(255,255,255,0.015);
}
[data-testid="stDataFrame"] [role="gridcell"] {
    font-size: 0.85rem !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
                
/* ─── Expanders ─────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-bottom-left-radius: var(--r-md) !important;
    border-bottom-right-radius: var(--r-md) !important;
}

/* ─── Alerts ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.875rem !important;
}

/* ─── Divider ───────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.75rem 0 !important; opacity: 1; }

/* ─── PO card ───────────────────────────────────────────────────────────── */
.po-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 1rem 1.15rem;
    transition: border-color 0.15s;
}
.po-card:hover { border-color: var(--border-hi); }
.po-name { font-weight: 700; color: var(--text); font-size: 0.95rem; }
.po-meta { color: var(--muted); font-size: 0.82rem; margin-top: 0.25rem; }

/* ─── Sidebar button ────────────────────────────────────────────────────── */
.sb-btn-wrap .stButton > button {
    background: transparent !important;
    color: var(--muted-hi) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.85rem !important;
    font-weight: 500 !important;
}
.sb-btn-wrap .stButton > button:hover {
    border-color: var(--blue) !important;
    color: var(--text) !important;
    background: var(--blue-dim) !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_kpis():
    try:
        from analytics.engines import get_store_kpis
        return get_store_kpis()
    except Exception:
        return {}

@st.cache_data(ttl=300)
def load_revenue(period: str = "D"):
    try:
        from analytics.engines import get_revenue_by_period
        df = get_revenue_by_period(period)
        if not df.empty and "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_category_margins():
    try:
        from analytics.engines import get_category_margins
        return get_category_margins()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_velocity(top_n: int = 10):
    try:
        from analytics.engines import get_product_velocity
        return get_product_velocity(top_n)
    except Exception:
        return {"top_sellers": pd.DataFrame(), "slow_movers": pd.DataFrame()}

@st.cache_data(ttl=300)
def load_inventory_health():
    try:
        from analytics.inventory import get_inventory_health_summary
        return get_inventory_health_summary()
    except Exception:
        return {}

@st.cache_data(ttl=300)
def load_dead_stock():
    try:
        from analytics.inventory import get_dead_stock
        return get_dead_stock()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_alerts():
    try:
        from analytics.inventory import get_low_stock_alerts
        return get_low_stock_alerts()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_restock():
    try:
        from analytics.inventory import get_restock_recommendations
        return get_restock_recommendations()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_kpi_deltas():
    """Computes month-over-month revenue/profit deltas dynamically."""
    try:
        from analytics.engines import get_revenue_by_period
        df = get_revenue_by_period("M")
        if len(df) < 2:
            return {"revenue_delta": None, "profit_delta": None}
        current  = df.iloc[-1]
        previous = df.iloc[-2]
        rev_delta = round(
            (current["total_revenue"] - previous["total_revenue"]) / previous["total_revenue"] * 100, 1
        ) if previous["total_revenue"] > 0 else 0
        profit_delta = round(
            (current["total_profit"] - previous["total_profit"]) / previous["total_profit"] * 100, 1
        ) if previous["total_profit"] > 0 else 0
        return {"revenue_delta": rev_delta, "profit_delta": profit_delta}
    except Exception:
        return {"revenue_delta": None, "profit_delta": None}

def ask_rag(question: str) -> dict:
    """Invokes RAG pipeline safely extracting attributes or keys dynamically."""
    try:
        from rag.pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        result   = pipeline.answer(question)
        
        def safe_get(obj, attr_or_key, default_val=None):
            if isinstance(obj, dict):
                return obj.get(attr_or_key, default_val)
            return getattr(obj, attr_or_key, default_val)

        return {
            "success":         safe_get(result, "success", True),
            "answer":          safe_get(result, "answer", str(result)),
            "sources":         safe_get(result, "sources", []),
            "route":           safe_get(result, "route", "N/A"),
            "grounded":        safe_get(result, "grounded", False),
            "grounding_score": safe_get(result, "grounding_score", 0.0),
            "duration_sec":    safe_get(result, "duration_sec", 0.0),
        }
    except Exception as e:
        return {"success": False, "answer": f"Error running RAG pipeline: {str(e)}", "sources": []}


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_pkr(value) -> str:
    try:
        v = float(value)
        if   v >= 1_000_000: return f"Rs.{v/1_000_000:.1f}M"
        elif v >= 1_000:     return f"Rs.{v/1_000:.0f}K"
        else:                return f"Rs.{v:,.0f}"
    except Exception:
        return "Rs.N/A"


def page_header(title: str, subtitle: str, icon_key: str, bg: str, color: str):
    """Full-width page header with a colored icon chip."""
    icon_html = svg(icon_key, size=22, color=color)
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-header-icon" style="background:{bg};">{icon_html}</div>'
        f'<div><h1 class="page-title">{title}</h1>'
        f'<p class="page-sub">{subtitle}</p></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def section_head(label: str, icon_key: str):
    icon_html = svg(icon_key, size=16, color="var(--blue)")
    st.markdown(
        f'<div class="section-head">{icon_html}<span>{label}</span></div>',
        unsafe_allow_html=True
    )


def kpi_card(
    label: str,
    value: str,
    delta: str = "",
    delta_type: str = "neutral",
    icon_key: str = "chart",
    icon_bg: str = "rgba(59,130,246,0.14)",
    icon_color: str = "#3B82F6",
    accent_glow: str = "rgba(59,130,246,0.06)",
):
    """Enterprise KPI card packed into single line to secure the HTML layout parser."""
    icon_html = svg(icon_key, size=17, color=icon_color)

    if delta_type == "positive":
        arr = svg("arrow_up", size=11, color="var(--green)", stroke=2.5)
        badge = f'<span class="kpi-badge badge-up">{arr}{delta}</span>'
    elif delta_type == "negative":
        arr = svg("arrow_down", size=11, color="var(--red)", stroke=2.5)
        badge = f'<span class="kpi-badge badge-down">{arr}{delta}</span>'
    elif delta:
        badge = f'<span class="kpi-badge badge-flat">{delta}</span>'
    else:
        badge = ""

    st.markdown(
        f'<div class="kpi-card" style="--accent-glow:{accent_glow};">'
        f'<div class="kpi-top"><span class="kpi-label">{label}</span>'
        f'<span class="kpi-icon" style="background:{icon_bg}; color:{icon_color};">{icon_html}</span></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-foot">{badge}</div></div>',
        unsafe_allow_html=True
    )


def secondary_button(label: str, key: str, full: bool = True) -> bool:
    st.markdown('<div data-btn="secondary">', unsafe_allow_html=True)
    r = st.button(label, key=key, use_container_width=full)
    st.markdown('</div>', unsafe_allow_html=True)
    return r

def danger_button(label: str, key: str, full: bool = True) -> bool:
    st.markdown('<div data-btn="danger">', unsafe_allow_html=True)
    r = st.button(label, key=key, use_container_width=full)
    st.markdown('</div>', unsafe_allow_html=True)
    return r


def alert_pill(text: str, cls: str, icon_key: str, icon_color: str):
    icon_html = svg(icon_key, size=16, color=icon_color)
    st.markdown(
        f'<div class="alert-pill {cls}">{icon_html}<span>{text}</span></div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Custom Query-Param Navigation Engine
# ══════════════════════════════════════════════════════════════════════════════

NAV_ITEMS = [
    ("overview",   "overview",  "Store Overview",   "var(--blue-dim)",   "var(--blue)"),
    ("inventory",  "inventory", "Inventory Intel",  "var(--amber-dim)",  "var(--amber)"),
    ("ai",         "ai",        "AI Assistant",     "var(--indigo-dim)", "var(--indigo)"),
    ("manage",     "manage",    "Manage Store",     "var(--green-dim)",  "var(--green)"),
]


def render_sidebar() -> str:
    """Renders highly customized sidebar links bound natively to state query parameters."""
    # Synchronize native URL router params with backend states
    if "page" in st.query_params:
        st.session_state.active_page = st.query_params["page"]
    elif "active_page" not in st.session_state:
        st.session_state.active_page = "overview"
        st.query_params["page"] = "overview"

    with st.sidebar:
        brand_svg = svg("store", size=19, color="white", stroke=1.8)
        st.markdown(
            f'<div class="sidebar-brand">'
            f'<div class="brand-logo">{brand_svg}</div>'
            f'<div><div class="brand-name">AI Retail Intel</div>'
            f'<div class="brand-sub">Enterprise BI Platform</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="nav-section-label">Main Navigation</div>', unsafe_allow_html=True)
        
        # Build functional link-driven list
        nav_html = ""
        for page_key, icon_key, label, bg, color in NAV_ITEMS:
            is_active = (st.session_state.active_page == page_key)
            active_cls = "active" if is_active else ""
            icon_html  = svg(icon_key, size=17, color=color)
            nav_html += (
                f'<a class="nav-item {active_cls}" href="?page={page_key}" target="_self">'
                f'<span class="nav-icon" style="background:{bg};">{icon_html}</span>'
                f'<span class="nav-label">{label}</span>'
                f'</a>'
            )
        st.markdown(nav_html, unsafe_allow_html=True)
        
        st.divider()
        kpis   = load_kpis()
        health = load_inventory_health()
        
        if kpis:
            st.markdown('<div class="nav-section-label">Live Metrics</div>', unsafe_allow_html=True)
            active_skus = kpis.get("total_products", 0)
            net_rev     = fmt_pkr(kpis.get("total_revenue", 0))
            avg_margin  = f'{kpis.get("overall_margin_pct", 0):.1f}%'
            st.markdown(
                f'<div class="sidebar-metrics">'
                f'<div class="sidebar-metric-row"><span class="sidebar-metric-label">Active SKUs</span><span class="sidebar-metric-val">{active_skus:,}</span></div>'
                f'<div class="sidebar-metric-row"><span class="sidebar-metric-label">Net Revenue</span><span class="sidebar-metric-val">{net_rev}</span></div>'
                f'<div class="sidebar-metric-row"><span class="sidebar-metric-label">Avg Margin</span><span class="sidebar-metric-val">{avg_margin}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        st.markdown('<div class="nav-section-label">System Health</div>', unsafe_allow_html=True)
        db_ok = bool(kpis)
        try:
            from rag.pipeline import get_rag_pipeline
            rag_ok = get_rag_pipeline().is_ready()
        except Exception:
            rag_ok = False
            
        db_icon = svg("db",  size=13, color="var(--green)" if db_ok  else "var(--red)")
        ai_icon = svg("cpu", size=13, color="var(--green)" if rag_ok else "var(--red)")
        st.markdown(
            f'<div style="padding: 0 1.25rem;">'
            f'<div class="status-row"><span class="status-dot {"on" if db_ok else "off"}"></span>{db_icon}<span>Database {"Connected" if db_ok else "Error"}</span></div>'
            f'<div class="status-row"><span class="status-dot {"on" if rag_ok else "off"}"></span>{ai_icon}<span>AI Engine {"Online" if rag_ok else "Offline"}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.divider()
        
        st.markdown('<div class="sb-btn-wrap">', unsafe_allow_html=True)
        if st.button("Sync Platform Data", use_container_width=True, key="sync_btn"):
            st.cache_data.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="sidebar-meta">'
            f'<span>v2.5.0 · Day 17</span>'
            f'<span>{datetime.now().strftime("%H:%M")}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    return st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STORE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_store_overview():
    page_header("Store Overview", "Revenue, profit, and product velocity across your entire catalog.", "overview", "var(--blue-dim)", "var(--blue)")

    kpis = load_kpis()
    if not kpis:
        st.warning("Could not load store KPIs — check your database connection.")
        return

    # ── Row 1: Financial ──
    deltas = load_kpi_deltas()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rd = deltas.get("revenue_delta")
        if rd is not None:
            delta_text = f"{'↑' if rd >= 0 else '↓'} {abs(rd)}% vs last month"
            delta_type = "positive" if rd >= 0 else "negative"
        else:
            delta_text, delta_type = "No prior month data", "neutral"
        kpi_card("Total Revenue", fmt_pkr(kpis.get("total_revenue", 0)), delta_text, delta_type,
                 "revenue", "rgba(16,185,129,0.14)", "#10B981", "rgba(16,185,129,0.07)")
    with c2:
        pf = deltas.get("profit_delta")
        if pf is not None:
            delta_text = f"{'↑' if pf >= 0 else '↓'} {abs(pf)}% vs last month"
            delta_type = "positive" if pf >= 0 else "negative"
        else:
            delta_text, delta_type = "No prior month data", "neutral"
        kpi_card("Total Profit", fmt_pkr(kpis.get("total_profit", 0)), delta_text, delta_type,
                 "trending", "rgba(59,130,246,0.14)", "#3B82F6", "rgba(59,130,246,0.06)")
    with c3:
        margin = kpis.get("overall_margin_pct", 0)
        kpi_card("Profit Margin", f"{margin:.1f}%",
                 "Healthy" if margin >= 20 else "Needs attention",
                 "positive" if margin >= 20 else "negative",
                 "scale", "rgba(99,102,241,0.14)", "#6366F1", "rgba(99,102,241,0.06)")
    with c4:
        kpi_card("Units Sold", f"{kpis.get('total_units_sold', 0):,}", "Stable momentum", "neutral",
                 "box", "rgba(245,158,11,0.14)", "#F59E0B", "rgba(245,158,11,0.06)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Inventory ──
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("Total Products", f"{kpis.get('total_products', 0):,}", "", "neutral",
                 "tag", "rgba(59,130,246,0.12)", "#3B82F6")
    with c6:
        kpi_card("Stock Units", f"{kpis.get('total_stock_units', 0):,}", "", "neutral",
                 "building", "rgba(99,102,241,0.12)", "#6366F1")
    with c7:
        kpi_card("Inventory Value", fmt_pkr(kpis.get("current_inventory_value", 0)), "", "neutral",
                 "wallet", "rgba(16,185,129,0.12)", "#10B981")
    with c8:
        kpi_card("Avg Order Value", fmt_pkr(kpis.get("avg_order_value", 0)), "", "neutral",
                 "cart", "rgba(245,158,11,0.12)", "#F59E0B")

    st.divider()

    # ── Revenue Trend ──
    section_head("Revenue Trend", "trending")
    period = st.radio("Period", ["Daily", "Weekly", "Monthly"], horizontal=True, label_visibility="collapsed", key="revenue_period")
    
    # Clean future warning discrepancies
    p_code = "ME" if period == "Monthly" else "W" if period == "Weekly" else "D"
    df_rev = load_revenue(p_code)

    if not df_rev.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_rev["sale_date"], y=df_rev["total_revenue"], name="Revenue",
            line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
            hovertemplate="<b>%{x}</b><br>Revenue: Rs. %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_rev["sale_date"], y=df_rev["total_profit"], name="Profit",
            line=dict(color="#10B981", width=2.5),
            hovertemplate="<b>%{x}</b><br>Profit: Rs. %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748B", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=True, gridcolor="#1E2D45", showline=False, tickfont=dict(size=11, color="#64748B")),
            yaxis=dict(showgrid=True, gridcolor="#1E2D45", showline=False, tickprefix="Rs. ", tickformat=",.0f", tickfont=dict(size=11, color="#64748B")),
            margin=dict(l=0, r=0, t=10, b=0), height=360, hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue data available.")

    st.divider()

    # ── Category Performance ──
    section_head("Category Performance", "chart")
    df_cat = load_category_margins()

    if not df_cat.empty:
        col_pie, col_bar = st.columns(2, gap="large")
        with col_pie:
            fig_pie = px.pie(df_cat, names="category", values="total_revenue", hole=0.62,
                             color_discrete_sequence=["#3B82F6","#10B981","#6366F1","#F59E0B","#EF4444","#8B5CF6","#EC4899"],
                             title="Revenue Share by Category")
            fig_pie.update_traces(textposition="outside", textinfo="label+percent", textfont_size=11)
            fig_pie.update_layout(
                plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B", family="Inter"),
                title_font=dict(color="#F1F5F9", size=14, family="Inter"),
                margin=dict(l=0, r=0, t=40, b=0), height=360, showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            df_sorted = df_cat.sort_values("margin_pct", ascending=True)
            # FIXED: Quoted #EF4444, #F59E0B, and #10B981 inside color_continuous_scale
            fig_bar = px.bar(df_sorted, x="margin_pct", y="category", orientation="h",
                             color="margin_pct", color_continuous_scale=[[0, "#EF4444"], [0.5, "#F59E0B"], [1, "#10B981"]],
                             title="Profit Margin % by Category", text="margin_pct")
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B", family="Inter"),
                title_font=dict(color="#F1F5F9", size=14, family="Inter"),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#1E2D45"),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=60, t=40, b=0), height=360,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Product Velocity ──
    section_head("Product Velocity", "zap")
    top_n    = st.slider("Products to show", 5, 20, 10, key="velocity_n")
    velocity = load_velocity(top_n)
    col_top, col_slow = st.columns(2, gap="large")

    with col_top:
        st.markdown("<p style='font-weight:700; font-size:0.9rem; color:var(--green); margin-bottom:0.75rem;'>🔥 Top Sellers</p>", unsafe_allow_html=True)
        df_top = velocity.get("top_sellers", pd.DataFrame())
        if not df_top.empty:
            disp = [c for c in ["product_name","category","units_per_day","total_revenue"] if c in df_top.columns]
            d = df_top[disp].copy()
            if "total_revenue"  in d: d["total_revenue"]  = d["total_revenue"].apply(lambda x: f"Rs. {x:,.0f}")
            if "units_per_day"  in d: d["units_per_day"]  = d["units_per_day"].apply(lambda x: f"{x:.2f}")
            d.columns = [c.replace("_"," ").title() for c in d.columns]
            st.dataframe(d, use_container_width=True, hide_index=True)

    with col_slow:
        st.markdown("<p style='font-weight:700; font-size:0.9rem; color:var(--muted); margin-bottom:0.75rem;'>🐌 Slow Movers</p>", unsafe_allow_html=True)
        df_slow = velocity.get("slow_movers", pd.DataFrame())
        if not df_slow.empty:
            disp = [c for c in ["product_name","category","units_per_day","total_revenue"] if c in df_slow.columns]
            d = df_slow[disp].copy()
            if "total_revenue"  in d: d["total_revenue"]  = d["total_revenue"].apply(lambda x: f"Rs. {x:,.0f}")
            if "units_per_day"  in d: d["units_per_day"]  = d["units_per_day"].apply(lambda x: f"{x:.4f}")
            d.columns = [c.replace("_"," ").title() for c in d.columns]
            st.dataframe(d, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INVENTORY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_inventory_intelligence():
    page_header("Inventory Intelligence", "Dead stock, low-stock alerts, and restock recommendations.", "inventory", "var(--amber-dim)", "var(--amber)")

    health = load_inventory_health()

    if health:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total Products", f"{health.get('total_products', 0):,}", "", "neutral",
                     "box", "rgba(59,130,246,0.14)", "#3B82F6")
        with c2:
            dead = health.get("dead_stock_products", 0)
            kpi_card("Dead Stock Items", str(dead),
                     "Requires action" if dead > 0 else "None detected",
                     "negative" if dead > 0 else "positive",
                     "skull", "rgba(239,68,68,0.14)", "#EF4444", "rgba(239,68,68,0.05)")
        with c3:
            crit = health.get("low_stock_critical", 0)
            kpi_card("Critical Alerts", str(crit),
                     "Order now" if crit > 0 else "All stocked",
                     "negative" if crit > 0 else "positive",
                     "bell", "rgba(245,158,11,0.14)", "#F59E0B", "rgba(245,158,11,0.05)")
        with c4:
            kpi_card("Restock Cost Est.", fmt_pkr(health.get("estimated_restock_cost_pkr", 0)), "", "neutral",
                     "card", "rgba(99,102,241,0.14)", "#6366F1")

        st.markdown("<br>", unsafe_allow_html=True)
        dead_val = health.get("dead_stock_value_pkr", 0)
        if dead_val > 0:
            alert_pill(
                f"<strong>Capital locked in dead stock: {fmt_pkr(dead_val)}</strong> — these items are not selling and tying up working capital.",
                "ap-red", "skull", "#EF4444"
            )

    st.divider()

    # ── Low Stock Alerts ──
    section_head("Low Stock Alerts", "bell")
    df_alerts = load_alerts()

    if df_alerts.empty:
        st.success("✅ All products are adequately stocked.")
    else:
        crit_n   = len(df_alerts[df_alerts["urgency"] == "CRITICAL"])
        high_n   = len(df_alerts[df_alerts["urgency"] == "HIGH"])
        med_n    = len(df_alerts[df_alerts["urgency"] == "MEDIUM"])
        total_n  = len(df_alerts)

        b1, b2, b3, b4 = st.columns(4)
        bell = svg("bell", 15, "currentColor")
        with b1: st.markdown(f'<div class="stat-pill ap-red"   >{bell}<span>Critical — {crit_n}</span></div>', unsafe_allow_html=True)
        with b2: st.markdown(f'<div class="stat-pill ap-amber" >{bell}<span>High — {high_n}</span></div>',     unsafe_allow_html=True)
        with b3: st.markdown(f'<div class="stat-pill ap-blue"  >{bell}<span>Medium — {med_n}</span></div>',    unsafe_allow_html=True)
        with b4: st.markdown(f'<div class="stat-pill ap-subtle">{bell}<span>Total — {total_n}</span></div>',   unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        urgency_filter = st.selectbox("Filter by urgency", ["All", "CRITICAL", "HIGH", "MEDIUM"], key="alert_filter")
        df_f = df_alerts if urgency_filter == "All" else df_alerts[df_alerts["urgency"] == urgency_filter]

        cols = ["product_name","category","current_stock","days_of_stock_remaining","urgency","suggested_reorder_qty","estimated_reorder_cost_pkr","supplier"]
        avail = [c for c in cols if c in df_f.columns]
        d = df_f[avail].copy()
        if "estimated_reorder_cost_pkr"  in d: d["estimated_reorder_cost_pkr"]  = d["estimated_reorder_cost_pkr"].apply(lambda x: f"Rs. {x:,.0f}")
        if "days_of_stock_remaining"     in d: d["days_of_stock_remaining"]     = d["days_of_stock_remaining"].apply(lambda x: f"{x:.1f} days")
        d.columns = [c.replace("_"," ").title() for c in d.columns]
        st.dataframe(d, use_container_width=True, hide_index=True)

    st.divider()

    # ── Dead Stock ──
    section_head("Dead Stock Analysis", "skull")
    df_dead = load_dead_stock()

    if df_dead.empty:
        st.success("✅ No dead stock detected.")
    else:
        col_tbl, col_chart = st.columns([1.2, 1], gap="large")
        with col_tbl:
            cols  = ["product_name","category","current_stock","units_sold","capital_locked_pkr","supplier"]
            avail = [c for c in cols if c in df_dead.columns]
            d = df_dead[avail].head(15).copy()
            if "capital_locked_pkr" in d: d["capital_locked_pkr"] = d["capital_locked_pkr"].apply(lambda x: f"Rs. {x:,.0f}")
            d.columns = [c.replace("_"," ").title() for c in d.columns]
            st.dataframe(d, use_container_width=True, hide_index=True)
        with col_chart:
            if "capital_locked_pkr" in df_dead.columns:
                top10 = df_dead.nlargest(10, "capital_locked_pkr")
                # FIXED: Quoted #EF4444 and #F59E0B inside color_continuous_scale
                fig = px.bar(top10, x="capital_locked_pkr", y="product_name", orientation="h",
                             color="capital_locked_pkr", color_continuous_scale=[[0, "#F59E0B"], [1, "#EF4444"]],
                             title="Top 10 Dead Stock by Capital Locked")
                fig.update_layout(
                    plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#64748B", family="Inter"),
                    title_font=dict(color="#F1F5F9", size=13, family="Inter"),
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=True, gridcolor="#1E2D45", tickprefix="Rs. ", tickformat=",.0f"),
                    yaxis=dict(showgrid=False, tickfont=dict(size=10)),
                    margin=dict(l=0, r=10, t=40, b=0), height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Restock Recommendations ──
    section_head("Restock Recommendations", "cart")
    df_restock = load_restock()

    if df_restock.empty:
        st.success("✅ No restocking needed at this time.")
    else:
        total_cost = df_restock["estimated_reorder_cost_pkr"].sum() if "estimated_reorder_cost_pkr" in df_restock.columns else 0
        alert_pill(
            f"<strong>{len(df_restock)} items</strong> need restocking — estimated total: <strong>{fmt_pkr(total_cost)}</strong>",
            "ap-subtle", "cart", "var(--blue)"
        )
        cols  = ["product_name","category","supplier","current_stock","suggested_reorder_qty","estimated_reorder_cost_pkr","urgency"]
        avail = [c for c in cols if c in df_restock.columns]
        d = df_restock[avail].copy()
        if "estimated_reorder_cost_pkr" in d: d["estimated_reorder_cost_pkr"] = d["estimated_reorder_cost_pkr"].apply(lambda x: f"Rs. {x:,.0f}")
        d.columns = [c.replace("_"," ").title() for c in d.columns]
        st.dataframe(d, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

def page_ai_assistant():
    page_header("AI Intelligence Assistant",
                "Query your store in plain English — grounded in your live database and internal docs.",
                "ai", "var(--indigo-dim)", "var(--indigo)")

    if "messages"  not in st.session_state: st.session_state.messages  = []
    if "rag_ready" not in st.session_state:
        try:
            from rag.pipeline import get_rag_pipeline
            st.session_state.rag_ready = get_rag_pipeline().is_ready()
        except Exception:
            st.session_state.rag_ready = False

    # Status
    import os
    provider_name = "Groq (Llama 3.1)" if os.getenv("LLM_PROVIDER", "ollama").lower() == "groq" else "phi3"
    st.markdown(f'<span class="kpi-badge badge-up" style="font-size:0.8rem;padding:0.3rem 0.75rem;">● Online — {provider_name}</span>', unsafe_allow_html=True)

    # ── Quick prompts ──
    section_head("Quick Actions", "zap")
    suggested = [
        "What should I restock this week?",
        "Which products are dead stock?",
        "What is my overall profit margin?",
        "Which category earns the most?",
        "Tell me about Tapal Danedar tea",
        "How is my business performing?",
    ]
    q_cols = st.columns(3)
    for i, q in enumerate(suggested):
        with q_cols[i % 3]:
            if secondary_button(f"↗ {q}", key=f"sq_{i}"):
                st.session_state.pending_question = q

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Chat history ──
    msg_icon = svg("msg", 13, "var(--blue)")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-ai">'
                f'<div class="chat-ai-header">{msg_icon} AI Assistant</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                with st.expander(f"📚 View Citations ({len(msg['sources'])} sources)", expanded=False):
                    for s in msg["sources"][:4]:
                        doc_type = s.get("doc_type", "unknown")
                        name     = s.get("product_name") or s.get("category") or "Store Data"
                        score    = s.get("score", 0)
                        preview  = s.get("preview", "")[:200]
                        st.markdown(
                            f'<div class="chat-source"><strong>[{doc_type.upper()}] {name}</strong> — Confidence: {score:.2%}</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(preview)

            if msg.get("grounding_score") is not None:
                gs    = msg["grounding_score"]
                color = "var(--green)" if gs >= 0.4 else "var(--amber)" if gs >= 0.2 else "var(--red)"
                st.markdown(
                    f'<div class="chat-meta">'
                    f'<span style="color:{color};">⚡ {gs:.0%} grounding</span>'
                    f'<span>·</span><span>Route: {msg.get("route","?")}</span>'
                    f'<span>·</span><span>{msg.get("duration","?")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # A question can come from a Quick Action button OR the chat box —
    # handle both the same way so clicking a Quick Action runs the query too.
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask about your store data…", key="chat_input")
    question = pending or user_input

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Computing response…"):
            if not st.session_state.rag_ready:
                response = {"success": False, "answer": "AI assistant is currently offline.", "sources": []}
            else:
                response = ask_rag(question)
        st.session_state.messages.append({
            "role":            "assistant",
            "content":         response.get("answer", "Sorry, no response generated."),
            "sources":         response.get("sources", []),
            "grounding_score": response.get("grounding_score"),
            "route":           response.get("route", "?"),
            "duration":        f"{response.get('duration_sec', 0):.1f}s",
        })
        st.rerun()

    if st.session_state.messages:
        if danger_button("🗑 Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MANAGE STORE
# ══════════════════════════════════════════════════════════════════════════════

def page_manage_store():
    page_header("Operations Management", "Manage your product catalog, adjust stock, and handle purchase orders.", "manage", "var(--green-dim)", "var(--green)")
    tab1, tab2 = st.tabs(["📦 Product Catalog", "🛒 Procurement Orders"])

    # ── Tab 1 ──
    with tab1:
        section_head("Search & Filter Catalog", "list")
        col_s, col_c, col_l = st.columns([2, 1, 1], gap="medium")
        with col_s:
            search_term = st.text_input("Search", placeholder="e.g. Tapal", label_visibility="collapsed")
        with col_c:
            cats = ["All Categories","Tea & Beverages","Spices & Masala","Dairy","Cooking Oil",
                    "Detergents","Personal Care","Instant Food","Snacks","Beverages","Condiments"]
            selected_cat = st.selectbox("Category", cats, label_visibility="collapsed")
        with col_l:
            show_low = st.checkbox("Low stock only")

        from database.db_manager import get_connection
        conn  = get_connection()
        query = """
            SELECT product_id, product_name, category, cost_price, selling_price,
                   stock, low_stock_threshold, supplier, created_at,
                   ROUND((selling_price-cost_price)*100.0/selling_price,1) AS margin_pct
            FROM products WHERE is_active=1
        """
        params = []
        if selected_cat != "All Categories":
            query += " AND category=?"; params.append(selected_cat)
        if search_term:
            query += " AND LOWER(product_name) LIKE LOWER(?)"; params.append(f"%{search_term}%")
        if show_low:
            query += " AND stock <= low_stock_threshold"
        query += " ORDER BY category, product_name"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        st.markdown(f"<p style='color:var(--muted); font-size:0.8rem; margin-bottom:0.75rem;'>{len(df)} records</p>", unsafe_allow_html=True)
        st.dataframe(df.drop(columns=["product_id"]), use_container_width=True, hide_index=True)
        st.divider()

        with st.expander("➕ Register New Product", expanded=False):
            with st.form("add_product_form"):
                fc1, fc2 = st.columns(2, gap="large")
                with fc1:
                    new_name     = st.text_input("Product Name *")
                    new_category = st.selectbox("Category *", cats[1:])
                    new_cost     = st.number_input("Cost Price (PKR) *", min_value=1.0, step=5.0)
                    new_selling  = st.number_input("Selling Price (PKR) *", min_value=1.0, step=5.0)
                with fc2:
                    new_stock     = st.number_input("Initial Stock", min_value=0, step=1)
                    new_supplier  = st.text_input("Supplier / Vendor")
                    new_threshold = st.number_input("Low Stock Threshold", min_value=1, value=10)

                if st.form_submit_button("Provision Product", use_container_width=True):
                    if not new_name:
                        st.error("Product name is required.")
                    elif new_selling <= new_cost:
                        st.error("Selling price must exceed cost price.")
                    else:
                        try:
                            from datetime import date
                            conn   = get_connection()
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO products (product_name,category,cost_price,selling_price,stock,supplier,low_stock_threshold,created_at) VALUES (?,?,?,?,?,?,?,?)",
                                (new_name, new_category, new_cost, new_selling, new_stock, new_supplier, new_threshold, date.today().isoformat())
                            )
                            conn.commit(); conn.close()
                            st.success(f"✅ '{new_name}' provisioned.")
                            st.cache_data.clear(); st.rerun()
                        except Exception as e:
                            st.error(f"Database error: {e}")

        st.divider()
        section_head("Fast Stock Adjustment", "box")

        if not df.empty:
            prod_opts    = dict(zip(df["product_name"], df["product_id"]))
            sel_product  = st.selectbox("Target Product", list(prod_opts.keys()), key="stock_adjust_product")
            adj1, adj2   = st.columns([2, 1], gap="medium")
            with adj1:
                adj_qty = st.number_input("Adjustment Quantity (negative to deduct)", value=0, step=1, key="adj_qty")
            with adj2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("Commit Adjustment", use_container_width=True):
                    if adj_qty != 0:
                        from datetime import date
                        pid    = prod_opts[sel_product]
                        conn   = get_connection(); cursor = conn.cursor()
                        cursor.execute("UPDATE products SET stock=MAX(0,stock+?), updated_at=? WHERE product_id=?",
                                       (adj_qty, date.today().isoformat(), pid))
                        conn.commit(); conn.close()
                        st.success(f"Stock adjusted by {adj_qty:+d} units.")
                        st.cache_data.clear(); st.rerun()

    # ── Tab 2 ──
    with tab2:
        from database.db_manager import get_connection
        section_head("Pending Inbound Orders", "list")

        conn       = get_connection()
        df_pending = pd.read_sql_query("""
            SELECT po.id, p.product_name, po.quantity_ordered, po.cost_per_unit,
                   po.total_cost, po.supplier, po.order_date
            FROM purchase_orders po
            JOIN products p ON po.product_id=p.product_id
            WHERE po.status='pending' ORDER BY po.order_date DESC
        """, conn)
        conn.close()

        if df_pending.empty:
            st.info("No pending orders.")
        else:
            for _, row in df_pending.iterrows():
                ci, cb = st.columns([4, 1])
                with ci:
                    st.markdown(
                        f'<div class="po-card">'
                        f'<div class="po-name">{row["product_name"]}</div>'
                        f'<div class="po-meta">{row["quantity_ordered"]} units from {row["supplier"]} · Rs. {row["total_cost"]:,.0f} · Ordered: {row["order_date"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with cb:
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    if st.button("✓ Received", key=f"recv_{row['id']}"):
                        from datetime import date
                        conn   = get_connection(); cursor = conn.cursor()
                        cursor.execute("UPDATE purchase_orders SET status='received', received_date=? WHERE id=?",
                                       (date.today().isoformat(), row['id']))
                        cursor.execute("UPDATE products SET stock=stock+?, updated_at=? WHERE product_id=(SELECT product_id FROM purchase_orders WHERE id=?)",
                                       (row['quantity_ordered'], date.today().isoformat(), row['id']))
                        conn.commit(); conn.close()
                        st.success("Order fulfilled. Inventory updated.")
                        st.cache_data.clear(); st.rerun()

        st.divider()

        with st.expander("➕ Create Procurement Order", expanded=False):
            conn        = get_connection()
            products_df = pd.read_sql_query(
                "SELECT product_id, product_name, supplier, cost_price FROM products WHERE is_active=1 ORDER BY product_name", conn)
            conn.close()

            with st.form("create_order_form"):
                prod_map     = dict(zip(products_df["product_name"], products_df["product_id"]))
                supplier_map = dict(zip(products_df["product_name"], products_df["supplier"].fillna("")))
                cost_map     = dict(zip(products_df["product_name"], products_df["cost_price"]))
                selected     = st.selectbox("Product *", list(prod_map.keys()))
                oc1, oc2     = st.columns(2)
                with oc1:
                    o_qty      = st.number_input("Order Volume *", min_value=1, value=50)
                    o_supplier = st.text_input("Vendor *", value=supplier_map.get(selected, ""))
                with oc2:
                    o_cost = st.number_input("Unit Cost (PKR) *", value=float(cost_map.get(selected, 0)), min_value=0.01)
                st.info(f"Projected Spend: **Rs. {o_qty * o_cost:,.0f}**")

                if st.form_submit_button("Initiate Purchase Order", use_container_width=True):
                    try:
                        from datetime import date
                        conn   = get_connection(); cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO purchase_orders (product_id,quantity_ordered,cost_per_unit,total_cost,supplier,order_date) VALUES (?,?,?,?,?,?)",
                            (prod_map[selected], o_qty, o_cost, round(o_qty*o_cost,2), o_supplier, date.today().isoformat())
                        )
                        conn.commit(); conn.close()
                        st.success(f"✅ PO created for {selected}.")
                        st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.divider()
        section_head("Procurement History", "list")
        conn       = get_connection()
        df_history = pd.read_sql_query("""
            SELECT po.id, p.product_name, po.quantity_ordered, po.total_cost,
                   po.supplier, po.order_date, po.status, po.received_date
            FROM purchase_orders po
            JOIN products p ON po.product_id=p.product_id
            ORDER BY po.order_date DESC LIMIT 50
        """, conn)
        conn.close()

        if not df_history.empty:
            spent = df_history[df_history["status"]=="received"]["total_cost"].sum()
            st.markdown(
                f"<p style='color:var(--muted);font-size:0.8rem;margin-bottom:0.75rem;'>"
                f"Fulfilled spend: <strong style='color:var(--text);'>Rs. {spent:,.0f}</strong></p>",
                unsafe_allow_html=True
            )
            st.dataframe(df_history, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    active = render_sidebar()

    if   active == "overview":   page_store_overview()
    elif active == "inventory":  page_inventory_intelligence()
    elif active == "ai":         page_ai_assistant()
    elif active == "manage":     page_manage_store()


if __name__ == "__main__":
    main()