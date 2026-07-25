"""
app.py — Customer Feedback Intelligence System
Light / Dark mode toggle · Centered layout · Auto-clean on upload
"""

import os
import time
import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.cleaner import inspect_dataframe, clean_dataframe
from utils.ai_enrichment import enrich_dataframe_streaming
from utils.analytics import (
    compute_summary_stats,
    build_category_bar_chart,
    build_sentiment_donut_chart,
    build_sentiment_line_chart,
    get_sample_messages,
)
from utils.report import generate_report
from utils.logger import build_cleaning_log

load_dotenv()

st.set_page_config(
    page_title="Customer Feedback Intelligence System",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Session State Init ───────────────────────────────────────────────────────
def _init():
    defaults = {
        "dark_mode": True,
        "raw_df": None, "cleaned_df": None, "enriched_df": None,
        "cleaning_log": None, "quality_stats": None,
        "report_bytes": None, "log_bytes": None,
        "ai_done": False, "clean_done": False, "filename": None,
        "analytics_stats": None, "sample_messages": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

dm = st.session_state.dark_mode   # shorthand

# ─── Theme Palette ───────────────────────────────────────────────────────────
if dm:
    T = {
        "bg":           "#0d1117",
        "card":         "#161b22",
        "card2":        "#1c2230",
        "nav_from":     "#0d1117",
        "nav_to":       "#1a2a4a",
        "nav_border":   "#30363d",
        "text":         "#e6edf3",
        "text_muted":   "#8b949e",
        "text_label":   "#8b949e",
        "border":       "#30363d",
        "metric_bg":    "#1c2230",
        "metric_border":"#30363d",
        "metric_val":   "#e6edf3",
        "accent":       "#58a6ff",
        "positive":     "#3fb950",
        "negative":     "#f85149",
        "neutral":      "#8b949e",
        "warn":         "#d29922",
        "upload_bg":    "#1c2230",
        "upload_border":"#30363d",
        "tbl_header":   "#1c2230",
        "tbl_row":      "#161b22",
        "tbl_row_hover":"#1c2230",
        "tbl_border":   "#30363d",
        "input_bg":     "#0d1117",
        "input_border": "#30363d",
        "expander_bg":  "#1c2230",
        "key_warn_bg":  "#2d2007",
        "key_warn_border":"#d29922",
        "key_warn_text":"#e3b341",
        "key_ok_bg":    "#0a2818",
        "key_ok_border":"#3fb950",
        "key_ok_text":  "#3fb950",
        "tab_list":     "#1c2230",
        "btn_label":    "☀️ Light",
        "footer":       "#484f58",
        "sidebar_bg":   "#161b22",
        "sidebar_text": "#e6edf3",
    }
else:
    T = {
        "bg":           "#eef2f7",
        "card":         "#ffffff",
        "card2":        "#f4f8ff",
        "nav_from":     "#1a3a6b",
        "nav_to":       "#1e4d8c",
        "nav_border":   "#1a3a6b",
        "text":         "#1e293b",
        "text_muted":   "#64748b",
        "text_label":   "#64748b",
        "border":       "#dce8f8",
        "metric_bg":    "#f4f8ff",
        "metric_border":"#dce8f8",
        "metric_val":   "#1a3a6b",
        "accent":       "#2563eb",
        "positive":     "#16a34a",
        "negative":     "#dc2626",
        "neutral":      "#64748b",
        "warn":         "#d97706",
        "upload_bg":    "#f4f8ff",
        "upload_border":"#a8c4e8",
        "tbl_header":   "#f0f5fb",
        "tbl_row":      "#ffffff",
        "tbl_row_hover":"#f7faff",
        "tbl_border":   "#dce8f8",
        "input_bg":     "#f4f8ff",
        "input_border": "#a8c4e8",
        "expander_bg":  "#f4f8ff",
        "key_warn_bg":  "#fef9ec",
        "key_warn_border":"#fde68a",
        "key_warn_text":"#92400e",
        "key_ok_bg":    "#f0fdf4",
        "key_ok_border":"#86efac",
        "key_ok_text":  "#166534",
        "tab_list":     "#eef2f7",
        "btn_label":    "🌙 Dark",
        "footer":       "#94a3b8",
        "sidebar_bg":   "#1a3a6b",
        "sidebar_text": "#e2e8f0",
    }

# ─── Inject CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Global reset ── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body,
[class*="css"],
.stApp,
.stApp > div,
section[data-testid="stSidebar"] ~ div,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stMain"],
div[data-testid="stMainBlockContainer"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
}}

/* ── Hide chrome ── */
#MainMenu, footer, header {{ visibility: hidden !important; }}

/* ── Centered container ── */
.block-container {{
    padding: 0 0 3rem 0 !important;
    max-width: 840px !important;
    margin: 0 auto !important;
}}

/* ─────────────── NAVBAR ─────────────── */
.top-navbar {{
    background: linear-gradient(90deg, {T['nav_from']} 0%, {T['nav_to']} 100%);
    padding: 0.8rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.4rem;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 3px 18px rgba(0,0,0,0.3);
}}
.nav-brand {{ display: flex; align-items: center; gap: 0.65rem; }}
.nav-icon  {{ background: rgba(255,255,255,0.15); border-radius: 9px; padding: 6px 9px; font-size: 1.15rem; }}
.nav-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; font-weight: 700; color: #ffffff !important; }}

/* ─────────────── CARDS ─────────────── */
.card {{
    background: {T['card']} !important;
    border: 1px solid {T['border']};
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 2px 16px rgba(0,0,0,{'0.25' if dm else '0.06'});
}}

/* ─────────────── SECTION TITLE ─────────────── */
.sec-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: {T['accent']} !important;
    margin-bottom: 0.85rem;
}}
.sec-sub {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {T['text']} !important;
    margin-bottom: 0.65rem;
}}

/* ─────────────── METRICS ─────────────── */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
}}
.metric-card {{
    background: {T['metric_bg']};
    border: 1px solid {T['metric_border']};
    border-radius: 10px;
    padding: 0.9rem 0.6rem;
    text-align: center;
}}
.metric-lbl {{
    color: {T['text_label']} !important;
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
}}
.metric-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: {T['metric_val']} !important;
    line-height: 1;
}}
.metric-val.ok     {{ color: {T['positive']} !important; }}
.metric-val.danger {{ color: {T['negative']} !important; }}
.metric-val.warn   {{ color: {T['warn']}     !important; }}
.metric-sub {{
    font-size: 0.67rem;
    color: {T['text_muted']} !important;
    margin-top: 0.25rem;
}}

/* ─────────────── BUTTONS ─────────────── */
.stButton > button {{
    background: linear-gradient(135deg, #1a3a6b 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.7rem 1.4rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.35) !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #15306a 0%, #1d4ed8 100%) !important;
    box-shadow: 0 6px 22px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:disabled {{
    background: {'#2d3748' if dm else '#cbd5e1'} !important;
    color: {'#4a5568' if dm else '#94a3b8'} !important;
    box-shadow: none !important;
    transform: none !important;
}}

/* ─────────────── DOWNLOAD BUTTONS ─────────────── */
.stDownloadButton > button {{
    background: {'#1c2230' if dm else '#1a3a6b'} !important;
    color: {T['accent']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 9px !important;
    padding: 0.6rem 0.9rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}}
.stDownloadButton > button:hover {{
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}}

/* ─────────────── FILE UPLOADER ─────────────── */
[data-testid="stFileUploader"] section {{
    background: {T['upload_bg']} !important;
    border: 2px dashed {T['upload_border']} !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {T['accent']} !important;
}}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {{
    color: {T['text']} !important;
}}

/* ─────────────── PROGRESS BAR ─────────────── */
.stProgress > div > div {{
    background: linear-gradient(90deg, #1a3a6b, #3b82f6) !important;
    border-radius: 999px !important;
}}
.stProgress > div {{
    background: {T['metric_bg']} !important;
    border-radius: 999px !important;
}}

/* ─────────────── EXPANDER ─────────────── */
[data-testid="stExpander"] {{
    background: {T['expander_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary {{
    color: {T['text']} !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] summary:hover {{
    color: {T['accent']} !important;
}}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] div {{
    color: {T['text']} !important;
}}

/* ─────────────── DATAFRAME / TABLE ─────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
    background: {T['tbl_header']} !important;
    color: {T['text']} !important;
}}
[data-testid="stDataFrame"] td {{
    background: {T['tbl_row']} !important;
    color: {T['text']} !important;
}}

/* ─────────────── METRICS (native st.metric) ─────────────── */
[data-testid="stMetric"] {{
    background: {T['metric_bg']} !important;
    border: 1px solid {T['metric_border']} !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
}}
[data-testid="stMetricLabel"] p {{
    color: {T['text_muted']} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    color: {T['text']} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}}

/* ─────────────── ALERTS ─────────────── */
.stAlert {{
    border-radius: 10px !important;
    background: {T['expander_bg']} !important;
    color: {T['text']} !important;
    border-color: {T['border']} !important;
}}
.stAlert p {{ color: {T['text']} !important; }}

/* ─────────────── TEXT INPUT / SELECT ─────────────── */
.stTextInput input,
.stSelectbox select,
.stSelectbox > div > div > div {{
    background: {T['input_bg']} !important;
    border: 1px solid {T['input_border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}
.stTextInput label,
.stSelectbox label {{
    color: {T['text']} !important;
    font-weight: 500 !important;
}}

/* ─────────────── SUCCESS / ERROR / INFO ─────────────── */
div[data-testid="stSuccessMessage"],
div[data-testid="stErrorMessage"],
div[data-testid="stInfoMessage"] {{
    border-radius: 10px !important;
}}
div[data-testid="stSuccessMessage"] p,
div[data-testid="stErrorMessage"] p,
div[data-testid="stInfoMessage"] p {{
    color: {T['text']} !important;
}}

/* ─────────────── SPINNER ─────────────── */
.stSpinner p {{ color: {T['text']} !important; }}
.stSpinner > div {{ border-top-color: {T['accent']} !important; }}

/* ─────────────── CAPTION ─────────────── */
.stCaption,
.stCaption p {{
    color: {T['text_muted']} !important;
}}

/* ─────────────── MARKDOWN ─────────────── */
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {{
    color: {T['text']} !important;
}}

/* ─────────────── TABS ─────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {T['tab_list']} !important;
    border-radius: 8px !important;
    gap: 4px;
    padding: 4px;
    border: 1px solid {T['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px !important;
    color: {T['text_muted']} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg,#1a3a6b,#2563eb) !important;
    color: #ffffff !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: {T['card']} !important;
    padding-top: 0.8rem !important;
}}

/* ─────────────── SIDEBAR ─────────────── */
[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {T['sidebar_text']} !important;
}}
[data-testid="stSidebar"] .stTextInput input {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #fff !important;
    font-family: monospace !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div > div {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
}}

/* ─────────────── CUSTOM HTML ELEMENTS ─────────────── */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: capitalize;
}}
.badge-positive {{ background: {'rgba(63,185,80,0.15)' if dm else '#dcfce7'}; color: {T['positive']}; }}
.badge-negative {{ background: {'rgba(248,81,73,0.15)' if dm else '#fee2e2'};  color: {T['negative']}; }}
.badge-neutral  {{ background: {'rgba(139,148,158,0.15)' if dm else '#f1f5f9'}; color: {T['neutral']};  }}

.key-warn {{
    background: {T['key_warn_bg']};
    border: 1px solid {T['key_warn_border']};
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: {T['key_warn_text']} !important;
    margin-bottom: 0.8rem;
}}
.key-ok {{
    background: {T['key_ok_bg']};
    border: 1px solid {T['key_ok_border']};
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-size: 0.8rem;
    color: {T['key_ok_text']} !important;
    margin-bottom: 0.8rem;
}}

.tbl-header {{
    display: grid;
    grid-template-columns: 85px 100px 1fr 90px;
    gap: 0.4rem;
    padding: 0.5rem 0.8rem;
    background: {T['tbl_header']};
    border-radius: 8px 8px 0 0;
    border: 1px solid {T['tbl_border']};
    font-size: 0.7rem;
    font-weight: 700;
    color: {T['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.tbl-row {{
    display: grid;
    grid-template-columns: 85px 100px 1fr 90px;
    gap: 0.4rem;
    padding: 0.6rem 0.8rem;
    border: 1px solid {T['tbl_border']};
    border-top: none;
    font-size: 0.82rem;
    color: {T['text']} !important;
    background: {T['tbl_row']};
    align-items: center;
    transition: background 0.12s;
}}
.tbl-row:hover {{ background: {T['tbl_row_hover']} !important; }}
.tbl-row:last-child {{ border-radius: 0 0 8px 8px; }}

.stat-muted {{ color: {T['text_muted']} !important; font-size: 0.78rem; }}
</style>
""", unsafe_allow_html=True)

# ─── NAVBAR with theme toggle ─────────────────────────────────────────────────
nav_col1, nav_col2 = st.columns([8, 1])
with nav_col1:
    st.markdown(f"""
    <div class="top-navbar">
      <div class="nav-brand">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Customer Feedback Intelligence System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with nav_col2:
    # Tiny toggle styled via custom CSS override
    st.markdown("<div style='padding-top:0.35rem;'>", unsafe_allow_html=True)
    if st.button(T["btn_label"], key="theme_toggle",
                 help="Switch between light and dark mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ─── SIDEBAR — API Key ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### ⚙️ Settings")
    st.markdown("---")
    env_key    = os.getenv("NVIDIA_API_KEY", "")
    sidebar_key = st.text_input(
        "🔑 NVIDIA API Key",
        value=env_key,
        type="password",
        placeholder="nvapi-xxxxxxxxxxxxxxxxxxxx",
        help="Auto-reads from .env · Enter here to override · Never shown in main UI",
    )
    model_choice = st.selectbox(
        "🤖 Model",
        options=[
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        index=0,
    )
    status_color = "#3fb950" if sidebar_key else "#d29922"
    status_icon  = "✅ API Key loaded" if sidebar_key else "⚠️ No API Key set"
    st.markdown(f'<p style="color:{status_color};font-size:0.8rem;">{status_icon}</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f'<p style="font-size:0.76rem;">Add to <code>.env</code>:<br>'
        f'<code>NVIDIA_API_KEY=nvapi-xxx</code></p>',
        unsafe_allow_html=True,
    )

nvidia_key = sidebar_key

# ═══════════════════════════════════════════════════════════════════════════════
# CARD 1 — Feedback Center / Upload
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📊 Feedback Center</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-sub">📁 Data Ingestion</div>', unsafe_allow_html=True)

if not nvidia_key:
    st.markdown(
        '<div class="key-warn">⚠️ <b>No NVIDIA API Key.</b> Open the ← sidebar and enter your key, '
        'or add to <code>.env</code> as <code>NVIDIA_API_KEY=nvapi-xxx</code></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="key-ok">✅ API Key ready — AI Analysis enabled</div>',
                unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Expected columns: id, timestamp, source, rating, feedback_text",
    label_visibility="collapsed",
)

# ── Auto-process on upload ────────────────────────────────────────────────────
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.filename:
        for k in ["cleaned_df","enriched_df","cleaning_log","quality_stats",
                  "report_bytes","log_bytes","analytics_stats","sample_messages"]:
            st.session_state[k] = None
        st.session_state.clean_done = False
        st.session_state.ai_done   = False
        st.session_state.filename  = uploaded_file.name

        try:
            raw_df = pd.read_csv(uploaded_file, dtype=str)
            for col in ["rating"]:
                if col in raw_df.columns:
                    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
            st.session_state.raw_df = raw_df

            with st.spinner("🧹 Loading & cleaning data…"):
                cleaned, log = clean_dataframe(raw_df)
                quality      = inspect_dataframe(raw_df)
                st.session_state.cleaned_df    = cleaned
                st.session_state.cleaning_log  = log
                st.session_state.quality_stats = quality
                st.session_state.clean_done    = True
                log_text = build_cleaning_log(log)
                st.session_state.log_bytes = log_text.encode("utf-8")

            removed = log['rows_loaded'] - log['final_rows']
            st.success(
                f"✅ **{uploaded_file.name}** — "
                f"{log['rows_loaded']:,} rows loaded, {removed:,} removed, "
                f"**{log['final_rows']:,} clean rows ready**"
            )
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
            st.code(traceback.format_exc())
            st.session_state.raw_df = None

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CARD 2 — Dataset Snapshot
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.clean_done and st.session_state.quality_stats:
    qs      = st.session_state.quality_stats
    log     = st.session_state.cleaning_log or {}
    total_r = qs.get("total_rows", 0)
    miss_v  = qs.get("missing_values", 0)
    dup_r   = qs.get("duplicate_rows", 0)
    blank_f = qs.get("blank_feedback", 0)
    final_r = log.get("final_rows", total_r)

    mc = lambda v, danger=True: "danger" if (v > 0 and danger) else ("warn" if v > 0 else "ok")

    st.markdown(f"""
    <div class="card">
      <div class="sec-title">📊 Dataset Snapshot</div>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-lbl">Total Entries</div>
          <div class="metric-val">{total_r:,}</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Missing Info</div>
          <div class="metric-val {mc(miss_v, danger=False)}">{miss_v:,}</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Duplicates</div>
          <div class="metric-val {mc(dup_r)}">{dup_r:,}</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Empty Comments</div>
          <div class="metric-val {mc(blank_f)}">{blank_f:,}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Cleaning Summary"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duplicates Removed",  log.get("duplicate_rows_removed", 0))
        c2.metric("Blank Removed",        log.get("blank_feedback_removed", 0))
        c3.metric("Timestamps Fixed",     log.get("invalid_timestamps_normalized", 0))
        c4.metric("Final Clean Rows",     f"{final_r:,}")

    with st.expander("🔍 Preview Cleaned Data (first 10 rows)"):
        st.dataframe(
            st.session_state.cleaned_df.head(10),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# CARD 3 — AI Analysis Button
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.clean_done and not st.session_state.ai_done:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🤖 AI Analysis</div>', unsafe_allow_html=True)

    n_rows    = len(st.session_state.cleaned_df)
    n_batches = (n_rows + 19) // 20
    st.markdown(
        f'<p class="stat-muted">{n_rows:,} rows ready · ~{n_batches} API calls '
        f'(batches of 20) · Model: {model_choice.split("/")[-1]}</p>',
        unsafe_allow_html=True,
    )

    if st.button("[ Perform AI Analysis ]", key="btn_analyze",
                 disabled=not nvidia_key):
        df_to_enrich = st.session_state.cleaned_df.copy()
        total_rows   = len(df_to_enrich)

        prog_bar    = st.progress(0)
        status_txt  = st.empty()
        elapsed_txt = st.empty()
        all_results = []
        t0 = time.time()

        try:
            for batch_res, done, total in enrich_dataframe_streaming(
                df_to_enrich, api_key=nvidia_key, model=model_choice
            ):
                all_results.extend(batch_res)
                pct     = done / total
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (total - done) / rate if rate > 0 else 0
                prog_bar.progress(pct)
                status_txt.markdown(f"**Analyzed {done:,} / {total:,} rows ({pct*100:.1f}%)**")
                elapsed_txt.markdown(
                    f"<span class='stat-muted'>⏱ {elapsed:.0f}s elapsed · "
                    f"{rate:.1f} rows/s · ETA {eta:.0f}s</span>",
                    unsafe_allow_html=True,
                )

            import pandas as _pd
            res_df   = _pd.DataFrame(all_results)
            enriched = df_to_enrich.copy()
            enriched["sentiment"] = res_df["sentiment"].values
            enriched["category"]  = res_df["category"].values
            enriched["summary"]   = res_df["summary"].values

            desired = ["id","timestamp","source","rating","feedback_text",
                       "sentiment","category","summary"]
            exist   = [c for c in desired if c in enriched.columns]
            extra   = [c for c in enriched.columns if c not in exist]
            enriched = enriched[exist + extra]

            st.session_state.enriched_df = enriched
            st.session_state.ai_done     = True

            prog_bar.progress(1.0)
            status_txt.markdown("**✅ AI analysis complete!**")
            elapsed_txt.empty()

            with st.spinner("📝 Generating report…"):
                stats   = compute_summary_stats(enriched)
                samples = get_sample_messages(enriched)
                rep     = generate_report(
                    enriched, st.session_state.cleaning_log, samples, stats
                )
                st.session_state.report_bytes    = rep
                st.session_state.analytics_stats = stats
                st.session_state.sample_messages = samples

            st.success(f"🎉 {total_rows:,} rows enriched in {time.time()-t0:.0f}s")
            st.rerun()

        except Exception as e:
            st.error(f"❌ AI enrichment failed: {e}")
            st.code(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CARD 4 — Dashboard (after AI)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.ai_done and st.session_state.enriched_df is not None:
    enriched = st.session_state.enriched_df
    stats    = st.session_state.analytics_stats or compute_summary_stats(enriched)
    samples  = st.session_state.sample_messages  or get_sample_messages(enriched)

    total = stats.get("total", len(enriched))
    pos_c = stats.get("sentiment_positive_count", 0)
    neg_c = stats.get("sentiment_negative_count", 0)
    neu_c = stats.get("sentiment_neutral_count", 0)
    pos_p = stats.get("sentiment_positive_pct", 0)
    neg_p = stats.get("sentiment_negative_pct", 0)
    neu_p = stats.get("sentiment_neutral_pct", 0)

    st.markdown(f"""
    <div class="card">
      <div class="sec-title">📊 Feedback Dashboard</div>
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-lbl">Total Analyzed</div>
          <div class="metric-val">{total:,}</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Positive 😊</div>
          <div class="metric-val ok">{pos_c:,}</div>
          <div class="metric-sub">{pos_p}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Negative 😞</div>
          <div class="metric-val danger">{neg_c:,}</div>
          <div class="metric-sub">{neg_p}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-lbl">Neutral 😐</div>
          <div class="metric-val warn">{neu_c:,}</div>
          <div class="metric-sub">{neu_p}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)

    _lyt = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"], family="Inter, sans-serif"),
        title=dict(font=dict(color=T["accent"], size=14)),
        legend=dict(font=dict(color=T["text"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=45, b=10),
        height=280,
    )
    _ax = dict(color=T["text_muted"], gridcolor=T["border"], zerolinecolor=T["border"])

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = build_category_bar_chart(enriched)
        fig.update_layout(**_lyt, xaxis=_ax, yaxis=_ax)
        fig.update_traces(textfont=dict(color=T["text"]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = build_sentiment_donut_chart(enriched)
        fig.update_layout(**_lyt)
        fig.update_traces(marker=dict(line=dict(color=T["card"], width=2)))
        for ann in fig.layout.annotations:
            ann.font.color = T["accent"]
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    fig = build_sentiment_line_chart(enriched)
    fig.update_layout(**{**_lyt, "height": 250,
                         "xaxis": _ax, "yaxis": _ax,
                         "hovermode": "x unified"})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Sample Feedback Logs ──────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Sample Feedback Logs</div>',
                unsafe_allow_html=True)

    tabs_labels = list(samples.keys()) if samples else []
    if tabs_labels:
        tabs = st.tabs([f"📌 {t}" for t in tabs_labels])
        for tab, cat in zip(tabs, tabs_labels):
            with tab:
                msgs = samples.get(cat, [])
                st.markdown("""
                <div class="tbl-header">
                  <span>Date</span><span>Customer ID</span>
                  <span>Message</span><span>Sentiment</span>
                </div>""", unsafe_allow_html=True)

                for msg in msgs:
                    ts_val  = msg.get("timestamp", "")
                    ts_str  = str(ts_val)[:10] if ts_val and str(ts_val) not in ("NaT","None","nan") else "N/A"
                    cust_id = str(msg.get("id", "N/A"))
                    fb      = str(msg.get("feedback_text", ""))
                    fb_s    = fb[:72] + ("…" if len(fb) > 72 else "")
                    sent    = str(msg.get("sentiment", "neutral")).lower()
                    st.markdown(f"""
                    <div class="tbl-row">
                      <span class="stat-muted">{ts_str}</span>
                      <span>{cust_id}</span>
                      <span>{fb_s}</span>
                      <span><span class="badge badge-{sent}">{sent.capitalize()}</span></span>
                    </div>""", unsafe_allow_html=True)

    with st.expander(f"📄 Full Enriched Dataset ({len(enriched):,} rows)"):
        st.dataframe(
            enriched, use_container_width=True, hide_index=True,
            column_config={
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "category":  st.column_config.TextColumn("Category",  width="medium"),
                "summary":   st.column_config.TextColumn("Summary",   width="large"),
                "rating":    st.column_config.NumberColumn("Rating",   format="%.1f"),
            }
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Download Row ──────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⬇️ Download Results</div>',
                unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇️ Cleaned CSV",
            data=enriched.to_csv(index=False).encode("utf-8"),
            file_name="customer_feedback_processed.csv",
            mime="text/csv", key="dl_csv",
        )
        st.caption("customer_feedback_processed.csv")
    with dl2:
        if st.session_state.report_bytes:
            st.download_button(
                "⬇️ Summary Report (DOCX)",
                data=st.session_state.report_bytes,
                file_name="customer_feedback_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_report",
            )
            st.caption("customer_feedback_report.docx")
    with dl3:
        if st.session_state.log_bytes:
            st.download_button(
                "⬇️ Cleaning Log (TXT)",
                data=st.session_state.log_bytes,
                file_name="cleaning_log.txt",
                mime="text/plain", key="dl_log",
            )
            st.caption("cleaning_log.txt")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:{T['footer']};font-size:0.72rem;padding:1rem 0 0.5rem;">
  Customer Feedback Intelligence System &nbsp;·&nbsp;
  NVIDIA NIM · Streamlit · Plotly · pandas
  &nbsp;|&nbsp; {'🌙 Dark Mode' if dm else '☀️ Light Mode'}
</div>
""", unsafe_allow_html=True)
