"""
app.py — Customer Feedback Intelligence System
Single-page Streamlit application.

Run: streamlit run app.py
"""

import io
import os
import time
import traceback

import numpy as np
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

# ─── Load .env if present ────────────────────────────────────────────────────
load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #060d1f;
    --bg-card:       #0d1b35;
    --bg-card2:      #0a1628;
    --accent-blue:   #3b82f6;
    --accent-indigo: #6366f1;
    --accent-violet: #7c3aed;
    --text-primary:  #e2e8f0;
    --text-muted:    #94a3b8;
    --border:        rgba(59,130,246,0.18);
    --positive:      #22c55e;
    --negative:      #ef4444;
    --neutral:       #94a3b8;
    --radius:        14px;
    --shadow:        0 4px 32px rgba(0,0,0,0.5);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary);
}
.stApp { background-color: var(--bg-primary) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1200px; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    border-radius: 0 0 var(--radius) var(--radius);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin: 0;
}

/* ── Section cards ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: 0.02em;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-blue);
}
.metric-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-value.warn { color: #f59e0b; }
.metric-value.danger { color: #ef4444; }
.metric-value.good { color: #22c55e; }

/* ── Sentiment badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: capitalize;
}
.badge-positive { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-negative { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-neutral  { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

/* ── Primary buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.7rem 1.8rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.25) !important;
    cursor: pointer !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: linear-gradient(135deg, #1e293b, #1e293b) !important;
    color: #475569 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f4c81, #1e3a8a) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 9px !important;
    padding: 0.6rem 1.2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: #fff !important;
    border-color: var(--accent-blue) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border-radius: 999px !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── API key input ── */
.stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', monospace !important;
}

/* ── Status indicator ── */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-done  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.status-pending { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
.status-idle  { background: #475569; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card2) !important;
    border-radius: 10px !important;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: #fff !important;
}

/* ── Selectbox / Sidebar ── */
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text-primary) !important;
}

/* ── Feedback sample rows ── */
.sample-row {
    background: var(--bg-card2);
    border-left: 3px solid var(--accent-blue);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.sample-text { font-size: 0.9rem; color: var(--text-primary); margin-bottom: 0.3rem; }
.sample-meta { font-size: 0.78rem; color: var(--text-muted); }

/* ── Plotly charts dark override ── */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div>
        <div style="font-size:2.2rem; line-height:1;">📊</div>
    </div>
    <div>
        <div class="hero-title">Customer Feedback Intelligence System</div>
        <div class="hero-sub">QuickCart · Upload → Clean → Enrich → Analyze → Download</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Session state initialisation ────────────────────────────────────────────
def _init_state():
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "enriched_df": None,
        "cleaning_log": None,
        "quality_stats": None,
        "report_bytes": None,
        "log_bytes": None,
        "ai_done": False,
        "clean_done": False,
        "filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — API Key + Upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔑 Configuration</div>', unsafe_allow_html=True)

api_env = os.getenv("NVIDIA_API_KEY", "")
api_col1, api_col2 = st.columns([3, 1])
with api_col1:
    nvidia_key = st.text_input(
        "NVIDIA API Key",
        value=api_env,
        type="password",
        placeholder="nvapi-xxxxxxxxxxxxxxxxxxxx",
        help="Get your key at https://integrate.api.nvidia.com",
        label_visibility="collapsed",
    )
with api_col2:
    model_choice = st.selectbox(
        "Model",
        options=[
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        index=0,
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Data Ingestion</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your customer feedback CSV",
    type=["csv"],
    help="Expected columns: id, timestamp, source, rating, feedback_text",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.filename:
        # New file uploaded — reset everything
        st.session_state.filename = uploaded_file.name
        st.session_state.clean_done = False
        st.session_state.ai_done = False
        st.session_state.enriched_df = None
        st.session_state.report_bytes = None
        st.session_state.log_bytes = None

        try:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = raw_df
            st.session_state.quality_stats = inspect_dataframe(raw_df)
            st.success(f"✅ Loaded **{uploaded_file.name}** — {len(raw_df):,} rows, {len(raw_df.columns)} columns")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.session_state.raw_df = None

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset Snapshot
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.raw_df is not None:
    qs = st.session_state.quality_stats or {}

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Dataset Snapshot</div>', unsafe_allow_html=True)

    total_r = qs.get("total_rows", 0)
    miss_v  = qs.get("missing_values", 0)
    dup_r   = qs.get("duplicate_rows", 0)
    blank_f = qs.get("blank_feedback", 0)
    inv_ts  = qs.get("invalid_timestamps", 0)

    miss_cls  = "warn"   if miss_v  > 0 else "good"
    dup_cls   = "danger" if dup_r   > 0 else "good"
    blank_cls = "danger" if blank_f > 0 else "good"
    ts_cls    = "warn"   if inv_ts  > 0 else "good"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Rows</div>
            <div class="metric-value">{total_r:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value {miss_cls}">{miss_v:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Duplicates</div>
            <div class="metric-value {dup_cls}">{dup_r:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Blank Feedback</div>
            <div class="metric-value {blank_cls}">{blank_f:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Column preview
    with st.expander("🔍 Preview Raw Data (first 10 rows)"):
        st.dataframe(
            st.session_state.raw_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Action Buttons
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.raw_df is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Processing Pipeline</div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)

    # ── Process Data ──────────────────────────────────────────────────────────
    with btn_col1:
        clean_icon = "✅" if st.session_state.clean_done else "⏳"
        if st.button(f"[ {clean_icon} Process & Clean Data ]", key="btn_clean"):
            with st.spinner("🧹 Cleaning data…"):
                try:
                    cleaned, log = clean_dataframe(st.session_state.raw_df)
                    st.session_state.cleaned_df = cleaned
                    st.session_state.cleaning_log = log
                    st.session_state.clean_done = True
                    # Reset AI if re-running
                    st.session_state.ai_done = False
                    st.session_state.enriched_df = None
                    st.session_state.report_bytes = None

                    # Build log bytes immediately
                    log_text = build_cleaning_log(log)
                    st.session_state.log_bytes = log_text.encode("utf-8")

                    st.success(
                        f"✅ Cleaning complete! "
                        f"{log['rows_loaded']:,} → {log['final_rows']:,} rows "
                        f"({log['rows_loaded'] - log['final_rows']:,} removed)"
                    )
                except Exception as e:
                    st.error(f"❌ Cleaning failed: {e}")
                    st.code(traceback.format_exc())

    # ── AI Analyse ────────────────────────────────────────────────────────────
    with btn_col2:
        ai_disabled = not st.session_state.clean_done
        ai_icon = "✅" if st.session_state.ai_done else "🤖"
        btn_ai_label = f"[ {ai_icon} Perform AI Analysis ]"

        if st.button(
            btn_ai_label,
            key="btn_ai",
            disabled=ai_disabled,
        ):
            if not nvidia_key:
                st.error("❌ Please enter your NVIDIA API key above.")
            else:
                df_to_enrich = st.session_state.cleaned_df.copy()
                total_rows = len(df_to_enrich)

                st.markdown(f"**🚀 Processing {total_rows:,} rows in batches of 20…**")

                progress_bar = st.progress(0)
                status_text  = st.empty()
                elapsed_text = st.empty()

                all_results = []
                start_time = time.time()
                error_count = 0

                try:
                    for batch_results, done, total in enrich_dataframe_streaming(
                        df_to_enrich,
                        api_key=nvidia_key,
                        model=model_choice,
                    ):
                        all_results.extend(batch_results)
                        pct = done / total
                        elapsed = time.time() - start_time
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (total - done) / rate if rate > 0 else 0

                        progress_bar.progress(pct)
                        status_text.markdown(
                            f"**Analyzed {done:,} / {total:,} rows** "
                            f"({pct*100:.1f}%)"
                        )
                        elapsed_text.markdown(
                            f"⏱ Elapsed: {elapsed:.0f}s | "
                            f"Rate: {rate:.1f} rows/s | "
                            f"ETA: {eta:.0f}s"
                        )

                    # Merge results back into the dataframe
                    results_df = pd.DataFrame(all_results)
                    enriched = df_to_enrich.copy()
                    enriched["sentiment"] = results_df["sentiment"].values
                    enriched["category"]  = results_df["category"].values
                    enriched["summary"]   = results_df["summary"].values

                    # Reorder columns to match spec
                    desired_cols = ["id", "timestamp", "source", "rating", "feedback_text",
                                    "sentiment", "category", "summary"]
                    existing = [c for c in desired_cols if c in enriched.columns]
                    extra = [c for c in enriched.columns if c not in existing]
                    enriched = enriched[existing + extra]

                    st.session_state.enriched_df = enriched
                    st.session_state.ai_done = True

                    # Build report + log
                    progress_bar.progress(1.0)
                    status_text.markdown("**✅ AI analysis complete!**")
                    elapsed_text.empty()

                    with st.spinner("📝 Generating report…"):
                        stats = compute_summary_stats(enriched)
                        samples = get_sample_messages(enriched)
                        report_bytes = generate_report(
                            enriched,
                            st.session_state.cleaning_log,
                            samples,
                            stats,
                        )
                        st.session_state.report_bytes = report_bytes
                        st.session_state.analytics_stats = stats
                        st.session_state.sample_messages = samples

                    st.success(
                        f"🎉 Analysis complete! "
                        f"{total_rows:,} rows enriched. "
                        f"Total time: {time.time() - start_time:.0f}s"
                    )

                except Exception as e:
                    st.error(f"❌ AI enrichment failed: {e}")
                    st.code(traceback.format_exc())

    # ── Cleaning summary (shown after clean) ──────────────────────────────────
    if st.session_state.clean_done and st.session_state.cleaning_log:
        log = st.session_state.cleaning_log
        with st.expander("📋 Cleaning Summary"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Duplicates Removed", log.get("duplicate_rows_removed", 0))
            c2.metric("Blank Feedback Removed", log.get("blank_feedback_removed", 0))
            c3.metric("Final Clean Rows", f"{log.get('final_rows', 0):,}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Timestamps Normalized", log.get("invalid_timestamps_normalized", 0))
            c5.metric("Timestamps Set Null", log.get("timestamps_set_null", 0))
            c6.metric("Ratings Coerced", log.get("ratings_coerced", 0))

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Dashboard (shown after AI analysis)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.ai_done and st.session_state.enriched_df is not None:
    enriched = st.session_state.enriched_df
    stats = getattr(st.session_state, "analytics_stats", compute_summary_stats(enriched))
    samples = getattr(st.session_state, "sample_messages", get_sample_messages(enriched))

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Feedback Intelligence Dashboard</div>', unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total = stats.get("total", len(enriched))
    pos_c = stats.get("sentiment_positive_count", 0)
    neg_c = stats.get("sentiment_negative_count", 0)
    neu_c = stats.get("sentiment_neutral_count", 0)
    pos_p = stats.get("sentiment_positive_pct", 0)
    neg_p = stats.get("sentiment_negative_pct", 0)
    neu_p = stats.get("sentiment_neutral_pct", 0)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Analyzed</div>
            <div class="metric-value">{total:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Positive 😊</div>
            <div class="metric-value good">{pos_c:,}</div>
            <div class="metric-label">{pos_p}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Negative 😞</div>
            <div class="metric-value danger">{neg_c:,}</div>
            <div class="metric-label">{neg_p}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Neutral 😐</div>
            <div class="metric-value" style="color:#94a3b8">{neu_c:,}</div>
            <div class="metric-label">{neu_p}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1: Bar + Donut ─────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        fig_bar = build_category_bar_chart(enriched)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        fig_donut = build_sentiment_donut_chart(enriched)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ── Chart Row 2: Line chart (full width) ─────────────────────────────────
    fig_line = build_sentiment_line_chart(enriched)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Sample Feedback Logs ──────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Sample Feedback Logs</div>', unsafe_allow_html=True)

    tabs_labels = list(samples.keys()) if samples else []
    if tabs_labels:
        tabs = st.tabs([f"📌 {t}" for t in tabs_labels])
        for tab, cat in zip(tabs, tabs_labels):
            with tab:
                msgs = samples.get(cat, [])
                if not msgs:
                    st.info("No samples available for this category.")
                    continue

                for msg in msgs:
                    sentiment = str(msg.get("sentiment", "neutral")).lower()
                    badge_cls = f"badge-{sentiment}"
                    ts_val = msg.get("timestamp", "")
                    ts_str = str(ts_val)[:10] if ts_val and str(ts_val) != "NaT" else "N/A"
                    fb_text = str(msg.get("feedback_text", ""))[:300]
                    summary = str(msg.get("summary", ""))
                    source = str(msg.get("source", "N/A")).replace("_", " ").title()
                    rating = msg.get("rating", "N/A")
                    cust_id = msg.get("id", "N/A")

                    st.markdown(f"""
                    <div class="sample-row">
                        <div class="sample-text">"{fb_text}"</div>
                        <div class="sample-meta">
                            → <em>{summary}</em><br>
                            <span class="badge {badge_cls}">{sentiment}</span>
                            &nbsp;|&nbsp; 📅 {ts_str}
                            &nbsp;|&nbsp; 🏷 {source}
                            &nbsp;|&nbsp; ⭐ {rating}
                            &nbsp;|&nbsp; ID: {cust_id}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Full enriched table
    with st.expander(f"📄 Full Enriched Dataset ({len(enriched):,} rows)"):
        st.dataframe(
            enriched,
            use_container_width=True,
            hide_index=True,
            column_config={
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "category":  st.column_config.TextColumn("Category",  width="medium"),
                "summary":   st.column_config.TextColumn("Summary",   width="large"),
                "rating":    st.column_config.NumberColumn("Rating", format="%.1f"),
            }
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Downloads
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.ai_done and st.session_state.enriched_df is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⬇️ Download Results</div>', unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # ── Cleaned & Enriched CSV ────────────────────────────────────────────────
    with dl_col1:
        csv_bytes = st.session_state.enriched_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name="customer_feedback_processed.csv",
            mime="text/csv",
            key="dl_csv",
        )
        st.caption("customer_feedback_processed.csv")

    # ── Summary Report DOCX ───────────────────────────────────────────────────
    with dl_col2:
        if st.session_state.report_bytes:
            st.download_button(
                label="⬇️ Download Summary Report",
                data=st.session_state.report_bytes,
                file_name="customer_feedback_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_report",
            )
            st.caption("customer_feedback_report.docx")
        else:
            st.button("⬇️ Report Not Ready", disabled=True, key="dl_report_dis")

    # ── Cleaning Log ──────────────────────────────────────────────────────────
    with dl_col3:
        if st.session_state.log_bytes:
            st.download_button(
                label="⬇️ Download Cleaning Log",
                data=st.session_state.log_bytes,
                file_name="cleaning_log.txt",
                mime="text/plain",
                key="dl_log",
            )
            st.caption("cleaning_log.txt")
        else:
            st.button("⬇️ Log Not Ready", disabled=True, key="dl_log_dis")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align:center;
    color: #475569;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(59,130,246,0.1);
    margin-top: 1rem;
">
    Customer Feedback Intelligence System &nbsp;·&nbsp;
    Powered by NVIDIA NIM API &nbsp;·&nbsp;
    Built with Streamlit + Plotly + python-docx
</div>
""", unsafe_allow_html=True)
