"""
utils/analytics.py
Aggregation and chart-data preparation from enriched dataframe.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple


# ─── Color palette ────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
}

CATEGORY_COLORS = [
    "#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"
]

CHART_BG = "rgba(0,0,0,0)"
CHART_FONT_COLOR = "#334155"
CHART_GRID_COLOR = "#e2e8f0"


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """High-level stats for the enriched dataframe."""
    total = len(df)
    stats = {"total": total}

    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts()
        for s in ("positive", "negative", "neutral"):
            count = int(vc.get(s, 0))
            pct = round(count / total * 100, 1) if total else 0
            stats[f"sentiment_{s}_count"] = count
            stats[f"sentiment_{s}_pct"] = pct

    if "category" in df.columns:
        stats["top_categories"] = (
            df["category"]
            .value_counts()
            .head(5)
            .to_dict()
        )

    return stats


def build_category_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top 5 complaint categories."""
    cat_counts = (
        df["category"]
        .value_counts()
        .head(5)
        .sort_values(ascending=True)
        .reset_index()
    )
    cat_counts.columns = ["category", "count"]

    fig = go.Figure(go.Bar(
        x=cat_counts["count"],
        y=cat_counts["category"],
        orientation="h",
        marker=dict(
            color=CATEGORY_COLORS[:len(cat_counts)],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=cat_counts["count"],
        textposition="outside",
        textfont=dict(color="#475569", size=12),
    ))

    fig.update_layout(
        title=dict(
            text="Top 5 Complaint Categories",
            font=dict(color=CHART_FONT_COLOR, size=16),
        ),
        xaxis=dict(
            title="Count",
            color=CHART_FONT_COLOR,
            gridcolor=CHART_GRID_COLOR,
            showgrid=True,
        ),
        yaxis=dict(
            color=CHART_FONT_COLOR,
            gridcolor=CHART_GRID_COLOR,
        ),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT_COLOR),
        margin=dict(l=20, r=60, t=50, b=20),
        height=320,
    )
    return fig


def build_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart for sentiment distribution."""
    sentiments = ["positive", "negative", "neutral"]
    vc = df["sentiment"].value_counts()
    values = [int(vc.get(s, 0)) for s in sentiments]
    colors = [SENTIMENT_COLORS[s] for s in sentiments]

    fig = go.Figure(go.Pie(
        labels=[s.capitalize() for s in sentiments],
        values=values,
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color="#0f172a", width=2),
        ),
        textfont=dict(color=CHART_FONT_COLOR, size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))

    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=18, color=CHART_FONT_COLOR),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(
            text="Sentiment Distribution",
            font=dict(color=CHART_FONT_COLOR, size=16),
        ),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT_COLOR),
        legend=dict(
            font=dict(color=CHART_FONT_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
    )
    return fig


def build_sentiment_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Line chart: sentiment counts grouped by week.
    Falls back to a message chart if timestamps are mostly missing.
    """
    has_ts = (
        "timestamp" in df.columns
        and df["timestamp"].notna().sum() > 20
    )

    if not has_ts:
        # Fallback: sentiment counts by source
        return _build_sentiment_by_source(df)

    tmp = df[df["timestamp"].notna()].copy()
    tmp["week"] = tmp["timestamp"].dt.to_period("W").dt.start_time

    weekly = (
        tmp.groupby(["week", "sentiment"])
        .size()
        .reset_index(name="count")
        .sort_values("week")
    )

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        mask = weekly["sentiment"] == sentiment
        subset = weekly[mask]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["week"],
            y=subset["count"],
            name=sentiment.capitalize(),
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{sentiment.capitalize()}</b><br>Week: %{{x|%b %d, %Y}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Sentiment Trend Over Time (Weekly)",
            font=dict(color=CHART_FONT_COLOR, size=16),
        ),
        xaxis=dict(
            title="Week",
            color=CHART_FONT_COLOR,
            gridcolor=CHART_GRID_COLOR,
            showgrid=True,
        ),
        yaxis=dict(
            title="Count",
            color=CHART_FONT_COLOR,
            gridcolor=CHART_GRID_COLOR,
        ),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT_COLOR),
        legend=dict(
            font=dict(color=CHART_FONT_COLOR),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=320,
        hovermode="x unified",
    )
    return fig


def _build_sentiment_by_source(df: pd.DataFrame) -> go.Figure:
    """Fallback: sentiment stacked bar by source."""
    if "source" not in df.columns:
        return go.Figure()

    grp = (
        df.groupby(["source", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        mask = grp["sentiment"] == sentiment
        subset = grp[mask]
        fig.add_trace(go.Bar(
            x=subset["source"],
            y=subset["count"],
            name=sentiment.capitalize(),
            marker_color=color,
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Sentiment by Source",
            font=dict(color=CHART_FONT_COLOR, size=16),
        ),
        xaxis=dict(color=CHART_FONT_COLOR, gridcolor=CHART_GRID_COLOR),
        yaxis=dict(color=CHART_FONT_COLOR, gridcolor=CHART_GRID_COLOR),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT_COLOR),
        legend=dict(font=dict(color=CHART_FONT_COLOR), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=50, b=30),
        height=320,
    )
    return fig


def get_sample_messages(df: pd.DataFrame, n_per_category: int = 3) -> Dict[str, List[Dict]]:
    """
    Return 2–3 representative messages per top-5 category.
    Prefers negative sentiment (most actionable for a report).
    """
    samples = {}
    top_cats = df["category"].value_counts().head(5).index.tolist()

    for cat in top_cats:
        subset = df[df["category"] == cat].copy()
        # Prefer negative, then neutral, then positive
        neg = subset[subset["sentiment"] == "negative"]
        neu = subset[subset["sentiment"] == "neutral"]
        pos = subset[subset["sentiment"] == "positive"]

        ordered = pd.concat([neg, neu, pos]).head(n_per_category)

        cols = ["id", "timestamp", "source", "rating", "feedback_text", "sentiment", "summary"]
        available_cols = [c for c in cols if c in ordered.columns]
        samples[cat] = ordered[available_cols].to_dict(orient="records")

    return samples
