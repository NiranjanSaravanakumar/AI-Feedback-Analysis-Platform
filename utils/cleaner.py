"""
utils/cleaner.py
Data quality inspection and cleaning for customer_feedback_raw.csv
Handles 1800+ rows with comprehensive trap detection.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Tuple, Dict, Any


# Values treated as "meaningless" feedback text
MEANINGLESS_VALUES = {
    "", "n/a", "na", "none", "null", ".", "-", "—", "–",
    "nan", "no comment", "no feedback", "nothing", "nill", "nil",
    "no", "yes", "ok", "okay", "N/A", "N/a"
}

VALID_SOURCES = {"support_ticket", "app_store_review", "survey_comment"}


def inspect_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run data quality checks without modifying the dataframe.
    Returns a dict of quality metrics shown in the UI snapshot.
    """
    stats = {}
    stats["total_rows"] = len(df)

    # Missing values (any column)
    stats["missing_values"] = int(df.isnull().sum().sum())

    # Exact duplicate rows (same id AND same feedback_text)
    if "id" in df.columns and "feedback_text" in df.columns:
        stats["duplicate_rows"] = int(
            df.duplicated(subset=["id", "feedback_text"], keep="first").sum()
        )
    else:
        stats["duplicate_rows"] = int(df.duplicated().sum())

    # Blank / meaningless feedback
    if "feedback_text" in df.columns:
        blank_mask = (
            df["feedback_text"].isna()
            | df["feedback_text"].astype(str).str.strip().str.lower().isin(MEANINGLESS_VALUES)
            | (df["feedback_text"].astype(str).str.strip().str.len() < 3)
        )
        stats["blank_feedback"] = int(blank_mask.sum())
    else:
        stats["blank_feedback"] = 0

    # Invalid / unparseable timestamps
    if "timestamp" in df.columns:
        ts_series = df["timestamp"].astype(str).str.strip()
        invalid_ts = ts_series.apply(lambda x: not _can_parse_timestamp(x))
        stats["invalid_timestamps"] = int(invalid_ts.sum())
    else:
        stats["invalid_timestamps"] = 0

    return stats


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full cleaning pipeline. Returns (cleaned_df, cleaning_log_dict).
    All operations are logged so the UI can show exactly what changed.
    """
    log = {
        "rows_loaded": len(df),
        "duplicate_rows_removed": 0,
        "blank_feedback_removed": 0,
        "invalid_timestamps_normalized": 0,
        "timestamps_set_null": 0,
        "ratings_coerced": 0,
        "source_normalized": 0,
        "final_rows": 0,
    }

    df = df.copy()

    # ── Step 1: Strip whitespace from all string columns ─────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)

    # ── Step 2: Remove exact duplicates (same id + same feedback_text) ───────
    before = len(df)
    if "id" in df.columns and "feedback_text" in df.columns:
        df = df.drop_duplicates(subset=["id", "feedback_text"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    log["duplicate_rows_removed"] = before - len(df)

    # ── Step 3: Drop blank / meaningless feedback_text ───────────────────────
    before = len(df)
    if "feedback_text" in df.columns:
        blank_mask = (
            df["feedback_text"].isna()
            | df["feedback_text"].astype(str).str.strip().str.lower().isin(MEANINGLESS_VALUES)
            | (df["feedback_text"].astype(str).str.strip().str.len() < 3)
        )
        df = df[~blank_mask].copy()
    log["blank_feedback_removed"] = before - len(df)

    # ── Step 4: Normalize source column ──────────────────────────────────────
    if "source" in df.columns:
        before_sources = df["source"].copy()
        df["source"] = df["source"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        # Map common variants
        source_map = {
            "support": "support_ticket",
            "ticket": "support_ticket",
            "support ticket": "support_ticket",
            "app_store": "app_store_review",
            "app store": "app_store_review",
            "appstore": "app_store_review",
            "review": "app_store_review",
            "survey": "survey_comment",
            "comment": "survey_comment",
            "survey comment": "survey_comment",
        }
        df["source"] = df["source"].apply(
            lambda x: source_map.get(x, x) if x not in VALID_SOURCES else x
        )
        # Anything still not valid → keep as-is (don't drop; AI can still process)
        changed = (df["source"] != before_sources).sum()
        log["source_normalized"] = int(changed)

    # ── Step 5: Parse & standardize timestamps ────────────────────────────────
    if "timestamp" in df.columns:
        parsed, fixed_count, null_count = _parse_timestamps(df["timestamp"])
        df["timestamp"] = parsed
        log["invalid_timestamps_normalized"] = fixed_count
        log["timestamps_set_null"] = null_count

    # ── Step 6: Normalize rating column ──────────────────────────────────────
    if "rating" in df.columns:
        coerced = _normalize_ratings(df["rating"])
        changed = (coerced != pd.to_numeric(df["rating"], errors="coerce")).sum()
        df["rating"] = coerced
        log["ratings_coerced"] = int(pd.isna(df["rating"]).sum())

    # ── Step 7: Reset index ───────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    log["final_rows"] = len(df)

    return df, log


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _can_parse_timestamp(value: str) -> bool:
    """Return True if the string can be parsed to a datetime."""
    if not value or value.lower() in ("nan", "none", "nat", "", "null"):
        return False
    try:
        pd.to_datetime(value, infer_datetime_format=True, dayfirst=False)
        return True
    except Exception:
        return False


def _parse_timestamps(series: pd.Series) -> Tuple[pd.Series, int, int]:
    """
    Try multiple strategies to parse mixed-format timestamps.
    Returns (parsed_series, count_fixed, count_set_null).
    """
    result = pd.Series([pd.NaT] * len(series), dtype="datetime64[ns]")
    fixed = 0
    nulled = 0

    # Common formats to try (order matters: more specific first)
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]

    for idx, raw in enumerate(series):
        raw_str = str(raw).strip()
        if not raw_str or raw_str.lower() in ("nan", "none", "nat", "null", ""):
            nulled += 1
            continue

        # Try epoch timestamp (numeric string)
        if re.fullmatch(r"\d{10,13}", raw_str):
            try:
                ts = int(raw_str)
                if ts > 1e12:
                    ts /= 1000
                result.iloc[idx] = pd.Timestamp(ts, unit="s")
                fixed += 1
                continue
            except Exception:
                pass

        # Try pandas infer (handles many formats automatically)
        parsed = False
        try:
            result.iloc[idx] = pd.to_datetime(raw_str, infer_datetime_format=True, dayfirst=False)
            fixed += 1
            parsed = True
        except Exception:
            pass

        if not parsed:
            # Try each explicit format
            for fmt in formats:
                try:
                    result.iloc[idx] = datetime.strptime(raw_str, fmt)
                    fixed += 1
                    parsed = True
                    break
                except Exception:
                    continue

        if not parsed:
            nulled += 1

    return result, fixed, nulled


def _normalize_ratings(series: pd.Series) -> pd.Series:
    """
    Convert ratings to float. Handles:
    - Numeric strings: "3", "4.5"
    - Word numbers: "five", "three"
    - Star strings: "★★★"
    - Out-of-range values: clip to [1, 5]
    """
    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }

    def parse_single(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s in ("nan", "none", "null", "n/a", ""):
            return np.nan
        # Star rating
        if "★" in s or "☆" in s:
            count = s.count("★") + s.count("☆")
            return float(count) if 1 <= count <= 5 else np.nan
        # Word
        if s in word_map:
            return float(word_map[s])
        # Numeric
        try:
            val_f = float(s)
            return float(np.clip(val_f, 1, 5))
        except Exception:
            return np.nan

    return series.apply(parse_single)
