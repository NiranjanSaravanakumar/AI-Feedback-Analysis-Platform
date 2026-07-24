"""
utils/logger.py
Builds the plain-text cleaning log for download.
"""
import datetime
from typing import Dict, Any


def build_cleaning_log(cleaning_log: Dict[str, Any]) -> str:
    """Return a formatted plain-text cleaning log."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_removed = (
        cleaning_log.get("rows_loaded", 0) - cleaning_log.get("final_rows", 0)
    )

    lines = [
        "=" * 55,
        "  CUSTOMER FEEDBACK INTELLIGENCE SYSTEM — CLEANING LOG",
        "=" * 55,
        f"  Generated At     : {now}",
        "-" * 55,
        "",
        "  INPUT",
        f"  Rows Loaded                    : {cleaning_log.get('rows_loaded', 'N/A')}",
        "",
        "  CLEANING OPERATIONS",
        f"  Duplicate Rows Removed         : {cleaning_log.get('duplicate_rows_removed', 0)}",
        f"  Blank Feedback Removed         : {cleaning_log.get('blank_feedback_removed', 0)}",
        f"  Source Values Normalized       : {cleaning_log.get('source_normalized', 0)}",
        f"  Timestamps Normalized          : {cleaning_log.get('invalid_timestamps_normalized', 0)}",
        f"  Timestamps Set Null            : {cleaning_log.get('timestamps_set_null', 0)}",
        f"  Ratings Coerced to Null        : {cleaning_log.get('ratings_coerced', 0)}",
        "",
        "  OUTPUT",
        f"  Total Rows Removed             : {rows_removed}",
        f"  Final Rows                     : {cleaning_log.get('final_rows', 'N/A')}",
        "",
        "=" * 55,
        "  NOTES",
        "  - Duplicates: exact match on (id, feedback_text)",
        "  - Blank: empty, whitespace, or meaningless values",
        "    (e.g. 'N/A', '.', '—', strings < 3 chars)",
        "  - Timestamps: 15+ formats attempted; failures set NaT",
        "  - Ratings: word/star formats coerced; out-of-range clipped",
        "=" * 55,
    ]
    return "\n".join(lines)
