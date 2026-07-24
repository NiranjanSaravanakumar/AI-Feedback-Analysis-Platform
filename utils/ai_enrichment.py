"""
utils/ai_enrichment.py
NVIDIA API enrichment for customer feedback — optimized for 1800+ rows.

Strategy:
  - Batch 20 rows per API call  →  ~90 calls for 1800 rows (not 1800 calls)
  - Retry once on transient errors
  - Fallback defaults on persistent failure
  - Yields (batch_results, progress_pct) for Streamlit progress bar
"""

import json
import re
import time
import logging
from typing import List, Dict, Any, Generator, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Fixed category list (DO NOT change) ─────────────────────────────────────
VALID_CATEGORIES = {"Billing", "App Bug", "Delivery", "Staff/Support", "Other"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

FALLBACK_RESULT = {
    "sentiment": "neutral",
    "category": "Other",
    "summary": "Could not process this entry.",
}

BATCH_SIZE = 20          # rows per API call
MAX_RETRIES = 2          # retry attempts per batch
RETRY_DELAY = 2.0        # seconds between retries


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_dataframe_streaming(
    df,
    api_key: str,
    model: str = "meta/llama-3.1-8b-instruct",
) -> Generator[Tuple[List[Dict], int, int], None, None]:
    """
    Generator that processes the dataframe in batches.

    Yields: (batch_results: List[dict], processed_count: int, total: int)

    Usage in Streamlit:
        results = []
        for batch, done, total in enrich_dataframe_streaming(df, api_key):
            results.extend(batch)
            progress_bar.progress(done / total)
    """
    client = _make_client(api_key)
    total = len(df)
    rows = df.to_dict(orient="records")

    processed = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]
        batch_results = _process_batch(client, batch_rows, model)
        processed += len(batch_rows)
        yield batch_results, processed, total


# ─────────────────────────────────────────────────────────────────────────────
# Private: client
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private: batch processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_batch(
    client: OpenAI,
    rows: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Send a batch of rows to NVIDIA API. Retry on failure.
    Returns a list of dicts with keys: sentiment, category, summary.
    """
    for attempt in range(MAX_RETRIES):
        try:
            prompt = _build_batch_prompt(rows)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,       # low temp for consistent structured output
                max_tokens=2000,
                top_p=0.9,
            )
            raw_text = response.choices[0].message.content.strip()
            parsed = _parse_batch_response(raw_text, len(rows))
            return parsed

        except Exception as e:
            logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"Batch permanently failed after {MAX_RETRIES} attempts.")

    # All retries exhausted — return fallbacks for entire batch
    return [FALLBACK_RESULT.copy() for _ in rows]


def _system_prompt() -> str:
    return """You are a customer feedback analyst for QuickCart, a food and grocery delivery app.

Your job is to analyze customer feedback messages and return structured JSON analysis.

Rules:
1. Sentiment must be exactly one of: positive, negative, neutral
2. Category must be exactly one of: Billing, App Bug, Delivery, Staff/Support, Other
   - Billing: payment issues, charges, refunds, subscriptions
   - App Bug: crashes, errors, slow app, login issues, technical problems
   - Delivery: late delivery, wrong order, missing items, delivery person issues
   - Staff/Support: customer service, rude staff, unhelpful support
   - Other: anything that doesn't clearly fit the above categories
3. Summary: one concise sentence (max 15 words) describing the actual issue
4. IMPORTANT: Trust the feedback TEXT over the star rating — ratings can be wrong
5. Always respond with valid JSON only, no extra text"""


def _build_batch_prompt(rows: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for a batch of rows.
    Returns a JSON array response.
    """
    items = []
    for i, row in enumerate(rows):
        text = str(row.get("feedback_text", "")).strip()
        source = str(row.get("source", "")).strip()
        rating = row.get("rating", "")
        rating_str = str(rating) if rating and str(rating) not in ("nan", "None", "") else "not provided"

        items.append(
            f'  {{"index": {i}, "text": {json.dumps(text)}, "source": "{source}", "rating": "{rating_str}"}}'
        )

    prompt = (
        "Analyze these customer feedback entries and return a JSON array.\n"
        "For each entry, provide sentiment, category, and summary.\n\n"
        "Input:\n[\n" + ",\n".join(items) + "\n]\n\n"
        "Return ONLY a JSON array with exactly this structure for each entry:\n"
        '[\n'
        '  {"index": 0, "sentiment": "...", "category": "...", "summary": "..."},\n'
        '  ...\n'
        ']'
    )
    return prompt


def _parse_batch_response(raw: str, expected_count: int) -> List[Dict[str, Any]]:
    """
    Parse the JSON array response from the API.
    Validates each entry and applies fallbacks for invalid values.
    """
    # Extract JSON array from the response (handle markdown code blocks)
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        logger.warning(f"No JSON array found in response: {raw[:200]}")
        return [FALLBACK_RESULT.copy() for _ in range(expected_count)]

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed: {e}. Raw: {raw[:200]}")
        return [FALLBACK_RESULT.copy() for _ in range(expected_count)]

    # Build index-keyed results
    result_map: Dict[int, Dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", -1)
        sentiment = str(item.get("sentiment", "")).strip().lower()
        category = str(item.get("category", "")).strip()
        summary = str(item.get("summary", "")).strip()

        # Validate & sanitize
        if sentiment not in VALID_SENTIMENTS:
            sentiment = "neutral"
        if category not in VALID_CATEGORIES:
            category = "Other"
        if not summary or len(summary) < 3:
            summary = "No summary available."
        # Truncate overly long summaries
        if len(summary) > 200:
            summary = summary[:197] + "..."

        result_map[idx] = {
            "sentiment": sentiment,
            "category": category,
            "summary": summary,
        }

    # Fill in any missing indices with fallback
    results = []
    for i in range(expected_count):
        results.append(result_map.get(i, FALLBACK_RESULT.copy()))

    return results
