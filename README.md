# 📊 Customer Feedback Intelligence System

> **QuickCart · Final-Round Practical Assessment**  
> End-to-end pipeline: Upload → Clean → AI Enrich → Visualize → Download

---

## 🚀 Quick Start (How to Run)

### 1. Prerequisites

- Python 3.9 or higher
- A **NVIDIA NIM API key** → [Get one free at integrate.api.nvidia.com](https://integrate.api.nvidia.com)

---

### 2. Create & Activate a Virtual Environment

A virtual environment isolates this project's packages from your global Python.
This prevents version conflicts with other projects and makes the setup reproducible.

```bash
cd d:\ANTIGRAVITY\Streamkit

# Create the venv (only once)
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# You should now see (venv) in your terminal prompt
```

> **Why venv?**  
> Without it, `pip install` writes packages into your **global Python**, which can break other projects that need different versions of the same library. With venv, this project gets its own clean, isolated Python environment. Delete the `venv/` folder to fully uninstall everything.

---

### 3. Install Dependencies (inside the venv)

```bash
# Make sure (venv) is active in your prompt first
pip install -r requirements.txt
```

---

### 4. (Optional) Set Your API Key via `.env`

```bash
# Copy the example file
copy .env.example .env

# Open .env and paste your key
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
```

> You can also enter the key directly in the app's text field — no `.env` file required.

---

### 5. Run the App

```bash
# Make sure (venv) is still active
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

### 6. Deactivate the venv (when done)

```bash
deactivate
```

---

### 5. Use the App (Step by Step)

| Step | Action | What Happens |
|------|--------|-------------|
| 1 | Enter NVIDIA API Key | App authenticates silently |
| 2 | Upload `customer_feedback_raw.csv` | Pandas reads & inspects it |
| 3 | Click **[ Process & Clean Data ]** | Cleaning pipeline runs |
| 4 | Click **[ Perform AI Analysis ]** | Batched NVIDIA API calls |
| 5 | View Dashboard | Charts + sample messages appear |
| 6 | Download | CSV + DOCX Report + Cleaning Log |

---

## 🗂️ Project Structure

```
Streamkit/
├── app.py                  ← Single-page Streamlit UI (all UI here)
├── requirements.txt        ← All Python dependencies
├── .env.example            ← API key template
│
└── utils/
    ├── cleaner.py          ← Pandas data quality + cleaning pipeline
    ├── ai_enrichment.py    ← NVIDIA API calls (batched, with retry)
    ├── analytics.py        ← Plotly chart builders + aggregations
    ├── report.py           ← DOCX summary report generator
    └── logger.py           ← Plain-text cleaning log builder
```

---

## ⚙️ How It Works

### Stage 1 — Data Ingestion

`pandas.read_csv()` loads the raw file. The `inspect_dataframe()` function immediately counts:
- Total rows
- Missing values (across all columns)
- Duplicate rows (matching both `id` AND `feedback_text`)
- Blank / meaningless feedback entries
- Unparseable timestamps

These numbers appear in the **Dataset Snapshot** cards before any changes are made.

---

### Stage 2 — Data Cleaning (`utils/cleaner.py`)

The cleaning pipeline runs **7 deterministic steps**, each logged:

| Step | What It Does | Why |
|------|-------------|-----|
| 1. Strip whitespace | Removes leading/trailing spaces from all strings | Prevents invisible duplicates |
| 2. Drop exact duplicates | Removes rows where `id` + `feedback_text` both match | Same review ID with same text = genuine duplicate |
| 3. Drop blank feedback | Removes empty, `N/A`, `.`, `—`, strings < 3 chars | Meaningless rows waste AI quota |
| 4. Normalize `source` | Lowercases, maps variants (e.g. `"ticket"` → `"support_ticket"`) | Consistent grouping for charts |
| 5. Parse timestamps | Tries 15+ date formats + epoch detection | Mixed formats are the #1 real-world trap |
| 6. Normalize ratings | Converts `"five"`, `"★★★"`, clamps to [1–5] | Ratings are often entered as text |
| 7. Reset index | Clean sequential index | Required for AI batch alignment |

**Key design decision:** Duplicate detection uses BOTH `id` AND `feedback_text`. A row with the same ID but different text is *not* a duplicate — it could be an edited resubmission.

---

### Stage 3 — AI Enrichment (`utils/ai_enrichment.py`)

For each cleaned row, the AI derives:

| Field | Values | Notes |
|-------|--------|-------|
| `sentiment` | `positive` / `negative` / `neutral` | Fixed list — no other values accepted |
| `category` | `Billing` / `App Bug` / `Delivery` / `Staff/Support` / `Other` | Fixed list per assessment rules |
| `summary` | One-line plain English | Max ~15 words, truncated to 200 chars |

**Batching strategy (critical for 1,800 rows):**

```
Naive approach:  1,800 API calls  →  very slow, expensive
Our approach:    20 rows per call →  ~90 API calls  →  90x faster
```

The prompt sends a JSON array of 20 feedback items and receives a JSON array of 20 results back. This uses the model's full context window efficiently.

**Reliability measures:**
- Each batch retried once on failure (with exponential backoff)
- JSON extracted with regex (handles markdown code block wrapping)
- Every output validated: any value outside the fixed lists → replaced with `neutral` / `Other`
- Fallback default applied per-row if a batch index is missing from the response

**The AI is explicitly instructed:** *Trust the feedback text over the star rating.* A 5-star entry saying "this app crashed" is **negative**.

---

### Stage 4 — Analytics (`utils/analytics.py`)

Three Plotly charts built from the enriched dataframe:

| Chart | Type | What It Shows |
|-------|------|--------------|
| Complaint Categories | Horizontal Bar | Top 5 categories by volume |
| Sentiment Distribution | Donut | Positive / Negative / Neutral breakdown |
| Sentiment Over Time | Line (weekly) | Trend over time; falls back to "by source" if timestamps are mostly null |

---

### Stage 5 — Download

| File | Contents |
|------|----------|
| `customer_feedback_processed.csv` | All cleaned rows + `sentiment`, `category`, `summary` columns |
| `customer_feedback_report.docx` | 6-section Word document (executive summary, quality stats, top categories, sentiment breakdown, example messages, AI notes) |
| `cleaning_log.txt` | Plain-text log of every operation and count |

---

## 🧠 Why I Made These Technology Choices

### Streamlit — not Flask/React/Django

**Why:** Streamlit lets one Python developer build a fully interactive data app without writing any HTML, CSS routing, or JavaScript. The entire pipeline (upload → clean → AI → charts → download) fits in a single `app.py` with no backend server, no API endpoints, and no frontend build step.

**Trade-off I knew about:** Streamlit re-runs the entire script on every user interaction. I solved this with `st.session_state` to cache `raw_df`, `cleaned_df`, `enriched_df`, and report bytes — so cleaning and AI results are never lost when the user clicks a button.

---

### NVIDIA NIM API — not OpenAI directly

**Why:** The assessment uses NVIDIA's API key. NVIDIA NIM exposes its models through an **OpenAI-compatible endpoint** — meaning the exact same `openai` Python SDK works with just a `base_url` change. No extra library, no SDK to learn. The `meta/llama-3.1-8b-instruct` model is fast, capable, and free-tier accessible.

**Why LLaMA-3.1-8B specifically:**
- Fast enough for batch processing
- Follows structured JSON instructions reliably
- Free-tier friendly for 1,800 rows of evaluation

The app also lets the user switch to LLaMA-70B or Nemotron-70B from a dropdown for higher accuracy at higher cost.

---

### Pandas — not SQL/Spark

**Why:** At 1,800 rows, Pandas is the right tool. It fits in memory in milliseconds. Spark and Dask have startup overhead that would dwarf the actual processing time for this dataset size. SQLite would add complexity without benefit since the output is a flat CSV anyway.

**Scale note:** If this grew to millions of rows, I would switch the timestamp parsing loop to vectorized `pd.to_datetime` with `errors="coerce"` and profile the cleaning bottlenecks before reaching for Spark.

---

### Plotly — not Matplotlib/Seaborn

**Why:** Plotly charts are **interactive by default** in Streamlit (hover tooltips, zoom, pan). Matplotlib/Seaborn produce static images. For a dashboard that a manager will explore, interactivity is essential. Plotly also supports dark themes natively.

---

### python-docx — not reportlab (PDF)

**Why:** DOCX files are **editable**. A manager can open the report in Word, add their own commentary, or paste it into a slide deck. PDFs look more polished but require reportlab's complex layout engine for tables and colored headers. DOCX produces a professional result with far less code.

**Trade-off:** DOCX requires Microsoft Word or LibreOffice to open. If PDF was required, I would add reportlab as a secondary download button.

---

## ⚖️ Trade-offs I'm Aware Of

| Trade-off | What I Did | What I'd Do With More Time |
|-----------|-----------|---------------------------|
| AI cost on 1,800 rows | Batched 20 rows/call (~90 calls total) | Add a cost estimator + user confirmation dialog |
| Timestamp parsing | Tried 15+ formats manually | Use `dateutil.parser` with explicit fallback chain |
| Report styling | Clean but minimal DOCX | Add chart screenshots embedded in the DOCX |
| Error recovery | Failed batches get fallback values | Retry queue + user-visible error table |
| No database | Output is CSV | SQLite with a `processed_runs` table for audit trail |
| Re-run on page refresh | Session state preserved in-memory | Add persistent caching with `@st.cache_data` |

---

## 🤖 AI Usage Log

*(Required by assessment section 6)*

| What I Asked the AI | What Was Wrong / Incomplete | How I Fixed It |
|--------------------|-----------------------------|----------------|
| Prompt to extract sentiment + category + summary | Early versions returned free-form text, not JSON | Added `"respond with valid JSON only"` to system prompt and regex extraction as fallback |
| Batch 20 rows in one API call | Model sometimes skipped indices in the response array | Added `result_map` keyed by `index` field; missing indices fall back to defaults |
| Timestamp parsing across 15 formats | `pd.to_datetime(infer=True)` silently swapped DD/MM | Added explicit format list tried in order; epoch detection separate |
| Deduplication logic | First version dropped any duplicate `id` — lost valid re-submissions | Changed to require BOTH `id` AND `feedback_text` to match |
| Rating normalization | Missed star-character ratings (`★★★`) | Added `"★" in s` check before numeric coercion |
| AI category validation | Model occasionally returned `"Technical Issue"` (not in fixed list) | Added validation loop: anything not in `VALID_CATEGORIES` → `"Other"` |

---

## 📋 Output Column Reference

| Column | Type | Notes |
|--------|------|-------|
| `id` | string | Original row ID |
| `timestamp` | datetime | ISO 8601, NaT if unparseable |
| `source` | string | Normalized: `support_ticket` / `app_store_review` / `survey_comment` |
| `rating` | float | 1.0–5.0, NaN if missing/invalid |
| `feedback_text` | string | Stripped, non-empty |
| `sentiment` | string | `positive` / `negative` / `neutral` |
| `category` | string | `Billing` / `App Bug` / `Delivery` / `Staff/Support` / `Other` |
| `summary` | string | One-line AI-generated summary |

---

## 🔮 What I Would Improve With More Time

1. **Embed Plotly charts as images in the DOCX report** (currently text-only tables)
2. **Add a cost estimator** before the AI step (tokens × price per token)
3. **SQLite audit table** — store each processing run with timestamp and row counts
4. **Async parallel batching** — use `asyncio` + NVIDIA's async client for 3-5× speedup
5. **Confidence scoring** — ask the model to return a `confidence: 0.0–1.0` field and flag low-confidence rows for human review
6. **Rating vs. sentiment conflict detection** — flag rows where `rating=5` but `sentiment=negative`
7. **Streaming enriched rows to disk** — avoid holding 1,800 enriched rows in memory at once

---

*Built for QuickCart · Customer Feedback Intelligence Assessment*
