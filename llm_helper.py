import json
import random
import time
import numpy as np
import pandas as pd
from google import genai
from google.genai import types


# ---------------------------------------------------------
# Transient-error retry wrapper
# ---------------------------------------------------------
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")


class TransientAPIError(RuntimeError):
    """Raised when Gemini keeps failing with a transient/overload error after all retries."""
    pass


def _call_gemini_with_retry(client, model, contents, config, max_retries=4, base_delay=2.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            last_error = e
            is_transient = any(marker in str(e) for marker in _TRANSIENT_MARKERS)
            if is_transient and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            if is_transient:
                raise TransientAPIError(
                    "Gemini is currently overloaded and kept rejecting the request "
                    f"even after {max_retries} attempts. This is a temporary issue on "
                    "Google's side, not a bug in the app — wait a minute and try again."
                ) from e
            raise
    raise last_error


# ---------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------
def _build_prompt(num_rows: int, total_documents: int | None = None) -> str:
    sample_rule = ""
    if total_documents and total_documents > num_rows:
        fraction = num_rows / total_documents
        sample_rule = f"""
10. IMPORTANT — SAMPLE, NOT THE FULL POPULATION: the dashboard represents
    approximately {total_documents} real underlying transactions/documents,
    but you are only generating {num_rows} rows (about {fraction:.2%} of that
    volume). Do NOT force the full-period KPI totals onto these {num_rows}
    rows — that produces impossibly large single orders (e.g. one row with
    tens of thousands of units, when a real single order would have a
    normal, human-scale quantity). Instead:
    - Generate rows with realistic, human-scale per-order quantities and
      amounts, consistent with what a single real transaction of this kind
      would plausibly look like.
    - The SUM of your {num_rows} rows should approximate roughly
      {fraction:.2%} of each KPI total shown on the dashboard (i.e. a
      proportional slice), not 100% of it.
"""
    return f"""
You are an expert data analyst and AI vision specialist reconstructing the
underlying dataset behind one or more dashboard screenshots.

Follow these rules strictly:

1. Only include a column if you can point to a visible label, axis, legend,
   or card title in the image(s) that names it. Do not invent columns that
   are not directly observable.

2. A KPI card that shows a single aggregate (e.g. a "LY" or "YTD" badge on a
   summary card) is a single overall number, not a per-row breakdown. Do NOT
   create a per-row column for it (e.g. "Country_Total_LY") unless the image
   also shows that value broken down per row (e.g. a chart or table with one
   LY figure per country/category). If you cannot assign a real, distinct
   value to every single row for a column, leave that column out entirely
   rather than filling it with nulls or placeholders.

3. Never output empty strings, "N/A", "null", or placeholder values for any
   cell. Every cell in every column you include must have a real, concrete
   value consistent with the rest of that row.

4. Infer the row-level columns needed to recreate the detailed charts/tables
   (e.g. Date, Region, Product, Category, Customer, Sales, Profit, Quantity),
   and generate realistic synthetic data across EXACTLY {num_rows} rows —
   not fewer, not more.

5. Every row must represent a DISTINCT real-world order/transaction event.
   Never repeat the exact same combination of (Date, Customer, Item) across
   two different rows, and never repeat the exact same combination of
   financial figures (Revenue, Cost, Profit) or the exact same Quantity
   across two different rows — even for the same customer. Every
   transaction should have its own plausible variation in date, quantity,
   and price, the way real sales data would never show the same customer
   buying the same item on the same day with the same quantity more than
   once.

6. Profit margin (profit divided by revenue) must vary meaningfully by
   product category, reflecting realistic cost structures — for example,
   electronics typically carry tighter margins than consumables or apparel.
   Do not use the same margin percentage across unrelated categories.

7. Keep basic arithmetic internally consistent within each row (e.g. a
   profit/cost figure should equal revenue minus cost, and a unit count times
   an implied unit price should roughly equal the row's revenue).

8. When the generated rows are aggregated (grouped, summed, averaged), the
   resulting totals must closely match the top-level KPI cards and chart
   totals shown in the image(s) — subject to rule 10 below if it applies.

9. Before finalizing, review your own column list and drop any column where
   you were unsure of the value for more than a couple of rows.
{sample_rule}
Return ONLY a JSON array of flat objects, one object per row, with identical
keys across all objects. No markdown, no commentary, no surrounding text.
"""


# ---------------------------------------------------------
# Post-processing / data hygiene
# ---------------------------------------------------------
def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "n/a", "na", "null", "none", "-"):
        return True
    return False


def clean_records(records: list, null_threshold: float = 0.0) -> tuple[list, list]:
    """
    Removes columns that are entirely (or mostly, per null_threshold) blank.

    null_threshold: fraction of rows allowed to be blank before a column is
    dropped. 0.0 (default) drops a column if even a single value is blank —
    use a higher value (e.g. 0.3) to tolerate some missing data.

    Returns (cleaned_records, dropped_column_names).
    """
    if not records:
        return records, []

    df = pd.DataFrame(records)
    total_rows = len(df)
    dropped = []

    for col in df.columns:
        blank_ratio = df[col].apply(_is_blank).mean() if total_rows else 1.0
        if blank_ratio > null_threshold:
            dropped.append(col)

    df = df.drop(columns=dropped)
    return df.to_dict(orient="records"), dropped


# ---------------------------------------------------------
# Column role detection
# ---------------------------------------------------------
def _guess_financial_columns(df: pd.DataFrame) -> dict:
    """
    Best-effort guess at which columns represent revenue, cost, and profit,
    based on common naming patterns. Returns keys that may be None if a
    role couldn't be identified — callers must check before using them.
    """
    cols_lower = {c: c.lower() for c in df.columns}

    def find(*keywords):
        for col, low in cols_lower.items():
            if pd.api.types.is_numeric_dtype(df[col]) and any(k in low for k in keywords):
                return col
        return None

    return {
        "revenue": find("revenue", "sales", "amount"),
        "cost": find("cogs", "cost"),
        "profit": find("profit", "margin_value", "grossprofit"),
        "category": next((c for c in df.columns if "categor" in c.lower()), None),
    }


def _guess_dimension_columns(df: pd.DataFrame) -> dict:
    """
    Best-effort guess at which columns identify a unique transaction: the
    date it happened, who bought, what was bought, and how many units.
    Used to detect and break up duplicate "orders" created either by the
    model itself or by row-count padding.
    """
    cols_lower = {c: c.lower() for c in df.columns}

    def find(*keywords):
        for col, low in cols_lower.items():
            if any(k in low for k in keywords):
                return col
        return None

    quantity_col = None
    for col, low in cols_lower.items():
        if pd.api.types.is_numeric_dtype(df[col]) and any(k in low for k in ("quantity", "qty", "units")):
            quantity_col = col
            break

    return {
        "date": find("date", "day"),
        "customer": find("customer", "client", "account"),
        "item": find("item", "product", "sku"),
        "quantity": quantity_col,
    }


# ---------------------------------------------------------
# Realism enforcement — fixes patterns an LLM (and naive row-count padding)
# tend to fall into even with a strict prompt:
#   - identical repeated (date, customer, item, quantity) "orders"
#   - a flat, unrealistic profit margin applied to every category alike
#   - totals that balloon once duplicate rows are added to hit num_rows
# ---------------------------------------------------------
def enforce_data_realism(
    records: list,
    num_rows: int,
    seed: int | None = None,
    total_documents: int | None = None,
) -> list:
    if not records:
        return records

    rng = np.random.default_rng(seed)
    raw_df = pd.DataFrame(records).reset_index(drop=True)

    fin = _guess_financial_columns(raw_df)
    dims = _guess_dimension_columns(raw_df)
    revenue_col, cost_col, profit_col, category_col = (
        fin["revenue"], fin["cost"], fin["profit"], fin["category"]
    )
    date_col, customer_col, item_col, quantity_col = (
        dims["date"], dims["customer"], dims["item"], dims["quantity"]
    )

    # --- Anchor targets are captured from the model's RAW output, BEFORE any
    #     row-count padding/trimming happens below. This is the key fix:
    #     previously these sums were captured *after* padding, so duplicating
    #     rows to reach num_rows silently inflated the "target" total that
    #     everything else got rescaled to match. ---
    original_revenue_sum = raw_df[revenue_col].sum() if revenue_col else None
    original_cost_sum = raw_df[cost_col].sum() if (revenue_col and cost_col) else None
    original_quantity_sum = raw_df[quantity_col].sum() if quantity_col else None

    # If the dashboard represents far more real transactions than num_rows,
    # treat the generated rows as a proportional SAMPLE: scale the anchor
    # totals down to that fraction, so each row keeps a human-scale,
    # realistic size instead of a few rows being forced to carry an entire
    # year's volume.
    if total_documents and total_documents > num_rows:
        sample_fraction = num_rows / total_documents
        if original_revenue_sum is not None:
            original_revenue_sum *= sample_fraction
        if original_cost_sum is not None:
            original_cost_sum *= sample_fraction
        if original_quantity_sum is not None:
            original_quantity_sum *= sample_fraction

    df = raw_df

    # --- 1. Force the row count to exactly match what was requested ---
    current_n = len(df)
    if current_n < num_rows:
        extra_idx = rng.integers(0, current_n, size=num_rows - current_n)
        df = pd.concat([df, df.iloc[extra_idx]], ignore_index=True)
    elif current_n > num_rows:
        keep_idx = rng.choice(current_n, size=num_rows, replace=False)
        df = df.iloc[sorted(keep_idx)].reset_index(drop=True)

    n = len(df)

    # --- 1b. Break up rows that are now literal duplicates of another row on
    #         the business key (date, customer, item). This catches both
    #         duplicates the model itself produced AND duplicates introduced
    #         by the padding step above. A real business essentially never
    #         places the exact same order twice, so every occurrence after
    #         the first gets a jittered quantity and a nudged date. ---
    key_cols = [c for c in (date_col, customer_col, item_col) if c]
    if key_cols:
        dup_mask = df.duplicated(subset=key_cols, keep="first")
        dup_positions = np.where(dup_mask.values)[0]

        if len(dup_positions) and quantity_col:
            q_noise = rng.normal(loc=1.0, scale=0.15, size=len(dup_positions))
            q_noise = np.clip(q_noise, 0.6, 1.5)
            df.loc[dup_positions, quantity_col] = np.maximum(
                1, (df.loc[dup_positions, quantity_col].values * q_noise).round()
            ).astype(int)

        if len(dup_positions) and date_col:
            try:
                parsed = pd.to_datetime(df[date_col], errors="coerce")
                offsets = rng.integers(-10, 11, size=len(dup_positions))
                new_dates = parsed.iloc[dup_positions] + pd.to_timedelta(offsets, unit="D")
                df.loc[dup_positions, date_col] = new_dates.dt.strftime("%Y-%m-%d")
            except Exception:
                pass  # date column wasn't parseable — leave as-is rather than corrupt it

    # If we can't confidently identify revenue + at least one of cost/profit,
    # skip financial realism adjustments — row-count/duplicate fixes above
    # still apply, but we won't risk corrupting columns we can't interpret.
    if revenue_col and (cost_col or profit_col):
        # --- 2. Break exact duplicate revenue figures with small jitter,
        #        then rescale to the pre-padding (and, if applicable,
        #        sample-scaled) anchor total — never to the inflated total
        #        that padding would otherwise produce. ---
        noise = rng.normal(loc=1.0, scale=0.06, size=n)
        noise = np.clip(noise, 0.85, 1.15)
        adjusted_revenue = df[revenue_col] * noise
        target_revenue_sum = original_revenue_sum if original_revenue_sum else adjusted_revenue.sum()
        rescale = target_revenue_sum / adjusted_revenue.sum() if adjusted_revenue.sum() else 1
        df[revenue_col] = (adjusted_revenue * rescale).round(2)

        # --- 3. Assign each category its own base profit margin (stable
        #        per category name), then add small row-level variance ---
        if category_col:
            categories = df[category_col].astype(str).unique()
            category_margins = {cat: rng.uniform(0.15, 0.45) for cat in categories}
            base_margin = df[category_col].astype(str).map(category_margins)
        else:
            base_margin = pd.Series(rng.uniform(0.15, 0.45), index=df.index)

        row_margin = (base_margin + rng.normal(0, 0.02, n)).clip(0.05, 0.65)

        if cost_col:
            adjusted_cost = df[revenue_col] * (1 - row_margin)
            target_cost_sum = original_cost_sum if original_cost_sum else adjusted_cost.sum()
            cost_rescale = target_cost_sum / adjusted_cost.sum() if adjusted_cost.sum() else 1
            df[cost_col] = (adjusted_cost * cost_rescale).round(2)

        # --- 4. Recompute profit so it always equals revenue minus cost ---
        if profit_col and cost_col:
            df[profit_col] = (df[revenue_col] - df[cost_col]).round(2)
        elif profit_col:
            df[profit_col] = (df[revenue_col] * row_margin).round(2)

    # --- 5. Keep Quantity anchored the same way as revenue/cost, so it
    #        doesn't silently balloon from padding either, and so the
    #        implied unit price (revenue / quantity) stays plausible. ---
    if quantity_col and original_quantity_sum:
        q_noise = rng.normal(loc=1.0, scale=0.05, size=n)
        q_noise = np.clip(q_noise, 0.9, 1.1)
        adjusted_qty = df[quantity_col] * q_noise
        q_rescale = original_quantity_sum / adjusted_qty.sum() if adjusted_qty.sum() else 1
        df[quantity_col] = np.maximum(1, (adjusted_qty * q_rescale).round()).astype(int)

    return df.to_dict(orient="records")


# ---------------------------------------------------------
# Error classification — turns raw exception text into a short,
# plain-language explanation the UI can show without exposing
# stack traces or raw API error payloads to end users.
# ---------------------------------------------------------
def classify_error(e: Exception) -> dict:
    text = str(e)

    if isinstance(e, TransientAPIError):
        return {
            "title": "Gemini is temporarily busy",
            "message": "The app already retried automatically a few times. "
                       "This is a short-lived issue on Google's side, not a problem "
                       "with your images — wait a moment and try again.",
            "technical": text,
        }

    if isinstance(e, json.JSONDecodeError):
        return {
            "title": "Couldn't read the model's response",
            "message": "Gemini returned something the app couldn't parse. This "
                       "usually clears up on retry — if it keeps happening, try a "
                       "sharper screenshot with clearer, more legible numbers.",
            "technical": text,
        }

    if any(k in text for k in ("API_KEY_INVALID", "PERMISSION_DENIED", "401", "403")):
        return {
            "title": "API key problem",
            "message": "Your Gemini API key looks invalid, expired, or missing the "
                       "right permissions. Double check it in your `.env` file or "
                       "Streamlit secrets.",
            "technical": text,
        }

    if any(k in text for k in ("QUOTA", "RESOURCE_EXHAUSTED", "429")):
        return {
            "title": "Usage limit reached",
            "message": "You've hit the request limit for your Gemini API key. "
                       "Check your quota or billing in Google AI Studio, or try "
                       "again after it resets.",
            "technical": text,
        }

    if any(k in text for k in ("DEADLINE_EXCEEDED", "timeout", "Timeout")):
        return {
            "title": "Request timed out",
            "message": "The request took too long to complete. Try again, or "
                       "upload fewer/smaller images at once.",
            "technical": text,
        }

    if any(k in text for k in ("ConnectionError", "Failed to establish", "Name or service not known", "NewConnectionError")):
        return {
            "title": "Couldn't reach Gemini",
            "message": "The app couldn't connect to Google's servers. Check your "
                       "internet connection and try again.",
            "technical": text,
        }

    return {
        "title": "Something went wrong",
        "message": "An unexpected error occurred while processing your request. "
                   "Try again, and if it keeps happening, reach out for help.",
        "technical": text,
    }


# ---------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------
def extract_dashboard_data(
    images_data: list,
    api_key: str,
    num_rows: int = 50,
    null_threshold: float = 0.0,
    total_documents: int | None = None,
) -> tuple[list, list]:
    """
    Extracts a synthetic tabular dataset matching the KPIs shown in the
    provided dashboard screenshot(s).

    total_documents: if you know (or can read off a KPI card, e.g.
    "Documents: 61,079") the real number of underlying transactions the
    dashboard represents, pass it here. When it's larger than num_rows,
    the generated rows are treated as a proportional sample instead of
    being forced to carry the full period's totals — this is what keeps
    individual row quantities/amounts realistic instead of absurdly large.

    Returns (records, dropped_columns) where dropped_columns lists any
    columns the model produced that were removed for being blank/unreliable.
    """
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(num_rows, total_documents=total_documents)

    contents = []
    for img_bytes, mime_type in images_data:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    contents.append(prompt)

    response = _call_gemini_with_retry(
        client=client,
        model='gemini-3.6-flash',
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )

    response_text = response.text.strip()

    # Defensive cleanup in case the model still wraps output in fences.
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]

    records = json.loads(response_text.strip())
    cleaned_records, dropped_columns = clean_records(records, null_threshold=null_threshold)
    realistic_records = enforce_data_realism(
        cleaned_records, num_rows=num_rows, total_documents=total_documents
    )

    return realistic_records, dropped_columns
