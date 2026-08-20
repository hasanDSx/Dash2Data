import json
import random
import numpy as np
import pandas as pd
from google import genai
from google.genai import types


# ---------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------
def _build_prompt(num_rows: int) -> str:
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

5. Never repeat the exact same combination of financial figures (e.g. Revenue,
   Cost, Profit) across two different rows, even for the same customer. Every
   transaction should have its own plausible variation in quantity and price,
   the way real sales data would never show two different products selling
   for the identical exact total.

6. Profit margin (profit divided by revenue) must vary meaningfully by
   product category, reflecting realistic cost structures — for example,
   electronics typically carry tighter margins than consumables or apparel.
   Do not use the same margin percentage across unrelated categories.

7. Keep basic arithmetic internally consistent within each row (e.g. a
   profit/cost figure should equal revenue minus cost, and a unit count times
   an implied unit price should roughly equal the row's revenue).

8. When the generated rows are aggregated (grouped, summed, averaged), the
   resulting totals must closely match the top-level KPI cards and chart
   totals shown in the image(s).

9. Before finalizing, review your own column list and drop any column where
   you were unsure of the value for more than a couple of rows.

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
# Numeric column detection
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


# ---------------------------------------------------------
# Realism enforcement — fixes patterns an LLM tends to fall into even
# with a strict prompt: identical repeated financial figures across rows,
# and a flat, unrealistic profit margin applied to every category alike.
# ---------------------------------------------------------
def enforce_data_realism(records: list, num_rows: int, seed: int | None = None) -> list:
    if not records:
        return records

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records).reset_index(drop=True)

    # --- 1. Force the row count to exactly match what was requested ---
    current_n = len(df)
    if current_n < num_rows:
        extra_idx = rng.integers(0, current_n, size=num_rows - current_n)
        df = pd.concat([df, df.iloc[extra_idx]], ignore_index=True)
    elif current_n > num_rows:
        keep_idx = rng.choice(current_n, size=num_rows, replace=False)
        df = df.iloc[sorted(keep_idx)].reset_index(drop=True)

    fin = _guess_financial_columns(df)
    revenue_col, cost_col, profit_col, category_col = (
        fin["revenue"], fin["cost"], fin["profit"], fin["category"]
    )

    # If we can't confidently identify revenue + at least one of cost/profit,
    # skip financial realism adjustments — row-count enforcement above still
    # applies, but we won't risk corrupting columns we can't interpret.
    if revenue_col and (cost_col or profit_col):
        n = len(df)
        original_revenue_sum = df[revenue_col].sum()
        original_cost_sum = df[cost_col].sum() if cost_col else None

        # --- 2. Break exact duplicate revenue figures with small jitter,
        #        then rescale so the overall total is preserved (keeps the
        #        dashboard-matching totals from the prompt requirement intact) ---
        noise = rng.normal(loc=1.0, scale=0.06, size=n)
        noise = np.clip(noise, 0.85, 1.15)
        adjusted_revenue = df[revenue_col] * noise
        rescale = original_revenue_sum / adjusted_revenue.sum() if adjusted_revenue.sum() else 1
        df[revenue_col] = (adjusted_revenue * rescale).round(2)

        # --- 3. Assign each category its own base profit margin (stable
        #        per category name), then add small row-level variance ---
        if category_col:
            categories = df[category_col].astype(str).unique()
            category_margins = {
                cat: rng.uniform(0.15, 0.45) for cat in categories
            }
            base_margin = df[category_col].astype(str).map(category_margins)
        else:
            base_margin = pd.Series(rng.uniform(0.15, 0.45), index=df.index)

        row_margin = (base_margin + rng.normal(0, 0.02, n)).clip(0.05, 0.65)

        if cost_col:
            adjusted_cost = df[revenue_col] * (1 - row_margin)
            if original_cost_sum:
                cost_rescale = original_cost_sum / adjusted_cost.sum() if adjusted_cost.sum() else 1
                adjusted_cost = adjusted_cost * cost_rescale
            df[cost_col] = adjusted_cost.round(2)

        # --- 4. Recompute profit so it always equals revenue minus cost ---
        if profit_col and cost_col:
            df[profit_col] = (df[revenue_col] - df[cost_col]).round(2)
        elif profit_col:
            df[profit_col] = (df[revenue_col] * row_margin).round(2)

    return df.to_dict(orient="records")


# ---------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------
def extract_dashboard_data(
    images_data: list,
    api_key: str,
    num_rows: int = 50,
    null_threshold: float = 0.0,
) -> tuple[list, list]:
    """
    Extracts a synthetic tabular dataset matching the KPIs shown in the
    provided dashboard screenshot(s).

    Returns (records, dropped_columns) where dropped_columns lists any
    columns the model produced that were removed for being blank/unreliable.
    """
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(num_rows)

    contents = []
    for img_bytes, mime_type in images_data:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    contents.append(prompt)

    response = client.models.generate_content(
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
    realistic_records = enforce_data_realism(cleaned_records, num_rows=num_rows)

    return realistic_records, dropped_columns
