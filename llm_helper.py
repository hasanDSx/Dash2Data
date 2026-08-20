import json
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
   and generate realistic synthetic data across approximately {num_rows} rows.

5. When the generated rows are aggregated (grouped, summed, averaged), the
   resulting totals must closely match the top-level KPI cards and chart
   totals shown in the image(s).

6. Before finalizing, review your own column list and drop any column where
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

    return cleaned_records, dropped_columns
