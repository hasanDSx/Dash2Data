"""LLM layer: dashboard screenshots -> DashboardSpec.

The model only *describes* the dashboard (columns, measures, visible numbers).
All data generation and number-matching is done by deterministic code.
"""
from __future__ import annotations

import json
import logging
import re
import time

from .config import (DEFAULT_MODEL, FALLBACK_MODELS, MAX_API_RETRIES, MAX_SPEC_REPAIRS,
                     RETRYABLE_CODES)
from .schemas import DashboardSpec, SpecError, load_spec

log = logging.getLogger(__name__)

PROMPT = """You are a senior BI analyst. The attached image(s) are screenshots of a Power BI / Tableau / Looker
dashboard. Do NOT generate any data rows. Produce a JSON *specification* of the dataset behind it.

Return ONLY one JSON object (no markdown, no commentary) with exactly these keys:

{
  "title": "short dashboard name",
  "fact_table": "Fact_Sales",                       // snake/pascal case, e.g. Fact_Orders
  "currency": "$",
  "date": {"name": "Order Date", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} | null,
  "dimensions": [
    {"name": "Region", "values": ["East", "West"]},
    {"name": "Category", "values": ["Technology", "Furniture"]},
    {"name": "Product", "parent": "Category", "values": ["Laptop", "Chair"],
     "parent_map": {"Laptop": "Technology", "Chair": "Furniture"}},
    {"name": "Customer", "values": ["Acme"], "cardinality": 800}
  ],
  "id_columns": [{"name": "OrderID"}],
  "fact_columns": [
    {"name": "Sales", "kind": "money"},
    {"name": "Cost", "kind": "money", "derived_from": "Sales", "derived_ratio": 0.7},
    {"name": "Quantity", "kind": "quantity"},
    {"name": "Discount", "kind": "rate", "min": 0, "max": 0.5}
  ],
  "measures": [
    {"name": "Total Sales", "kind": "agg", "agg": "SUM", "column": "Sales", "format": "currency"},
    {"name": "Total Cost", "kind": "agg", "agg": "SUM", "column": "Cost", "format": "currency"},
    {"name": "Total Profit", "kind": "diff", "left": "Total Sales", "right": "Total Cost", "format": "currency"},
    {"name": "Profit Margin", "kind": "ratio", "num": "Total Profit", "den": "Total Sales", "format": "percent"},
    {"name": "Orders", "kind": "agg", "agg": "DISTINCTCOUNT", "column": "OrderID", "format": "number"},
    {"name": "Online Sales", "kind": "agg", "agg": "SUM", "column": "Sales", "format": "currency",
     "filters": [{"column": "Channel", "value": "Online"}]}
  ],
  "targets": [
    {"measure": "Total Sales", "value": 1200000, "source": "kpi", "label": "Total Sales card"},
    {"measure": "Total Sales", "value": 400000, "source": "chart", "label": "Sales by Region",
     "filters": [{"column": "Region", "value": "East"}], "estimated": false}
  ],
  "notes": "assumptions you made"
}

RULES
1. RAW COLUMNS ONLY. Never create a column for something computable from other columns: profit, margin, ratios,
   averages per order, growth, YoY, % of total, rankings, running totals. Those are MEASURES (kind ratio/diff/sum or
   agg). The dataset must stay narrow; derived metrics are delivered later as DAX.
2. If a raw numeric column is a fixed fraction of another (e.g. Cost ~ 70% of Sales), use derived_from/derived_ratio.
3. Dimension "values": list every category/label you can read (axes, legends, slicers, table rows). Use
   "parent"/"parent_map" for hierarchies (Category -> Product). A dimension may have at most one child.
   Use "cardinality" only when a distinct-count KPI implies far more members than are visible.
4. measure.kind: "agg" (agg: SUM|AVERAGE|MIN|MAX|COUNT|DISTINCTCOUNT|COUNTROWS, plus column; COUNTROWS has no
   column), "ratio" (num, den = measure names), "diff" (left, right), "sum" (items). Composite measures must
   reference other measures by name. format: currency|number|percent|decimal.
5. fact_columns.kind: money | quantity | rate (0..1 fractions) | number.
6. TARGETS are the numbers visible on the dashboard. Add one target for every KPI card and for every readable bar,
   point, slice, label or table cell of the charts (up to 150). Chart values use "filters" with the group's
   dimension/value. If a slicer or page filter is active, add it to the "filters" of every affected target.
   Time axes: use the reserved filter columns Year, Quarter (Q1..Q4), Month (Jan..Dec), MonthNumber, YearMonth
   (YYYY-MM) - they exist automatically when "date" is set. Do not declare them as dimensions.
7. Numbers must be plain JSON numbers: expand 1.2M -> 1200000, 3.4K -> 3400. Percentages are FRACTIONS
   (23% -> 0.23). If a value is only estimable from an axis scale, give your best estimate and set "estimated": true.
8. Several images belong to the same dataset: reuse the same column and measure names across them.
9. Only describe what is visible or clearly implied. Do not invent charts.
"""


def _client(api_key: str):
    from google import genai  # lazy: keeps the rest of the package importable without the SDK
    return genai.Client(api_key=api_key.strip())


def _call(client, models: list, contents: list):
    from google.genai import types
    from google.genai.errors import APIError

    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    last: Exception | None = None
    for model in models:
        for attempt in range(MAX_API_RETRIES):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except APIError as e:
                if getattr(e, "code", None) in RETRYABLE_CODES:
                    last = e
                    log.warning("Gemini %s attempt %d failed (%s); retrying", model, attempt + 1, e.code)
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
    assert last is not None
    raise last


def parse_json_text(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    data = json.loads(t)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise SpecError("Model returned JSON that is not an object.")
    return data


def extract_spec(images_data: list, api_key: str, model: str | None = None, hints: str = ""):
    """images_data: [(bytes, mime_type), ...] -> (DashboardSpec, warnings, raw_json_text)."""
    from google.genai import types

    client = _client(api_key)
    models = [model or DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != (model or DEFAULT_MODEL)]
    parts = [types.Part.from_bytes(data=b, mime_type=mt) for b, mt in images_data]
    prompt = PROMPT + (f"\nUSER HINTS (trust these): {hints.strip()}\n" if hints.strip() else "")

    contents = parts + [prompt]
    last_err: Exception | None = None
    raw = ""
    for attempt in range(MAX_SPEC_REPAIRS + 1):
        response = _call(client, models, contents)
        raw = response.text or ""
        try:
            spec, warnings = load_spec(parse_json_text(raw))
            if attempt:
                warnings.insert(0, f"Spec needed {attempt} automatic repair pass(es).")
            return spec, warnings, raw
        except (json.JSONDecodeError, SpecError, KeyError, TypeError, ValueError) as e:
            last_err = e
            log.warning("Spec invalid on attempt %d: %s", attempt + 1, e)
            contents = parts + [
                prompt,
                f"\nYour previous answer was rejected: {e}\nPrevious answer:\n{raw[:6000]}\n"
                "Return the corrected, complete JSON object only.",
            ]
    raise SpecError(f"Could not obtain a valid dashboard specification from the model: {last_err}")
