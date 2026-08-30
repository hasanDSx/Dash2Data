import json
import random
import re
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
# Prompt construction — domain-agnostic. Says nothing about "sales"
# specifically; works the same for a sales, HR, marketing, or ops dashboard.
# ---------------------------------------------------------
def _build_prompt(num_rows: int, total_documents: int | None = None) -> str:
    sample_rule = ""
    if total_documents and total_documents > num_rows:
        fraction = num_rows / total_documents
        sample_rule = f"""
10. IMPORTANT — SAMPLE, NOT THE FULL POPULATION: the dashboard represents
    approximately {total_documents} real underlying records/transactions,
    but you are only generating {num_rows} rows (about {fraction:.2%} of that
    volume). Do NOT force the full-period KPI totals onto these {num_rows}
    rows — that produces impossibly large single records (e.g. one row
    worth what should be hundreds of real ones). Instead:
    - Generate rows with realistic, human-scale per-record values, consistent
      with what a single real record of this kind would plausibly look like.
    - The SUM of your {num_rows} rows should approximate roughly
      {fraction:.2%} of each KPI total shown on the dashboard (a
      proportional slice), not 100% of it.
"""
    return f"""
You are an expert data analyst and AI vision specialist reconstructing the
underlying row-level dataset behind one or more dashboard screenshots. The
dashboard could be about any domain — sales, HR, marketing, operations,
finance, logistics, etc. Infer the schema entirely from what is visible;
do not assume any fixed set of column names.

Follow these rules strictly:

1. Only include a column if you can point to a visible label, axis, legend,
   or card title in the image(s) that names it. Do not invent columns that
   are not directly observable.

2. A KPI card that shows a single aggregate (e.g. a "LY" or "YTD" badge on a
   summary card) is a single overall number, not a per-row breakdown. Do NOT
   create a per-row column for it unless the image also shows that value
   broken down per row (e.g. a chart or table with one figure per
   category/entity). If you cannot assign a real, distinct value to every
   single row for a column, leave that column out entirely rather than
   filling it with nulls or placeholders.

3. Never output empty strings, "N/A", "null", or placeholder values for any
   cell. Every cell in every column you include must have a real, concrete
   value consistent with the rest of that row.

4. Infer whatever row-level columns are needed to recreate the detailed
   charts/tables shown (dates, entities/customers/employees, categories,
   products/items, amounts, counts, statuses — whatever applies to this
   specific dashboard), and generate realistic synthetic data across EXACTLY
   {num_rows} rows — not fewer, not more.

5. Every row must represent a DISTINCT real-world record/event. Never repeat
   the exact same combination of (date, entity, category/item) across two
   different rows, and never repeat the exact same combination of numeric
   values (e.g. amount, cost, profit, or any other measure columns) — even
   for the same entity. Every record should have its own plausible
   variation, the way real data would never show the same entity with the
   same category on the same day with the same numbers more than once.

6. If there is an ID / document / reference / order-number style column,
   treat each distinct ID as ONE real-world record: every row sharing the
   same ID must also share the exact same date and the exact same entity
   (customer/employee/etc.) — an ID can never span two different dates or
   two different entities. If you generate multiple rows that logically
   belong to the same real event (e.g. multiple line items on one invoice),
   they must share both the ID and the date; anything else must get its own
   distinct ID.

7. If multiple numeric columns are related (e.g. a total that equals another
   column minus/plus a third), keep that arithmetic internally consistent
   within each row. Rates, ratios, or margins derived from those columns
   should vary meaningfully across categories/segments rather than being a
   single flat percentage applied everywhere.

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
# Generic schema inference — no assumptions about domain or naming
# convention beyond broad, cross-domain synonym lists. This is what makes
# the pipeline reusable for any dashboard, not just a sales dashboard.
# ---------------------------------------------------------
def _find_col(df: pd.DataFrame, keywords, numeric_only: bool = False) -> str | None:
    cols_lower = {c: c.lower() for c in df.columns}
    for col, low in cols_lower.items():
        if numeric_only and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(k in low for k in keywords):
            return col
    return None


def _guess_measure_columns(df: pd.DataFrame) -> dict:
    """
    Guess which numeric columns form a (total, part_a, part_b) triplet such
    as Revenue/Cost/Profit, Income/Expense/Net, Budget/Spend/Remaining, etc.
    Tries broad cross-domain keywords first; if that doesn't confidently
    resolve, falls back to detecting the relationship purely from the
    numbers themselves (total ≈ a ± b across most rows), so the pipeline
    still works even with unfamiliar column names.
    """
    total_col = _find_col(df, ("revenue", "sales", "income", "amount", "value", "budget"), numeric_only=True)
    minus_col = _find_col(df, ("cogs", "cost", "expense", "spend"), numeric_only=True)
    plus_col = _find_col(df, ("profit", "margin_value", "grossprofit", "net", "surplus", "earnings"), numeric_only=True)
    category_col = _find_col(df, ("categor", "segment", "type", "group", "department"))

    if total_col and (minus_col or plus_col):
        return {"total": total_col, "minus": minus_col, "plus": plus_col, "category": category_col}

    # Fallback: search all numeric-triplets for an arithmetic relationship.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for total_c in numeric_cols:
        for a in numeric_cols:
            if a == total_c:
                continue
            for b in numeric_cols:
                if b in (total_c, a):
                    continue
                tol = max(df[total_c].abs().mean() * 0.02, 1e-6)
                if ((df[total_c] - (a and df[a] - df[b])).abs() < tol).mean() > 0.9:
                    return {"total": total_c, "minus": b, "plus": a, "category": category_col}
    return {"total": total_col, "minus": minus_col, "plus": plus_col, "category": category_col}


def _guess_dimension_columns(df: pd.DataFrame) -> dict:
    """
    Guess which columns identify a unique record: when it happened, who/what
    it's about, what it concerns, a quantity/count, and any ID/reference
    column tying line items together. Broad synonyms make this domain
    agnostic (works for sales orders, HR records, support tickets, etc).
    """
    date_col = _find_col(df, ("date", "day", "period", "timestamp"))
    entity_col = _find_col(df, ("customer", "client", "account", "employee", "vendor", "user", "member", "company"))
    item_col = _find_col(df, ("item", "product", "sku", "service", "material", "task"))
    quantity_col = _find_col(df, ("quantity", "qty", "units", "count", "volume"), numeric_only=True)

    id_col = None
    id_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ("doc", "invoice", "order", "reference", "transaction", "ticket", "receipt", "record", "id", "no", "number"))
    ]
    for c in id_candidates:
        vals = df[c].astype(str)
        # A genuine ID/document column: alphanumeric-code-like, and NOT
        # unique on every single row necessarily (line items can share one),
        # but also not identical everywhere.
        looks_code_like = vals.str.match(r'^[A-Za-z]*\d+[A-Za-z]*$').mean() > 0.8
        reasonable_cardinality = 0 < df[c].nunique() < len(df)
        if looks_code_like or reasonable_cardinality:
            id_col = c
            break

    return {
        "date": date_col,
        "entity": entity_col,
        "item": item_col,
        "quantity": quantity_col,
        "doc_id": id_col,
    }


def _next_ids(existing_ids: pd.Series, count: int) -> list:
    """
    Generate `count` brand-new IDs that follow the same alphanumeric pattern
    as existing_ids (prefix + zero-padded number + suffix), continuing after
    the highest number already in use. Falls back to a generic scheme if no
    numeric pattern is detected, so this works for any ID format.
    """
    pattern = re.compile(r'^(\D*)(\d+)(\D*)$')
    parsed = []
    for val in existing_ids.astype(str):
        m = pattern.match(val)
        if m:
            parsed.append((m.group(1), int(m.group(2)), len(m.group(2)), m.group(3)))

    if parsed:
        prefix, _, width, suffix = max(parsed, key=lambda t: t[1])
        start = max(p[1] for p in parsed) + 1
        return [f"{prefix}{str(start + i).zfill(width)}{suffix}" for i in range(count)]

    base = str(existing_ids.iloc[0]) if len(existing_ids) else "REC"
    return [f"{base}-NEW{i + 1}" for i in range(count)]


# ---------------------------------------------------------
# Realism enforcement — domain-agnostic fixes for patterns an LLM (and
# naive row-count padding) tend to fall into:
#   - identical repeated (date, entity, item, quantity) "records"
#   - an ID/document column spanning multiple dates or entities, which is
#     never valid for ANY kind of document (invoice, ticket, record...)
#   - a flat, unrealistic rate/margin applied to every category alike
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

    measures = _guess_measure_columns(raw_df)
    dims = _guess_dimension_columns(raw_df)
    total_col, minus_col, plus_col, category_col = (
        measures["total"], measures["minus"], measures["plus"], measures["category"]
    )
    date_col, entity_col, item_col, quantity_col, doc_id_col = (
        dims["date"], dims["entity"], dims["item"], dims["quantity"], dims["doc_id"]
    )

    # --- Anchors captured from the RAW model output, BEFORE any row-count
    #     padding/trimming. Duplicating rows later must never inflate these. ---
    original_total_sum = raw_df[total_col].sum() if total_col else None
    original_minus_sum = raw_df[minus_col].sum() if (total_col and minus_col) else None
    original_quantity_sum = raw_df[quantity_col].sum() if quantity_col else None

    if total_documents and total_documents > num_rows:
        fraction = num_rows / total_documents
        if original_total_sum is not None:
            original_total_sum *= fraction
        if original_minus_sum is not None:
            original_minus_sum *= fraction
        if original_quantity_sum is not None:
            original_quantity_sum *= fraction

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

    # --- 1b. Break up rows that are literal duplicates of another row on the
    #         (date, entity, item) business key — whether the model produced
    #         them or padding introduced them. Every occurrence after the
    #         first gets a jittered quantity and a nudged date. ---
    key_cols = [c for c in (date_col, entity_col, item_col) if c]
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
                pass

    # --- 1c. Enforce document/ID integrity, generically: whatever the ID
    #         column is called (Doc_No, Invoice_ID, Ticket #, ...), every row
    #         sharing an ID must share the same date/entity/item. If step 1b
    #         (or the model itself) left an ID spanning multiple dates or
    #         entities, give every row after the first occurrence in that ID
    #         group a fresh ID that follows the existing naming pattern. ---
    if doc_id_col and key_cols:
        rows_needing_new_id = []
        for _doc_id, group in df.groupby(doc_id_col, sort=False):
            if len(group) > 1 and group[key_cols].nunique(dropna=False).gt(1).any():
                rows_needing_new_id.extend(group.index[1:].tolist())
        if rows_needing_new_id:
            new_ids = _next_ids(df[doc_id_col], len(rows_needing_new_id))
            df.loc[rows_needing_new_id, doc_id_col] = new_ids

    # If we can't confidently identify a total + at least one related
    # column, skip financial realism adjustments — the structural fixes
    # above still apply regardless.
    if total_col and (minus_col or plus_col):
        # --- 2. Break exact duplicate totals with small jitter, then
        #        rescale to the pre-padding (and, if applicable,
        #        sample-scaled) anchor — never to an inflated padded total. ---
        noise = rng.normal(loc=1.0, scale=0.06, size=n)
        noise = np.clip(noise, 0.85, 1.15)
        adjusted_total = df[total_col] * noise
        target_total_sum = original_total_sum if original_total_sum else adjusted_total.sum()
        rescale = target_total_sum / adjusted_total.sum() if adjusted_total.sum() else 1
        df[total_col] = (adjusted_total * rescale).round(2)

        # --- 3. Per-category rate/margin, varied rather than flat ---
        if category_col:
            categories = df[category_col].astype(str).unique()
            category_rates = {cat: rng.uniform(0.15, 0.45) for cat in categories}
            base_rate = df[category_col].astype(str).map(category_rates)
        else:
            base_rate = pd.Series(rng.uniform(0.15, 0.45), index=df.index)
        row_rate = (base_rate + rng.normal(0, 0.02, n)).clip(0.05, 0.65)

        if minus_col:
            adjusted_minus = df[total_col] * (1 - row_rate)
            target_minus_sum = original_minus_sum if original_minus_sum else adjusted_minus.sum()
            minus_rescale = target_minus_sum / adjusted_minus.sum() if adjusted_minus.sum() else 1
            df[minus_col] = (adjusted_minus * minus_rescale).round(2)

        # --- 4. Recompute the derived column so the arithmetic always holds ---
        if plus_col and minus_col:
            df[plus_col] = (df[total_col] - df[minus_col]).round(2)
        elif plus_col:
            df[plus_col] = (df[total_col] * row_rate).round(2)

    # --- 5. Keep quantity anchored the same way, so it doesn't silently
    #        balloon from padding either. ---
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
    provided dashboard screenshot(s). Works for any dashboard domain — the
    schema (columns, entities, measures) is inferred purely from the images
    and from the generated data itself, not assumed in advance.

    total_documents: if you know (or can read off a KPI card, e.g.
    "Documents: 61,079" / "Tickets: 4,200" / any count) the real number of
    underlying records the dashboard represents, pass it here. When it's
    larger than num_rows, generated rows are treated as a proportional
    sample instead of being forced to carry the full period's totals.

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
