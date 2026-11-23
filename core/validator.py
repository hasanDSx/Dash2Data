"""Compare what the dashboard shows (targets) with what the generated data produces."""
from __future__ import annotations

import math

import pandas as pd

from .config import TOL_CLOSE, TOL_MATCH
from .measures import Evaluator, filter_label, filters_to_dict
from .schemas import DashboardSpec

COLUMNS = ["Source", "Label", "Measure", "Filter", "Target", "Actual", "Abs Error", "Error %", "Status", "Note"]


def validate(ev: Evaluator, spec: DashboardSpec) -> tuple[pd.DataFrame, dict]:
    mm = spec.measure_map()
    rows = []
    for t in spec.targets:
        actual = ev.value(t.measure, filters_to_dict(t.filters))
        if isinstance(actual, float) and math.isnan(actual):
            err_abs, err_pct, status = math.nan, math.nan, "⚠️ not computable"
        else:
            err_abs = abs(actual - t.value)
            err_pct = (err_abs / abs(t.value) if t.value else err_abs) * 100.0
            status = ("✅ match" if err_pct / 100 <= TOL_MATCH
                      else "🟡 close" if err_pct / 100 <= TOL_CLOSE else "❌ off")
        m = mm[t.measure]
        note = []
        if t.estimated:
            note.append("value estimated from chart")
        if m.kind == "agg" and m.agg in ("MIN", "MAX"):
            note.append("MIN/MAX are not calibrated")
        rows.append({
            "Source": t.source, "Label": t.label or t.measure, "Measure": t.measure,
            "Filter": filter_label(t.filters), "Target": t.value, "Actual": actual,
            "Abs Error": err_abs, "Error %": err_pct, "Status": status, "Note": "; ".join(note),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    errs = df["Error %"].dropna()
    n = len(df)
    summary = {
        "targets": n,
        "match": int((df["Status"] == "✅ match").sum()),
        "close": int((df["Status"] == "🟡 close").sum()),
        "off": int((df["Status"] == "❌ off").sum()),
        "not_computable": int((df["Status"] == "⚠️ not computable").sum()),
        "mean_error_pct": float(errs.mean()) if len(errs) else math.nan,
        "median_error_pct": float(errs.median()) if len(errs) else math.nan,
        "max_error_pct": float(errs.max()) if len(errs) else math.nan,
        "match_rate": (float((df["Status"] == "✅ match").sum()) / n) if n else 0.0,
    }
    return df, summary
