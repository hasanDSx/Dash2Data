"""Evaluate structural measures against a (flat) DataFrame.

This mirrors what DAX will compute in Power BI, so we can verify that the generated
data reproduces the dashboard numbers *before* the user ever opens Power BI.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .schemas import MONTHS, DashboardSpec, MeasureSpec

_MONTH_KEYS = [m.lower() for m in MONTHS]


def canon_value(col: str, val) -> str:
    """Canonical (lower-case) text used to compare a filter value with a column."""
    v = str(val).strip().lower()
    try:
        if col == "Month":
            if v.isdigit() and 1 <= int(v) <= 12:
                return _MONTH_KEYS[int(v) - 1]
            if v.isalpha() and v[:3] in _MONTH_KEYS:
                return v[:3]
        elif col == "MonthNumber":
            if v.isalpha() and v[:3] in _MONTH_KEYS:
                return str(_MONTH_KEYS.index(v[:3]) + 1)
            return str(int(float(v)))
        elif col == "Quarter":
            m = re.search(r"(\d)", v)
            if m:
                return f"q{m.group(1)}"
        elif col == "Year":
            return str(int(float(v)))
        elif col == "YearMonth":
            return pd.Timestamp(v).strftime("%Y-%m")
    except (ValueError, TypeError):
        pass
    return v


def filters_to_dict(filters) -> dict:
    return {f.column: f.value for f in filters}


def filter_label(filters) -> str:
    return ", ".join(f"{f.column}={f.value}" for f in filters) if filters else "(all data)"


def deps_of(m: MeasureSpec) -> list:
    return {"ratio": [m.num, m.den], "diff": [m.left, m.right], "sum": m.items}.get(m.kind, [])


def is_additive(m: MeasureSpec, mm: dict) -> bool:
    """True if the measure adds up across groups (so group shares are meaningful)."""
    if m.kind == "agg":
        return m.agg in ("SUM", "COUNT", "COUNTROWS")
    if m.kind in ("sum", "diff"):
        return all(is_additive(mm[d], mm) for d in deps_of(m) if d in mm)
    return False


class Evaluator:
    """Evaluates measures in a filter context over a flat table."""

    def __init__(self, flat: pd.DataFrame, measures: dict):
        self.flat = flat
        self.measures = measures
        self._norm: dict = {}
        self._masks: dict = {}

    def _norm_col(self, col: str) -> np.ndarray:
        if col not in self._norm:
            self._norm[col] = self.flat[col].astype(str).str.strip().str.lower().to_numpy()
        return self._norm[col]

    def mask(self, filters: dict) -> np.ndarray:
        key = tuple(sorted((c, canon_value(c, v)) for c, v in filters.items()))
        if key not in self._masks:
            m = np.ones(len(self.flat), dtype=bool)
            for col, val in key:
                if col not in self.flat.columns:
                    return np.zeros(len(self.flat), dtype=bool)
                m &= self._norm_col(col) == val
            self._masks[key] = m
        return self._masks[key]

    def numeric(self, col: str) -> np.ndarray:
        return pd.to_numeric(self.flat[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    def value(self, name: str, filters: dict | None = None) -> float:
        m = self.measures[name]
        ctx = dict(filters or {})
        if m.kind == "agg":
            ctx.update(filters_to_dict(m.filters))       # CALCULATE overrides context
            mask = self.mask(ctx)
            if m.agg == "COUNTROWS":
                return float(mask.sum())
            if m.agg in ("COUNT", "DISTINCTCOUNT"):
                s = self.flat[m.column][mask].dropna()
                return float(s.nunique() if m.agg == "DISTINCTCOUNT" else len(s))
            vals = self.numeric(m.column)[mask]
            vals = vals[~np.isnan(vals)]
            if m.agg == "SUM":
                return float(vals.sum())
            if len(vals) == 0:
                return math.nan
            return float({"AVERAGE": np.mean, "MIN": np.min, "MAX": np.max}[m.agg](vals))
        if m.kind == "ratio":
            n, d = self.value(m.num, ctx), self.value(m.den, ctx)
            return math.nan if (math.isnan(n) or math.isnan(d) or d == 0) else n / d
        if m.kind == "diff":
            return self.value(m.left, ctx) - self.value(m.right, ctx)
        return float(sum(self.value(i, ctx) for i in m.items))


def make_evaluator(flat: pd.DataFrame, spec: DashboardSpec) -> Evaluator:
    return Evaluator(flat, spec.measure_map())
