"""Calibration: make the generated rows reproduce the dashboard numbers.

Pipeline
--------
1. ``derive_column_targets``  — translate dashboard targets (KPIs, chart values, ratios,
   differences...) into plain *column sum / average targets* per filter context. Ratio
   and difference measures are solved algebraically (e.g. Margin=23% & Sales=1.2M
   => Cost total = 924K).
2. ``calibrate``              — iterative proportional fitting (raking): rows are scaled
   multiplicatively, group by group, until every group total hits its target.
   Multiplicative scaling keeps values positive and preserves the shape of the
   generated distribution. Finally rounding residuals are redistributed so totals are
   exact at the column's precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .measures import Evaluator, deps_of, filters_to_dict
from .schemas import DashboardSpec


@dataclass(frozen=True)
class ColumnTarget:
    column: str
    agg: str                 # SUM | AVERAGE
    filters: tuple           # ((column, value), ...)
    value: float


# ------------------------------------------------------------------ 1. target resolution
def _solve(spec: DashboardSpec, known: dict) -> None:
    """Fixed-point algebra over composite measures. Mutates ``known`` (name -> value)."""
    changed = True
    while changed:
        changed = False
        for m in spec.measures:
            if m.kind == "agg":
                continue
            t = known.get(m.name)
            if m.kind == "ratio":
                n, d = known.get(m.num), known.get(m.den)
                if t is not None and d is not None and n is None:
                    known[m.num] = t * d; changed = True
                elif t is not None and n is not None and d is None and t != 0:
                    known[m.den] = n / t; changed = True
                elif t is None and n is not None and d not in (None, 0):
                    known[m.name] = n / d; changed = True
            elif m.kind == "diff":
                a, b = known.get(m.left), known.get(m.right)
                if t is not None and a is not None and b is None:
                    known[m.right] = a - t; changed = True
                elif t is not None and b is not None and a is None:
                    known[m.left] = t + b; changed = True
                elif t is None and a is not None and b is not None:
                    known[m.name] = a - b; changed = True
            elif m.kind == "sum":
                vals = [known.get(i) for i in m.items]
                missing = [i for i, v in zip(m.items, vals) if v is None]
                if t is not None and len(missing) == 1:
                    known[missing[0]] = t - sum(v for v in vals if v is not None); changed = True
                elif t is None and not missing:
                    known[m.name] = float(sum(vals)); changed = True


_STRUCTURAL = ("COUNTROWS", "COUNT", "DISTINCTCOUNT")


def derive_column_targets(spec: DashboardSpec, ev: Evaluator | None = None, done=()) -> tuple[list, list]:
    """Translate dashboard targets into column-level SUM/AVERAGE targets.

    When ``ev`` is given, measures that depend only on *already calibrated* columns (``done``) or on
    structural counts are seeded from the current data, so ratios such as Bonus % = Bonus / Salary
    can be solved even if Total Salary was never printed on the dashboard.
    """
    mm = spec.measure_map()
    needed = {d for m in spec.measures if m.kind != "agg" for d in deps_of(m)}
    contexts: dict = {}
    for t in spec.targets:
        key = tuple(sorted((f.column, f.value) for f in t.filters))
        contexts.setdefault(key, {})[t.measure] = t.value

    out, warns, seen = [], [], set()
    for ctx, known in contexts.items():
        known = dict(known)
        if ev is not None:
            for name in needed:
                m = mm[name]
                if name in known or m.kind != "agg":
                    continue
                if m.agg in _STRUCTURAL or m.column in done:
                    v = ev.value(name, dict(ctx))
                    if v == v and math.isfinite(v):
                        known[name] = v
        _solve(spec, known)
        for name, value in known.items():
            m = mm[name]
            if m.kind != "agg" or m.agg not in ("SUM", "AVERAGE"):
                continue
            eff = {**dict(ctx), **filters_to_dict(m.filters)}
            key = (m.column, m.agg, tuple(sorted(eff.items())))
            if key in seen:
                continue
            seen.add(key)
            if value < 0:
                warns.append(f"Skipped negative {m.agg} target for '{m.column}' in {dict(eff) or 'all data'}.")
                continue
            out.append(ColumnTarget(m.column, m.agg, key[2], float(value)))
    return out, warns


# ------------------------------------------------------------------ 2. IPF
def _topo(spec: DashboardSpec) -> list:
    fmap, order, done = spec.fact_map(), [], set()

    def visit(name):
        if name in done:
            return
        done.add(name)
        p = fmap[name].derived_from
        if p:
            visit(p)
        order.append(name)

    for c in spec.fact_columns:
        visit(c.name)
    return order


def _bounds(col):
    lo = col.min if col.min is not None else (0.0 if col.kind != "number" else -math.inf)
    if col.kind == "rate":
        lo = col.min if col.min is not None else 0.0
    hi = col.max if col.max is not None else (1.0 if col.kind == "rate" else math.inf)
    return lo, hi


def _build_constraints(x, targets, ev: Evaluator, warns: list, col: str):
    """-> list of (cols_tuple, mask, sum_target)."""
    cons = []
    for t in targets:
        mask = ev.mask(dict(t.filters))
        cnt = int((mask & ~np.isnan(x)).sum())
        if cnt == 0:
            warns.append(f"No rows available for {col} target {dict(t.filters) or 'all data'} "
                         f"(raise the row count to cover it).")
            continue
        total = t.value * cnt if t.agg == "AVERAGE" else t.value
        cons.append((tuple(sorted(c for c, _ in t.filters)), mask, float(total)))
    return cons


def _ipf(x, cons, lo, hi, max_iter=400, tol=1e-10):
    x = x.copy()
    groups = [c for c in cons if c[0]]
    totals = [c for c in cons if not c[0]]
    covered = np.zeros(len(x), dtype=bool)
    for _, mask, _ in groups:
        covered |= mask
    free = ~covered & ~np.isnan(x)

    for _ in range(max_iter):
        for _, mask, target in groups:
            cur = np.nansum(x[mask])
            if cur > 0:
                x[mask] *= target / cur
        np.clip(x, lo, hi, out=x)
        for _, _, target in totals:
            nz = ~np.isnan(x)
            if free.any() and np.nansum(x[free]) > 0:
                resid = target - np.nansum(x[nz & covered])
                if resid > 0:
                    x[free] *= resid / np.nansum(x[free])
                    continue
            cur = np.nansum(x)
            if cur > 0:
                x[nz] *= target / cur
        np.clip(x, lo, hi, out=x)
        worst = 0.0
        for _, mask, target in cons:
            if target:
                worst = max(worst, abs(np.nansum(x[mask]) - target) / abs(target))
        if worst < tol:
            break
    return x, free


def _distribute(x, idx, diff, step, decimals, lo, hi):
    n = int(round(diff / step))
    if n == 0 or len(idx) == 0:
        return
    sign, n = (1 if n > 0 else -1), abs(n)
    order = idx[np.argsort(-np.nan_to_num(x[idx]))]
    base, rem = divmod(n, len(order))
    x[order] += sign * step * base
    x[order[:rem]] += sign * step
    x[order] = np.round(np.clip(x[order], lo, hi), decimals)


def _round_and_fix(x, cons, free, decimals, lo, hi):
    step = 10.0 ** -decimals
    x = np.round(x, decimals)
    groups = sorted((c for c in cons if c[0]), key=lambda c: int(c[1].sum()))
    for _, mask, target in groups:
        idx = np.flatnonzero(mask & ~np.isnan(x))
        diff = round(target, decimals) - np.nansum(x[idx])
        if abs(diff) <= max(0.02 * abs(target), step * len(idx)):
            _distribute(x, idx, diff, step, decimals, lo, hi)
    for _, mask, target in (c for c in cons if not c[0]):
        idx = np.flatnonzero(free) if (free.any() and groups) else (np.arange(len(x)) if not groups else np.array([], int))
        idx = idx[~np.isnan(x[idx])] if len(idx) else idx
        diff = round(target, decimals) - np.nansum(x)
        if len(idx) and abs(diff) <= max(0.02 * abs(target), step * len(idx)):
            _distribute(x, idx, diff, step, decimals, lo, hi)
    return x


def calibrate(flat, spec: DashboardSpec, column_targets: list, ev: Evaluator) -> list:
    """Rescale numeric fact columns in-place so aggregates match the targets.

    Columns are processed parents-first. Before each column its targets are (re)derived from the
    dashboard numbers *plus* the values already fixed in earlier columns.
    """
    import pandas as pd

    warns: list = []
    fmap = spec.fact_map()
    x = {c: pd.to_numeric(flat[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan).copy()
         for c in fmap}
    meta, done = {}, set()
    for name in _topo(spec):
        col = fmap[name]
        lo, hi = _bounds(col)
        flat[name] = x[name]                       # let the evaluator see current values
        targets = [t for t in derive_column_targets(spec, ev, done)[0] if t.column == name]
        before = x[name].copy()
        cons = _build_constraints(x[name], targets, ev, warns, name)
        free = ~np.isnan(x[name])
        if cons:
            x[name], free = _ipf(x[name], cons, lo, hi)
        meta[name] = (cons, free, lo, hi)
        flat[name] = x[name]
        done.add(name)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(before > 0, x[name] / before, 1.0)
        for child in spec.fact_columns:           # keep derived columns consistent
            if child.derived_from == name:
                x[child.name] = x[child.name] * ratio
                flat[child.name] = x[child.name]

    for name in _topo(spec):
        col = fmap[name]
        cons, free, lo, hi = meta[name]
        if cons:
            x[name] = _round_and_fix(x[name], cons, free, col.decimals, lo, hi)
        else:
            x[name] = np.round(x[name], col.decimals)

    for name, arr in x.items():
        col = fmap[name]
        if col.decimals == 0:
            flat[name] = pd.array(arr, dtype="Int64")
        else:
            flat[name] = arr
    return list(dict.fromkeys(warns))
