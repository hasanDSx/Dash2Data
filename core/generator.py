"""Raw row generation (pre-calibration).

Only *raw* columns are generated: dates, dimensions, ids and numeric fact columns.
Derived metrics (profit, margin, AOV, YoY...) are NOT columns; they are DAX measures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MAX_ROWS, MIN_ROWS
from .measures import is_additive
from .schemas import MONTHS, DashboardSpec


@dataclass
class Plan:
    n_rows: int
    id_unique: dict = field(default_factory=dict)     # id column -> distinct count
    dim_card: dict = field(default_factory=dict)      # dimension -> cardinality
    notes: list = field(default_factory=list)


def plan_generation(spec: DashboardSpec, num_rows: int, auto_rows: bool) -> Plan:
    mm = spec.measure_map()
    count_rows, distinct = [], {}
    for t in spec.targets:
        if t.filters:
            continue
        m = mm[t.measure]
        if m.kind != "agg" or m.filters:
            continue
        if m.agg == "COUNTROWS":
            count_rows.append(int(round(t.value)))
        elif m.agg == "DISTINCTCOUNT" and m.column:
            distinct[m.column] = max(distinct.get(m.column, 0), int(round(t.value)))

    n = max(MIN_ROWS, int(num_rows))
    required = max(count_rows + list(distinct.values()) + [0])
    notes = []
    if auto_rows and required:
        n = max(count_rows) if count_rows else max(n, required)
        if n != num_rows:
            notes.append(f"Row count set to {min(n, MAX_ROWS):,} to match count KPIs.")
    elif required > n:
        notes.append(f"Count KPIs need >= {required:,} rows; only {n:,} requested, so they cannot match.")
    if n > MAX_ROWS:
        notes.append(f"Row count capped at {MAX_ROWS:,} (needed {n:,}).")
        n = MAX_ROWS

    plan = Plan(n_rows=n, notes=notes)
    dims = spec.dim_map()
    for col, target in distinct.items():
        if col in dims:
            plan.dim_card[col] = max(1, min(target, n))
        else:
            plan.id_unique[col] = max(1, min(target, n))
    return plan


# ------------------------------------------------------------------ dimension sampling
def _members(dim, card: int | None) -> list:
    members = list(dim.values)
    total = card or dim.cardinality or (len(members) or 6)
    total = max(total, len(members))
    members += [f"{dim.name} {i:04d}" for i in range(1, total - len(members) + 1)]
    return members


def _parent_lookup(spec: DashboardSpec, members: dict) -> dict:
    """dim name -> {member -> parent member} (explicit map first, then round-robin)."""
    out = {}
    for d in spec.dimensions:
        if not d.parent:
            continue
        pm_members = members[d.parent]
        mapping, rr = {}, 0
        for m in members[d.name]:
            if m in d.parent_map and d.parent_map[m] in pm_members:
                mapping[m] = d.parent_map[m]
            else:
                mapping[m] = pm_members[rr % len(pm_members)]
                rr += 1
        out[d.name] = mapping
    return out


def _target_shares(spec: DashboardSpec, dim_name: str) -> dict:
    """Shares per member from additive single-filter targets on this dimension."""
    mm = spec.measure_map()
    per_measure: dict = {}
    for t in spec.targets:
        if len(t.filters) == 1 and t.filters[0].column == dim_name and t.value > 0 \
                and is_additive(mm[t.measure], mm):
            per_measure.setdefault(t.measure, {})[t.filters[0].value] = t.value
    if not per_measure:
        return {}
    best = max(per_measure.values(), key=len)
    tot = sum(best.values())
    return {k: v / tot for k, v in best.items()}


def _leaf_weights(spec, dim, members, parent_lookup) -> np.ndarray:
    lower = {m.lower(): m for m in members[dim.name]}
    own = {lower[k.lower()]: v for k, v in _target_shares(spec, dim.name).items() if k.lower() in lower}
    mem = members[dim.name]
    w = np.ones(len(mem))
    if own:
        floor = 0.3 * float(np.mean(list(own.values())))
        w = np.array([own.get(m, floor) for m in mem], dtype=float)
    else:   # inherit from the nearest ancestor that has targets
        dmap = spec.dim_map()
        cur, chain = dim, {m: m for m in mem}
        while cur.parent:
            mp = parent_lookup[cur.name]
            chain = {leaf: mp[chain[leaf]] for leaf in mem}
            cur = dmap[cur.parent]
            shares = _target_shares(spec, cur.name)
            if shares:
                lowers = {m.lower(): m for m in members[cur.name]}
                sh = {lowers[k.lower()]: v for k, v in shares.items() if k.lower() in lowers}
                kids: dict = {}
                for leaf in mem:
                    kids[chain[leaf]] = kids.get(chain[leaf], 0) + 1
                floor = 0.3 * float(np.mean(list(sh.values())))
                w = np.array([sh.get(chain[m], floor) / kids[chain[m]] for m in mem], dtype=float)
                break
    return w / w.sum()



def _exact_codes(spec: DashboardSpec, dim, mem: list, w: np.ndarray, n: int, rng) -> np.ndarray | None:
    """Exact per-member row counts when the dashboard shows row counts (e.g. headcount) by this dimension."""
    mm = spec.measure_map()
    lower = {m.lower(): i for i, m in enumerate(mem)}
    counts = np.zeros(len(mem), dtype=int)
    listed = np.zeros(len(mem), dtype=bool)
    for t in spec.targets:
        if len(t.filters) == 1 and t.filters[0].column == dim.name:
            m = mm[t.measure]
            if m.kind == "agg" and m.agg == "COUNTROWS" and not m.filters \
                    and t.filters[0].value.lower() in lower and t.value >= 0:
                i = lower[t.filters[0].value.lower()]
                counts[i], listed[i] = int(round(t.value)), True
    if not listed.any():
        return None
    if counts.sum() > n:                                   # targets exceed available rows: scale down
        counts = np.floor(counts * n / counts.sum()).astype(int)
    rest = n - int(counts.sum())
    if rest > 0:
        free = ~listed
        p = w * free if free.any() and (w * free).sum() > 0 else counts.astype(float) + 1e-9
        counts += rng.multinomial(rest, p / p.sum())
    codes = np.repeat(np.arange(len(mem)), counts)
    rng.shuffle(codes)
    return codes


def _short_prefix(name: str) -> str:
    base = re.sub(r"[_\s]*id$", "", name, flags=re.I)
    return (re.sub(r"[^A-Za-z]", "", base)[:3].upper() or "ID")


def add_date_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = pd.to_datetime(df[col])
    df["Year"] = d.dt.year.astype("int64")
    df["Quarter"] = "Q" + d.dt.quarter.astype(str)
    df["MonthNumber"] = d.dt.month.astype("int64")
    df["Month"] = d.dt.month.map(lambda m: MONTHS[m - 1])
    df["YearMonth"] = d.dt.strftime("%Y-%m")
    return df


# ------------------------------------------------------------------ main entry
def generate_flat(spec: DashboardSpec, plan: Plan, column_targets: list, seed: int,
                  include_nulls: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = plan.n_rows
    cols: dict = {}

    # ---- date -----------------------------------------------------------
    if spec.date:
        start, end = pd.Timestamp(spec.date.start), pd.Timestamp(spec.date.end)
        days = (end - start).days + 1
        dates = (start + pd.to_timedelta(rng.integers(0, days, n), unit="D")).to_numpy().astype("datetime64[ns]").copy()
        periods = pd.period_range(start, end, freq="M")
        perm = rng.permutation(n)
        for row, per in zip(perm, periods):             # guarantee every month has rows
            lo, hi = max(per.start_time, start), min(per.end_time.normalize(), end)
            pick = lo + pd.Timedelta(days=int(rng.integers(0, (hi - lo).days + 1)))
            dates[row] = np.datetime64(pick.to_datetime64(), "ns")
        cols[spec.date.name] = pd.DatetimeIndex(dates).normalize()

    # ---- dimensions -----------------------------------------------------
    members = {d.name: _members(d, plan.dim_card.get(d.name)) for d in spec.dimensions}
    plookup = _parent_lookup(spec, members)
    parent_names = {d.parent for d in spec.dimensions if d.parent}
    dmap = spec.dim_map()
    for d in spec.dimensions:
        if d.name in parent_names:
            continue                                      # derived from its leaf below
        w = _leaf_weights(spec, d, members, plookup)
        mem = members[d.name]
        codes = _exact_codes(spec, d, mem, w, n, rng)
        if codes is None:
            codes = rng.choice(len(mem), size=n, p=w)
            must_cover = len(mem) if d.name in plan.dim_card else len(d.values)   # distinct-count KPIs need every member
            listed = min(must_cover, n)
            rows = rng.permutation(n)[:listed]            # guarantee every listed member appears
            codes[rows] = np.arange(listed)
        leaf = np.array(mem, dtype=object)[codes]
        cols[d.name] = leaf
        cur, col = d, leaf
        while cur.parent:
            mp = plookup[cur.name]
            col = np.array([mp[v] for v in col], dtype=object)
            cur = dmap[cur.parent]
            cols[cur.name] = col

    # ---- ids ------------------------------------------------------------
    for i in spec.id_columns:
        uniq = max(1, min(plan.id_unique.get(i.name, n), n))
        idx = np.concatenate([np.arange(uniq), rng.integers(0, uniq, n - uniq)])
        rng.shuffle(idx)
        prefix = i.prefix or _short_prefix(i.name)
        cols[i.name] = np.array([f"{prefix}-{k + 1:06d}" for k in idx], dtype=object)

    # ---- numeric fact columns (topologically: parents before derived) ----------
    totals = {t.column: t for t in column_targets if not t.filters}
    fmap, done = spec.fact_map(), {}

    def make(name: str) -> np.ndarray:
        if name in done:
            return done[name]
        c = fmap[name]
        lo = c.min if c.min is not None else (0.0 if c.kind != "number" else 0.0)
        hi = c.max if c.max is not None else (1.0 if c.kind == "rate" else np.inf)
        if c.derived_from:
            parent = make(c.derived_from)
            ratio = c.derived_ratio if c.derived_ratio else 0.6
            x = parent * ratio * rng.lognormal(0, 0.08, n)
        else:
            t = totals.get(name)
            typical = (t.value / n if t.agg == "SUM" else t.value) if t else c.typical
            if c.kind == "rate":
                lo = c.min if c.min is not None else 0.0
                span = hi - lo
                typical = typical if typical is not None else lo + span / 2
                frac = float(np.clip((typical - lo) / span, 0.03, 0.97))
                x = lo + span * rng.beta(frac * 20, (1 - frac) * 20, n)
            elif c.kind == "quantity":
                typical = max(typical if typical is not None else 5.0, 1.0)
                x = 1 + rng.poisson(max(typical - 1, 0.2), n).astype(float)
            else:
                typical = typical if typical is not None else 100.0
                sigma = 0.6
                x = rng.lognormal(np.log(max(typical, 1e-9)) - sigma ** 2 / 2, sigma, n)
        done[name] = np.clip(x, lo, hi)
        return done[name]

    for c in spec.fact_columns:
        cols[c.name] = make(c.name)

    df = pd.DataFrame(cols)

    if include_nulls:                                      # messy-data realism (fact columns only)
        for c in spec.fact_columns:
            frac = rng.uniform(0.03, 0.05)
            df.loc[rng.random(n) < frac, c.name] = np.nan

    if spec.date:
        df = add_date_parts(df, spec.date.name)
    return df
