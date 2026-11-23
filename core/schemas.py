"""Dashboard specification.

The LLM does NOT generate data. It extracts a *specification* describing:
  * the raw columns needed (dimensions, ids, numeric fact columns)
  * the measures shown on the dashboard (defined structurally, later rendered to DAX)
  * the numeric targets visible on the dashboard (KPI cards, chart values)

Everything numeric is produced afterwards by deterministic code (see generator/calibrator).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from .config import MAX_ROWS
from .numbers import parse_number

DATE_PARTS = ("Year", "Quarter", "MonthNumber", "Month", "YearMonth")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMERIC_AGGS = ("SUM", "AVERAGE", "MIN", "MAX")
ALL_AGGS = NUMERIC_AGGS + ("COUNT", "DISTINCTCOUNT", "COUNTROWS")
FACT_KINDS = ("money", "quantity", "rate", "number")
MEASURE_KINDS = ("agg", "ratio", "diff", "sum")
MEASURE_FORMATS = ("currency", "number", "percent", "decimal")


class SpecError(ValueError):
    """Raised when the spec is unusable (not merely imperfect)."""


# --------------------------------------------------------------------------- dataclasses
@dataclass
class Filter:
    column: str
    value: str


@dataclass
class DimensionSpec:
    name: str
    values: list = field(default_factory=list)
    parent: Optional[str] = None
    parent_map: dict = field(default_factory=dict)   # child value -> parent value
    cardinality: Optional[int] = None                # total members (>= len(values))


@dataclass
class DateSpec:
    name: str = "Date"
    start: str = "2023-01-01"
    end: str = "2023-12-31"


@dataclass
class FactColumnSpec:
    name: str
    kind: str = "money"
    min: Optional[float] = None
    max: Optional[float] = None
    decimals: Optional[int] = None
    typical: Optional[float] = None
    derived_from: Optional[str] = None   # generated as parent * ratio (e.g. Cost = Sales * 0.7)
    derived_ratio: Optional[float] = None


@dataclass
class IdColumnSpec:
    name: str
    prefix: str = ""


@dataclass
class MeasureSpec:
    name: str
    kind: str = "agg"
    agg: Optional[str] = None
    column: Optional[str] = None
    filters: list = field(default_factory=list)      # list[Filter] (CALCULATE filters)
    num: Optional[str] = None
    den: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    items: list = field(default_factory=list)
    format: str = "number"
    description: str = ""


@dataclass
class TargetSpec:
    measure: str
    value: float
    filters: list = field(default_factory=list)      # list[Filter] (context, e.g. Region=East)
    source: str = "kpi"                              # kpi | chart
    label: str = ""
    estimated: bool = False                          # value eyeballed from a chart axis


@dataclass
class DashboardSpec:
    title: str = "Dashboard"
    fact_table: str = "Fact_Sales"
    currency: str = "$"
    date: Optional[DateSpec] = None
    dimensions: list = field(default_factory=list)
    fact_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    measures: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    # convenience lookups -------------------------------------------------
    def measure_map(self) -> dict:
        return {m.name: m for m in self.measures}

    def dim_map(self) -> dict:
        return {d.name: d for d in self.dimensions}

    def fact_map(self) -> dict:
        return {c.name: c for c in self.fact_columns}

    def filter_columns(self) -> set:
        cols = {d.name for d in self.dimensions}
        if self.date:
            cols |= set(DATE_PARTS)
        return cols


# --------------------------------------------------------------------------- parsing helpers
def safe_name(text: str, default: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(text or "").strip()).strip("_")
    return out or default


def _s(x) -> str:
    return str(x).strip() if x is not None else ""


def _opt_num(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return parse_number(x)
    except ValueError:
        return None


def _list(x) -> list:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _parse_filters(raw) -> list:
    out = []
    if isinstance(raw, dict):
        raw = [{"column": k, "value": v} for k, v in raw.items()]
    for f in _list(raw):
        if isinstance(f, dict) and _s(f.get("column")) and f.get("value") is not None:
            out.append(Filter(_s(f["column"]), _s(f["value"])))
    return out


def parse_spec(data: dict) -> tuple[DashboardSpec, list]:
    """Leniently convert raw JSON into a DashboardSpec. Bad items are dropped with warnings."""
    if not isinstance(data, dict):
        raise SpecError("Spec root must be a JSON object.")
    w: list = []
    spec = DashboardSpec(
        title=_s(data.get("title")) or "Dashboard",
        fact_table=safe_name(data.get("fact_table"), "Fact_Sales"),
        currency=_s(data.get("currency")) or "$",
        notes=_s(data.get("notes")),
    )

    d = data.get("date")
    if isinstance(d, dict) and d.get("start") and d.get("end"):
        spec.date = DateSpec(_s(d.get("name")) or "Date", _s(d["start"]), _s(d["end"]))

    for raw in _list(data.get("dimensions")):
        if not isinstance(raw, dict) or not _s(raw.get("name")):
            w.append("Dropped a dimension without a name.")
            continue
        card = _opt_num(raw.get("cardinality"))
        pm = raw.get("parent_map") if isinstance(raw.get("parent_map"), dict) else {}
        spec.dimensions.append(DimensionSpec(
            name=_s(raw["name"]),
            values=[_s(v) for v in _list(raw.get("values")) if _s(v)],
            parent=_s(raw.get("parent")) or None,
            parent_map={_s(k): _s(v) for k, v in pm.items()},
            cardinality=int(card) if card else None,
        ))

    for raw in _list(data.get("fact_columns")):
        if not isinstance(raw, dict) or not _s(raw.get("name")):
            w.append("Dropped a fact column without a name.")
            continue
        kind = _s(raw.get("kind")).lower()
        dec = _opt_num(raw.get("decimals"))
        spec.fact_columns.append(FactColumnSpec(
            name=_s(raw["name"]),
            kind=kind if kind in FACT_KINDS else "money",
            min=_opt_num(raw.get("min")), max=_opt_num(raw.get("max")),
            decimals=int(dec) if dec is not None else None,
            typical=_opt_num(raw.get("typical")),
            derived_from=_s(raw.get("derived_from")) or None,
            derived_ratio=_opt_num(raw.get("derived_ratio")),
        ))

    for raw in _list(data.get("id_columns")):
        if isinstance(raw, str):
            raw = {"name": raw}
        if isinstance(raw, dict) and _s(raw.get("name")):
            spec.id_columns.append(IdColumnSpec(_s(raw["name"]), _s(raw.get("prefix"))))

    for raw in _list(data.get("measures")):
        if not isinstance(raw, dict) or not _s(raw.get("name")):
            w.append("Dropped a measure without a name.")
            continue
        kind = _s(raw.get("kind")).lower() or "agg"
        agg = _s(raw.get("agg")).upper() or None
        fmt = _s(raw.get("format")).lower()
        spec.measures.append(MeasureSpec(
            name=_s(raw["name"]),
            kind=kind if kind in MEASURE_KINDS else "agg",
            agg=agg, column=_s(raw.get("column")) or None,
            filters=_parse_filters(raw.get("filters")),
            num=_s(raw.get("num")) or None, den=_s(raw.get("den")) or None,
            left=_s(raw.get("left")) or None, right=_s(raw.get("right")) or None,
            items=[_s(i) for i in _list(raw.get("items")) if _s(i)],
            format=fmt if fmt in MEASURE_FORMATS else "number",
            description=_s(raw.get("description")),
        ))

    for raw in _list(data.get("targets")):
        if not isinstance(raw, dict) or not _s(raw.get("measure")):
            w.append("Dropped a target without a measure.")
            continue
        try:
            value = parse_number(raw.get("value"))
        except ValueError:
            w.append(f"Dropped target '{_s(raw.get('measure'))}': value {raw.get('value')!r} is not numeric.")
            continue
        src = _s(raw.get("source")).lower()
        spec.targets.append(TargetSpec(
            measure=_s(raw["measure"]), value=value,
            filters=_parse_filters(raw.get("filters")),
            source=src if src in ("kpi", "chart") else "kpi",
            label=_s(raw.get("label")), estimated=bool(raw.get("estimated", False)),
        ))
    return spec, w


# --------------------------------------------------------------------------- sanitisation
def _dedupe(items: list, key=lambda x: x.name, warn: Optional[list] = None, what: str = "item") -> list:
    seen, out = set(), []
    for it in items:
        k = key(it).lower()
        if k in seen:
            if warn is not None:
                warn.append(f"Duplicate {what} '{key(it)}' ignored.")
            continue
        seen.add(k)
        out.append(it)
    return out


def sanitize_spec(spec: DashboardSpec) -> tuple[DashboardSpec, list]:
    """Repair what can be repaired, drop what can't, and report everything."""
    w: list = []

    # ---- date ---------------------------------------------------------
    if spec.date:
        try:
            s, e = pd.Timestamp(spec.date.start), pd.Timestamp(spec.date.end)
            if s > e:
                s, e = e, s
                w.append("Date range was reversed; swapped start and end.")
            if (e - s).days > 366 * 30:
                e = s + pd.Timedelta(days=366 * 30)
                w.append("Date range capped at 30 years.")
            spec.date.start, spec.date.end = s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")
            spec.date.name = safe_name(spec.date.name, "Date").replace("_", " ").strip() or "Date"
        except Exception:
            w.append("Date range could not be parsed; date column removed.")
            spec.date = None

    reserved = {p.lower() for p in DATE_PARTS} if spec.date else set()
    used = {spec.date.name.lower()} if spec.date else set()

    # ---- dimensions ---------------------------------------------------
    dims = []
    for d in _dedupe(spec.dimensions, warn=w, what="dimension"):
        if d.name.lower() in reserved:
            w.append(f"Dimension '{d.name}' is covered by the Date table; removed.")
            continue
        if d.name.lower() in used:
            w.append(f"Dimension '{d.name}' collides with another column; removed.")
            continue
        used.add(d.name.lower())
        d.values = list(dict.fromkeys(d.values))
        if d.cardinality is not None:
            d.cardinality = max(len(d.values), min(int(d.cardinality), MAX_ROWS))
        dims.append(d)
    spec.dimensions = dims
    dmap = spec.dim_map()

    parents_used: dict = {}
    for d in spec.dimensions:
        if d.parent is None:
            continue
        if d.parent not in dmap or d.parent == d.name:
            w.append(f"Dimension '{d.name}': unknown parent '{d.parent}' ignored.")
            d.parent = None
        elif d.parent in parents_used:
            w.append(f"'{d.parent}' already parents '{parents_used[d.parent]}'; "
                     f"'{d.name}' will be a sibling instead.")
            d.parent = None
        else:
            parents_used[d.parent] = d.name
    for d in spec.dimensions:                       # break cycles
        seen, cur = {d.name}, d.parent
        while cur:
            if cur in seen:
                w.append(f"Hierarchy cycle at '{d.name}'; parent removed.")
                d.parent = None
                break
            seen.add(cur)
            cur = dmap[cur].parent
    for d in spec.dimensions:                       # parent_map hygiene
        if d.parent:
            p = dmap[d.parent]
            d.parent_map = {k: v for k, v in d.parent_map.items() if k and v}
            for v in set(d.parent_map.values()):
                if v not in p.values:
                    p.values.append(v)
            for k in d.parent_map:
                if k not in d.values:
                    d.values.append(k)
        else:
            d.parent_map = {}

    # ---- id columns ---------------------------------------------------
    ids = []
    for i in _dedupe(spec.id_columns, warn=w, what="id column"):
        if i.name.lower() in used or i.name.lower() in reserved:
            w.append(f"ID column '{i.name}' collides with another column; removed.")
            continue
        used.add(i.name.lower())
        ids.append(i)
    spec.id_columns = ids

    # ---- fact columns -------------------------------------------------
    facts = []
    for c in _dedupe(spec.fact_columns, warn=w, what="fact column"):
        if c.name.lower() in used or c.name.lower() in reserved:
            w.append(f"Fact column '{c.name}' collides with another column; removed.")
            continue
        used.add(c.name.lower())
        if c.min is not None and c.max is not None and c.min > c.max:
            c.min, c.max = c.max, c.min
        if c.decimals is None:
            c.decimals = {"quantity": 0, "rate": 4}.get(c.kind, 2)
        c.decimals = max(0, min(int(c.decimals), 6))
        facts.append(c)
    spec.fact_columns = facts
    fmap = spec.fact_map()
    for c in spec.fact_columns:                     # derived_from hygiene + cycles
        if c.derived_from and (c.derived_from not in fmap or c.derived_from == c.name):
            w.append(f"Column '{c.name}': unknown derived_from '{c.derived_from}' ignored.")
            c.derived_from = None
    for c in spec.fact_columns:
        seen, cur = {c.name}, c.derived_from
        while cur:
            if cur in seen:
                w.append(f"Derived-column cycle at '{c.name}'; link removed.")
                c.derived_from = None
                break
            seen.add(cur)
            cur = fmap[cur].derived_from

    # ---- measures -----------------------------------------------------
    spec.measures = _dedupe(spec.measures, warn=w, what="measure")
    all_cols = ({c.name for c in spec.fact_columns} | {i.name for i in spec.id_columns}
                | {d.name for d in spec.dimensions})
    filter_cols = spec.filter_columns()

    for m in list(spec.measures):                   # auto-add unknown numeric columns
        if m.kind == "agg" and m.agg in NUMERIC_AGGS and m.column and m.column not in all_cols:
            if m.column.lower() in used or m.column.lower() in reserved:
                continue
            spec.fact_columns.append(FactColumnSpec(name=m.column, kind="money", decimals=2))
            all_cols.add(m.column)
            used.add(m.column.lower())
            w.append(f"Measure '{m.name}' uses unknown column '{m.column}'; added it as a numeric fact column.")
    fmap = spec.fact_map()

    def problem(m: MeasureSpec, names: set) -> Optional[str]:
        if m.kind == "agg":
            if m.agg not in ALL_AGGS:
                return f"unsupported aggregation {m.agg!r}"
            if m.agg != "COUNTROWS":
                if not m.column:
                    return "missing column"
                if m.agg in NUMERIC_AGGS and m.column not in fmap:
                    return f"'{m.column}' is not a numeric column"
                if m.agg in ("COUNT", "DISTINCTCOUNT") and m.column not in all_cols:
                    return f"unknown column '{m.column}'"
            for f in m.filters:
                if f.column not in filter_cols:
                    return f"filter column '{f.column}' is not a dimension or date part"
            return None
        refs = {"ratio": [m.num, m.den], "diff": [m.left, m.right], "sum": m.items}[m.kind]
        if not refs or any(r is None for r in refs):
            return "missing operand"
        if any(r not in names or r == m.name for r in refs):
            return "references an unknown measure"
        return None

    changed = True
    while changed:
        changed = False
        names = {m.name for m in spec.measures}
        for m in list(spec.measures):
            why = problem(m, names)
            if why:
                w.append(f"Measure '{m.name}' dropped: {why}.")
                spec.measures.remove(m)
                changed = True
    # cycles among composite measures
    mm = spec.measure_map()

    def deps(m):
        return {"ratio": [m.num, m.den], "diff": [m.left, m.right], "sum": m.items}.get(m.kind, [])

    for m in list(spec.measures):
        stack, seen = list(deps(m)), set()
        while stack:
            cur = stack.pop()
            if cur == m.name:
                w.append(f"Measure '{m.name}' dropped: circular definition.")
                spec.measures.remove(m)
                break
            if cur in seen or cur not in mm:
                continue
            seen.add(cur)
            stack.extend(deps(mm[cur]))
    mm = spec.measure_map()

    # ---- targets ------------------------------------------------------
    clean, seen_keys, added = [], set(), 0
    dmap = spec.dim_map()
    for t in spec.targets:
        if t.measure not in mm:
            w.append(f"Target for unknown measure '{t.measure}' dropped.")
            continue
        bad = [f.column for f in t.filters if f.column not in filter_cols]
        if bad:
            w.append(f"Target '{t.measure}' dropped: unknown filter column(s) {bad}.")
            continue
        key = (t.measure, tuple(sorted((f.column, f.value.lower()) for f in t.filters)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        m = mm[t.measure]
        if m.kind == "ratio" and m.format == "percent" and abs(t.value) > 1.5:
            w.append(f"Target '{t.measure}' = {t.value:g} looks like a percentage; converted to a fraction.")
            t.value /= 100.0
        for f in t.filters:                          # unseen dimension member -> add it
            if f.column in dmap and f.value.lower() not in {v.lower() for v in dmap[f.column].values}:
                dmap[f.column].values.append(f.value)
                added += 1
        clean.append(t)
    spec.targets = clean
    if added:
        w.append(f"Added {added} dimension member(s) that appeared only in chart targets.")

    if not spec.targets:
        raise SpecError("No usable numeric targets were found on the dashboard.")
    if not spec.measures:
        raise SpecError("No usable measures were found on the dashboard.")
    return spec, w


def load_spec(data: dict) -> tuple[DashboardSpec, list]:
    spec, w1 = parse_spec(data)
    spec, w2 = sanitize_spec(spec)
    return spec, w1 + w2
