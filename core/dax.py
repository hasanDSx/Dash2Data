"""DAX generation.

Dashboard measures are rendered *deterministically* from their structural definition (so the
DAX is always syntactically valid and identical in meaning to what we validated in pandas).
On top of that we recommend extra measures (time intelligence, % of total, ranking, per-order
averages) so the dataset itself can stay lean: derived metrics live in the model as measures,
not as extra columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .measures import canon_value, is_additive
from .model import DataModel
from .schemas import DATE_PARTS, MONTHS, DashboardSpec, MeasureSpec


@dataclass
class DaxMeasure:
    name: str
    expression: str
    format_string: str
    folder: str
    description: str
    origin: str                      # "dashboard" | "recommended"
    validated: bool = False          # True when the number was checked against the dashboard
    target: Optional[float] = None
    actual: Optional[float] = None


# ---------------------------------------------------------------- quoting
def q_table(t: str) -> str:
    return "'" + t.replace("'", "''") + "'"


def q_col(t: str, c: str) -> str:
    return f"{q_table(t)}[{c.replace(']', ']]')}]"


def q_str(v: str) -> str:
    return '"' + str(v).replace('"', '""') + '"'


def fmt_string(fmt: str, currency: str) -> str:
    if fmt == "percent":
        return "0.0%"
    if fmt == "decimal":
        return "#,0.00"
    if fmt == "currency":
        return "".join("\\" + ch for ch in currency) + "#,0"
    return "#,0"


class DaxBuilder:
    def __init__(self, spec: DashboardSpec, model: DataModel):
        self.spec, self.model = spec, model
        self.mm = spec.measure_map()
        taken = {c.lower() for c in model.column_table}
        self.names: dict = {}
        used: set = set()
        for m in spec.measures:
            name = m.name
            if name.lower() in taken:
                name = f"Total {m.name}"
            while name.lower() in taken or name.lower() in used:
                name += " (Measure)"
            used.add(name.lower())
            self.names[m.name] = name

    # -- helpers
    def ref(self, measure_name: str) -> str:
        return f"[{self.names[measure_name]}]"

    def col(self, c: str) -> str:
        return q_col(self.model.column_table[c], c)

    def date_ref(self) -> str:
        return q_col(self.model.date_table, self.model.date_col)

    def filter_expr(self, column: str, value: str) -> str:
        """Single-column boolean filter for CALCULATE."""
        if column in DATE_PARTS:
            v = canon_value(column, value)
            if self.model.mode == "star":
                ref = q_col(self.model.date_table, column)
                if column in ("Year", "MonthNumber"):
                    return f"{ref} = {int(v)}"
                if column == "Month":
                    return f"{ref} = {q_str(MONTHS[[m.lower() for m in MONTHS].index(v)])}"
                if column == "Quarter":
                    return f"{ref} = {q_str(v.upper())}"
                return f"{ref} = {q_str(v)}"
            d = self.date_ref()
            if column == "Year":
                return f"YEAR ( {d} ) = {int(v)}"
            if column == "MonthNumber":
                return f"MONTH ( {d} ) = {int(v)}"
            if column == "Month":
                return f'FORMAT ( {d}, "mmm" ) = {q_str(MONTHS[[m.lower() for m in MONTHS].index(v)])}'
            if column == "Quarter":
                return f'"Q" & ROUNDUP ( MONTH ( {d} ) / 3, 0 ) = {q_str(v.upper())}'
            return f'FORMAT ( {d}, "yyyy-mm" ) = {q_str(v)}'
        actual = self.model.members.get(column, {}).get(str(value).lower(), str(value))
        return f"{self.col(column)} = {q_str(actual)}"

    # -- expression for a dashboard measure
    def expression(self, m: MeasureSpec) -> str:
        if m.kind == "agg":
            if m.agg == "COUNTROWS":
                base = f"COUNTROWS ( {q_table(self.model.fact_table)} )"
            else:
                base = f"{m.agg} ( {self.col(m.column)} )"
            if not m.filters:
                return base
            filt = ",\n    ".join(self.filter_expr(f.column, f.value) for f in m.filters)
            return f"CALCULATE (\n    {base},\n    {filt}\n)"
        if m.kind == "ratio":
            return f"DIVIDE ( {self.ref(m.num)}, {self.ref(m.den)} )"
        if m.kind == "diff":
            return f"{self.ref(m.left)} - {self.ref(m.right)}"
        return " + ".join(self.ref(i) for i in m.items)


def _target_lookup(validation: Optional[pd.DataFrame], spec: DashboardSpec) -> dict:
    out: dict = {}
    if validation is None:
        return out
    for _, r in validation.iterrows():
        if r["Filter"] == "(all data)" and r["Measure"] not in out:
            out[r["Measure"]] = (r["Target"], r["Actual"])
    return out


def build_dax(spec: DashboardSpec, model: DataModel, validation: Optional[pd.DataFrame] = None,
              recommended: bool = True) -> list:
    b = DaxBuilder(spec, model)
    lookup = _target_lookup(validation, spec)
    cur = spec.currency
    out: list = []

    for m in spec.measures:
        t, a = lookup.get(m.name, (None, None))
        out.append(DaxMeasure(
            name=b.names[m.name], expression=b.expression(m), format_string=fmt_string(m.format, cur),
            folder="Dashboard KPIs", description=m.description or f"Measure shown on the dashboard ({m.name}).",
            origin="dashboard", validated=t is not None, target=t, actual=a))
    if not recommended:
        return out

    existing = {d.name.lower() for d in out}
    taken = {c.lower() for c in model.column_table} | existing

    def add(name, expr, fmt, folder, desc):
        n = name
        while n.lower() in taken:
            n += " (Rec)"
        taken.add(n.lower())
        out.append(DaxMeasure(n, expr, fmt, folder, desc, "recommended"))
        return n

    mm = b.mm
    money = [m for m in spec.measures if m.kind == "agg" and m.agg == "SUM" and not m.filters][:4]
    counts = [m for m in spec.measures if m.kind == "agg" and m.agg in ("DISTINCTCOUNT", "COUNTROWS")
              and not m.filters][:2]
    chart_dims: dict = {}
    for t in spec.targets:
        if t.source == "chart" and len(t.filters) == 1 and t.filters[0].column in spec.dim_map():
            chart_dims.setdefault(t.measure, [])
            if t.filters[0].column not in chart_dims[t.measure]:
                chart_dims[t.measure].append(t.filters[0].column)

    # existing ratios already covering num/den pairs
    ratio_pairs = {(m.num, m.den) for m in spec.measures if m.kind == "ratio"}

    for m in money:
        r = b.ref(m.name)
        label = b.names[m.name]
        base_fmt = fmt_string(m.format, cur)
        for dim in chart_dims.get(m.name, [])[:2] or chart_dims.get(
                next((x.name for x in spec.measures if x.kind != "agg" and m.name in
                      (x.left, x.num)), ""), [])[:1]:
            d = b.col(dim)
            add(f"{label} % of Total ({dim})", f"DIVIDE ( {r}, CALCULATE ( {r}, ALLSELECTED ( {d} ) ) )",
                "0.0%", "Recommended / Share", f"Share of {label} within the current selection, by {dim}.")
            add(f"{label} Rank ({dim})", f"RANKX ( ALLSELECTED ( {d} ), {r}, , DESC, DENSE )",
                "0", "Recommended / Share", f"Rank of each {dim} by {label}.")
        if model.date_col:
            dr = b.date_ref()
            ytd = add(f"{label} YTD", f"TOTALYTD ( {r}, {dr} )", base_fmt,
                      "Recommended / Time Intelligence", f"Year-to-date {label}.")
            py = add(f"{label} PY", f"CALCULATE ( {r}, SAMEPERIODLASTYEAR ( {dr} ) )",
                     base_fmt,
                     "Recommended / Time Intelligence", f"{label} for the same period last year.")
            add(f"{label} YoY %", f"DIVIDE ( {r} - [{py}], [{py}] )", "0.0%",
                "Recommended / Time Intelligence", f"Year-over-year growth of {label}.")
        for c in counts:
            if (m.name, c.name) in ratio_pairs:
                continue
            add(f"{label} per {b.names[c.name]}", f"DIVIDE ( {r}, {b.ref(c.name)} )",
                base_fmt, "Recommended / Ratios",
                f"Average {label} per {b.names[c.name]}.")
    return out


def to_dax_text(measures: list, model: DataModel, title: str = "") -> str:
    head = [
        f"// Dash2Data - DAX measures{(' for ' + title) if title else ''}",
        "// Create a disconnected table named _Measures (Enter data > blank) and add each measure to it,",
        "// or paste into Tabular Editor / DAX Studio. Measures tagged 'recommended' are NOT validated",
        "// against the dashboard; they are derived suggestions that replace extra columns.",
        f"// Model mode: {model.mode}. Tables: {', '.join(model.tables)}",
        "",
    ]
    if model.mode == "star" and model.date_table:
        head += [f"// Mark {model.date_table} as a Date table ({model.date_col}); sort Month by MonthNumber.", ""]
    body = []
    for m in measures:
        expr = m.expression.replace("\n", "\n    ")
        status = "validated vs dashboard" if m.validated else m.origin
        body.append(f"// [{m.folder}] {m.description}  ({status}; format: {m.format_string})")
        body.append(f"{m.name} =\n    {expr}\n")
    return "\n".join(head + body)
