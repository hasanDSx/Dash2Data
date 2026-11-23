"""Turn the calibrated flat table into either a single table or a Power BI-style star schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .schemas import DATE_PARTS, DashboardSpec, safe_name

DATE_TABLE = "Dim_Date"


@dataclass
class DataModel:
    mode: str                                   # "star" | "flat"
    fact_table: str
    tables: dict = field(default_factory=dict)  # name -> DataFrame (insertion order = export order)
    relationships: list = field(default_factory=list)
    column_table: dict = field(default_factory=dict)   # column -> owning table
    date_col: Optional[str] = None
    date_table: Optional[str] = None            # table holding the date (Dim_Date or fact)
    members: dict = field(default_factory=dict) # column -> {lower: actual member text}


def _leaves(spec: DashboardSpec) -> list:
    parents = {d.parent for d in spec.dimensions if d.parent}
    return [d for d in spec.dimensions if d.name not in parents]


def _chain(spec: DashboardSpec, leaf) -> list:
    dmap, out, cur = spec.dim_map(), [], leaf
    while cur:
        out.append(cur.name)
        cur = dmap.get(cur.parent) if cur.parent else None
    return out


def _calendar(spec: DashboardSpec) -> pd.DataFrame:
    days = pd.date_range(spec.date.start, spec.date.end, freq="D")
    cal = pd.DataFrame({spec.date.name: days})
    from .generator import add_date_parts
    return add_date_parts(cal, spec.date.name)


def build_model(flat: pd.DataFrame, spec: DashboardSpec, mode: str = "star") -> DataModel:
    date = spec.date.name if spec.date else None
    id_cols = [i.name for i in spec.id_columns]
    fact_cols = [c.name for c in spec.fact_columns]
    sort_by = [date] if date else []
    base = flat.sort_values(sort_by, kind="stable").reset_index(drop=True) if sort_by else flat.copy()

    if mode == "flat":
        dim_cols = [d.name for d in spec.dimensions]
        cols = ([date] if date else []) + dim_cols + id_cols + fact_cols
        model = DataModel("flat", spec.fact_table, date_col=date, date_table=spec.fact_table if date else None)
        model.tables[spec.fact_table] = base[cols].copy()
        model.column_table = {c: spec.fact_table for c in cols}
    else:
        fact = base[(([date] if date else [])) + id_cols].copy()
        model = DataModel("star", spec.fact_table, date_col=date)
        used = set(base.columns)
        for leaf in _leaves(spec):
            chain = _chain(spec, leaf)
            tname = f"Dim_{safe_name(leaf.name, 'Dim')}"
            key = f"{safe_name(leaf.name, 'Dim')}Key"
            while key in used:
                key += "_ID"
            used.add(key)
            dim_df = (base[chain].drop_duplicates(subset=[leaf.name])
                      .sort_values(leaf.name).reset_index(drop=True))
            dim_df.insert(0, key, range(1, len(dim_df) + 1))
            fact[key] = base[leaf.name].map(dict(zip(dim_df[leaf.name], dim_df[key]))).astype("int64")
            model.tables[tname] = dim_df
            for c in [key] + chain:
                model.column_table[c] = tname
            model.relationships.append({"from_table": spec.fact_table, "from_column": key,
                                        "to_table": tname, "to_column": key, "cardinality": "many-to-one"})
        for c in fact_cols:
            fact[c] = base[c]
        model.tables = {spec.fact_table: fact, **model.tables}
        for c in fact.columns:
            model.column_table.setdefault(c, spec.fact_table)
        if date:
            cal = _calendar(spec)
            model.tables[DATE_TABLE] = cal
            model.date_table = DATE_TABLE
            for c in cal.columns:
                model.column_table[c] = DATE_TABLE
            model.relationships.append({"from_table": spec.fact_table, "from_column": date,
                                        "to_table": DATE_TABLE, "to_column": date, "cardinality": "many-to-one"})

    # member lookup so DAX filters use the exact spelling of dimension values
    for d in spec.dimensions:
        if d.name in base.columns:
            model.members[d.name] = {str(v).lower(): str(v) for v in base[d.name].dropna().unique()}
    return model
