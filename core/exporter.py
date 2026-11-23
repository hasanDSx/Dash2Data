"""Exports: CSV, multi-sheet Excel, DAX file and a one-click ZIP bundle."""
from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import DATE_FORMATS
from .dax import DaxMeasure, to_dax_text
from .model import DataModel
from .schemas import DashboardSpec

_HEADER_FILL = PatternFill("solid", fgColor="1F6FEB")
_HEADER_FONT = Font(bold=True, color="FFFFFF")


def _fmt_dates(df: pd.DataFrame, date_format: str) -> pd.DataFrame:
    py_fmt = DATE_FORMATS[date_format][0]
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime(py_fmt)
    return out


def table_csv(df: pd.DataFrame, date_format: str) -> bytes:
    return _fmt_dates(df, date_format).to_csv(index=False).encode("utf-8-sig")


def measures_frame(measures: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Measure": m.name, "Folder": m.folder, "Origin": m.origin,
        "Validated": "yes" if m.validated else "no", "Dashboard value": m.target,
        "Generated value": m.actual, "Format": m.format_string,
        "Description": m.description, "DAX": m.expression,
    } for m in measures])


def relationships_frame(model: DataModel) -> pd.DataFrame:
    cols = ["from_table", "from_column", "to_table", "to_column", "cardinality"]
    return pd.DataFrame(model.relationships, columns=cols)


def _sheet_name(name: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"
    out, i = base, 2
    while out.lower() in used:
        out = f"{base[:28]}_{i}"
        i += 1
    used.add(out.lower())
    return out


def _style(ws, df: pd.DataFrame, date_format: str) -> None:
    excel_fmt = DATE_FORMATS[date_format][1]
    for cell in ws[1]:
        cell.fill, cell.font = _HEADER_FILL, _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    for idx, col in enumerate(df.columns, start=1):
        sample = ["" if pd.isna(v) else str(v) for v in df[col].head(200)]
        width = min(max([len(str(col))] + [len(s) for s in sample]) + 2, 60)
        ws.column_dimensions[get_column_letter(idx)].width = max(width, 10)
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            for row in ws.iter_rows(min_row=2, min_col=idx, max_col=idx):
                row[0].number_format = excel_fmt
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions


def build_excel(model: DataModel, measures: list, validation: pd.DataFrame, date_format: str) -> bytes:
    buf, used = io.BytesIO(), set()
    extras = {
        "DAX Measures": measures_frame(measures),
        "Validation": validation,
    }
    if model.relationships:
        extras["Relationships"] = relationships_frame(model)
    with pd.ExcelWriter(buf, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        for name, df in {**model.tables, **extras}.items():
            sheet = _sheet_name(name, used)
            df.to_excel(writer, index=False, sheet_name=sheet)
            _style(writer.sheets[sheet], df, date_format)
            if name == "DAX Measures":
                writer.sheets[sheet].column_dimensions[get_column_letter(list(df.columns).index("DAX") + 1)].width = 70
    return buf.getvalue()


def _readme(model: DataModel, spec: DashboardSpec, summary: dict) -> str:
    lines = [
        f"# {spec.title} - Dash2Data export", "",
        f"Match rate vs dashboard: {summary['match']}/{summary['targets']} targets within 0.5% "
        f"(mean error {summary['mean_error_pct']:.4f}%).", "",
        "## Import into Power BI", "",
        "1. Get data > Text/CSV, load every CSV in this folder.",
    ]
    if model.mode == "star":
        lines += ["2. Model view: create the relationships listed in `relationships.csv` "
                  "(many-to-one, single direction, dimension -> fact).",
                  f"3. Mark `{model.date_table}` as a Date table and sort `Month` by `MonthNumber`."]
    else:
        lines += ["2. Single flat table: no relationships needed."]
    lines += ["4. Create a table named `_Measures` and add every measure from `measures.dax`.",
              "5. Rebuild the dashboard visuals - the numbers should match the original.", "",
              "Synthetic data: values are generated to reproduce dashboard aggregates; "
              "they are not real records."]
    return "\n".join(lines)


def build_zip(model: DataModel, measures: list, validation: pd.DataFrame, summary: dict,
              spec: DashboardSpec, date_format: str, base: str) -> bytes:
    base = re.sub(r"[^\w\-]+", "_", base).strip("_") or "dash2data"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in model.tables.items():
            z.writestr(f"{base}/{name}.csv", table_csv(df, date_format))
        z.writestr(f"{base}/{base}.xlsx", build_excel(model, measures, validation, date_format))
        z.writestr(f"{base}/measures.dax", to_dax_text(measures, model, spec.title))
        z.writestr(f"{base}/validation_report.csv", validation.to_csv(index=False).encode("utf-8-sig"))
        if model.relationships:
            z.writestr(f"{base}/relationships.csv", relationships_frame(model).to_csv(index=False))
        z.writestr(f"{base}/spec.json", json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))
        z.writestr(f"{base}/README.md", _readme(model, spec, summary))
    return buf.getvalue()
