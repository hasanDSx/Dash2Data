from __future__ import annotations

import json
import logging
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.config import DATE_FORMATS, DEFAULT_MODEL, MAX_ROWS, MIN_ROWS
from core.dax import build_dax, to_dax_text
from core.exporter import build_excel, build_zip, table_csv
from core.extractor import extract_spec
from core.model import build_model
from core.pipeline import build_dataset
from core.schemas import SpecError, load_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
load_dotenv()

st.set_page_config(page_title="Dash2Data - Dashboard to Dataset",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- styling
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
.custom-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px;
               margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,.15); }
.badge-green, .badge-blue, .badge-purple, .badge-timer { font-size: 11px; padding: 3px 8px; border-radius: 12px;
               font-weight: 600; display: inline-block; }
.badge-green  { background: rgba(46,160,67,.15);  color: #3fb950; border: 1px solid rgba(46,160,67,.4); }
.badge-blue   { background: rgba(56,139,253,.15); color: #58a6ff; border: 1px solid rgba(56,139,253,.4); }
.badge-purple { background: rgba(163,113,247,.15); color: #bc8cff; border: 1px solid rgba(163,113,247,.4); }
.badge-timer  { background: rgba(210,153,34,.15); color: #d29922; border: 1px solid rgba(210,153,34,.4);
                font-size: 12px; padding: 4px 10px; }
.social-link { display: flex; align-items: center; gap: 8px; color: #58a6ff !important; text-decoration: none;
               margin-top: 8px; font-size: 13px; }
.social-link:hover { color: #79c0ff !important; }
div.stButton > button[kind="primary"] { background: linear-gradient(135deg,#238636 0%,#2ea043 100%);
               border: 1px solid rgba(240,246,252,.1); border-radius: 8px; font-weight: 600; }
div.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg,#2ea043 0%,#3fb950 100%);
               box-shadow: 0 4px 12px rgba(46,160,67,.3); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background-color: #0d1117 !important;
               border-color: #30363d !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def resolve_api_key(user_key: str) -> str | None:
    if user_key.strip():
        return user_key.strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def friendly_error(e: Exception) -> None:
    msg = str(e)
    if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
        st.warning("⏳ Gemini is busy or rate-limited right now. Please try again in a minute or two.")
    elif isinstance(e, SpecError):
        st.error(f"The dashboard could not be understood: {msg}")
        st.info("Try a clearer / higher-resolution screenshot, or add a hint in the box above the button.")
    elif "API key" in msg or "API_KEY" in msg or "401" in msg or "403" in msg:
        st.error("🔑 The API key was rejected. Check it in the sidebar > Advanced.")
    elif "404" in msg or "NOT_FOUND" in msg:
        st.error(f"The model name was not found: {msg}. Change it in the sidebar > Advanced.")
    else:
        st.error(f"An error occurred during processing: {msg}")
        st.info("📩 If this persists, contact **hasan.m.abdelaty@gmail.com**.")


def run_build(spec_dict: dict, settings: dict):
    spec, warns = load_spec(spec_dict)
    result = build_dataset(spec, num_rows=settings["rows"], auto_rows=settings["auto_rows"],
                           include_nulls=settings["nulls"], seed=settings["seed"])
    st.session_state["result"] = result
    st.session_state["spec_warnings"] = warns
    st.session_state["result_id"] = st.session_state.get("result_id", 0) + 1
    st.session_state.pop("downloads", None)
    return result


def get_bundle(result, mode: str, recommended: bool):
    key = (st.session_state["result_id"], mode, recommended)
    cache = st.session_state.setdefault("bundle_cache", {})
    if key not in cache:
        cache.clear()
        model = build_model(result.flat, result.spec, mode)
        measures = build_dax(result.spec, model, result.validation, recommended=recommended)
        cache[key] = (model, measures)
    return cache[key]


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<h2 style='font-size:20px;font-weight:700;margin-bottom:20px;'>⚙️ Control Panel</h2>",
                unsafe_allow_html=True)

    adv_model = DEFAULT_MODEL
    with st.expander("🔑 Advanced (API key & model)"):
        user_key = st.text_input("Your Gemini API key (optional)", type="password",
                                 help="Used only for this session. Leave empty to use the server's key.")
        adv_model = st.text_input("Model name", value=DEFAULT_MODEL,
                                  help="Any Gemini model that accepts images.")

    st.markdown(f"""
    <div class="custom-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <strong style="color:#f0f6fc;font-size:14px;">Engine Details</strong>
        <span class="badge-green">Vision AI</span></div>
      <p style="font-size:13px;margin:0;color:#8b949e;"><strong>Model:</strong>
         <code style="color:#58a6ff;">{adv_model}</code></p>
      <ul style="font-size:12px;color:#8b949e;padding-left:18px;margin:8px 0;">
        <li>AI extracts the dashboard <em>spec</em></li>
        <li>Code generates &amp; calibrates the rows</li>
        <li>Every KPI is validated numerically</li></ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:14px;font-weight:600;color:#f0f6fc;margin-bottom:8px;'>🛠️ Dataset Parameters</p>",
                unsafe_allow_html=True)
    num_rows = st.number_input("Number of rows:", min_value=MIN_ROWS, max_value=MAX_ROWS, value=50, step=10,
                               help="Used when the dashboard has no count KPIs, or when auto-size is off.")
    auto_rows = st.checkbox("Auto-size rows to match count KPIs", value=True,
                            help="If the dashboard shows 'Orders: 3,420' or 'Headcount: 600', the dataset gets "
                                 "enough rows to reproduce those counts exactly.")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1,
                           help="Same seed = same dataset. Change it for a different random draw.")
    include_nulls = st.checkbox("Allow synthetic nulls", value=False,
                                help="Adds ~3-5% missing values to numeric columns (totals are still calibrated).")
    date_format = st.selectbox("Date format", list(DATE_FORMATS), help="Applied to CSV and Excel exports.")

    st.markdown("<p style='font-size:14px;font-weight:600;color:#f0f6fc;margin:12px 0 8px;'>🧱 Output Model</p>",
                unsafe_allow_html=True)
    mode_label = st.radio("Schema", ["Star schema (Power BI)", "Single flat table"],
                          help="Star schema = Fact + Dimension tables + Date table, as real Power BI models.")
    mode = "star" if mode_label.startswith("Star") else "flat"
    recommended = st.checkbox("Include recommended DAX measures", value=True,
                              help="YTD / PY / YoY, % of total, ranking, per-order averages. These live in the model "
                                   "as measures, so the dataset needs no extra columns.")
    export_name = st.text_input("Export base name", value="extracted_dashboard_data")

    st.markdown("""
    <div class="custom-card" style="margin-top:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <strong style="color:#f0f6fc;font-size:14px;">Hasan</strong><span class="badge-blue">Developer</span></div>
      <p style="font-size:12px;color:#8b949e;margin-bottom:12px;">BI &amp; Machine Learning Specialist</p>
      <a class="social-link" href="https://github.com/hasanDSx" target="_blank">GitHub Profile</a>
      <a class="social-link" href="https://www.linkedin.com/in/hasanmuhammad06/" target="_blank">LinkedIn Profile</a>
      <a class="social-link" href="mailto:hasan.m.abdelaty@gmail.com">Contact Email</a>
    </div>""", unsafe_allow_html=True)

settings = {"rows": int(num_rows), "auto_rows": auto_rows, "nulls": include_nulls, "seed": int(seed)}

# --------------------------------------------------------------------------- header + upload
st.markdown("""
<div style="margin-bottom:24px;">
  <span class="badge-purple" style="margin-bottom:8px;">Automated Data Re-engineering</span>
  <h1 style="font-size:28px;font-weight:700;color:#f0f6fc;margin:0;">📊 Dash2Data Workspace</h1>
  <p style="color:#8b949e;font-size:14px;margin-top:4px;">Upload dashboard screenshots. The AI reads the dashboard,
  then code rebuilds a star-schema dataset whose aggregates match the visuals - plus the DAX measures to rebuild it.</p>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose dashboard images (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"],
                            accept_multiple_files=True)
if uploaded:
    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        cols[i % len(cols)].image(f, caption=f"Screenshot {i + 1}", width="stretch")

hints = st.text_area("Optional hints for the AI", placeholder="e.g. Currency is EGP. Data covers 2022-2024. "
                     "The slicer 'Year' is set to 2023.", height=70)

if st.button("🚀 Extract & Generate Dataset", type="primary", width="stretch"):
    api_key = resolve_api_key(user_key)
    if not api_key:
        st.error("🔑 API key is missing. Add it in the sidebar (Advanced), `.env`, or Streamlit secrets.")
    elif not uploaded:
        st.warning("⚠️ Please upload at least one dashboard screenshot first.")
    else:
        try:
            t0 = time.time()
            with st.status("Working...", expanded=True) as status:
                st.write("1/2 Reading the dashboard with Gemini Vision...")
                images = [(f.getvalue(), f.type) for f in uploaded]
                spec, spec_warns, _raw = extract_spec(images, api_key, adv_model, hints)
                st.session_state["spec_text"] = json.dumps(spec.to_dict(), indent=2, ensure_ascii=False)
                st.write(f"   found {len(spec.measures)} measures and {len(spec.targets)} dashboard numbers")
                st.write("2/2 Generating rows, calibrating and validating...")
                run_build(spec.to_dict(), settings)
                st.session_state["spec_warnings"] = spec_warns + st.session_state["spec_warnings"]
                st.session_state["elapsed"] = round(time.time() - t0, 2)
                status.update(label="Done", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001 - surfaced to the user
            logging.getLogger("app").exception("Pipeline failed")
            friendly_error(e)

# --------------------------------------------------------------------------- results
if "result" in st.session_state:
    result = st.session_state["result"]
    s = result.summary
    model, measures = get_bundle(result, mode, recommended)

    head_l, head_r = st.columns([3, 1])
    head_l.markdown(f"""
    <div style="display:flex;gap:10px;align-items:center;margin-top:20px;">
      <h3 style="font-size:18px;color:#f0f6fc;margin:0;">✅ {result.spec.title}</h3>
      <span class="badge-timer">⚡ {st.session_state.get('elapsed', 0)}s</span></div>""", unsafe_allow_html=True)
    if head_r.button("🔄 Regenerate (no AI call)", width="stretch",
                     help="Rebuild with the current sidebar settings and the spec shown in the Spec tab."):
        try:
            run_build(json.loads(st.session_state.get("spec_text", json.dumps(result.spec.to_dict()))), settings)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            friendly_error(e)

    c = st.columns(5)
    c[0].metric("Dashboard numbers matched", f"{s['match']}/{s['targets']}")
    c[1].metric("Mean error", f"{s['mean_error_pct']:.4f}%" if s["mean_error_pct"] == s["mean_error_pct"] else "n/a")
    c[2].metric("Worst error", f"{s['max_error_pct']:.3f}%" if s["max_error_pct"] == s["max_error_pct"] else "n/a")
    c[3].metric("Rows", f"{result.n_rows:,}")
    c[4].metric("DAX measures", len(measures))

    if s["off"]:
        st.warning(f"{s['off']} number(s) could not be matched - see the Validation tab. Usually the dashboard "
                   "numbers conflict with each other or a value was misread; fix them in the Spec tab and regenerate.")
    all_warns = list(dict.fromkeys(st.session_state.get("spec_warnings", []) + result.warnings))
    if all_warns:
        with st.expander(f"⚠️ Notes ({len(all_warns)})"):
            for w in all_warns:
                st.write("- " + w)

    tab_data, tab_val, tab_dax, tab_spec, tab_exp = st.tabs(
        ["📋 Data", "✅ Validation", "🧮 DAX Measures", "🧠 Spec", "⬇️ Export"])

    with tab_data:
        names = list(model.tables)
        subtabs = st.tabs([f"{n} ({len(model.tables[n]):,})" for n in names])
        for tab, n in zip(subtabs, names):
            with tab:
                st.dataframe(model.tables[n].head(1000), width="stretch", hide_index=True)
                if len(model.tables[n]) > 1000:
                    st.caption("Showing the first 1,000 rows. Exports contain everything.")
        if model.relationships:
            with st.expander("🔗 Relationships to create in Power BI"):
                st.dataframe(pd.DataFrame(model.relationships), width="stretch", hide_index=True)

    with tab_val:
        only_bad = st.checkbox("Show only mismatches", value=False)
        v = result.validation
        if only_bad:
            v = v[v["Status"] != "✅ match"]
        st.dataframe(v, width="stretch", hide_index=True, column_config={
            "Target": st.column_config.NumberColumn(format="%.4f"),
            "Actual": st.column_config.NumberColumn(format="%.4f"),
            "Abs Error": st.column_config.NumberColumn(format="%.4f"),
            "Error %": st.column_config.NumberColumn(format="%.5f"),
        })
        st.caption("Status: match <= 0.5% error, close <= 2%. 'Estimated' targets were read from chart scales.")

    with tab_dax:
        st.caption("Dashboard measures are validated against the dashboard. Recommended measures are "
                   "suggestions that replace extra columns - they are not validated.")
        folders: dict = {}
        for m in measures:
            folders.setdefault(m.folder, []).append(m)
        for folder, items in folders.items():
            with st.expander(f"{folder} ({len(items)})", expanded=folder == "Dashboard KPIs"):
                for m in items:
                    badge = "✅ validated" if m.validated else ("dashboard" if m.origin == "dashboard" else "suggested")
                    st.markdown(f"**{m.name}** &nbsp; `{badge}` &nbsp; <span style='color:#8b949e'>{m.description}</span>",
                                unsafe_allow_html=True)
                    st.code(f"{m.name} =\n    " + m.expression.replace("\n", "\n    "), language="text")
        st.download_button("📥 Download measures.dax", to_dax_text(measures, model, result.spec.title),
                           file_name="measures.dax", mime="text/plain")

    with tab_spec:
        st.caption("What the AI understood. Edit anything (a misread number, a missing category) and press "
                   "**Regenerate** at the top - no new AI call is needed.")
        st.text_area("Dashboard specification (JSON)", key="spec_text", height=420)

    with tab_exp:
        sig = (st.session_state["result_id"], mode, recommended, date_format, export_name)
        if st.session_state.get("downloads", {}).get("sig") != sig:
            if st.button("📦 Prepare downloads", type="primary", width="stretch"):
                with st.spinner("Building files..."):
                    st.session_state["downloads"] = {
                        "sig": sig,
                        "zip": build_zip(model, measures, result.validation, s, result.spec, date_format, export_name),
                        "xlsx": build_excel(model, measures, result.validation, date_format),
                        "csv": table_csv(model.tables[model.fact_table], date_format),
                    }
                st.rerun()
            st.info("Click to build the ZIP (all CSVs + Excel + DAX + validation report), the Excel workbook and the CSV.")
        else:
            d = st.session_state["downloads"]
            c1, c2, c3 = st.columns(3)
            c1.download_button("📦 Everything (.zip)", d["zip"], file_name=f"{export_name}.zip",
                               mime="application/zip", width="stretch")
            c2.download_button("📊 Excel workbook (.xlsx)", d["xlsx"], file_name=f"{export_name}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               width="stretch")
            c3.download_button(f"📄 {model.fact_table}.csv", d["csv"], file_name=f"{export_name}.csv",
                               mime="text/csv", width="stretch")
