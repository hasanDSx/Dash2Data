import os
import io
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llm_helper import extract_dashboard_data, classify_error

# ---------------------------------------------------------
# Environment & API Key Configuration
# ---------------------------------------------------------
# 1. Load local environment variables from .env
load_dotenv()

# 2. Safely retrieve API Key (Cloud Secrets -> Local .env)
API_KEY = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="Dash2Data - Dashboard to Dataset Extractor",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern Styling (CSS Injection)
#
# Design tokens — one accent (blue), one neutral scale, semantic
# colors reserved for status only. No gradients, no emoji, no
# decorative shadows.
# ---------------------------------------------------------
st.markdown("""
    <style>
    :root {
        --bg-page: #0d1117;
        --bg-surface: #12161c;
        --bg-surface-2: #161b22;
        --border-default: #2a303a;
        --text-primary: #e6e8eb;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent: #4c8dff;
        --accent-bg: rgba(76, 141, 255, 0.10);
        --accent-border: rgba(76, 141, 255, 0.35);
        --success: #3fb950;
        --success-bg: rgba(63, 185, 80, 0.10);
        --success-border: rgba(63, 185, 80, 0.30);
        --warning: #d29922;
        --warning-bg: rgba(210, 153, 34, 0.10);
        --warning-border: rgba(210, 153, 34, 0.30);
        --danger: #e5534b;
        --danger-bg: rgba(229, 83, 75, 0.10);
        --danger-border: rgba(229, 83, 75, 0.30);
    }

    .stApp {
        background-color: var(--bg-page);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface-2) !important;
        border-right: 1px solid var(--border-default);
    }

    /* Section eyebrow — small caps label above a heading */
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--accent);
        background-color: var(--accent-bg);
        border: 1px solid var(--accent-border);
        border-radius: 6px;
        padding: 3px 10px;
    }

    .custom-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 13px;
        font-weight: 500;
        color: var(--text-primary);
        margin: 0;
    }

    .card-subtitle {
        font-size: 12px;
        color: var(--text-secondary);
        margin: 2px 0 0 0;
    }

    .tag {
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
        white-space: nowrap;
    }
    .tag-accent {
        background-color: var(--accent-bg);
        color: var(--accent);
        border: 1px solid var(--accent-border);
    }
    .tag-success {
        background-color: var(--success-bg);
        color: var(--success);
        border: 1px solid var(--success-border);
    }
    .tag-warning {
        background-color: var(--warning-bg);
        color: var(--warning);
        border: 1px solid var(--warning-border);
    }

    .field-note {
        font-size: 11px;
        color: var(--text-muted);
        border-top: 1px solid var(--border-default);
        padding-top: 8px;
        margin-top: 10px;
    }

    .social-link {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-secondary) !important;
        text-decoration: none;
        margin-top: 8px;
        font-size: 13px;
        transition: color 0.15s ease;
    }
    .social-link:hover {
        color: var(--accent) !important;
    }
    .social-link .dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background-color: var(--text-muted);
        flex-shrink: 0;
    }

    div.stButton > button[kind="primary"] {
        background-color: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #3d7de0;
        border-color: #3d7de0;
    }

    div[data-baseweb="input"] {
        background-color: var(--bg-page) !important;
        border-color: var(--border-default) !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--bg-page) !important;
        border-color: var(--border-default) !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Status card — a single, consistently styled way to show success,
# info, warning, and error states. Replaces raw st.error/st.warning
# calls so users never see a technical stack trace or API payload.
# ---------------------------------------------------------
def render_status_card(kind: str, title: str, message: str, technical: str | None = None) -> None:
    palette = {
        "danger": ("var(--danger)", "var(--danger-bg)", "var(--danger-border)"),
        "warning": ("var(--warning)", "var(--warning-bg)", "var(--warning-border)"),
        "success": ("var(--success)", "var(--success-bg)", "var(--success-border)"),
        "info": ("var(--accent)", "var(--accent-bg)", "var(--accent-border)"),
    }
    text_color, bg_color, border_color = palette.get(kind, palette["info"])

    st.markdown(f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_color};
                    border-left: 3px solid {text_color}; border-radius: 8px;
                    padding: 12px 16px; margin: 10px 0;">
            <p style="margin: 0 0 3px 0; font-size: 14px; font-weight: 500; color: {text_color};">{title}</p>
            <p style="margin: 0; font-size: 13px; color: var(--text-secondary); line-height: 1.55;">{message}</p>
        </div>
    """, unsafe_allow_html=True)

    if technical:
        with st.expander("Technical details"):
            st.code(technical, language="text")


# ---------------------------------------------------------
# Sidebar Section
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<p style='font-size: 15px; font-weight: 500; color: var(--text-primary); margin-bottom: 18px;'>Control panel</p>",
        unsafe_allow_html=True
    )

    # Engine Details Card
    st.markdown("""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <p class="card-title">Engine details</p>
                <span class="tag tag-accent">Vision AI</span>
            </div>
            <p style="font-size: 13px; margin: 0 0 8px 0; color: var(--text-secondary);">
                Model: <code style="color: var(--accent);">Gemini 3.6 Flash</code>
            </p>
            <ul style="font-size: 12px; color: var(--text-secondary); padding-left: 18px; margin: 0;">
                <li>Multimodal structure extraction</li>
                <li>KPI and metric matching</li>
            </ul>
            <p class="field-note">Notice model issues? Contact via email.</p>
        </div>
    """, unsafe_allow_html=True)

    # Extraction Settings Card
    st.markdown(
        "<p style='font-size: 13px; font-weight: 500; color: var(--text-primary); margin-bottom: 8px;'>Extraction parameters</p>",
        unsafe_allow_html=True
    )

    num_rows = st.number_input(
        "Number of rows to generate",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
        help="Specify how many records should be synthesized while preserving overall visual totals."
    )

    export_filename = st.text_input(
        "Export file base name",
        value="extracted_dashboard_data",
        help="Custom prefix for output CSV and Excel files."
    )

    date_format = st.selectbox(
        "Preferred date format",
        ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
        help="Standardize extracted date columns into this format."
    )

    include_nulls = st.checkbox(
        "Allow synthetic noise / nulls",
        value=False,
        help="Simulate real-world messy dataset behavior by introducing occasional missing values."
    )

    null_tolerance = st.slider(
        "Column null tolerance",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="A column is dropped automatically if more than this percentage of its values are blank or unreliable. Keep at 0 to require every column to be fully populated."
    )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Developer Card
    st.markdown("""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <p class="card-title">Hasan</p>
                <span class="tag tag-accent">Developer</span>
            </div>
            <p class="card-subtitle" style="margin-bottom: 10px;">BI and machine learning specialist</p>
            <a class="social-link" href="https://github.com/hasanDSx" target="_blank">
                <span class="dot"></span> GitHub profile
            </a>
            <a class="social-link" href="https://www.linkedin.com/in/hasanmuhammed14/" target="_blank">
                <span class="dot"></span> LinkedIn profile
            </a>
            <a class="social-link" href="mailto:hasan.m.abdelaty@gmail.com">
                <span class="dot"></span> Contact email
            </a>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------
st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="eyebrow">Automated data re-engineering</span>
        <h1 style="font-size: 26px; font-weight: 500; color: var(--text-primary); margin: 12px 0 6px 0;">Dash2Data workspace</h1>
        <p style="color: var(--text-secondary); font-size: 14px; margin: 0; max-width: 640px; line-height: 1.6;">
            Upload dashboard screenshots. Gemini Vision reverse-engineers the underlying tabular datasets that match your visual KPIs.
        </p>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose dashboard images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        cols[idx].image(file, caption=f"Screenshot {idx + 1}", use_container_width=True)

st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

if st.button("Extract data and reconstruct dataset", type="primary", use_container_width=True):
    if not API_KEY:
        render_status_card(
            "danger",
            "API key not configured",
            "Add your Gemini API key to the `.env` file or your Streamlit secrets, then try again."
        )
    elif not uploaded_files:
        render_status_card(
            "warning",
            "No screenshot uploaded",
            "Upload at least one dashboard image (PNG, JPG, or JPEG) before extracting data."
        )
    else:
        with st.spinner("Analyzing dashboard visuals and reconstructing mathematical relationships..."):
            try:
                start_time = time.time()

                images_data = [(file.getvalue(), file.type) for file in uploaded_files]

                data_json, dropped_columns = extract_dashboard_data(
                    images_data=images_data,
                    api_key=API_KEY,
                    num_rows=num_rows,
                    null_threshold=null_tolerance / 100
                )

                df = pd.DataFrame(data_json).reset_index(drop=True)
                elapsed_time = round(time.time() - start_time, 2)

                render_status_card("success", "Dataset ready", "The dataset was synthesized and validated against the dashboard's numbers.")

                if dropped_columns:
                    render_status_card(
                        "warning",
                        "Some columns were dropped",
                        "These had too many blank or unreliable values, so they were removed "
                        f"automatically: {', '.join(dropped_columns)}. Raise the column null "
                        "tolerance in the sidebar if you'd rather keep them."
                    )

                # Keep the raw extraction around so manual edits below always
                # start from the same baseline if the user resets.
                st.session_state["raw_df"] = df

                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px; margin-bottom: 10px;">
                        <p style="font-size: 16px; font-weight: 500; color: var(--text-primary); margin: 0;">Dataset preview</p>
                        <span class="tag tag-warning">Execution time: {elapsed_time}s</span>
                    </div>
                """, unsafe_allow_html=True)

                st.dataframe(df, use_container_width=True)

            except Exception as e:
                info = classify_error(e)
                render_status_card("danger", info["title"], info["message"], technical=info["technical"])

# ---------------------------------------------------------
# Additional Controls — manual review and adjustment before export
# ---------------------------------------------------------
if "raw_df" in st.session_state:
    base_df = st.session_state["raw_df"]

    st.markdown(
        "<p style='font-size: 16px; font-weight: 500; color: var(--text-primary); margin: 28px 0 4px 0;'>Additional controls</p>"
        "<p style='font-size: 13px; color: var(--text-secondary); margin: 0 0 14px 0;'>"
        "Review and adjust the generated data before you download it."
        "</p>",
        unsafe_allow_html=True
    )

    with st.expander("Select columns to include", expanded=False):
        selected_columns = st.multiselect(
            "Columns to keep in the export",
            options=list(base_df.columns),
            default=list(base_df.columns)
        )

    with st.expander("Edit values directly", expanded=False):
        st.caption("Double-click a cell to edit it. Right-click a row to delete it.")
        edited_df = st.data_editor(
            base_df[selected_columns] if selected_columns else base_df,
            use_container_width=True,
            num_rows="dynamic",
            key="dataset_editor"
        )

    final_df = (edited_df if selected_columns else base_df).reset_index(drop=True)

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    csv_data = final_df.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="Download CSV dataset",
        data=csv_data,
        file_name=f"{export_filename}.csv",
        mime="text/csv",
        use_container_width=True
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Dashboard Data')
    excel_data = excel_buffer.getvalue()

    col2.download_button(
        label="Download Excel workbook (.xlsx)",
        data=excel_data,
        file_name=f"{export_filename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
