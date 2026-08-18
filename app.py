import os
import io
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llm_helper import extract_dashboard_data

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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern Styling (CSS Injection)
# ---------------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    .custom-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .badge-green {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-blue {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-purple {
        background-color: rgba(163, 113, 247, 0.15);
        color: #bc8cff;
        border: 1px solid rgba(163, 113, 247, 0.4);
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-timer {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }

    .social-link {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #58a6ff !important;
        text-decoration: none;
        margin-top: 8px;
        font-size: 13px;
        transition: color 0.2s ease;
    }
    .social-link:hover {
        color: #79c0ff !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }

    div[data-baseweb="input"] {
        background-color: #0d1117 !important;
        border-color: #30363d !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0d1117 !important;
        border-color: #30363d !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Section
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='font-size: 20px; font-weight: 700; margin-bottom: 20px;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    
    # Engine Details Card
    st.markdown("""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong style="color: #f0f6fc; font-size: 14px;">Engine Details</strong>
                <span class="badge-green">Vision AI</span>
            </div>
            <p style="font-size: 13px; margin: 0; color: #8b949e;">
                <strong>Model:</strong> <code style="color: #58a6ff;">Gemini 3.6 Flash</code>
            </p>
            <ul style="font-size: 12px; color: #8b949e; padding-left: 18px; margin-top: 8px; margin-bottom: 8px;">
                <li>Multimodal Structure Extraction</li>
                <li>KPI & Metric Matching</li>
            </ul>
            <div style="font-size: 11px; color: #6e7681; border-top: 1px solid #21262d; padding-top: 8px; margin-top: 6px;">
                💡 <em>Notice model issues? Contact via email.</em>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Extraction Settings Card
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: #f0f6fc; margin-bottom: 8px;'>🛠️ Extraction Parameters</p>", unsafe_allow_html=True)
    
    num_rows = st.number_input(
        "Number of rows to generate:", 
        min_value=10, 
        max_value=1000, 
        value=50, 
        step=10,
        help="Specify how many records should be synthesized while preserving overall visual totals."
    )
    
    export_filename = st.text_input(
        "Export File Base Name:", 
        value="extracted_dashboard_data",
        help="Custom prefix for output CSV and Excel files."
    )

    date_format = st.selectbox(
        "Preferred Date Format:",
        ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"],
        help="Standardize extracted date columns into this format."
    )

    include_nulls = st.checkbox(
        "Allow Synthetic Noise / Nulls", 
        value=False,
        help="Simulate real-world messy dataset behavior by introducing occasional missing values."
    )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Developer Card
    st.markdown("""
        <div class="custom-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="color: #f0f6fc; font-size: 14px;">Hasan</strong>
                <span class="badge-blue">Developer</span>
            </div>
            <p style="font-size: 12px; color: #8b949e; margin-bottom: 12px;">BI & Machine Learning Specialist</p>
            <a class="social-link" href="https://github.com/hasanDSx" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="15" height="15" style="filter: invert(0.8);"> GitHub Profile
            </a>
            <a class="social-link" href="https://www.linkedin.com/in/hasanmuhammed14/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="15" height="15"> LinkedIn Profile
            </a>
            <a class="social-link" href="mailto:hasan.m.abdelaty@gmail.com">
                <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="15" height="15"> Contact Email
            </a>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------
st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="badge-purple" style="margin-bottom: 8px;">Automated Data Re-engineering</span>
        <h1 style="font-size: 28px; font-weight: 700; color: #f0f6fc; margin: 0;">📊 Dash2Data Workspace</h1>
        <p style="color: #8b949e; font-size: 14px; margin-top: 4px;">Upload dashboard screenshots. Gemini Vision will reverse-engineer underlying tabular datasets matching visual KPIs.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose dashboard images (PNG, JPG, JPEG):", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        cols[idx].image(file, caption=f"Screenshot {idx + 1}", use_container_width=True)

st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

if st.button("🚀 Extract Data & Reconstruct Dataset", type="primary", use_container_width=True):
    if not API_KEY:
        st.error("🔑 API Key is missing! Please configure it in your `.env` file or Streamlit Secrets.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one dashboard screenshot first.")
    else:
        with st.spinner("Analyzing dashboard visuals and reconstructing mathematical relationships..."):
            try:
                start_time = time.time()

                images_data = [(file.getvalue(), file.type) for file in uploaded_files]

                data_json = extract_dashboard_data(
                    images_data=images_data,
                    api_key=API_KEY,
                    num_rows=num_rows
                )

                df = pd.DataFrame(data_json)
                elapsed_time = round(time.time() - start_time, 2)

                st.success("✅ Dataset synthesized and validated successfully!")

                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px; margin-bottom: 10px;">
                        <h3 style="font-size: 18px; color: #f0f6fc; margin: 0;">📋 Dataset Preview</h3>
                        <span class="badge-timer">⚡ Execution Time: <strong>{elapsed_time}s</strong></span>
                    </div>
                """, unsafe_allow_html=True)

                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)

                csv_data = df.to_csv(index=False).encode('utf-8')
                col1.download_button(
                    label="📥 Download CSV Dataset",
                    data=csv_data,
                    file_name=f"{export_filename}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Dashboard Data')
                excel_data = excel_buffer.getvalue()

                col2.download_button(
                    label="📊 Download Excel Workbook (.xlsx)",
                    data=excel_data,
                    file_name=f"{export_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.info("📩 If this issue persists, please reach out to **hasan.m.abdelaty@gmail.com**.")