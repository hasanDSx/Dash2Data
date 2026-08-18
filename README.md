# 📊 Dash2Data — Dashboard to Dataset Re-engineering Workspace

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini Vision](https://img.shields.io/badge/Gemini%20Vision-3.6%20Flash-8E44AD?style=for-the-badge&logo=google&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)

**An AI-powered web application that reverse-engineers dashboard screenshots into structured tabular datasets (CSV/Excel) while preserving visual metrics and KPIs.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Author](#author)

---

## 🔍 Overview <a id="overview"></a>

**Dash2Data** solves a common problem in Business Intelligence (BI) and Data Analytics: having high-level visual dashboard screenshots (Power BI, Tableau, Excel) without access to the underlying raw data.

Powered by **Google Gemini Vision**, Dash2Data inspects uploaded screenshots, interprets visual KPIs, charts, and tables, and synthesizes clean, tabular datasets that preserve mathematical relationships and overall metrics.

---

## 🌟 Key Features <a id="key-features"></a>

| Feature | Description |
|---|---|
| 👁️ **Multimodal Vision AI** | Leverages the Gemini Vision model for structure recognition and metric extraction. |
| 🎛️ **Customizable Synthesis** | Control output row counts, date formatting (`YYYY-MM-DD`, `DD/MM/YYYY`, etc.), and file naming. |
| 🧪 **Synthetic Noise Toggle** | Option to introduce controlled missing/null values to simulate realistic messy datasets. |
| 📥 **Dual Export Formats** | One-click export to clean **CSV** and multi-sheet **Excel Workbooks (.xlsx)**. |
| 🔒 **Enterprise-Grade Security** | Production-ready environment handling using `.env` locally and Streamlit Secrets in cloud deployment. |
| 🎨 **Modern Dark UI** | Custom CSS layout styled with developer-focused themes and real-time execution timing. |

---

## 🛠️ System Architecture <a id="system-architecture"></a>

```
┌─────────────────────────┐       ┌──────────────────────────┐       ┌──────────────────────────┐
│                          │       │                          │       │                          │
│   User Screenshot(s)     │ ───▶  │  Dash2Data (Streamlit)   │ ───▶  │   Google Gemini Vision   │
│  (PNG / JPG Dashboards)  │       │   App & Parameters       │       │   Structured Extraction  │
│                          │       │                          │       │                          │
└─────────────────────────┘       └──────────────────────────┘       └──────────────────────────┘
                                                                                    │
                                                                                    ▼
┌─────────────────────────┐       ┌──────────────────────────┐       ┌──────────────────────────┐
│                          │       │                          │       │                          │
│   Structured Export      │ ◀───  │   Data Frame Processing  │ ◀───  │  Extracted JSON Payload  │
│  (CSV / Excel Workbook)  │       │     (Pandas & OpenPyXL)  │       │                          │
│                          │       │                          │       │                          │
└─────────────────────────┘       └──────────────────────────┘       └──────────────────────────┘
```

---

## 🚀 Getting Started <a id="getting-started"></a>

### Prerequisites

- Python 3.10 or higher
- A free **Google Gemini API Key** ([get one here](https://aistudio.google.com/app/apikey))

### Run It Locally

```bash
# 1. Clone the repository
git clone <repository-url>
cd Dash2Data

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Deploy Your Own Copy (Streamlit Cloud)

1. Fork this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**, then select your fork, branch, and `app.py` as the main file.
4. Under **Advanced settings → Secrets**, add your API key:

   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

5. Click **Deploy**.

Your Gemini API key is only used to call Google's API on your behalf and is never shared or stored anywhere else.

---

## 📁 Project Structure <a id="project-structure"></a>

```
Dash2Data/
│
├── .env                     # Local environment variables (git-ignored)
├── .gitignore               # Git exclusion rules (secrets, environment, cache)
├── README.md                # Project documentation
├── app.py                   # Streamlit UI, styling, and application workflow
├── llm_helper.py             # Gemini API integration & JSON extraction logic
└── requirements.txt          # Python package dependencies
```

---

## 👤 Author <a id="author"></a>

Made by **Hasan Muhammad** — BI & Machine Learning Specialist

[GitHub Profile](https://github.com/hasanDSx) • [LinkedIn Profile](https://www.linkedin.com/)
