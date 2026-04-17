"""
VannVakt — Water Cut Intelligence

Multi-signal water breakthrough and production performance detector.

Signals monitored:
- Water cut (primary)
- Oil decline
- GOR (Gas-Oil Ratio)
- Field baseline deviation
- Trend acceleration

Two analysis modes:
- Historical:     complete time-series datasets (Volve NCS 2007-2016)
- Live Snapshot:  current production snapshots (Florida DEP 2026)

The app detects which mode to use based on data density.
Data cleaning is handled separately in the cleaning notebooks.
This app assumes clean input data.

Deployment:
- Local:            add ANTHROPIC_API_KEY to .env
- Streamlit Cloud:  add ANTHROPIC_API_KEY to app secrets
- Without API key:  app runs fully, chat is disabled

Author: Anuri Nwagbara
GitHub: github.com/Anuri-ops
Data:   Equinor Volve NCS (open) + Florida DEP 2026 (public)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────
# api key — works locally and on Streamlit Cloud
# ─────────────────────────────
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    try:
        _api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        _api_key = None

client = None
if _api_key:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_api_key)
    except Exception:
        client = None

st.set_page_config(
    page_title="VannVakt",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────
# styling
# ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #1a1d21;
    color: #d1d2d3;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

section[data-testid="stSidebar"] {
    background: #19171d !important;
    border-right: 1px solid #2d2f33;
    min-width: 260px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.brand-block { padding: 20px 16px 14px 16px; border-bottom: 1px solid #2d2f33; }
.brand-title { font-size: 20px; font-weight: 700; color: #fff; margin: 0; }
.brand-sub {
    font-size: 10px; color: #6b7280;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.12em; text-transform: uppercase; margin: 3px 0 0 0;
}
.mode-badge {
    display: inline-block; padding: 4px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em; margin-bottom: 16px;
}
.mode-historical { background: #0d2e1e; color: #2eb67d !important; border: 1px solid #1a4a30; }
.mode-live       { background: #1e1a08; color: #ecb22e !important; border: 1px solid #5a4a1a; }
.sidebar-label {
    padding: 14px 16px 4px 16px;
    font-size: 10px; font-weight: 600; color: #6b7280;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.meta-block { padding: 10px 16px; border-top: 1px solid #2d2f33; margin-top: 8px; }
.meta-row { display: flex; justify-content: space-between; padding: 3px 0; }
.meta-key { font-size: 10px; color: #6b7280; font-family: 'IBM Plex Mono', monospace; min-width: 80px; }
.meta-val { font-size: 10px; color: #c9cacc; font-family: 'IBM Plex Mono', monospace; text-align: right; flex: 1; }
.meta-na  { font-size: 10px; color: #4a4d52; font-family: 'IBM Plex Mono', monospace; text-align: right; flex: 1; font-style: italic; }
.about-block { padding: 10px 16px; border-top: 1px solid #2d2f33; margin-top: 8px; }
.about-text  { font-size: 10px; color: #4a4d52; line-height: 1.5; margin: 0 0 12px 0; }
.source-link { font-size: 10px; color: #4a4d52; display: block; margin-top: 10px; }
.section-title {
    font-size: 11px; font-weight: 600; color: #6b7280;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0 0 8px 0; border-bottom: 1px solid #2d2f33;
    margin-bottom: 14px; margin-top: 24px;
}
.mode-note {
    background: #1e2128; border: 1px solid #2d2f33; border-radius: 6px;
    padding: 10px 14px; margin: 0 0 16px 0;
    font-size: 12px; color: #9ea3a8; line-height: 1.6;
    font-family: 'IBM Plex Mono', monospace;
}
.diag-card {
    background: #1e2128; border-left: 3px solid #e01e5a;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    margin: 5px 0; font-size: 13px; color: #c9cacc; line-height: 1.5;
}
.diag-card.warn { border-left-color: #ecb22e; }
.diag-card.ok   { border-left-color: #2eb67d; }
.diag-card.info { border-left-color: #1264a3; }
.action-card {
    background: #141c2e; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 8px 14px; margin: 4px 0;
    font-size: 12px; color: #7cb9e8; line-height: 1.5;
}
.dq-banner {
    background: #1e1a08; border: 1px solid #5a4a1a; border-radius: 6px;
    padding: 10px 14px; margin: 8px 0 4px 0;
    font-size: 11px; color: #ecb22e;
    font-family: 'IBM Plex Mono', monospace; line-height: 1.6;
}
div[data-testid="stMetricContainer"] {
    background: #222529 !important; border: 1px solid #2d2f33 !important;
    border-radius: 8px !important; padding: 14px 18px !important;
}
div[data-testid="stMetricContainer"] label {
    color: #6b7280 !important; font-size: 10px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stMetricContainer"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important; font-weight: 600 !important;
}
div[data-testid="stMetricContainer"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #222529 !important; border: 1px solid #2d2f33 !important;
    color: #d1d2d3 !important; font-size: 13px !important;
}
div[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important; padding: 8px 0 !important;
}
div[data-testid="stChatInput"] {
    background: #222529 !important; border: 1px solid #3a3d42 !important;
    border-radius: 12px !important; padding: 4px 8px !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.15) !important;
}
div[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #d1d2d3 !important;
    font-size: 14px !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #6b7280 !important; }
div[data-testid="stChatInput"] button {
    background: #4fc3f7 !important; border-radius: 8px !important; color: #000 !important;
}
div[data-testid="stExpander"] {
    background: #1e2128 !important; border: 1px solid #2d2f33 !important;
    border-radius: 8px !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #1a1d21; }
::-webkit-scrollbar-thumb { background: #3d3944; border-radius: 3px; }

/* ── Force dark mode everywhere ── */

/* Main app background */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #1a1d21 !important;
}

/* Block container */
[data-testid="block-container"] {
    background-color: #1a1d21 !important;
}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #19171d !important;
}

/* Dataframes */
[data-testid="stDataFrame"], .dataframe, iframe {
    background-color: #222529 !important;
    color: #d1d2d3 !important;
}
[data-testid="stDataFrame"] th {
    background-color: #2d2f33 !important;
    color: #9ea3a8 !important;
}
[data-testid="stDataFrame"] td {
    background-color: #222529 !important;
    color: #d1d2d3 !important;
    border-color: #2d2f33 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #1e2128 !important;
    border: 1px solid #2d2f33 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    background-color: #1e2128 !important;
    color: #d1d2d3 !important;
}
[data-testid="stExpander"] > div {
    background-color: #1e2128 !important;
}

/* Selectbox dropdown */
[data-testid="stSelectbox"] > div > div,
div[role="listbox"],
div[role="option"] {
    background-color: #222529 !important;
    color: #d1d2d3 !important;
    border-color: #2d2f33 !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    background-color: transparent !important;
}
[data-testid="stRadio"] label {
    color: #d1d2d3 !important;
}

/* Text elements */
p, span, label, div {
    color: inherit !important;
}

/* Warnings and info boxes */
[data-testid="stAlert"] {
    background-color: #1e2128 !important;
    border-color: #2d2f33 !important;
}

/* Plotly charts */
.js-plotly-plot, .plotly {
    background-color: #16181d !important;
}

/* Caption text */
[data-testid="stCaptionContainer"] {
    color: #9ea3a8 !important;
}
/* Warning/info box text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #d1d2d3 !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Markdown text */
.stMarkdown p {
    color: #d1d2d3 !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #4fc3f7 !important;
}

/* Tooltips */
[data-testid="stTooltipIcon"] {
    color: #6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# constants
# ─────────────────────────────
PLOT_BG  = "#16181d"
GRID_COL = "#252830"
BLUE     = "#4fc3f7"
AMBER    = "#ffb74d"
RED      = "#ef5350"
GREEN    = "#66bb6a"
PURPLE   = "#b39ddb"

# ─────────────────────────────
# breakthrough config
# ─────────────────────────────
# Thresholds separated from code so they can be reviewed independently.
# Historical: derived from Volve data — natural WC variation < 8%/month
# Live: adjusted for Florida field maturity (avg WC > 90%)
BREAKTHROUGH_CONFIG = {
    "historical": {
        "wc_jump":    0.10,
        "wc_floor":   0.20,
        "rolling_wc": 0.30,
        "wc_high":    0.80,
        "wc_critical":0.90,
        "wc_trend":   0.05,
    },
    "live": {
        "wc_jump":    0.01,
        "wc_floor":   0.70,
        "high_wc":    0.80,
        "wc_high":    0.80,
        "wc_critical":0.90,
    }
}

VOLVE_META = {
    "15/9-F-1 C":  {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2013,"td_m":4079},
    "15/9-F-11":   {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2013,"td_m":4758},
    "15/9-F-12":   {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2007,"td_m":3496},
    "15/9-F-14":   {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2007,"td_m":3730},
    "15/9-F-15 D": {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2013,"td_m":4671},
    "15/9-F-5":    {"type":"Producer","trajectory":"Deviated","reservoir":"Hugin Fm. (Upper Jurassic)","drive":"Water drive","spud":2007,"td_m":3762},
}

# ─────────────────────────────
# schema validation
# ─────────────────────────────
VOLVE_SCHEMA = {
    "required_columns": ["Date","Wellbore name","Year","Month","Oil","Gas","Water","On Stream","Water_Cut"],
    "numeric_columns":  ["Year","Month","Oil","Gas","Water","On Stream","Water_Cut"],
    "ranges": {
        "Water_Cut": (0.0, 1.0),
        "Oil":       (0.0, None),
        "Gas":       (0.0, None),
        "Water":     (0.0, None),
        "Month":     (1, 12),
    }
}

FLORIDA_SCHEMA = {
    "required_columns": ["OilField","Lease_Id","WellName","Operator","APINumber","County","Oil","Gas","Water","DaysOn"],
    "numeric_columns":  ["Oil","Gas","Water","DaysOn"],
    "ranges": {
        "Oil":    (0.0, None),
        "Gas":    (0.0, None),
        "Water":  (0.0, None),
        "DaysOn": (0, 31),
    }
}

def validate_schema(df, schema, dataset_name):
    errors, warnings = [], []
    missing = [c for c in schema["required_columns"] if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return False, errors, warnings
    for col in schema["numeric_columns"]:
        if col not in df.columns: continue
        n_bad = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n_bad > 0:
            warnings.append(f"{col}: {n_bad} non-numeric values — will be set to 0")
    for col, (lo, hi) in schema["ranges"].items():
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if lo is not None and (vals < lo).any():
            errors.append(f"{col}: {(vals < lo).sum()} values below minimum ({lo})")
        if hi is not None and (vals > hi).any():
            errors.append(f"{col}: {(vals > hi).sum()} values above maximum ({hi})")
    if len(df) == 0:
        errors.append("Dataset is empty — no rows found")
    name_col = "Wellbore name" if "Wellbore name" in df.columns else "WellName"
    if name_col in df.columns:
        null_n = df[name_col].isna().sum()
        if null_n == len(df):
            errors.append(f"{name_col}: all values are null — file may be malformed")
        elif null_n > 0:
            warnings.append(f"{name_col}: {null_n} rows with no well name — will be excluded")
    return len(errors) == 0, errors, warnings

def show_validation(passed, errors, warnings, name):
    if not passed:
        st.error(f"**{name} — schema validation failed**")
        for e in errors:
            st.markdown(f"- ❌ {e}")
        st.caption("Check the file is correct and columns match the expected format.")
        st.stop()
    if warnings:
        with st.expander(f"⚠ {name} — {len(warnings)} data warning(s)"):
            for w in warnings:
                st.markdown(f"- ⚠ {w}")

# ─────────────────────────────
# mode detection
# ─────────────────────────────
def detect_mode(df):
    pts = df.groupby("Wellbore name").size().median()
    return "historical" if pts > 12 else "live"

# ─────────────────────────────
# data loaders
# ─────────────────────────────
@st.cache_data
def load_volve():
    df = pd.read_csv("cleaned_vannvakt_data.csv", parse_dates=["Date"])
    passed, errors, warnings = validate_schema(df, VOLVE_SCHEMA, "Volve NCS")
    show_validation(passed, errors, warnings, "Volve NCS")
    df = df.sort_values(["Wellbore name","Date"]).reset_index(drop=True)
    df["WC_Change"]  = df.groupby("Wellbore name")["Water_Cut"].diff()
    df["WC_Rolling"] = df.groupby("Wellbore name")["Water_Cut"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df["Oil_Change"] = df.groupby("Wellbore name")["Oil"].diff()
    if "Gas" in df.columns and df["Gas"].sum() > 0:
        df["GOR"] = np.where(df["Oil"]>0, df["Gas"]/df["Oil"], np.nan)
    else:
        df["GOR"] = np.nan
    cfg = BREAKTHROUGH_CONFIG["historical"]
    df["Sudden_BT"]    = (df["WC_Change"]>=cfg["wc_jump"])&(df["Water_Cut"]>=cfg["wc_floor"])&(df["Oil_Change"]<0)
    df["Sustained_BT"] = df["WC_Rolling"]>=cfg["rolling_wc"]
    def classify(r):
        if r["Sudden_BT"] and r["Sustained_BT"]: return "Sudden + Sustained"
        if r["Sudden_BT"]:   return "Sudden"
        if r["Sustained_BT"]: return "Sustained"
        return None
    df["Breakthrough_Type"] = df.apply(classify, axis=1)
    df["Mode"]    = "historical"
    df["Dataset"] = "Volve NCS (2007–2016)"
    return df

@st.cache_data
def load_florida():
    fp = "2026_WELL_PRODUCTION_DATA_REPORT_WEBSITE.xlsx"
    def read_sheet(sheet, label, days_in_month):
        raw = pd.read_excel(fp, sheet_name=sheet, header=None)
        hrow = next(i for i,r in raw.iterrows() if "WellName" in str(r.values))
        raw.columns = raw.iloc[hrow]
        raw = raw.iloc[hrow+1:].copy()
        raw.columns = ["OilField","Lease_Id","WellName","Operator",
                       "APINumber","County","Oil","Gas","Water","DaysOn"]
        raw["OilField"] = raw["OilField"].ffill()
        raw = raw[raw["WellName"].notna()].copy()
        for c in ["Oil","Gas","Water","DaysOn"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
        raw["Date"]          = pd.to_datetime(label, format="%b-%Y")
        raw["Month"]         = label
        raw["Days_In_Month"] = days_in_month
        raw["Shut_In"]       = raw["DaysOn"] == 0
        raw["Partial_Month"] = (raw["DaysOn"]>0)&(raw["DaysOn"]<days_in_month-2)
        raw["Oil_Daily"]   = np.where(raw["DaysOn"]>0, raw["Oil"]/raw["DaysOn"], 0)
        raw["Water_Daily"] = np.where(raw["DaysOn"]>0, raw["Water"]/raw["DaysOn"], 0)
        raw["Gas_Daily"]   = np.where(raw["DaysOn"]>0, raw["Gas"]/raw["DaysOn"], 0)
        total = raw["Oil"]+raw["Water"]
        raw["Water_Cut"]   = np.where(total>0, raw["Water"]/total, 0)
        if raw["Gas"].sum() > 0:
            raw["GOR"] = np.where(raw["Oil_Daily"]>0, raw["Gas_Daily"]/raw["Oil_Daily"], np.nan)
        else:
            raw["GOR"] = np.nan
        return raw
    jan = read_sheet("JANUARY","Jan-2026",31)
    feb = read_sheet("FEBRUARY","Feb-2026",28)
    passed_j, errors_j, warnings_j = validate_schema(jan, FLORIDA_SCHEMA, "Florida DEP (January)")
    show_validation(passed_j, errors_j, warnings_j, "Florida DEP (January)")
    passed_f, errors_f, warnings_f = validate_schema(feb, FLORIDA_SCHEMA, "Florida DEP (February)")
    show_validation(passed_f, errors_f, warnings_f, "Florida DEP (February)")
    active = pd.concat([jan,feb], ignore_index=True)
    active = active[active["DaysOn"]>0].copy()
    shutin = pd.concat([jan,feb], ignore_index=True)
    shutin = shutin[shutin["Shut_In"]].copy()
    active = active.sort_values(["WellName","Date"]).reset_index(drop=True)
    active["WC_Change"]        = active.groupby("WellName")["Water_Cut"].diff()
    active["Oil_Change_Daily"] = active.groupby("WellName")["Oil_Daily"].diff()
    cfg_live = BREAKTHROUGH_CONFIG["live"]
    active["Sustained_BT"] = active["Water_Cut"] >= cfg_live["high_wc"]
    active["Sudden_BT"]    = (
        (active["WC_Change"] >= cfg_live["wc_jump"]) &
        (active["Water_Cut"] >= cfg_live["wc_floor"]) &
        (active["Oil_Change_Daily"] < 0)
    )
    def classify(r):
        if r["Sudden_BT"] and r["Sustained_BT"]: return "Sudden + Sustained"
        if r["Sudden_BT"]:    return "Sudden"
        if r["Sustained_BT"]: return "Sustained"
        return None
    active["Breakthrough_Type"] = active.apply(classify, axis=1)
    active["Wellbore name"] = active["WellName"]
    active["Mode"]    = "live"
    active["Dataset"] = "Florida DEP (Live 2026)"
    return active, shutin

# ─────────────────────────────
# analytics helpers
# ─────────────────────────────
def compute_forecast(well_df, periods=6):
    df_sorted = well_df.dropna(subset=["Water_Cut"]).sort_values("Date")
    if len(df_sorted) < 6:
        return None, None
    recent = df_sorted.tail(6).copy()
    recent["t"] = np.arange(len(recent))
    slope = np.polyfit(recent["t"], recent["Water_Cut"], 1)
    last_date = df_sorted["Date"].iloc[-1]
    last_wc   = df_sorted["Water_Cut"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq="MS")[1:]
    future_wc    = [min(last_wc + slope[0]*(i+1), 1.0) for i in range(periods)]
    return future_dates, future_wc

def compute_field_baseline(df):
    return df.groupby("Date")["Water_Cut"].mean().reset_index()

def health_score(row, df_full):
    score = 100

    wc = row.get("Water_Cut", 0)

    # water cut severity
    if wc > 0.95:   score -= 40
    elif wc > 0.90: score -= 30
    elif wc > 0.80: score -= 20
    elif wc > 0.60: score -= 10

    # breakthrough penalty
    bt = row.get("Breakthrough_Type", None)
    if bt == "Sudden + Sustained": score -= 25
    elif bt == "Sudden":           score -= 15
    elif bt == "Sustained":        score -= 10

    # well history
    well_rows = df_full[df_full["Wellbore name"]==row["Wellbore name"]].sort_values("Date")

    # water cut trend — None-safe (0.0 is valid, not falsy)
    recent_chg = None
    if "WC_Change" in well_rows.columns and len(well_rows) > 0:
        recent_chg = well_rows["WC_Change"].tail(3).mean()
    if recent_chg is not None:
        if recent_chg > 0.05:   score -= 15
        elif recent_chg > 0.01: score -= 5

    # oil decline — None-safe
    oil_col = "Oil_Change_Daily" if "Oil_Change_Daily" in well_rows.columns else "Oil_Change"
    oil_chg = None
    if oil_col in well_rows.columns and len(well_rows) > 0:
        oil_chg = well_rows[oil_col].tail(3).mean()
    if oil_chg is not None and oil_chg < 0:
        score -= 10

    return max(0, min(100, score))

def data_confidence(row):
    """
    How reliable is the data for this well?
    Separate from health score — distinguishes a bad well from bad data.
    """
    score = 100
    if row.get("Partial_Month", False):   score -= 20
    if row.get("DaysOn", 31) < 20:        score -= 15
    if pd.isna(row.get("GOR", None)):     score -= 10
    return max(0, score)

# ─────────────────────────────
# charts
# ─────────────────────────────
def base_layout(title, y1_title, height=340):
    return dict(
        title=dict(text=title, font=dict(size=12, color="#9ea3a8"), x=0),
        height=height,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Mono", color="#9ea3a8", size=10),
        yaxis=dict(title=y1_title, gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=0,r=0,t=36,b=0),
    )

def chart_historical_trend(well_df, selected, events, df_all):
    fig = go.Figure()
    baseline = compute_field_baseline(df_all)
    fig.add_trace(go.Scatter(
        x=baseline["Date"], y=baseline["Water_Cut"]*100,
        name="Field avg", line=dict(color="#3d3944", dash="dot", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=well_df["Date"], y=well_df["WC_Rolling"]*100,
        name="3-month avg", line=dict(color=AMBER, dash="dash", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=well_df["Date"], y=well_df["Water_Cut"]*100,
        name="Water Cut (%)", line=dict(color=BLUE, width=2)
    ))
    if not events.empty:
        fig.add_trace(go.Scatter(
            x=events["Date"], y=events["Water_Cut"]*100,
            mode="markers", name="Breakthrough",
            marker=dict(color=RED, size=9, symbol="x-thin", line=dict(width=2))
        ))
    fut_dates, fut_wc = compute_forecast(well_df)
    if fut_dates is not None:
        fig.add_trace(go.Scatter(
            x=fut_dates, y=[v*100 for v in fut_wc],
            name="Linear forecast", line=dict(color=PURPLE, dash="dot", width=1.5), opacity=0.7
        ))
        fig.add_vrect(
            x0=well_df["Date"].iloc[-1], x1=fut_dates[-1],
            fillcolor=PURPLE, opacity=0.05, line_width=0
        )
    fig.add_hline(y=80, line_dash="dot", line_color=RED, opacity=0.4,
                  annotation_text="80% threshold", annotation_font_size=9)
    fig.update_layout(**base_layout(f"Water Cut Trend 2007–2016 — {selected}", "Water Cut (%)"))
    return fig

def chart_historical_dual(well_df, selected):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(
        x=well_df["Date"], y=well_df["Water_Cut"]*100,
        name="Water Cut (%)", line=dict(color=BLUE, width=2)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=well_df["Date"], y=well_df["Oil"],
        name="Oil Rate (Sm³/day)", line=dict(color=AMBER, width=2)
    ), secondary_y=True)
    fig.update_layout(**base_layout(f"Oil Rate vs Water Cut — {selected}", "Water Cut (%)"))
    fig.update_yaxes(title_text="Water Cut (%)", secondary_y=False, gridcolor=GRID_COL)
    fig.update_yaxes(title_text="Oil Rate (Sm³/day)", secondary_y=True, gridcolor=GRID_COL)
    return fig

def chart_gor(well_df, selected, mode):
    if "GOR" not in well_df.columns or well_df["GOR"].dropna().empty or well_df["GOR"].dropna().sum() == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=well_df["Date"], y=well_df["GOR"],
        name="GOR", line=dict(color=GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(102,187,106,0.1)"
    ))
    unit = "Sm³/Sm³" if mode=="historical" else "Mcf/bbl"
    fig.update_layout(**base_layout(f"Gas-Oil Ratio — {selected}", f"GOR ({unit})"))
    return fig

def chart_live_bars(well_df, selected, df_all):
    months  = well_df["Month"].tolist() if "Month" in well_df.columns else ["Period 1","Period 2"]
    wc_vals = (well_df["Water_Cut"]*100).tolist()
    colors_bar = [RED if v>80 else AMBER if v>60 else BLUE for v in wc_vals]
    field_avg  = df_all[df_all["DaysOn"]>0].groupby("Month")["Water_Cut"].mean()*100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months, y=wc_vals, name="Water Cut (%)",
        marker_color=colors_bar, marker_line_width=0
    ))
    avg_vals = [field_avg.get(m, None) for m in months]
    fig.add_trace(go.Scatter(
        x=months, y=avg_vals, name="Field avg",
        mode="markers+lines", line=dict(color="#9ea3a8", dash="dot", width=1.5),
        marker=dict(size=8, color="#9ea3a8")
    ))
    fig.add_hline(y=80, line_dash="dot", line_color=RED, opacity=0.4,
                  annotation_text="80% threshold", annotation_font_size=9)
    fig.update_layout(**base_layout(
        f"Water Cut Snapshot — Jan vs Feb 2026 — {selected[:20]}", "Water Cut (%)"))
    return fig

def chart_live_dual(well_df, selected):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    months = well_df["Month"].tolist() if "Month" in well_df.columns else ["Period 1","Period 2"]
    fig.add_trace(go.Bar(
        x=months, y=well_df["Water_Cut"]*100,
        name="Water Cut (%)", marker_color=BLUE, opacity=0.8
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=months, y=well_df["Oil_Daily"],
        name="Oil Rate (bbl/day)", marker_color=AMBER, opacity=0.8
    ), secondary_y=True)
    fig.update_layout(**base_layout(f"Oil Rate vs Water Cut — {selected[:20]}", "Water Cut (%)"))
    fig.update_yaxes(title_text="Water Cut (%)", secondary_y=False, gridcolor=GRID_COL)
    fig.update_yaxes(title_text="Oil Rate (bbl/day)", secondary_y=True, gridcolor=GRID_COL)
    return fig

def chart_bubble_all(latest_df, mode):
    oil_col   = "Oil_Daily" if "Oil_Daily" in latest_df.columns else "Oil"
    oil_label = "Oil Rate (bbl/day)" if "Oil_Daily" in latest_df.columns else "Oil Rate (Sm³/day)"
    status = latest_df["Water_Cut"].apply(
        lambda x: "Critical (>90%)" if x>0.9 else "High (>80%)" if x>0.8 else "Normal"
    )
    color_map = {"Critical (>90%)":RED,"High (>80%)":AMBER,"Normal":GREEN}
    fig = go.Figure()
    for s, color in color_map.items():
        mask = status==s
        if mask.any():
            sub = latest_df[mask]
            fig.add_trace(go.Scatter(
                x=sub["Water_Cut"]*100, y=sub[oil_col],
                mode="markers+text", name=s,
                text=sub["Wellbore name"].str[-10:],
                textposition="top center",
                textfont=dict(size=8, color="#9ea3a8"),
                marker=dict(color=color, size=12, opacity=0.85,
                            line=dict(width=1, color="rgba(0,0,0,0.3)"))
            ))
    fig.add_vline(x=80, line_dash="dot", line_color=RED, opacity=0.3)
    title = "All Wells — Water Cut vs Oil Rate" + (
        " (Historical)" if mode=="historical" else " (Jan–Feb 2026)")
    fig.update_layout(**base_layout(title, oil_label, height=360))
    fig.update_xaxes(title_text="Water Cut (%)", gridcolor=GRID_COL)
    return fig

# ─────────────────────────────
# tools
# ─────────────────────────────
def tool_well_status(well_name, df, mode):
    w = df[df["Wellbore name"]==well_name]
    if w.empty: return f"Well {well_name} not found."
    r   = w.sort_values("Date").iloc[-1]
    wc  = r["Water_Cut"]
    chg = r.get("WC_Change", 0) or 0
    bt  = r.get("Breakthrough_Type","None") or "None"
    gor = r.get("GOR", None)
    gor_str = f"\nGOR: {gor:.2f}" if gor is not None and not pd.isna(gor) else ""
    if mode == "historical":
        oil_str = f"{r.get('Oil',0):,.0f} Sm³/day"
        period  = "most recent month in dataset"
    else:
        oil_str = f"{r.get('Oil_Daily',0):.1f} bbl/day (normalised)"
        period  = r.get("Month","latest month")
        if r.get("Partial_Month"):
            oil_str += f" — partial month ({r.get('DaysOn','?')} days only)"
    return (
        f"Well: {well_name}\nPeriod: {period}\n"
        f"Water Cut: {wc*100:.1f}%\nWC Change: {chg*100:+.2f}%\n"
        f"Oil Rate: {oil_str}{gor_str}\nBreakthrough: {bt}\n"
        f"Status: {'CRITICAL' if wc>0.9 else 'HIGH' if wc>0.75 else 'NORMAL'}"
    )

def tool_critical_wells(df, mode):
    latest = df.sort_values("Date").groupby("Wellbore name").last().reset_index()
    crit   = latest[(latest["Water_Cut"]>0.80)|(latest["Breakthrough_Type"].notna())]
    crit   = crit.sort_values("Water_Cut", ascending=False)
    if crit.empty: return "No critical wells detected."
    context = "historical trend analysis" if mode=="historical" else "Jan–Feb 2026 snapshot"
    lines = [f"Critical wells ({context}):"]
    for _,r in crit.head(8).iterrows():
        p = " [partial month]" if r.get("Partial_Month") else ""
        lines.append(f"- {r['Wellbore name']}: WC={r['Water_Cut']*100:.1f}%, {r.get('Breakthrough_Type','—')}{p}")
    return "\n".join(lines)

def tool_data_quality(df, mode, shutin=None):
    lines = [f"Data Quality — {mode.upper()} MODE"]
    lines.append(f"Active wells: {df['Wellbore name'].nunique()}")
    if shutin is not None:
        lines.append(f"Shut-in excluded: {shutin['WellName'].nunique()} (DaysOn=0)")
    if mode == "live":
        partial = df[df["Partial_Month"]==True] if "Partial_Month" in df.columns else pd.DataFrame()
        if len(partial)>0:
            lines.append(f"Partial month wells ({len(partial)} records — volumes normalised):")
            for _,r in partial.drop_duplicates("WellName").iterrows():
                lines.append(f"  {r['WellName']}: {r.get('DaysOn','?')} days in {r.get('Month','?')}")
        lines.append("Water cut is ratio-based — reliable for all wells.")
    else:
        lines.append("298 clean monthly records across 6 producer wells.")
        cfg = BREAKTHROUGH_CONFIG["historical"]
        lines.append(f"Sudden threshold: WC jump >= {cfg['wc_jump']*100:.0f}%/month")
        lines.append(f"Sustained threshold: rolling avg >= {cfg['rolling_wc']*100:.0f}%")
    return "\n".join(lines)

def tool_field_summary(df, mode, shutin=None):
    latest = df.sort_values("Date").groupby("Wellbore name").last().reset_index()
    crit   = latest[latest["Water_Cut"]>0.80]
    context = "Full field history 2007–2016" if mode=="historical" else "Snapshot Jan–Feb 2026"
    return (
        f"Field Summary ({context}):\n"
        f"Active wells: {len(latest)}\n"
        f"Shut-in excluded: {shutin['WellName'].nunique() if shutin is not None else 'N/A'}\n"
        f"High water cut (>80%): {len(crit)}\n"
        f"Avg water cut: {latest['Water_Cut'].mean()*100:.1f}%\n"
        f"Highest WC: {latest.loc[latest['Water_Cut'].idxmax(),'Wellbore name']} "
        f"({latest['Water_Cut'].max()*100:.1f}%)\n"
        f"Dataset: {df['Dataset'].iloc[0]}"
    )

TOOLS = [
    {"name":"well_status",
     "description":"Get water cut, oil rate, GOR, breakthrough status for a specific well",
     "input_schema":{"type":"object","properties":{"well_name":{"type":"string"}},"required":["well_name"]}},
    {"name":"critical_wells",
     "description":"List wells with high water cut or breakthrough flags",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"data_quality",
     "description":"Get data quality report — exclusions, normalisations, threshold values",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"field_summary",
     "description":"Field-wide overview — active wells, average water cut, critical count",
     "input_schema":{"type":"object","properties":{}}},
]

def run_agent(question, df, mode, history, shutin=None):
    # safety guard — app works without API key, chat is disabled
    if client is None:
        return "Chat is disabled. Add ANTHROPIC_API_KEY to your .env file or Streamlit secrets to enable this feature."
    system = (
        f"You are VannVakt. You help production engineers read well data. "
        f"Current mode: {mode.upper()}. "
        f"{'This is historical data (2007-2016). You can discuss trends, GOR, breakthrough events, and long-term patterns.' if mode=='historical' else 'This is a live snapshot (Jan-Feb 2026 only). Only 2 data points per well. Focus on current state and month-on-month change — not trends.'} "
        f"Use tools to get numbers before you answer. "
        f"If a well has partial month data, say so. Keep answers short and specific."
    )
    msgs = [{"role":m["role"],"content":m["content"]} for m in history if m["role"] in {"user","assistant"}]
    msgs.append({"role":"user","content":question})
    for _ in range(5):
        resp = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=500,
            system=system, tools=TOOLS, messages=msgs
        )
        if resp.stop_reason == "end_turn":
            return "\n".join(b.text for b in resp.content if hasattr(b,"text")).strip()
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                n, inp = b.name, b.input
                if n=="well_status":      r = tool_well_status(inp["well_name"], df, mode)
                elif n=="critical_wells": r = tool_critical_wells(df, mode)
                elif n=="data_quality":   r = tool_data_quality(df, mode, shutin)
                elif n=="field_summary":  r = tool_field_summary(df, mode, shutin)
                else: r = f"Unknown tool: {n}"
                results.append({"type":"tool_result","tool_use_id":b.id,"content":r})
        msgs.append({"role":"assistant","content":resp.content})
        msgs.append({"role":"user","content":results})
    return "Unable to complete analysis."

# ─────────────────────────────
# sidebar
# ─────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-block">'
        '<p class="brand-title">💧 VannVakt</p>'
        '<p class="brand-sub">Water Cut Intelligence</p>'
        '</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    dataset = st.radio("", ["Volve NCS (Historical)", "Florida DEP (Live 2026)"],
                       label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Well</div>', unsafe_allow_html=True)

# ─────────────────────────────
# load
# ─────────────────────────────
shutin_df = None
if "Florida" in dataset:
    try:
        df, shutin_df = load_florida()
    except Exception as e:
        st.error(f"Florida data error: {e}"); st.stop()
else:
    try:
        df = load_volve()
    except Exception as e:
        st.error(f"Volve data error: {e}"); st.stop()

mode = detect_mode(df)

with st.sidebar:
    wells    = sorted(df["Wellbore name"].unique())
    selected = st.selectbox("", wells, label_visibility="collapsed")
    st.markdown('<div class="sidebar-label">Well Info</div>', unsafe_allow_html=True)
    if mode == "historical":
        m = VOLVE_META.get(selected, {})
        st.markdown(f"""<div class="meta-block">
            <div class="meta-row"><span class="meta-key">Type</span><span class="meta-val">{m.get('type','—')}</span></div>
            <div class="meta-row"><span class="meta-key">Trajectory</span><span class="meta-val">{m.get('trajectory','—')}</span></div>
            <div class="meta-row"><span class="meta-key">Reservoir</span><span class="meta-val">{m.get('reservoir','—')}</span></div>
            <div class="meta-row"><span class="meta-key">Drive</span><span class="meta-val">{m.get('drive','—')}</span></div>
            <div class="meta-row"><span class="meta-key">Spud</span><span class="meta-val">{m.get('spud','—')}</span></div>
            <div class="meta-row"><span class="meta-key">TD</span><span class="meta-val">{m.get('td_m','—')} m</span></div>
            <div class="meta-row"><span class="meta-key">Field</span><span class="meta-val">Volve, Block 15/9</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        w_row    = df[df["Wellbore name"]==selected].iloc[0] if len(df[df["Wellbore name"]==selected])>0 else None
        county   = w_row["County"]    if w_row is not None else "—"
        field    = w_row["OilField"]  if w_row is not None else "—"
        operator = str(w_row["Operator"])[:22] if w_row is not None else "—"
        api      = w_row["APINumber"] if w_row is not None else "—"
        st.markdown(f"""<div class="meta-block">
            <div class="meta-row"><span class="meta-key">County</span><span class="meta-val">{county}</span></div>
            <div class="meta-row"><span class="meta-key">Field</span><span class="meta-val">{field}</span></div>
            <div class="meta-row"><span class="meta-key">Operator</span><span class="meta-val">{operator}</span></div>
            <div class="meta-row"><span class="meta-key">API No.</span><span class="meta-val">{api}</span></div>
            <div class="meta-row"><span class="meta-key">Reservoir</span><span class="meta-na">not available</span></div>
            <div class="meta-row"><span class="meta-key">Drive</span><span class="meta-na">not available</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-block">
        <p class="about-text">Tracks water breakthrough and production decline across multiple signals.</p>
        <p class="source-link" style="margin-top:8px;">· Volve NCS — Equinor (open)<br>· Florida DEP — 2026 (public)</p>
    </div>""", unsafe_allow_html=True)
    if not _api_key:
        st.warning("No API key — chat disabled", icon="⚠")

# ─────────────────────────────
# main
# ─────────────────────────────
if mode == "historical":
    st.markdown('<span class="mode-badge mode-historical">HISTORICAL ANALYSIS MODE — Volve NCS 2007–2016</span>', unsafe_allow_html=True)
    st.markdown('<div class="mode-note">Complete 9-year dataset. Multi-signal analysis: water cut trend, oil decline, GOR, field baseline deviation, and linear forecast. Thresholds defined in BREAKTHROUGH_CONFIG.</div>', unsafe_allow_html=True)
else:
    st.markdown('<span class="mode-badge mode-live">LIVE SNAPSHOT MODE — Florida DEP Jan–Feb 2026</span>', unsafe_allow_html=True)
    st.markdown('<div class="mode-note">Two months of data only (Jan–Feb 2026). Showing current state, field comparison, and month-on-month change. Trend analysis not applicable with 2 data points. Volumes shown as daily rates (bbl/day).</div>', unsafe_allow_html=True)

# engineering logic expander
with st.expander("Engineering logic — how breakthrough is detected"):
    if mode == "historical":
        cfg = BREAKTHROUGH_CONFIG["historical"]
        st.markdown(f"""
**Sudden breakthrough** — all three must be true:
- Water cut rises ≥ {cfg['wc_jump']*100:.0f}% in one month
- Water cut is already above {cfg['wc_floor']*100:.0f}%
- Oil production is declining

**Sustained breakthrough:**
- 3-month rolling average water cut ≥ {cfg['rolling_wc']*100:.0f}%

**Why these thresholds?**
Natural month-to-month variation in stable Volve wells is below 8%.
A {cfg['wc_jump']*100:.0f}% jump flags a real change, not noise.

**Secondary signals:**
GOR rising = reservoir pressure dropping.
Field baseline deviation = well performing worse than field average.
Linear forecast = direction of travel over next 6 months.
""")
    else:
        cfg = BREAKTHROUGH_CONFIG["live"]
        st.markdown(f"""
**Live snapshot mode — two months of data only (Jan–Feb 2026).**

**Sustained flag:** current water cut ≥ {cfg['high_wc']*100:.0f}%

**Dual-signal flag** — all three must be true:
- Water cut rose ≥ {cfg['wc_jump']*100:.0f}% month-on-month
- Water cut is already above {cfg['wc_floor']*100:.0f}%
- Daily oil rate is declining

**Why different from Volve?**
Florida wells are in late field life — average water cut above 90%.
A 10% change is essentially impossible at that level.
{cfg['wc_jump']*100:.0f}% month-on-month is meaningful at high water cut.

**Volumes** — shown as daily rates (bbl/day = monthly total ÷ DaysOn).
Water cut is ratio-based — not affected by operating days.
""")

# data quality banners
well_df = df[df["Wellbore name"]==selected].sort_values("Date")
if mode == "live":
    if "Partial_Month" in well_df.columns and well_df["Partial_Month"].any():
        for _,pr in well_df[well_df["Partial_Month"]==True].iterrows():
            st.markdown(
                f'<div class="dq-banner">⚠ {selected} — partial month in {pr.get("Month","?")}: '
                f'{pr.get("DaysOn","?")} days only. Water cut reliable. Volumes normalised to daily rates.</div>',
                unsafe_allow_html=True
            )
    if shutin_df is not None:
        with st.expander(f"⚠ {shutin_df['WellName'].nunique()} shut-in wells excluded — click to see why"):
            st.markdown(
                "Wells with DaysOn=0 have no production and no meaningful water cut. "
                "Excluded from analysis, reported here for transparency. "
                "65 wells shut-in in both months — likely permanently inactive."
            )

# KPI row
st.markdown('<div class="section-title">Field Overview</div>', unsafe_allow_html=True)
latest_all = df.sort_values("Date").groupby("Wellbore name").last().reset_index()
active_n   = len(latest_all[latest_all["Water_Cut"]>0])
crit_n     = len(latest_all[latest_all["Water_Cut"]>0.80])
flagged_n  = df[df["Breakthrough_Type"].notna()]["Wellbore name"].nunique()
avg_wc     = latest_all[latest_all["Water_Cut"]>0]["Water_Cut"].mean()

if mode == "live" and shutin_df is not None:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Active Wells", active_n)
    c2.metric("Avg Water Cut", f"{avg_wc*100:.1f}%")
    c3.metric("High WC (>80%)", crit_n, delta="need review", delta_color="inverse")
    c4.metric("Flagged Wells", flagged_n)
    c5.metric("Shut-in (excl.)", shutin_df["WellName"].nunique())
else:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Wells", active_n)
    c2.metric("Records", len(df))
    c3.metric("Wells Flagged", flagged_n)
    peak = latest_all.loc[latest_all["Water_Cut"].idxmax()]
    c4.metric("Highest WC", f"{peak['Water_Cut']*100:.1f}%", delta=str(peak["Wellbore name"]))

# well charts
st.markdown('<div class="section-title">Well Analysis</div>', unsafe_allow_html=True)
events = well_df[well_df["Sudden_BT"]==True] if "Sudden_BT" in well_df.columns else pd.DataFrame()

col1, col2 = st.columns(2)
with col1:
    if mode == "historical":
        st.plotly_chart(chart_historical_trend(well_df, selected, events, df), use_container_width=True)
    else:
        st.plotly_chart(chart_live_bars(well_df, selected, df), use_container_width=True)
with col2:
    if mode == "historical":
        st.plotly_chart(chart_historical_dual(well_df, selected), use_container_width=True)
    else:
        st.plotly_chart(chart_live_dual(well_df, selected), use_container_width=True)

# GOR chart
gor_fig = chart_gor(well_df, selected, mode)
if gor_fig is not None:
    st.markdown('<div class="section-title">Gas-Oil Ratio</div>', unsafe_allow_html=True)
    st.caption("Rising GOR can indicate reservoir pressure decline — a secondary signal alongside water cut.")
    st.plotly_chart(gor_fig, use_container_width=True)

# all wells bubble
st.plotly_chart(chart_bubble_all(latest_all[latest_all["Water_Cut"]>0], mode), use_container_width=True)

# forecast note
if mode == "historical":
    fut_dates, fut_wc = compute_forecast(well_df)
    if fut_dates is not None and fut_wc:
        st.markdown('<div class="section-title">Water Cut Forecast</div>', unsafe_allow_html=True)
        last_wc  = well_df["Water_Cut"].iloc[-1]*100
        proj_wc  = fut_wc[-1]*100
        direction = "rising" if proj_wc > last_wc else "stable or falling"
        st.markdown(
            f'<div class="diag-card info">Linear projection based on last 6 months of data. '
            f'If the current trend continues, water cut reaches <b>{proj_wc:.1f}%</b> '
            f'in 6 months ({direction}). '
            f'Straight-line extrapolation only — not a reservoir simulation.</div>',
            unsafe_allow_html=True
        )

# diagnosis
st.markdown('<div class="section-title">Diagnosis</div>', unsafe_allow_html=True)
latest  = well_df.iloc[-1]
wc      = latest["Water_Cut"]
bt      = latest.get("Breakthrough_Type", None)
actions = []
cfg_d   = BREAKTHROUGH_CONFIG[mode] if mode in BREAKTHROUGH_CONFIG else BREAKTHROUGH_CONFIG["historical"]

if mode == "historical":
    wc_chg  = (well_df["WC_Change"].tail(3).mean() or 0) if "WC_Change" in well_df.columns else 0
    oil_chg = (well_df["Oil_Change"].tail(3).mean() or 0) if "Oil_Change" in well_df.columns else 0
    if wc > cfg_d["wc_high"] and oil_chg < 0:
        st.markdown('<div class="diag-card">Water cut above 80% with declining oil — breakthrough is affecting this well.</div>', unsafe_allow_html=True)
        actions.append("Review water shut-off or zonal isolation options")
    if wc_chg and wc_chg > cfg_d["wc_trend"]:
        st.markdown('<div class="diag-card warn">Water cut rising faster than 5% per month — check for coning or channelling.</div>', unsafe_allow_html=True)
        actions.append("Monitor closely — escalate if trend continues")
    if wc > cfg_d["wc_high"] and wc_chg and abs(wc_chg) < 0.01:
        st.markdown('<div class="diag-card info">Water cut is high but stable — looks like mature water production rather than a new event.</div>', unsafe_allow_html=True)
        actions.append("Continue monitoring — assess economics of continued production")
    if bt:
        st.markdown(f'<div class="diag-card">Breakthrough event on record: <b>{bt}</b>.</div>', unsafe_allow_html=True)
    if not (wc>cfg_d["wc_high"] or (wc_chg and wc_chg>cfg_d["wc_trend"]) or bt):
        st.markdown('<div class="diag-card ok">Water cut and oil rate look stable. No anomaly flags triggered.</div>', unsafe_allow_html=True)
        actions.append("Continue routine monitoring")
else:
    wc_chg  = well_df["WC_Change"].dropna().iloc[-1] if "WC_Change" in well_df.columns and well_df["WC_Change"].notna().any() else None
    oil_chg = well_df["Oil_Change_Daily"].dropna().iloc[-1] if "Oil_Change_Daily" in well_df.columns and well_df["Oil_Change_Daily"].notna().any() else None
    if latest.get("Partial_Month", False):
        st.markdown(f'<div class="diag-card info">Partial month data ({latest.get("DaysOn","?")} days). Water cut reliable. Oil normalised to daily rate.</div>', unsafe_allow_html=True)
    if wc > cfg_d["wc_critical"]:
        st.markdown(f'<div class="diag-card">Water cut at {wc*100:.1f}% — this well is producing mostly water. Review economics.</div>', unsafe_allow_html=True)
        actions.append("Evaluate water shut-off or workover — check if production is still economic")
    elif wc > cfg_d["wc_high"]:
        st.markdown(f'<div class="diag-card warn">Water cut at {wc*100:.1f}% — most of what this well produces is water.</div>', unsafe_allow_html=True)
        actions.append("Review well economics — assess cost of continued water handling")
    if wc_chg is not None and wc_chg > cfg_d["wc_jump"]:
        st.markdown(f'<div class="diag-card warn">Water cut rose {wc_chg*100:.2f}% from January to February.</div>', unsafe_allow_html=True)
        actions.append("Watch for continued rise next month")
    if oil_chg is not None and oil_chg < 0:
        st.markdown(f'<div class="diag-card warn">Daily oil rate fell {abs(oil_chg):.1f} bbl/day from January to February.</div>', unsafe_allow_html=True)
        actions.append("Review production parameters")
    if bt:
        st.markdown(f'<div class="diag-card">Dual signal detected: <b>{bt}</b> — rising water cut and declining oil together.</div>', unsafe_allow_html=True)
    if not (wc>cfg_d["wc_high"] or (wc_chg and wc_chg>cfg_d["wc_jump"]) or (oil_chg and oil_chg<0) or bt):
        st.markdown('<div class="diag-card ok">Water cut and oil rate look stable between January and February.</div>', unsafe_allow_html=True)
        actions.append("Continue routine monitoring")

if actions:
    st.markdown("**Recommended actions:**")
    for a in actions:
        st.markdown(f'<div class="action-card">→ {a}</div>', unsafe_allow_html=True)

# well health score + data confidence
st.markdown('<div class="section-title">Well Health Score</div>', unsafe_allow_html=True)
st.caption("Health: 0 = critical · 100 = stable. Confidence: how reliable the data is for this well.")

health = latest_all[latest_all["Water_Cut"]>0].copy()
health["Health_Score"] = health.apply(lambda r: health_score(r, df), axis=1).round(0).astype(int)
health["Health_Rating"] = health["Health_Score"].apply(
    lambda s: "🔴 Critical" if s<40 else "🟠 Watch" if s<70 else "🟢 Stable"
)
health["Confidence"] = health.apply(data_confidence, axis=1).astype(int)
health["Confidence_Label"] = health["Confidence"].apply(
    lambda s: "✅ High" if s>=90 else "⚠ Medium" if s>=70 else "⚠ Low"
)
health_display = health[["Wellbore name","Health_Score","Health_Rating","Confidence","Confidence_Label","Water_Cut","Breakthrough_Type"]].copy()
health_display["Water_Cut"] = (health_display["Water_Cut"]*100).round(1)
health_display.columns = ["Well","Health Score","Rating","Data Confidence","Reliability","Water Cut (%)","Breakthrough"]
st.dataframe(health_display.sort_values("Health Score"), use_container_width=True, hide_index=True)

# summary table
st.markdown('<div class="section-title">All Wells Summary</div>', unsafe_allow_html=True)
oil_col = "Oil_Daily" if "Oil_Daily" in latest_all.columns else "Oil"
oil_lbl = "Oil (bbl/day)" if "Oil_Daily" in latest_all.columns else "Oil (Sm³/day)"
summary = latest_all[["Wellbore name","Water_Cut",oil_col,"Breakthrough_Type"]].copy()
summary["Water_Cut"] = (summary["Water_Cut"]*100).round(1)
summary[oil_col]     = summary[oil_col].round(1)
summary["Status"]    = summary["Water_Cut"].apply(
    lambda x: "🔴 Critical" if x>90 else "🟠 High" if x>75 else "🟢 Normal")
summary.columns = ["Well","Water Cut (%)",oil_lbl,"Breakthrough","Status"]
st.dataframe(summary.sort_values("Water Cut (%)",ascending=False), use_container_width=True, hide_index=True)

# ─────────────────────────────
# chat
# ─────────────────────────────
if "messages" not in st.session_state or st.session_state.get("last_mode") != mode:
    st.session_state.messages  = [{"role":"assistant","content":
        "Ask me about any well, breakthrough signals, or field status.\n\n"
        "Try: *Which wells are most critical?* or *What is the status of well X?*"}]
    st.session_state.last_mode = mode

st.markdown('<div class="section-title">Ask VannVakt</div>', unsafe_allow_html=True)

if not _api_key:
    st.warning(
        "Chat is disabled — no API key found. "
        "Add ANTHROPIC_API_KEY to your .env file to enable questions.",
        icon="⚠"
    )
else:
    mode_hint = (
        "Trend analysis · GOR · Breakthrough detection · Forecast · 2007–2016"
        if mode == "historical" else
        "Current state · Field comparison · Month-on-month · Jan–Feb 2026"
    )
    st.caption(mode_hint)

    for msg in st.session_state.messages:
        avatar = "💧" if msg["role"] == "assistant" else "🛢️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if not _api_key:
        st.warning(
            "Chat is disabled — no API key found. "
            "Add ANTHROPIC_API_KEY to your .env file to enable questions.",
            icon="⚠️"
        )

if _api_key:
    prompt = st.chat_input("Ask about wells, water cut, GOR, or data quality...")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="🛢️"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="💧"):
            with st.spinner(""):
                reply = run_agent(prompt, df, mode, st.session_state.messages[:-1], shutin_df)
            st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})

st.markdown("---")
st.caption("VannVakt · Built by Anuri Nwagbara · github.com/Anuri-ops")
