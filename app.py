import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.prediction import load_model as _load_model, predict as _predict

st.set_page_config(
    page_title="CarWise - Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'menu_open' not in st.session_state:
    st.session_state.menu_open = False

def toggle_menu():
    st.session_state.menu_open = not st.session_state.menu_open

def navigate_to(page):
    st.session_state.page = page
    st.session_state.menu_open = False

# ----------------------------------------
# PALETTE
# ----------------------------------------
INK, NAVY, BLUE, PAPER = "#1B2430", "#2F3B4C", "#3E6E9E", "#EDF1F5"
TREND = "#6D8FB3"   # muted blue trendline
SOFT  = "#9DB7D1"   # light blue accent

# ----------------------------------------
# DATA & MODEL LOADING
# ----------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/car_details.csv")
    df = df.drop_duplicates()
    df["brand"] = df["name"].str.split().str[0]
    df["brand"] = df["brand"].replace({"Land": "Land Rover", "OpelCorsa": "Opel"})
    df["car_age"] = 2020 - df["year"]
    return df

@st.cache_resource
def load_model():
    return _load_model()

# ----------------------------------------
# CHART HELPERS
# ----------------------------------------
def add_numpy_trendline(fig, x_data, y_data, color=TREND):
    """Add a linear OLS trendline using numpy - no statsmodels required."""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if mask.sum() < 2:
        return fig
    coeffs = np.polyfit(x_data[mask], y_data[mask], 1)
    x_line = np.linspace(x_data[mask].min(), x_data[mask].max(), 200)
    y_line = np.polyval(coeffs, x_line)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name="Trend",
        showlegend=True,
    ))
    return fig

def style_fig(fig, height=360, yaxis=None, xaxis=None, **extra):
    """One consistent Plotly style. Axis overrides are MERGED (no duplicate kwargs)."""
    grid = dict(gridcolor="rgba(62,110,158,0.12)", linecolor="rgba(62,110,158,0.2)",
                zeroline=False, automargin=True)
    fig.update_layout(
        height=height, autosize=True,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=15, color=INK), x=0.02, xanchor="left"),
        margin=dict(l=16, r=16, t=56, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#FFFFFF")),
        **extra,
    )
    fig.update_xaxes(**{**grid, **(xaxis or {})})
    fig.update_yaxes(**{**grid, **(yaxis or {})})
    return fig

def chart_card(fig):
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False, "responsive": True})

# ----------------------------------------
# GLOBAL CSS
# ----------------------------------------
st.markdown("""
<style>
/* 1. GLOBALS */
.stApp { background-color: #EDF1F5; }
.block-container {
    max-width: 1200px !important;
    padding-top: 32px !important;
    padding-bottom: 64px !important;
    margin: 0 auto;
}
header[data-testid="stHeader"] { display: none; }
a.header-anchor { display: none !important; }
* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1B2430; }
p { font-size: 16px; line-height: 1.6; }

/* 2. NAVBAR */
div[data-testid="stHorizontalBlock"]:has(div#brand-logo) {
    background-color: #1B2430; border-radius: 8px; padding: 8px 32px;
    align-items: center; justify-content: space-between;
}
div[data-testid="stHorizontalBlock"]:has(div#brand-logo) > div {
    width: auto !important; flex: 0 1 auto !important;
}
div#brand-logo { color: #EDF1F5 !important; font-size: 28px; font-weight: 800; margin: 0; padding: 8px 0; }
div[data-testid="stHorizontalBlock"]:has(div#brand-logo) button {
    background: transparent !important; border: none !important;
    color: #EDF1F5 !important; font-size: 16px; padding: 8px 16px; box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:has(div#brand-logo) button * { color: #EDF1F5 !important; }
div[data-testid="stHorizontalBlock"]:has(div#brand-logo) button:hover { color: #9DB7D1 !important; }

/* =========================================================
   3. NAV DROPDOWN — FIXED OVERLAY
   IMPORTANT:
   Only target the innermost Streamlit block containing
   #nav-menu-marker. Do NOT affect the whole page.
   ========================================================= */

div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) {
    position: fixed !important;
    top: 104px;
    right: max(16px, calc((100vw - 1200px) / 2 + 32px));
    width: 260px !important;

    background-color: #EDF1F5 !important;
    border: 1px solid rgba(62, 110, 158, 0.2);
    border-radius: 8px;
    padding: 16px;

    box-shadow: 0 16px 40px rgba(27, 36, 48, 0.15);

    z-index: 9999 !important;
}


/* Menu buttons */
div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) button {
    width: 100% !important;

    background: transparent !important;
    border: none !important;

    color: #1B2430 !important;

    text-align: left !important;
    justify-content: flex-start !important;

    padding: 10px 16px !important;

    font-weight: 500;
    border-radius: 6px;
}


/* Hover state */
div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) button:hover {
    background-color: rgba(62, 110, 158, 0.12) !important;
    color: #2F3B4C !important;
}


/* Try Predictor button */
div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) button[kind="primary"] {
    background-color: #3E6E9E !important;
    color: #EDF1F5 !important;

    justify-content: center !important;
    text-align: center !important;

    margin-top: 12px;
}


/* Try Predictor text */
div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) button[kind="primary"] * {
    color: #EDF1F5 !important;
}


/* Try Predictor hover */
div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker):not(
    :has(div[data-testid="stVerticalBlock"] div#nav-menu-marker)
) button[kind="primary"]:hover {
    background-color: #2F3B4C !important;
}

/* 4. BUTTONS */
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background-color: #3E6E9E !important; color: #EDF1F5 !important;
    border: none; padding: 8px 24px; font-weight: 500; border-radius: 6px;
}
div.stButton > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] > button * { color: #EDF1F5 !important; }
div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover { background-color: #2F3B4C !important; }
div.stButton > button[kind="secondary"] {
    background-color: transparent; color: #1B2430 !important;
    border: 1px solid #3E6E9E; padding: 8px 24px; font-weight: 500; border-radius: 6px;
}
div.stButton > button[kind="secondary"]:hover { background-color: rgba(62, 110, 158, 0.12); }

/* ================================================
   DROPDOWNS / INPUTS - readable in all states
   ================================================ */
div[data-baseweb="select"] > div { background:#FFFFFF !important; border:1px solid rgba(62,110,158,0.4) !important; }
div[data-baseweb="select"] * { color:#1B2430 !important; }
div[data-baseweb="select"] svg { fill:#3E6E9E !important; }

div[data-baseweb="popover"] > div { background:#FFFFFF !important; border:1px solid rgba(62,110,158,0.3); border-radius:8px; }
div[data-baseweb="popover"] ul { background:#FFFFFF !important; }
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] { background:#FFFFFF !important; color:#1B2430 !important; }
div[data-baseweb="popover"] [role="option"] * { color:#1B2430 !important; }
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][data-highlighted="true"] { background:rgba(62,110,158,0.14) !important; }
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background:rgba(62,110,158,0.22) !important; font-weight:600; }

div[data-testid="stNumberInput"] input { background:#FFFFFF !important; color:#1B2430 !important; }
div[data-testid="stNumberInput"] button { background:#F3F6F9 !important; color:#1B2430 !important; }
div[data-testid="stNumberInput"] > div > div { border-color: rgba(62,110,158,0.4) !important; }
div[data-testid="stSlider"] label { color: #1B2430 !important; }

/* CHART CARDS */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#FFFFFF; border:1px solid rgba(62,110,158,0.2) !important;
    border-radius:12px; padding:8px 8px 0 8px; overflow:hidden;
    box-shadow:0 8px 24px rgba(27,36,48,0.05);
}
div[data-testid="stPlotlyChart"] { width:100% !important; }

/* 5. SPACING */
.spacer-100 { height: 100px; }
.spacer-60  { height: 60px; }
.spacer-48  { height: 48px; }
.spacer-32  { height: 32px; }

/* HERO */
.hero-eyebrow { font-size: 13px; font-weight: 600; color: #3E6E9E; letter-spacing: 1.5px; margin-bottom: 32px; text-transform: uppercase; }
.hero-title   { font-size: 60px; font-weight: 600; color: #1B2430; line-height: 1.1; margin: 0 0 32px 0; letter-spacing: -1px; }
.hero-desc    { font-size: 17px; color: #2F3B4C; line-height: 1.6; max-width: 90%; margin-bottom: 48px; }

/* HERO DASHBOARD VISUAL */
.dash { background:#FFFFFF; border:1px solid rgba(62,110,158,0.25); border-radius:16px; padding:24px;
        box-shadow:0 24px 48px rgba(27,36,48,0.08); max-width:460px; margin:0 auto; }
.dash-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.dash-brand { font-size:12px; font-weight:700; letter-spacing:2px; }
.dash-tag { font-size:11px; color:#3E6E9E; background:rgba(62,110,158,0.1); padding:3px 10px; border-radius:20px; }
.dash-main { display:flex; align-items:center; gap:16px; background:#EDF1F5; border-radius:12px; padding:16px; }
.dash-car { width:44%; flex-shrink:0; }
.dash-lbl { font-size:12px; color:#3E6E9E; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
.dash-val { font-size:28px; font-weight:700; letter-spacing:-0.5px; }
.dash-sub { font-size:12px; color:#2F3B4C; }
.dash-chart { margin-top:16px; }
.dash-chart svg { width:100%; height:70px; display:block; margin-top:6px; }
.dash-kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px; padding-top:16px;
             border-top:1px solid rgba(62,110,158,0.15); }
.dash-kpis b { display:block; font-size:18px; }
.dash-kpis span { font-size:12px; color:#3E6E9E; }

/* SECTION HEADERS */
.sec-title    { font-size: 36px; font-weight: 600; color: #1B2430; margin: 0 0 16px 0; letter-spacing: -0.5px; }
.sec-subtitle { font-size: 16px; color: #2F3B4C; margin: 0 0 48px 0; line-height: 1.6; }

/* CAPABILITIES */
.cap-grid  { display: grid; grid-template-columns: repeat(4, 1fr); gap: 40px; }
.cap-block { display: flex; flex-direction: column; }
.cap-num   { font-size: 13px; font-weight: 600; color: #3E6E9E; margin-bottom: 16px; border-bottom: 1px solid rgba(62, 110, 158,0.2); padding-bottom: 16px; }
.cap-title { font-size: 20px; font-weight: 600; color: #1B2430; margin: 0 0 12px 0; }
.cap-desc  { font-size: 15px; color: #2F3B4C; line-height: 1.5; margin: 0; }

/* KPI STRIP */
.kpi-grid  { display: grid; grid-template-columns: repeat(5, 1fr); gap: 32px; }
.kpi-block { border-left: 2px solid rgba(62, 110, 158,0.3); padding-left: 24px; }
.kpi-val   { font-size: 32px; font-weight: 600; color: #1B2430; margin: 0 0 8px 0; }
.kpi-lbl   { font-size: 15px; color: #2F3B4C; font-weight: 500; margin: 0 0 8px 0; line-height: 1.3; }
.kpi-sub   { font-size: 13px; color: #3E6E9E; margin: 0; line-height: 1.4; display: block; }

/* WORKFLOW */
.wf-grid  { display: grid; grid-template-columns: repeat(8, 1fr); gap: 16px; }
.wf-item  { display: flex; flex-direction: column; position: relative; }
.wf-num   { font-size: 12px; font-weight: 600; color: #EDF1F5; background-color: #3E6E9E; width: 28px; height: 28px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; z-index: 2; }
.wf-num, .wf-num * { color: #EDF1F5 !important; }
.wf-title { font-size: 14px; font-weight: 500; color: #1B2430; line-height: 1.4; }
.wf-line  { position: absolute; top: 14px; left: 28px; right: -16px; height: 1px; background-color: rgba(62, 110, 158,0.3); z-index: 1; }
.wf-item:last-child .wf-line { display: none; }

/* FOOTER */
.footer-wrap     { background-color: #1B2430; padding: 64px 48px; border-radius: 8px; margin-top: 100px; text-align: left; }
.footer-title    { font-size: 24px; font-weight: 600; color: #EDF1F5 !important; margin: 0 0 8px 0; }
.footer-subtitle { font-size: 16px; color: #9DB7D1 !important; margin: 0 0 32px 0; }
.footer-note     { font-size: 15px; color: #EDF1F5 !important; opacity: 0.8; margin: 0; }

/* PREDICTOR RESULT */
.result-panel      { background-color: #1B2430; border-radius: 12px; padding: 40px 48px; margin-top: 32px; }
.result-model-lbl  { font-size: 12px; font-weight: 600; color: #9DB7D1 !important; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px; }
.result-price      { font-size: 52px; font-weight: 600; color: #EDF1F5 !important; margin: 0 0 16px 0; letter-spacing: -1px; }
.result-disclaimer { font-size: 14px; color: rgba(237, 241, 245,0.7) !important; margin: 0; }

/* BACK BUTTON STRIP */
.back-strip { display: flex; align-items: center; margin-bottom: 32px; }

/* VEHICLE SUMMARY GRID */
.summary-grid  { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-top: 32px; }
.summary-item  { border-left: 2px solid rgba(62, 110, 158,0.4); padding-left: 16px; }
.summary-label { font-size: 12px; color: #3E6E9E; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.summary-val   { font-size: 16px; font-weight: 500; color: #1B2430; }

/* SEGMENT CARDS */
.seg-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 48px; }
.seg-card { border: 1px solid rgba(62, 110, 158,0.2); border-radius: 10px; padding: 32px 28px; background-color: white; }
.seg-badge { display: inline-block; font-size: 11px; font-weight: 600; color: #3E6E9E; background-color: rgba(62, 110, 158,0.1); border-radius: 20px; padding: 4px 12px; margin-bottom: 16px; letter-spacing: 0.5px; text-transform: uppercase; }
.seg-name  { font-size: 20px; font-weight: 600; color: #1B2430; margin: 0 0 24px 0; }
.seg-row   { display: flex; justify-content: space-between; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(62, 110, 158,0.1); }
.seg-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.seg-lbl   { font-size: 13px; color: #2F3B4C; }
.seg-val   { font-size: 13px; font-weight: 600; color: #1B2430; }

/* MODEL COMPARISON TABLE */
.model-table { width: 100%; border-collapse: collapse; margin-bottom: 32px; }
.model-table th { font-size: 12px; font-weight: 600; color: #3E6E9E; text-transform: uppercase; letter-spacing: 1px; padding: 12px 16px; border-bottom: 2px solid rgba(62, 110, 158,0.2); text-align: left; }
.model-table td { font-size: 15px; padding: 14px 16px; border-bottom: 1px solid rgba(62, 110, 158,0.1); color: #1B2430; }
.model-table tr.sel td { background-color: rgba(62, 110, 158,0.07); font-weight: 600; }
.model-table tr.sel td:first-child::after { content: " ✓ Selected"; font-size: 11px; color: #3E6E9E; font-weight: 600; }

/* UTILITIES */
.section-divider { height: 1px; background-color: rgba(62, 110, 158,0.15); margin: 64px 0; }
.info-banner { background-color: rgba(62, 110, 158,0.08); border-left: 3px solid #3E6E9E; border-radius: 0 6px 6px 0; padding: 12px 16px; font-size: 14px; color: #2F3B4C; margin-bottom: 16px; }
.warn-banner { background-color: rgba(27, 36, 48,0.06); border-left: 3px solid #1B2430; border-radius: 0 6px 6px 0; padding: 12px 16px; font-size: 14px; color: #1B2430; margin-bottom: 16px; }

/* RESPONSIVE */
@media (max-width: 1024px) {
    .cap-grid { grid-template-columns: repeat(2, 1fr); }
    .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    .wf-grid  { grid-template-columns: repeat(4, 1fr); }
    .wf-line  { display: none; }
    .seg-grid { grid-template-columns: 1fr 1fr; }
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
    .cap-grid { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: 1fr; }
    .kpi-block { border-left: none; border-top: 2px solid rgba(62, 110, 158,0.3); padding-left: 0; padding-top: 16px; }
    .wf-grid  { grid-template-columns: repeat(2, 1fr); }
    .hero-title { font-size: 44px; }
    .seg-grid { grid-template-columns: 1fr; }
    .summary-grid { grid-template-columns: 1fr 1fr; }
    .result-price { font-size: 40px; }
    .dash-main { flex-direction: column; text-align: center; }
    .dash-car { width: 70%; }
    div[data-testid="stHorizontalBlock"]:has(div#brand-logo) { padding: 8px 16px; }
    div[data-testid="stVerticalBlock"]:has(div#nav-menu-marker) { right: 16px; width: calc(100% - 32px); max-width: 260px; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# NAVBAR
# ----------------------------------------
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    st.markdown("<div id='brand-logo'>🚗 CarWise</div>", unsafe_allow_html=True)
with nav_col2:
    st.button("☰ Menu", key="btn_hamburger", on_click=toggle_menu)

# ----------------------------------------
# DROPDOWN MENU
# ----------------------------------------
if st.session_state.menu_open:
    with st.container():
        st.markdown("<div id='nav-menu-marker' style='display:none;'></div>", unsafe_allow_html=True)
        st.button("Home",             on_click=navigate_to, args=("Home",))
        st.button("Price Predictor",  on_click=navigate_to, args=("Price Predictor",))
        st.button("Market Analysis",  on_click=navigate_to, args=("Market Analysis",))
        st.button("Car Segmentation", on_click=navigate_to, args=("Car Segmentation",))
        st.button("Insights",         on_click=navigate_to, args=("Insights",))
        st.markdown("<hr style='margin:12px 0; border:0; border-top:1px solid rgba(62, 110, 158,0.2);'>", unsafe_allow_html=True)
        st.button("Try Predictor", on_click=navigate_to, args=("Price Predictor",), type="primary")


# =================== HOME ===================
if st.session_state.page == "Home":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([1.2, 1], gap="large")
    with h_col1:
        st.markdown("""
            <div class="hero-eyebrow">CARWISE • USED CAR INTELLIGENCE</div>
            <h1 class="hero-title">Know what your car<br>is really worth.</h1>
            <p class="hero-desc">CarWise combines historical CarDekho data, machine learning, market analysis,
            and car segmentation to provide an estimated used-car selling price and meaningful market insights.</p>
        """, unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.button("Predict My Car's Price",  type="primary",   on_click=navigate_to, args=("Price Predictor",), use_container_width=True)
        with btn_c2:
            st.button("Explore Market Analysis", type="secondary", on_click=navigate_to, args=("Market Analysis",), use_container_width=True)
    with h_col2:
        st.markdown("""
<div class="dash">
<div class="dash-top"><span class="dash-brand">CARWISE</span><span class="dash-tag">Sample preview</span></div>
<div class="dash-main">
<svg viewBox="0 0 200 80" class="dash-car" xmlns="http://www.w3.org/2000/svg">
<path d="M10 58 L22 40 Q28 30 42 28 L82 24 Q96 12 120 12 L146 14 Q160 18 170 36 L188 42 Q194 46 194 56 L194 60 L10 60 Z" fill="#2F3B4C"/>
<path d="M92 26 Q102 17 120 17 L142 19 Q150 22 156 34 L92 34 Z" fill="#EDF1F5" opacity=".85"/>
<circle cx="52" cy="62" r="13" fill="#1B2430"/><circle cx="52" cy="62" r="5" fill="#3E6E9E"/>
<circle cx="148" cy="62" r="13" fill="#1B2430"/><circle cx="148" cy="62" r="5" fill="#3E6E9E"/>
</svg>
<div class="dash-val-wrap">
<div class="dash-lbl">Estimated Value</div>
<div class="dash-val">₹ 4,85,000</div>
<div class="dash-sub">Random Forest estimate</div>
</div>
</div>
<div class="dash-chart">
<div class="dash-lbl">Price vs car age</div>
<svg viewBox="0 0 300 70" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#3E6E9E" stop-opacity=".35"/><stop offset="1" stop-color="#3E6E9E" stop-opacity="0"/></linearGradient></defs>
<path d="M0 10 C40 14 70 22 110 32 S190 52 300 62 L300 70 L0 70 Z" fill="url(#g)"/>
<path d="M0 10 C40 14 70 22 110 32 S190 52 300 62" fill="none" stroke="#3E6E9E" stroke-width="2.5"/>
<circle cx="110" cy="32" r="4" fill="#1B2430"/>
</svg>
</div>
<div class="dash-kpis">
<div><b>7</b><span>Features</span></div>
<div><b>3</b><span>Segments</span></div>
<div><b>3,577</b><span>Records</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='spacer-100'></div>", unsafe_allow_html=True)

    st.markdown("<h2 class='sec-title'>Explore CarWise</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="cap-grid">
            <div class="cap-block"><div class="cap-num">01</div><div class="cap-title">Price Predictor</div><p class="cap-desc">Estimate used-car selling prices with the trained Random Forest model.</p></div>
            <div class="cap-block"><div class="cap-num">02</div><div class="cap-title">Market Analysis</div><p class="cap-desc">Explore real pricing patterns in the CarDekho dataset.</p></div>
            <div class="cap-block"><div class="cap-num">03</div><div class="cap-title">Car Segmentation</div><p class="cap-desc">Discover three data-driven used-car segments.</p></div>
            <div class="cap-block"><div class="cap-num">04</div><div class="cap-title">Insights</div><p class="cap-desc">Compare models and understand feature importance.</p></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='spacer-100'></div>", unsafe_allow_html=True)

    st.markdown("<h2 class='sec-title'>Built on Real Data</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="kpi-grid">
            <div class="kpi-block"><div class="kpi-val">3,577</div><div class="kpi-lbl">Clean records</div></div>
            <div class="kpi-block"><div class="kpi-val">8</div><div class="kpi-lbl">Attributes</div></div>
            <div class="kpi-block"><div class="kpi-val">763</div><div class="kpi-lbl">Duplicates removed</div></div>
            <div class="kpi-block"><div class="kpi-val">7</div><div class="kpi-lbl">Prediction features</div><span class="kpi-sub">car_age, km_driven, fuel, seller_type, transmission, owner, brand</span></div>
            <div class="kpi-block"><div class="kpi-val">58.9%</div><div class="kpi-lbl">Test-set R²</div><span class="kpi-sub">Variance explained on test set</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='spacer-100'></div>", unsafe_allow_html=True)

    st.markdown("<h2 class='sec-title'>DWM Workflow</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="wf-grid">
            <div class="wf-item"><div class="wf-num">1</div><div class="wf-title">Data<br>Understanding</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">2</div><div class="wf-title">Data<br>Cleaning</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">3</div><div class="wf-title">Preprocessing</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">4</div><div class="wf-title">EDA</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">5</div><div class="wf-title">K-Means<br>Clustering</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">6</div><div class="wf-title">Regression</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">7</div><div class="wf-title">Model<br>Evaluation</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">8</div><div class="wf-title">Streamlit<br>Dashboard</div></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer-wrap">
            <div class="footer-title">CarWise</div>
            <div class="footer-subtitle">Used Car Price Prediction &amp; Market Analysis</div>
            <div class="footer-note">DWM Mini Project</div>
        </div>
    """, unsafe_allow_html=True)


# =================== PRICE PREDICTOR ===================
elif st.session_state.page == "Price Predictor":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)

    back_col, title_col = st.columns([1, 6])
    with back_col:
        st.button("← Back", on_click=navigate_to, args=("Home",), type="secondary")

    st.markdown("<h1 class='sec-title' style='margin-bottom:8px;'>Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sec-subtitle'>Enter your vehicle details to estimate its selling price. Car age is computed automatically from the year you enter.</p>", unsafe_allow_html=True)

    model, model_err = load_model()
    if model_err:
        st.error(model_err)
        st.stop()

    df_full       = load_data()
    brands        = sorted(df_full["brand"].unique().tolist())
    fuels         = sorted(df_full["fuel"].unique().tolist())
    seller_types  = sorted(df_full["seller_type"].unique().tolist())
    transmissions = sorted(df_full["transmission"].unique().tolist())
    owners        = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    year_min      = int(df_full["year"].min())
    year_max      = int(df_full["year"].max())

    with st.form("predictor_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand     = st.selectbox("Brand", brands, index=None, placeholder="Select brand")
            year      = st.number_input("Year of Manufacture", min_value=1990, max_value=2020, value=2015, step=1,
                                        help="Used to compute car age: car_age = 2020 − year")
            km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=1_000_000, value=50000, step=1000)
            fuel      = st.selectbox("Fuel Type", fuels, index=None, placeholder="Select fuel type")
        with c2:
            seller_type  = st.selectbox("Seller Type", seller_types, index=None, placeholder="Select seller type")
            transmission = st.selectbox("Transmission", transmissions, index=None, placeholder="Select transmission")
            owner        = st.selectbox("Ownership", owners, index=None, placeholder="Select ownership")

        submitted = st.form_submit_button("Estimate Price", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if brand        == "— select —": errors.append("Please select a brand.")
        if fuel         == "— select —": errors.append("Please select a fuel type.")
        if seller_type  == "— select —": errors.append("Please select a seller type.")
        if transmission == "— select —": errors.append("Please select a transmission.")
        if owner        == "— select —": errors.append("Please select an ownership category.")
        if year < year_min or year > year_max:
            errors.append(f"Year must be between {year_min} and {year_max}.")
        if km_driven < 0:
            errors.append("Kilometres driven cannot be negative.")

        if errors:
            for err in errors:
                st.markdown(f"<div class='warn-banner'>{err}</div>", unsafe_allow_html=True)
        else:
            car_age = 2020 - int(year)
            try:
                predicted_price = _predict(
                    model=model,
                    car_age=car_age,
                    km_driven=int(km_driven),
                    fuel=fuel,
                    seller_type=seller_type,
                    transmission=transmission,
                    owner=owner,
                    brand=brand,
                )
                st.markdown(f"""
                    <div class="result-panel">
                        <div class="result-model-lbl">Model: Random Forest</div>
                        <div class="result-price">₹ {predicted_price:,.0f}</div>
                        <div class="result-disclaimer">
                            This is a historical market estimate based on CarDekho data — not a guaranteed selling price.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:13px;font-weight:600;color:#3E6E9E;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;'>Vehicle Summary</p>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="summary-grid">
                        <div class="summary-item"><div class="summary-label">Brand</div><div class="summary-val">{brand}</div></div>
                        <div class="summary-item"><div class="summary-label">Year / Car Age</div><div class="summary-val">{int(year)} &nbsp;·&nbsp; {car_age} yrs old</div></div>
                        <div class="summary-item"><div class="summary-label">Kilometres Driven</div><div class="summary-val">{int(km_driven):,} km</div></div>
                        <div class="summary-item"><div class="summary-label">Fuel</div><div class="summary-val">{fuel}</div></div>
                        <div class="summary-item"><div class="summary-label">Seller Type</div><div class="summary-val">{seller_type}</div></div>
                        <div class="summary-item"><div class="summary-label">Transmission</div><div class="summary-val">{transmission}</div></div>
                        <div class="summary-item"><div class="summary-label">Ownership</div><div class="summary-val">{owner}</div></div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)
                pa_col, hm_col, _ = st.columns([2, 2, 4])
                with pa_col:
                    if st.button("Predict Another Car", type="primary", use_container_width=True):
                        st.rerun()
                with hm_col:
                    st.button("← Back to Home", on_click=navigate_to, args=("Home",),
                              type="secondary", use_container_width=True)

            except Exception as e:
                st.markdown(f"<div class='warn-banner'>Prediction error: {e}</div>", unsafe_allow_html=True)


# =================== MARKET ANALYSIS ===================
elif st.session_state.page == "Market Analysis":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)

    st.button("← Back", on_click=navigate_to, args=("Home",), type="secondary")
    st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)

    st.markdown("<h1 class='sec-title' style='margin-bottom:8px;'>Market Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sec-subtitle'>Explore observed pricing patterns across the CarDekho dataset. Charts reflect historical data — they describe observed relationships, not causal ones.</p>", unsafe_allow_html=True)

    df = load_data()

    with st.expander("🔍 Filters", expanded=True):
        row1_c1, row1_c2, row1_c3, row1_c4, row1_c5 = st.columns(5)
        with row1_c1:
            sel_fuel   = st.selectbox("Fuel Type",    ["All"] + sorted(df["fuel"].unique().tolist()),         key="mkt_fuel")
        with row1_c2:
            sel_trans  = st.selectbox("Transmission", ["All"] + sorted(df["transmission"].unique().tolist()), key="mkt_trans")
        with row1_c3:
            sel_seller = st.selectbox("Seller Type",  ["All"] + sorted(df["seller_type"].unique().tolist()),  key="mkt_seller")
        with row1_c4:
            sel_owner  = st.selectbox("Owner",        ["All"] + sorted(df["owner"].unique().tolist()),        key="mkt_owner")
        with row1_c5:
            sel_brand  = st.selectbox("Brand",        ["All"] + sorted(df["brand"].unique().tolist()),        key="mkt_brand")

        year_range = st.slider(
            "Year Range",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=(int(df["year"].min()), int(df["year"].max())),
            key="mkt_year"
        )

    dff = df.copy()
    if sel_fuel   != "All": dff = dff[dff["fuel"]         == sel_fuel]
    if sel_trans  != "All": dff = dff[dff["transmission"] == sel_trans]
    if sel_seller != "All": dff = dff[dff["seller_type"]  == sel_seller]
    if sel_owner  != "All": dff = dff[dff["owner"]        == sel_owner]
    if sel_brand  != "All": dff = dff[dff["brand"]        == sel_brand]
    dff = dff[(dff["year"] >= year_range[0]) & (dff["year"] <= year_range[1])]

    st.markdown(
        f"<p style='font-size:14px;color:#3E6E9E;margin-bottom:32px;'>"
        f"<strong>{len(dff):,}</strong> vehicles match your filters.</p>",
        unsafe_allow_html=True
    )

    if len(dff) == 0:
        st.markdown("<div class='info-banner'>No data matches the selected filters. Try adjusting them.</div>", unsafe_allow_html=True)
    else:
        sample = dff.sample(min(len(dff), 1000), random_state=42)

        def avg_bar(col, title, color=BLUE, angle=0):
            d = (dff.groupby(col)["selling_price"].mean().reset_index()
                    .sort_values("selling_price", ascending=False))
            f = px.bar(d, x=col, y="selling_price", title=title, color_discrete_sequence=[color],
                       labels={col: col.replace("_", " ").title(), "selling_price": "Avg Selling Price (₹)"})
            f.update_xaxes(tickangle=angle)
            return style_fig(f)

        def scatter(x, title, xl, color):
            f = px.scatter(sample, x=x, y="selling_price", opacity=0.5, title=title,
                           labels={x: xl, "selling_price": "Selling Price (₹)"},
                           color_discrete_sequence=[color])
            add_numpy_trendline(f, sample[x].values, sample["selling_price"].values, color=TREND)
            return style_fig(f)

        brand_counts = (dff.groupby("brand").size().reset_index(name="count")
                           .sort_values("count", ascending=False).head(15))
        f_brand = px.bar(brand_counts, x="count", y="brand", orientation="h",
                         title="Top 15 Brands by Vehicle Count",
                         labels={"count": "Vehicles", "brand": ""},
                         color_discrete_sequence=[BLUE])
        style_fig(f_brand, yaxis=dict(categoryorder="total ascending"))

        f_price = px.histogram(dff, x="selling_price", nbins=50, title="Selling Price Distribution",
                               labels={"selling_price": "Selling Price (₹)"},
                               color_discrete_sequence=[BLUE])
        f_price.update_yaxes(title_text="Number of Cars")

        f_year = px.bar(dff.groupby("year").size().reset_index(name="count"), x="year", y="count",
                        title="Number of Listings by Year",
                        labels={"year": "Year", "count": "Listings"},
                        color_discrete_sequence=[NAVY])

        f_km = px.histogram(dff, x="km_driven", nbins=40, title="Kilometres Driven Distribution",
                            labels={"km_driven": "Kilometres Driven"},
                            color_discrete_sequence=[NAVY])
        f_km.update_yaxes(title_text="Number of Cars")

        grid = [
            (style_fig(f_price), style_fig(f_year)),
            (style_fig(f_km),    scatter("car_age", "Selling Price vs Car Age", "Car Age (years)", BLUE)),
            (scatter("km_driven", "Selling Price vs Kilometres Driven", "Kilometres Driven", NAVY), f_brand),
            (avg_bar("fuel", "Average Price by Fuel Type"),
             avg_bar("transmission", "Average Price by Transmission", NAVY)),
            (avg_bar("seller_type", "Average Price by Seller Type"),
             avg_bar("owner", "Average Price by Owner", NAVY, 20)),
        ]
        for left, right in grid:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                chart_card(left)
            with c2:
                chart_card(right)
            st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)


# =================== CAR SEGMENTATION ===================
elif st.session_state.page == "Car Segmentation":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)
    st.button("← Back", on_click=navigate_to, args=("Home",), type="secondary")
    st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)

    st.markdown("<h1 class='sec-title' style='margin-bottom:8px;'>Car Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sec-subtitle'>K-Means (K=3) grouped vehicles by year, kilometres driven, and selling price into three distinct market segments.</p>", unsafe_allow_html=True)

    segments = [
        {"label": "Segment 0", "name": "Older Budget Cars",    "count": 1610, "pct": "45.01%", "avg_year": "2009", "avg_km": "99,531 km", "avg_price": "₹2,50,929"},
        {"label": "Segment 1", "name": "Newer Mid-Range Cars", "count": 1874, "pct": "52.39%", "avg_year": "2016", "avg_km": "44,194 km", "avg_price": "₹5,53,303"},
        {"label": "Segment 2", "name": "Premium Cars",         "count":   93, "pct":  "2.60%", "avg_year": "2016", "avg_km": "49,945 km", "avg_price": "₹27,34,398"},
    ]

    cards_html = "".join(
    f'<div class="seg-card">'
    f'<div class="seg-badge">{seg["label"]}</div>'
    f'<div class="seg-name">{seg["name"]}</div>'
    f'<div class="seg-row"><span class="seg-lbl">Vehicles</span>'
    f'<span class="seg-val">{seg["count"]:,} ({seg["pct"]})</span></div>'
    f'<div class="seg-row"><span class="seg-lbl">Avg Year</span>'
    f'<span class="seg-val">{seg["avg_year"]}</span></div>'
    f'<div class="seg-row"><span class="seg-lbl">Avg KM Driven</span>'
    f'<span class="seg-val">{seg["avg_km"]}</span></div>'
    f'<div class="seg-row"><span class="seg-lbl">Avg Selling Price</span>'
    f'<span class="seg-val">{seg["avg_price"]}</span></div>'
    f'</div>'
    for seg in segments
)

    st.markdown(f'<div class="seg-grid">{cards_html}</div>',unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:22px;font-weight:600;color:#1B2430;margin-bottom:8px;'>Analytical Market Segment Visualisation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;color:#2F3B4C;margin-bottom:32px;'>Approximate segment boundaries applied for visualisation. This is separate from the price prediction model.</p>", unsafe_allow_html=True)

    df = load_data()

    def assign_segment(row):
        if row["selling_price"] >= 1_500_000:
            return "Premium Cars"
        elif row["year"] >= 2013:
            return "Newer Mid-Range Cars"
        else:
            return "Older Budget Cars"

    df["segment"] = df.apply(assign_segment, axis=1)
    sample = df.sample(min(len(df), 1500), random_state=42)

    fig = px.scatter(
        sample, x="year", y="selling_price", color="segment",
        color_discrete_map={
            "Older Budget Cars":    INK,
            "Newer Mid-Range Cars": BLUE,
            "Premium Cars":         SOFT,
        },
        opacity=0.7,
        title="Year vs Selling Price — Analytical Market Segments",
        labels={"year": "Year", "selling_price": "Selling Price (₹)", "segment": "Segment"},
        hover_data=["brand"] if "brand" in sample.columns else None,
    )
    style_fig(fig, height=440)
    chart_card(fig)

    st.markdown("""
        <div class="info-banner">
            <strong>Note:</strong> "Analytical Market Segment" is derived from K-Means clustering for exploratory analysis only.
            It is not an input to, or output of, the price prediction model.
        </div>
    """, unsafe_allow_html=True)


# =================== INSIGHTS ===================
elif st.session_state.page == "Insights":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)
    st.button("← Back", on_click=navigate_to, args=("Home",), type="secondary")
    st.markdown("<div class='spacer-32'></div>", unsafe_allow_html=True)

    st.markdown("<h1 class='sec-title' style='margin-bottom:8px;'>Model Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sec-subtitle'>Compare regression models evaluated on the test set, and understand which features contribute most to the Random Forest predictions.</p>", unsafe_allow_html=True)

    model_rows = [
        {"Model": "Linear Regression", "MAE": 179642.88, "RMSE": 384722.37, "R2": 0.5405, "sel": False},
        {"Model": "Decision Tree",      "MAE": 195085.31, "RMSE": 471391.30, "R2": 0.3102, "sel": False},
        {"Model": "Random Forest",      "MAE": 157686.48, "RMSE": 363871.46, "R2": 0.5890, "sel": True},
    ]

    rows_html = ""
    for m in model_rows:
        cls = "sel" if m["sel"] else ""
        rows_html += f"<tr class='{cls}'><td>{m['Model']}</td><td>₹{m['MAE']:,.2f}</td><td>₹{m['RMSE']:,.2f}</td><td>{m['R2']:.4f}</td></tr>"

    st.markdown(f"""
        <table class="model-table">
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div class="info-banner">
            The Random Forest model explains approximately <strong>58.9%</strong> of the variance in selling price
            on the test set (R² = 0.5890). R² is a measure of explained variance — not a classification accuracy metric.
        </div>
    """, unsafe_allow_html=True)

    df_models = pd.DataFrame(model_rows)
    color_map = {"Linear Regression": NAVY, "Decision Tree": NAVY, "Random Forest": BLUE}

    cmp1, cmp2 = st.columns(2, gap="medium")
    with cmp1:
        fig = px.bar(df_models, x="Model", y="MAE",
                     title="MAE by Model (lower is better)",
                     labels={"MAE": "MAE (₹)"},
                     color="Model", color_discrete_map=color_map)
        style_fig(fig, showlegend=False)
        chart_card(fig)
    with cmp2:
        fig = px.bar(df_models, x="Model", y="R2",
                     title="R² by Model (higher is better)",
                     labels={"R2": "R² Score"},
                     color="Model", color_discrete_map=color_map)
        style_fig(fig, showlegend=False)
        chart_card(fig)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:22px;font-weight:600;color:#1B2430;margin-bottom:8px;'>Random Forest Feature Importance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;color:#2F3B4C;margin-bottom:32px;'>Importance reflects each feature's relative contribution to the model — not a causal claim.</p>", unsafe_allow_html=True)

    fi_df = pd.DataFrame({
        "Feature":    ["Transmission", "Brand",  "Car Age", "KM Driven", "Fuel",   "Seller Type", "Owner"],
        "Importance": [0.2476,          0.2400,   0.2217,    0.1336,      0.1157,   0.0235,        0.0179],
    }).sort_values("Importance")

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 title="Random Forest Feature Importance",
                 labels={"Importance": "Relative Importance", "Feature": ""},
                 color_discrete_sequence=[BLUE])
    style_fig(fig, height=400, yaxis=dict(categoryorder="total ascending"))
    chart_card(fig)

    st.markdown("""
        <div class="footer-wrap">
            <div class="footer-title">CarWise</div>
            <div class="footer-subtitle">Used Car Price Prediction &amp; Market Analysis</div>
            <div class="footer-note">DWM Mini Project</div>
        </div>
    """, unsafe_allow_html=True)