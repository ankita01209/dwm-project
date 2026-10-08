import sys
import os
import base64
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

def navigate_to(page):
    st.session_state.page = page

# ----------------------------------------
# PALETTE  (warm cream / navy-plum — per design spec)
# ----------------------------------------
INK    = "#393646"   # Deep Navy-Plum
NAVY   = "#4F4557"   # Secondary Mauve
BLUE   = "#6D5D6E"   # Replaced by Copper in new spec, kept for compat till Plotly update
PAPER  = "#F4EEE0"   # Warm Cream
TREND  = "#8A7A8B"   # Trendline muted
COPPER = "#C47A52"   # SINGLE ACCENT for active nav/metrics
SOFT   = "#A89BAA"   # Lighter accent

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
    grid = dict(gridcolor="rgba(79,69,87,0.10)", linecolor="rgba(79,69,87,0.10)",
                zeroline=False, automargin=True, title_font=dict(size=11, color="#6D5D6E"))
    
    layout_kwargs = dict(
        height=height, autosize=True,
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", size=13, color=INK),
        title=dict(font=dict(size=13, color=INK, family="Inter", weight="bold"), x=0.02, xanchor="left"),
        margin=dict(l=16, r=16, t=48, b=16),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#FFFFFF"))
    )
    if extra.get('showlegend') is not False:
        layout_kwargs['legend'] = dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15)
        
    layout_kwargs.update(extra)
    fig.update_layout(**layout_kwargs)
    
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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

/* ── RESET & GLOBALS ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background-image: linear-gradient(to bottom, #F7F2E8, #F4EEE0); color: #393646; }
.block-container {
    max-width: 1240px !important;
    padding: 0 32px 80px !important;
    margin: 0 auto;
}
header[data-testid="stHeader"] { display: none !important; }
a.header-anchor { display: none !important; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
p  { font-size: 15.5px; line-height: 1.6; color: #4F4557 !important; }

/* ── PERSISTENT NAVBAR ── */
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) {
    background: #393646;
    border-radius: 12px;
    padding: 0 20px;
    margin-top: 20px;
    margin-bottom: 0;
    align-items: center;
    flex-wrap: nowrap;
    border-bottom: 1px solid rgba(196,122,82,0.15);
    height: 66px;
    display: flex;
}
/* brand */
div#cw-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 24px; font-weight: 400;
    color: #F4EEE0 !important;
    letter-spacing: -0.3px;
    white-space: nowrap;
    line-height: normal;
    display: flex;
    align-items: center;
    gap: 8px;
}
/* Neutralize ALL wrappers inside the navbar that may have white backgrounds */
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) div[data-testid^="stBase"],
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) div[class^="stButton"],
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) .stButton {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* nav buttons — all buttons inside navbar */
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) button,
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) div.stButton > button {
    background: transparent !important;
    border: none !important;
    color: rgba(244,238,224,0.72) !important;
    font-family: 'Inter', sans-serif;
    font-size: 12.5px; font-weight: 500;
    padding: 0 8px;
    border-radius: 0 !important;
    box-shadow: none !important;
    -webkit-box-shadow: none !important;
    transition: color 0.2s ease, background 0.2s ease;
    white-space: nowrap;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
}
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) button:hover,
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) div.stButton > button:hover {
    color: #F4EEE0 !important;
    background: rgba(244,238,224,0.06) !important;
    transform: translateY(-1px);
}
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) button *,
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) div.stButton > button * {
    color: inherit !important;
    background: transparent !important;
}

/* active page nav button (primary kind) — copper underline, no fill */
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:not(:nth-child(7)) button[kind="primary"],
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:not(:nth-child(7)) div.stButton > button[kind="primary"] {
    background: transparent !important;
    color: #C47A52 !important;
    font-weight: 600;
    border-bottom: 2px solid #C47A52 !important;
    border-radius: 0 !important;
    padding-bottom: 2px;
}

/* CTA "Try Predictor" — 7th column, filled copper */
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:nth-child(7) button[kind="primary"],
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:nth-child(7) div.stButton > button[kind="primary"] {
    background: #C47A52 !important;
    color: #F4EEE0 !important;
    font-weight: 600;
    padding: 8px 18px !important;
    border-radius: 8px !important;
    border-bottom: none !important;
    height: 40px;
}
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:nth-child(7) button[kind="primary"]:hover,
div[data-testid="stHorizontalBlock"]:has(div#cw-navbar) > div[data-testid="column"]:nth-child(7) div.stButton > button[kind="primary"]:hover {
    background: #A8653E !important;
    transform: translateY(-1px);
}

/* ── GENERAL BUTTONS ── */
div.stButton > button[kind="primary"] {
    background: #C97B4F !important; color: #3A3547 !important;
    border: none; font-weight: 600; border-radius: 7px;
    padding: 9px 24px;
    transition: background 0.18s ease, transform 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    background: #A25F3A !important;
    transform: translateY(-1px);
}
div.stButton > button[kind="primary"] * { color: #3A3547 !important; background: transparent !important; }

/* ── FORM PREDICT BUTTON ── */
div[data-testid="stFormSubmitButton"] > button {
    background: #C97B4F !important; color: #3A3547 !important;
    border: none; font-weight: 600; border-radius: 7px;
    padding: 12px 24px;
    width: 100% !important;
    transition: background 0.18s ease, transform 0.15s ease;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #A25F3A !important;
    transform: translateY(-1px);
}
div[data-testid="stFormSubmitButton"] > button * { color: #3A3547 !important; background: transparent !important; }

div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #393646 !important;
    border: 1.5px solid #C97B4F !important;
    font-weight: 500; border-radius: 7px; padding: 8px 22px;
    transition: background 0.18s ease, border-color 0.18s ease;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(201,123,79,0.09) !important;
    border-color: #A25F3A !important;
}

/* ── INPUTS / SELECTS ── */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #A89BAA !important;
    border-radius: 8px !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:focus-within { border-color: #C47A52 !important; box-shadow: none !important; }
div[data-baseweb="select"] * { color: #393646 !important; }
div[data-baseweb="select"] svg { fill: #A89BAA !important; }
div[data-baseweb="popover"] > div { background: #FFFFFF !important; border: 1px solid rgba(109,93,110,0.25); border-radius: 8px; }
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] { background: #FFFFFF !important; color: #393646 !important; }
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][data-highlighted="true"] { background: rgba(109,93,110,0.12) !important; }
div[data-baseweb="popover"] [role="option"][aria-selected="true"] { background: rgba(109,93,110,0.2) !important; font-weight: 600; }
div[data-testid="stNumberInput"] input { background: transparent !important; color: #393646 !important; }
div[data-testid="stNumberInput"] > div > div { background: #FFFFFF !important; border: 1px solid #A89BAA !important; border-radius: 8px !important; box-shadow: none !important;}
div[data-testid="stNumberInput"] > div > div:focus-within { border-color: #C47A52 !important; }
div[data-testid="stSlider"] label { color: #393646 !important; }

/* ── CHART CARDS ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid rgba(109,93,110,0.14) !important;
    border-radius: 12px;
    padding: 24px 20px 8px;
    box-shadow: none;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 20px rgba(57,54,70,0.04);
}

/* ── SPACING HELPERS ── */
.spacer-100 { height:100px; } .spacer-60  { height:60px; }
.spacer-48  { height:48px;  } .spacer-32  { height:32px; }
.section-divider { height:1px; background:rgba(109,93,110,0.14); margin:56px 0; }

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4, h5, h6 { 
    font-family: 'DM Serif Display', serif !important; 
    color: #393646 !important; 
    font-weight: 400 !important; 
}
.eyebrow, .hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #C47A52; margin-bottom: 20px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 68px; font-weight: 400;
    color: #393646; line-height: 1.0;
    margin: 0 0 28px; letter-spacing: -1.5px;
}
.hero-desc {
    font-size: 16px; color: #4F4557;
    line-height: 1.6; max-width: 560px; margin-bottom: 44px;
}
.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 38px; font-weight: 400;
    color: #393646; margin: 0 0 14px; letter-spacing: -0.5px;
}
.sec-subtitle { font-size: 15.5px; color: #4F4557; margin: 0 0 44px; line-height: 1.65; }
.page-eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #6D5D6E; margin-bottom: 10px;
}

/* Center page properly bounded */
div.block-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding-top: 14px !important;
}

/* ── HERO DASHBOARD CARD ── */
.dash {
    background: #393646; border-radius: 18px;
    padding: 32px 32px 28px; max-width: 440px; margin: 0 auto;
    box-shadow: 0 14px 36px rgba(58,53,71,0.22);
}
.dash-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; }
.dash-brand { font-family:'Inter',sans-serif; font-size:11px; font-weight:700; color:#A89BAA !important; letter-spacing:2px; text-transform:uppercase; }
.dash-main { background:rgba(255,255,255,0.03); border-radius:12px; padding:24px 20px; margin-bottom:20px; text-align:center; border:1px solid rgba(196,122,82,0.15); }
.dash-car { width:180px; margin:-10px auto 16px; display:block; }
.dash-val { font-family:'DM Serif Display',serif; font-size:44px; font-weight:400; color:#C47A52 !important; letter-spacing:-1px; line-height:1; }
.dash-sub { font-size:12px; color:rgba(244,238,224,0.55) !important; margin-top:8px; text-transform:uppercase; letter-spacing:1px; }
.dash-chart { margin-bottom:24px; padding:0 20px; }
.dash-chart svg { width:100%; height:40px; display:block; }
.dash-kpis { display:flex; justify-content:space-between; padding-top:20px; border-top:1px solid rgba(244,238,224,0.1); }
.dash-kpis span { font-size:10px; font-weight:700; color:#A89BAA !important; text-transform:uppercase; letter-spacing:1px; }
.dash-kpis b { font-family:'DM Serif Display',serif; font-size:16px; color:#F4EEE0 !important; margin-right:4px; }

/* ── FEATURE CARDS ── */
.cap-grid  { display:grid; grid-template-columns:repeat(4,1fr); gap:20px; margin-top:20px; }
.cap-block { background:transparent; padding:20px 0 0 20px; border-left:1px solid rgba(109,93,110,0.2); transition:transform 0.2s ease, border-color 0.2s ease; position:relative; }
.cap-block:hover { transform:translateY(-2px); border-color:#C47A52; }
.cap-block:hover .cap-num { color:#C47A52; }
.cap-num   { font-family:'DM Serif Display',serif; font-size:48px; font-weight:400; color:rgba(109,93,110,0.15); margin:0 0 16px; line-height:1; transition:color 0.2s ease; }
.cap-title { font-size:16px; font-weight:700; color:#393646; margin:0 0 10px; }
.cap-desc  { font-size:14px; color:#4F4557; line-height:1.55; margin:0; }

/* ── KPI STRIP ── */
.kpi-grid  { display:grid; grid-template-columns:repeat(5,1fr); gap:0; border-top:1px solid rgba(109,93,110,0.2); border-bottom:1px solid rgba(109,93,110,0.2); padding:32px 0; margin-top:16px; }
.kpi-block { padding:0 32px; border-right:1px solid rgba(109,93,110,0.14); background:transparent; }
.kpi-block:last-child { border-right:none; padding-right:0; }
.kpi-block:first-child { padding-left:0; }
.kpi-val   { font-family:'DM Serif Display',serif; font-size:36px; font-weight:400; color:#393646; margin:0 0 8px; line-height:1; }
.kpi-val.highlight { color:#C47A52; }
.kpi-lbl   { font-size:12px; font-weight:700; color:#6D5D6E; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 4px; }
.kpi-sub   { font-size:12px; color:#A89BAA; line-height:1.4; display:block; }

/* ── DWM WORKFLOW ── */
.wf-grid  { display:flex; justify-content:space-between; align-items:flex-start; margin-top:32px; position:relative; }
.wf-item  { flex:1; text-align:center; position:relative; padding:0 10px; }
.wf-num   { font-size:18px; font-family:'DM Serif Display',serif; color:#6D5D6E; margin-bottom:12px; background:#F7F2E8; position:relative; z-index:2; display:inline-block; padding:0 8px; }
.wf-item:last-child .wf-num { color:#C47A52; }
.wf-title { font-size:11px; font-weight:600; color:#4F4557; text-transform:uppercase; letter-spacing:1px; line-height:1.4; }
.wf-line  { position:absolute; top:9px; left:50%; right:-50%; height:1px; background:rgba(109,93,110,0.2); z-index:1; }
.wf-item:last-child .wf-line { display:none; }

/* ── FOOTER ── */
.footer-wrap { background:#393646; border-radius:12px; padding:56px 48px; margin-top:80px; }
.footer-brand { font-family:'DM Serif Display',serif; font-size:26px; font-weight:400; color:#F4EEE0 !important; margin:0 0 6px; }
.footer-sub   { font-size:14px; color:#A89BAA !important; margin:0 0 28px; }
.footer-note  { font-size:13px; color:rgba(244,238,224,0.6) !important; margin:0; }

/* ── PRICE PREDICTOR RESULT ── */
.result-panel { background:#393646; border-radius:14px; padding:44px 52px; margin-top:32px; }
.result-label { font-size:10px; font-weight:700; color:#A89BAA !important; letter-spacing:2.5px; text-transform:uppercase; margin-bottom:12px; }
.result-price {
    font-family:'DM Serif Display',serif;
    font-size:64px; font-weight:400;
    color:#F4EEE0 !important; margin:0 0 16px; letter-spacing:-2px; line-height:1;
}
.result-disc { font-size:13px; color:rgba(244,238,224,0.55) !important; margin:0; line-height:1.5; }

/* ── VEHICLE SUMMARY ── */
.summary-grid  { display:grid; grid-template-columns:repeat(4,1fr); gap:24px; margin-top:32px; }
.summary-item  { border-left:2px solid rgba(109,93,110,0.3); padding-left:14px; }
.summary-label { font-size:10px; font-weight:700; color:#6D5D6E; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px; }
.summary-val   { font-size:15px; font-weight:500; color:#393646; }

/* ── SEGMENT CARDS ── */
.seg-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:44px; }
.seg-card { border:1px solid rgba(109,93,110,0.18); border-radius:12px; padding:32px 26px; background:#FFFFFF;
            transition:transform 0.2s ease, box-shadow 0.2s ease; }
.seg-card:hover { transform:translateY(-3px); box-shadow:0 12px 32px rgba(57,54,70,0.1); }
.seg-badge { display:inline-block; font-size:10px; font-weight:700; color:#6D5D6E; background:rgba(109,93,110,0.1); border-radius:20px; padding:3px 11px; margin-bottom:16px; letter-spacing:1px; text-transform:uppercase; }
.seg-name  { font-family:'DM Serif Display',serif; font-size:20px; font-weight:400; color:#393646; margin:0 0 20px; }
.seg-big   { font-family:'DM Serif Display',serif; font-size:38px; font-weight:400; color:#393646; margin:0 0 2px; line-height:1; }
.seg-pct   { font-size:12px; font-weight:600; color:#6D5D6E; margin:0 0 20px; }
.seg-row   { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid rgba(109,93,110,0.09); }
.seg-row:last-child { border-bottom:none; padding-bottom:0; }
.seg-lbl   { font-size:13px; color:#4F4557; }
.seg-val   { font-size:13px; font-weight:600; color:#393646; }

/* ── BANNERS ── */
.info-banner { background:rgba(109,93,110,0.07); border-left:3px solid #6D5D6E; border-radius:0 7px 7px 0; padding:12px 16px; font-size:14px; color:#4F4557; margin-bottom:16px; }
.warn-banner { background:rgba(57,54,70,0.05); border-left:3px solid #393646; border-radius:0 7px 7px 0; padding:12px 16px; font-size:14px; color:#393646; margin-bottom:12px; }

/* ── RESPONSIVE ── */
@media (max-width:1024px) {
    .cap-grid  { grid-template-columns:repeat(2,1fr); }
    .kpi-grid  { grid-template-columns:repeat(3,1fr); }
    .kpi-block { border-bottom:1px solid rgba(109,93,110,0.14); }
    .wf-grid   { grid-template-columns:repeat(4,1fr); }
    .wf-line   { display:none; }
    .seg-grid  { grid-template-columns:1fr 1fr; }
    .summary-grid { grid-template-columns:repeat(2,1fr); }
}
@media (max-width:768px) {
    .cap-grid  { grid-template-columns:1fr; }
    .kpi-grid  { grid-template-columns:1fr 1fr; }
    .wf-grid   { grid-template-columns:repeat(2,1fr); }
    .hero-title { font-size:44px; }
    .seg-grid  { grid-template-columns:1fr; }
    .summary-grid { grid-template-columns:1fr 1fr; }
    .result-price { font-size:48px; }
    .dash-main { flex-direction:column; text-align:center; }
    .dash-car  { width:65%; }
    .result-panel { padding:32px 28px; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# PERSISTENT HORIZONTAL NAVBAR
# ----------------------------------------
_p = st.session_state.page
_nb, _n1, _n2, _n3, _n4, _n5, _cta = st.columns([1.65, 0.7, 1.15, 1.15, 1.25, 0.75, 1.05], gap="small")

with _nb:
    st.markdown("<div id='cw-navbar'></div><div id='cw-brand'><svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='#C97B4F' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polygon points='12 2 2 7 12 12 22 7 12 2'></polygon><polyline points='2 17 12 22 22 17'></polyline><polyline points='2 12 12 17 22 12'></polyline></svg>CarWise</div>", unsafe_allow_html=True)
with _n1:
    st.button("Home",             on_click=navigate_to, args=("Home",),             key="nav_home",  type="primary" if _p=="Home" else "secondary")
with _n2:
    st.button("Price Predictor",  on_click=navigate_to, args=("Price Predictor",),  key="nav_pred",  type="primary" if _p=="Price Predictor" else "secondary")
with _n3:
    st.button("Market Analysis",  on_click=navigate_to, args=("Market Analysis",),  key="nav_mkt",   type="primary" if _p=="Market Analysis" else "secondary")
with _n4:
    st.button("Car Segmentation", on_click=navigate_to, args=("Car Segmentation",), key="nav_seg",   type="primary" if _p=="Car Segmentation" else "secondary")
with _n5:
    st.button("Insights",         on_click=navigate_to, args=("Insights",),         key="nav_ins",   type="primary" if _p=="Insights" else "secondary")
with _cta:
    st.button("Try Predictor", on_click=navigate_to, args=("Price Predictor",), key="nav_cta", type="primary")

# =================== HOME ===================
if st.session_state.page == "Home":

    # Dynamically inject base64 hero background CSS
    try:
        with open(os.path.join("assets", "hero_bg.webp"), "rb") as f:
            hero_webp = base64.b64encode(f.read()).decode()
        with open(os.path.join("assets", "hero_bg.jpg"), "rb") as f:
            hero_jpg = base64.b64encode(f.read()).decode()
        hero_css = f"""
        <style>
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stMarkdownContainer"] > div#hero-marker) {{
            background-image: 
                linear-gradient(90deg, rgba(246,241,231,0.96) 0%, rgba(246,241,231,0.85) 40%, rgba(246,241,231,0.2) 100%),
                -webkit-image-set(url(data:image/webp;base64,{hero_webp}) 1x, url(data:image/jpeg;base64,{hero_jpg}) 1x);
            background-size: cover;
            background-position: center right;
            background-repeat: no-repeat;
            border-radius: 16px;
            padding: 56px 48px;
            margin-bottom: 24px;
            margin-top: 24px;
        }}
        @media (max-width:768px) {{
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stMarkdownContainer"] > div#hero-marker) {{
                background-image: 
                    linear-gradient(180deg, rgba(246,241,231,0.96) 0%, rgba(246,241,231,0.85) 30%, rgba(246,241,231,0.2) 100%),
                    -webkit-image-set(url(data:image/webp;base64,{hero_webp}) 1x, url(data:image/jpeg;base64,{hero_jpg}) 1x);
                background-position: center top;
                padding: 40px 32px;
            }}
        }}
        </style>
        """
    except Exception:
        hero_css = """
        <style>
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stMarkdownContainer"] > div#hero-marker) {
            background-color: #F6F1E7;
            background-image: radial-gradient(circle at center right, rgba(201,123,79,0.1), transparent 50%),
                              repeating-linear-gradient(45deg, rgba(201,123,79,0.03) 0, rgba(201,123,79,0.03) 1px, transparent 1px, transparent 10px);
            border-radius: 16px;
            padding: 56px 48px;
            margin-bottom: 24px;
            margin-top: 24px;
        }
        </style>
        """
    
    st.markdown(hero_css, unsafe_allow_html=True)

    hero_con = st.container()
    with hero_con:
        st.markdown("<div id='hero-marker' aria-hidden='true'></div>", unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([1.22, 1], gap="large")
        with h_col1:
            st.markdown("""
                <div class="hero-eyebrow" style="color:#C97B4F;">CARWISE / MARKET INTELLIGENCE</div>
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
    <div class="dash-top"><span class="dash-brand">CARWISE / SAMPLE PREVIEW</span></div>
    <div class="dash-main">
    <svg viewBox="0 0 200 80" class="dash-car" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 58 L22 40 Q28 30 42 28 L82 24 Q96 12 120 12 L146 14 Q160 18 170 36 L188 42 Q194 46 194 56 L194 60 L10 60 Z" fill="#4F4557"/>
    <path d="M92 26 Q102 17 120 17 L142 19 Q150 22 156 34 L92 34 Z" fill="#F4EEE0" opacity="1"/>
    <circle cx="52" cy="62" r="13" fill="#252331"/><circle cx="52" cy="62" r="5" fill="#C97B4F"/>
    <circle cx="148" cy="62" r="13" fill="#252331"/><circle cx="148" cy="62" r="5" fill="#C97B4F"/>
    </svg>
    <div class="dash-val" style="color:#C97B4F !important;">₹ 4,85,000</div>
    <div class="dash-sub">Random Forest estimate</div>
    </div>
    <div class="dash-chart">
    <svg viewBox="0 0 300 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#C97B4F" stop-opacity=".35"/><stop offset="1" stop-color="#C97B4F" stop-opacity="0"/></linearGradient></defs>
    <path d="M0 20 C40 24 70 8 110 16 S190 32 300 12 L300 40 L0 40 Z" fill="url(#g)"/>
    <path d="M0 20 C40 24 70 8 110 16 S190 32 300 12" fill="none" stroke="#C97B4F" stroke-width="2.5"/>
    </svg>
    </div>
    <div class="dash-kpis">
    <span><b>7</b>FEATURES</span>
    <span><b>3</b>SEGMENTS</span>
    <span><b>3,577</b>RECORDS</span>
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
            <div class="kpi-block"><div class="kpi-val highlight">58.9%</div><div class="kpi-lbl">Test-set R²</div><span class="kpi-sub">Variance explained on test set</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='spacer-100'></div>", unsafe_allow_html=True)

    st.markdown("<h2 class='sec-title'>DWM Workflow</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="wf-grid">
            <div class="wf-item"><div class="wf-num">01</div><div class="wf-title">Data<br>Understanding</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">02</div><div class="wf-title">Data<br>Cleaning</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">03</div><div class="wf-title">Preprocessing</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">04</div><div class="wf-title">EDA</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">05</div><div class="wf-title">K-Means<br>Clustering</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">06</div><div class="wf-title">Regression</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">07</div><div class="wf-title">Model<br>Evaluation</div><div class="wf-line"></div></div>
            <div class="wf-item"><div class="wf-num">08</div><div class="wf-title">Streamlit<br>Dashboard</div></div>
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

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)
    
    st.button("← Back to CarWise", on_click=navigate_to, args=("Home",), type="secondary", key="pred_back")
    
    st.markdown("""
        <style>
        .st-key-pred_back button {
            background: transparent !important; color: #A89BAA !important;
            border: none !important; padding: 0 !important; font-size: 13px !important; font-weight: 500 !important;
            transition: color 0.2s ease !important; height: auto !important; margin: 0 !important; line-height: 1 !important;
        }
        .st-key-pred_back button:hover {
            color: #C47A52 !important; transform: none !important;
        }
        </style>
        <div class="page-eyebrow" style="color:#C47A52;margin-top:24px;">CARWISE / PREDICT</div>
        <h1 class="hero-title" style="font-size:54px;margin-bottom:16px;">Price Predictor</h1>
        <p class="hero-desc" style="max-width:100%;margin-bottom:48px;">Enter your vehicle details to estimate its selling price. Car age is computed automatically from the year you enter.</p>
    """, unsafe_allow_html=True)

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
    
    pred_left, pred_right = st.columns([1.1, 1], gap="large")

    with pred_left:
        st.markdown("<h3 style='font-family:\"DM Serif Display\",serif;font-size:28px;color:#393646;margin-bottom:20px;font-weight:400;'>Tell us about the car.</h3>", unsafe_allow_html=True)
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

    with pred_right:
        if submitted:
            errors = []
            if brand        is None: errors.append("Please select a brand.")
            if fuel         is None: errors.append("Please select a fuel type.")
            if seller_type  is None: errors.append("Please select a seller type.")
            if transmission is None: errors.append("Please select a transmission.")
            if owner        is None: errors.append("Please select an ownership category.")
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
                        <div style="background: linear-gradient(145deg, #1B2430 0%, #393646 100%); border-radius: 16px; padding: 48px 40px; box-shadow: inset 0 2px 10px rgba(255,255,255,0.05), 0 32px 64px rgba(57,54,70,0.18);">
                            <div style="font-size:10px;font-weight:700;color:#A89BAA;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">ESTIMATED SELLING PRICE</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:62px;color:#C47A52;line-height:1;margin-bottom:20px;letter-spacing:-1px;">₹ {predicted_price:,.0f}</div>
                            <div style="font-size:12px;color:#A89BAA;border-top:1px solid rgba(255,255,255,0.1);padding-top:16px;line-height:1.5;">
                                This is a historical market estimate based on CarDekho data — not a guaranteed selling price.<br><br>
                                <strong>Model:</strong> Random Forest<br>
                                <strong>Summary:</strong> {brand} · {car_age} yrs old · {int(km_driven):,} km · {transmission}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
                <div style="background: #FFFFFF; border: 1px solid rgba(109,93,110,0.18); border-radius: 16px; padding: 60px 40px; text-align:center; height: 100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:11px;font-weight:700;color:#6D5D6E;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">YOUR ESTIMATE</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:36px;color:rgba(109,93,110,0.15);position:relative;z-index:2;">₹ --,---</div>
                    <div style="font-size:13px;color:#6D5D6E;margin-top:16px;max-width:240px;margin-left:auto;margin-right:auto;">Fill out the form on the left to see your vehicle's estimated selling price.</div>
                </div>
            """, unsafe_allow_html=True)

# =================== MARKET ANALYSIS ===================
elif st.session_state.page == "Market Analysis":

    st.markdown("<div class='spacer-60'></div>", unsafe_allow_html=True)
    
    st.button("← Back to CarWise", on_click=navigate_to, args=("Home",), type="secondary", key="mkt_back")
    
    st.markdown("""
        <style>
        .st-key-mkt_back button {
            background: transparent !important; color: #A89BAA !important;
            border: none !important; padding: 0 !important; font-size: 13px !important; font-weight: 500 !important;
            transition: color 0.2s ease !important; height: auto !important; margin: 0 !important; line-height: 1 !important;
        }
        .st-key-mkt_back button:hover {
            color: #C47A52 !important; transform: none !important;
        }
        </style>
        <div class="page-eyebrow" style="color:#C47A52;margin-top:24px;">CARWISE / MARKET INTELLIGENCE</div>
        <h1 class="hero-title" style="font-size:54px;margin-bottom:16px;">Market Analysis</h1>
        <p class="hero-desc" style="max-width:100%;margin-bottom:48px;">Explore observed pricing patterns across the CarDekho dataset. Charts reflect historical data — they describe observed relationships, not causal ones.</p>
    """, unsafe_allow_html=True)

    df = load_data()

    st.markdown("<div style='border-top:1px solid rgba(109,93,110,0.15);padding-top:24px;margin-bottom:8px;'></div>", unsafe_allow_html=True)

    f_cols = st.columns(5)
    with f_cols[0]:
        sel_fuel   = st.selectbox("Fuel Type",    ["All"] + sorted(df["fuel"].unique().tolist()),         key="mkt_fuel")
    with f_cols[1]:
        sel_trans  = st.selectbox("Transmission", ["All"] + sorted(df["transmission"].unique().tolist()), key="mkt_trans")
    with f_cols[2]:
        sel_seller = st.selectbox("Seller Type",  ["All"] + sorted(df["seller_type"].unique().tolist()),  key="mkt_seller")
    with f_cols[3]:
        sel_owner  = st.selectbox("Owner",        ["All"] + sorted(df["owner"].unique().tolist()),        key="mkt_owner")
    with f_cols[4]:
        sel_brand  = st.selectbox("Brand",        ["All"] + sorted(df["brand"].unique().tolist()),        key="mkt_brand")

    year_range = st.slider(
        "Year Range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(int(df["year"].min()), int(df["year"].max())),
        key="mkt_year"
    )

    st.markdown("<div style='border-bottom:1px solid rgba(109,93,110,0.15);padding-bottom:16px;margin-bottom:32px;'></div>", unsafe_allow_html=True)

    dff = df.copy()
    if sel_fuel   != "All": dff = dff[dff["fuel"]         == sel_fuel]
    if sel_trans  != "All": dff = dff[dff["transmission"] == sel_trans]
    if sel_seller != "All": dff = dff[dff["seller_type"]  == sel_seller]
    if sel_owner  != "All": dff = dff[dff["owner"]        == sel_owner]
    if sel_brand  != "All": dff = dff[dff["brand"]        == sel_brand]
    dff = dff[(dff["year"] >= year_range[0]) & (dff["year"] <= year_range[1])]

    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:32px;'>"
        f"<span style='font-family:\"DM Serif Display\",serif;font-size:42px;color:#393646;line-height:1;'>{len(dff):,}</span>"
        f"<span style='font-size:12px;font-weight:700;color:#6D5D6E;text-transform:uppercase;letter-spacing:1px;'>VEHICLES MATCH YOUR FILTERS</span>"
        f"</div>",
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
    st.button("← Back to CarWise", on_click=navigate_to, args=("Home",), type="secondary", key="seg_back")
    
    st.markdown("""
        <style>
        .st-key-seg_back button {
            background: transparent !important; color: #A89BAA !important;
            border: none !important; padding: 0 !important; font-size: 13px !important; font-weight: 500 !important;
            transition: color 0.2s ease !important; height: auto !important; margin: 0 !important; line-height: 1 !important;
        }
        .st-key-seg_back button:hover {
            color: #C47A52 !important; transform: none !important;
        }
        .seg-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-bottom:44px; }
        </style>
        <div class="page-eyebrow" style="color:#C47A52;margin-top:24px;">CARWISE / SEGMENTATION</div>
        <h1 class="hero-title" style="font-size:54px;margin-bottom:16px;">Car Segmentation</h1>
        <p class="hero-desc" style="max-width:100%;margin-bottom:48px;">K-Means (K=3) grouped vehicles by year, kilometres driven, and selling price into three distinct market segments.</p>
    """, unsafe_allow_html=True)

    segments = [
        {"idx": "01", "label": "Segment 0", "name": "Older Budget Cars",    "count": 1610, "pct": "45.01%", "avg_year": "2009", "avg_km": "99,531 km", "avg_price": "₹2,50,929", "style": "background:#FFFFFF;border:1px solid rgba(109,93,110,0.15);color:#393646;", "nm_col":"#393646", "lbl_col":"#6D5D6E", "div_col":"rgba(109,93,110,0.15)"},
        {"idx": "02", "label": "Segment 1", "name": "Newer Mid-Range Cars", "count": 1874, "pct": "52.39%", "avg_year": "2016", "avg_km": "44,194 km", "avg_price": "₹5,53,303", "style": "background:#393646;border:none;color:#F4EEE0;", "nm_col":"#F4EEE0", "lbl_col":"#A89BAA", "div_col":"rgba(244,238,224,0.1)"},
        {"idx": "03", "label": "Segment 2", "name": "Premium Cars",         "count":   93, "pct":  "2.60%", "avg_year": "2016", "avg_km": "49,945 km", "avg_price": "₹27,34,398", "style": "background:linear-gradient(145deg, #1B2430 0%, #393646 100%);border:1px solid rgba(196,122,82,0.25);color:#F4EEE0;", "nm_col":"#C47A52", "lbl_col":"#A89BAA", "div_col":"rgba(244,238,224,0.1)"},
    ]

    cards_html = "".join(
    f'<div style="border-radius:12px;padding:32px 28px;{seg["style"]}">'
    f'<div style="font-family:\'DM Serif Display\',serif;font-size:42px;opacity:0.25;margin-bottom:8px;line-height:1;">{seg["idx"]}</div>'
    f'<div style="font-size:10px;font-weight:700;color:{seg["lbl_col"]};margin-bottom:12px;letter-spacing:1px;text-transform:uppercase;">{seg["label"]}</div>'
    f'<div style="font-family:\'DM Serif Display\',serif;font-size:26px;font-weight:400;color:{seg["nm_col"]};margin-bottom:24px;line-height:1.2;">{seg["name"]}</div>'
    f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid {seg["div_col"]};padding:10px 0;"><span style="font-size:13px;color:{seg["lbl_col"]};">Vehicles</span><span style="font-size:13px;font-weight:600;">{seg["count"]:,} ({seg["pct"]})</span></div>'
    f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid {seg["div_col"]};padding:10px 0;"><span style="font-size:13px;color:{seg["lbl_col"]};">Avg Year</span><span style="font-size:13px;font-weight:600;">{seg["avg_year"]}</span></div>'
    f'<div style="display:flex;justify-content:space-between;border-bottom:1px solid {seg["div_col"]};padding:10px 0;"><span style="font-size:13px;color:{seg["lbl_col"]};">Avg KM Driven</span><span style="font-size:13px;font-weight:600;">{seg["avg_km"]}</span></div>'
    f'<div style="display:flex;justify-content:space-between;padding-top:10px;"><span style="font-size:13px;color:{seg["lbl_col"]};">Avg Selling Price</span><span style="font-size:13px;font-weight:600;">{seg["avg_price"]}</span></div>'
    f'</div>'
    for seg in segments
    )

    st.markdown(f'<div class="seg-grid">{cards_html}</div>', unsafe_allow_html=True)

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
    st.button("← Back to CarWise", on_click=navigate_to, args=("Home",), type="secondary", key="ins_back")
    
    st.markdown("""
        <style>
        .st-key-ins_back button {
            background: transparent !important; color: #A89BAA !important;
            border: none !important; padding: 0 !important; font-size: 13px !important; font-weight: 500 !important;
            transition: color 0.2s ease !important; height: auto !important; margin: 0 !important; line-height: 1 !important;
        }
        .st-key-ins_back button:hover {
            color: #C47A52 !important; transform: none !important;
        }
        .model-table { width:100%; border-collapse:collapse; margin-bottom:28px; }
        .model-table th { font-size:11px; font-weight:700; color:#6D5D6E; text-transform:uppercase; letter-spacing:1.5px; padding:12px 16px; border-bottom:1px solid rgba(109,93,110,0.18); text-align:left; background:transparent;}
        .model-table td { font-size:14.5px; padding:13px 16px; border-bottom:1px solid rgba(109,93,110,0.09); color:#393646; background:transparent;}
        .model-table tr.sel td { background:rgba(196,122,82,0.06); font-weight:600; border-bottom:1px solid rgba(196,122,82,0.25); }
        .model-table tr.sel td:first-child { border-left:3px solid #C47A52; }
        </style>
        <div class="page-eyebrow" style="color:#C47A52;margin-top:24px;">CARWISE / MODEL PERFORMANCE</div>
        <h1 class="hero-title" style="font-size:54px;margin-bottom:16px;">Model Insights</h1>
        <p class="hero-desc" style="max-width:100%;margin-bottom:48px;">Compare regression models evaluated on the test set, and understand which features contribute most to the Random Forest predictions.</p>
    """, unsafe_allow_html=True)

    model_rows = [
        {"Model": "Linear Regression", "MAE": 179642.88, "RMSE": 384722.37, "R2": 0.5405, "sel": False},
        {"Model": "Decision Tree",      "MAE": 195085.31, "RMSE": 471391.30, "R2": 0.3102, "sel": False},
        {"Model": "Random Forest",      "MAE": 157686.48, "RMSE": 363871.46, "R2": 0.5890, "sel": True},
    ]

    rows_html = ""
    for m in model_rows:
        cls = "sel" if m["sel"] else ""
        icon = " <span style='color:#C47A52;margin-left:6px;font-size:12px;'>✓</span>" if m["sel"] else ""
        rows_html += f"<tr class='{cls}'><td>{m['Model']}{icon}</td><td>₹{m['MAE']:,.2f}</td><td>₹{m['RMSE']:,.2f}</td><td>{m['R2']:.4f}</td></tr>"

    st.markdown(f"""
        <table class="model-table">
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div class="info-banner" style="background:transparent; border-left:2px solid rgba(109,93,110,0.3); padding-left:16px; margin-bottom:32px; color:#4F4557;">
            The Random Forest model explains approximately <strong>58.9%</strong> of the variance in selling price
            on the test set (R² = 0.5890). R² is a measure of explained variance — not a classification accuracy metric.
        </div>
    """, unsafe_allow_html=True)

    df_models = pd.DataFrame(model_rows)
    color_map = {"Linear Regression": NAVY, "Decision Tree": NAVY, "Random Forest": COPPER}

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

    bar_colors = [NAVY] * len(fi_df)
    bar_colors[-1] = COPPER
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 title="Random Forest Feature Importance",
                 labels={"Importance": "Relative Importance", "Feature": ""})
    fig.update_traces(marker_color=bar_colors)
    style_fig(fig, height=400, yaxis=dict(categoryorder="total ascending"))
    chart_card(fig)

    st.markdown("""
        <div class="footer-wrap">
            <div class="footer-left">
                <div class="footer-title">CarWise</div>
                <div class="footer-subtitle">Used Car Price Prediction &amp; Market Analysis</div>
            </div>
            <div class="footer-note">DWM Mini Project</div>
        </div>
    """, unsafe_allow_html=True)