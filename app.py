import sys
import os
import textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.ui.styles import inject_global_styles
from src.ui.pages.home import render_home
from src.ui.pages.price_predictor import render_price_predictor
from src.ui.pages.market_analysis import render_market_analysis
from src.ui.pages.segmentation import render_segmentation
from src.ui.pages.insights import render_insights

st.set_page_config(
    page_title="CarWise - Used Car Journal",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injection of our massive CSS journal architecture
inject_global_styles()

if 'page' not in st.session_state:
    st.session_state.page = "Home"

def navigate_to(page):
    st.session_state.page = page

# ----------------------------------------
# PERSISTENT JOURNAL NAVBAR (FLEXBOX)
# ----------------------------------------
def render_navbar():
    st.markdown('<div id="cw-nav-marker" style="display:none;"></div>', unsafe_allow_html=True)
    
    col_logo, col_links, col_cta = st.columns([1, 4, 1])
    p = st.session_state.page
    
    with col_logo:
        st.markdown(textwrap.dedent("""
        <div class="nav-logo-group">
            <svg width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='#C97B4F' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
                <path d="M10 20 L4 20 Q2 20 2 18 L2 6 Q2 4 4 4 L14 4 Q16 4 18 6 L22 12 L22 18 Q22 20 20 20 L20 20" stroke-width="1.5"></path>
                <circle cx="7" cy="20" r="3" fill="#F6F1E7"></circle>
                <circle cx="17" cy="20" r="3" fill="#F6F1E7"></circle>
            </svg>
            CarWise
        </div>
        """), unsafe_allow_html=True)
        
    with col_links:
        l1, l2, l3, l4, l5 = st.columns(5)
        with l1: st.button("Home", on_click=navigate_to, args=("Home",), use_container_width=True, type="primary" if p == "Home" else "secondary")
        with l2: st.button("Price Predictor", on_click=navigate_to, args=("Price Predictor",), use_container_width=True, type="primary" if p == "Price Predictor" else "secondary")
        with l3: st.button("Market Analysis", on_click=navigate_to, args=("Market Analysis",), use_container_width=True, type="primary" if p == "Market Analysis" else "secondary")
        with l4: st.button("Car Segmentation", on_click=navigate_to, args=("Car Segmentation",), use_container_width=True, type="primary" if p == "Car Segmentation" else "secondary")
        with l5: st.button("Insights", on_click=navigate_to, args=("Insights",), use_container_width=True, type="primary" if p == "Insights" else "secondary")
        
    with col_cta:
        st.button("Try Predictor", on_click=navigate_to, args=("Price Predictor",), type="primary", use_container_width=True, key="try_pred_btn")
    
    st.markdown("""
    <style>
    /* Turn the master layout wrapping these columns into a true Flexbox Navbar */
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
        background: rgba(246, 241, 231, 0.95);
        backdrop-filter: blur(12px);
        padding: 16px 32px;
        position: sticky;
        top: 0;
        z-index: 1000;
        border-bottom: 1px solid var(--line);
        margin-left: -32px;
        margin-right: -32px;
        width: calc(100% + 64px) !important;
    }
    
    /* Make the links row internally flex without wrapping */
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
        align-items: center !important;
    }
    
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1),
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) {
        flex: 0 0 auto !important;
        width: auto !important;
    }
    
    /* Clean up the native buttons inside the nav to look like textual links */
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton > button[kind="secondary"] {
        border: none !important;
        background: transparent !important;
        color: var(--ink) !important;
        box-shadow: none !important;
        font-weight: 500;
    }
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton > button[kind="primary"] {
        border: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: var(--terracotta-dark) !important;
        border-bottom: 2px solid var(--terracotta) !important;
        box-shadow: none !important;
        font-weight: 600;
        padding-bottom: 8px;
    }
    
    /* The CTA Try Predictor button on the right */
    div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div.stButton > button {
        background: var(--terracotta) !important;
        color: var(--paper) !important;
        border-radius: var(--radius-btn) !important;
        border: none !important;
        padding: 10px 20px !important;
    }
    
    @media (max-width: 900px) {
        div[data-testid="stVerticalBlock"]:has(> div > div > #cw-nav-marker) > div:nth-child(2) > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
            display: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

render_navbar()

# ----------------------------------------
# ROUTING
# ----------------------------------------
if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "Price Predictor":
    render_price_predictor()
elif st.session_state.page == "Market Analysis":
    render_market_analysis()
elif st.session_state.page == "Car Segmentation":
    render_segmentation()
elif st.session_state.page == "Insights":
    render_insights()
else:
    st.info("Module under active 'Data Journal' redesign. Please select 'Home'.")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown(
    "<style>"
    ".footer-wrap{background:var(--plum);border-radius:12px;padding:56px 48px;"
    "margin-top:80px;position:relative;overflow:hidden}"
    ".footer-bg{position:absolute;top:0;left:0;right:0;bottom:0;opacity:.05;"
    "background-image:url(\"data:image/svg+xml,%3Csvg width='100' height='100' "
    "viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath "
    "d='M0%2050%20Q%2025%2040%2050%2050%20T%20100%2050' fill='none' "
    "stroke='%23FFF' stroke-width='1'/%3E%3C/svg%3E\");background-size:150px}"
    ".f-title{font-family:'Fraunces',serif;font-size:28px;color:var(--paper);"
    "z-index:2;position:relative}"
    ".f-sub{font-size:15px;color:rgba(246,241,231,.7);z-index:2;position:relative;"
    "margin-top:4px;margin-bottom:32px}"
    ".f-credit{font-size:12px;color:rgba(246,241,231,.5);"
    "font-family:'JetBrains Mono',monospace;z-index:2;position:relative}"
    ".f-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;"
    "position:relative;z-index:2;margin-top:48px;"
    "border-top:1px solid rgba(246,241,231,.1);padding-top:32px}"
    ".f-link{color:rgba(246,241,231,.8);font-size:14px;text-decoration:none;"
    "display:block;margin-bottom:12px;transition:color .2s}"
    ".f-link:hover{color:var(--terracotta)}"
    "</style>",
    unsafe_allow_html=True,
)
st.html(
    '<div class="footer-wrap">'
    '<div class="footer-bg"></div>'
    '<div class="f-title">CarWise</div>'
    '<div class="f-sub">Used Car Price Prediction &amp; Market Analysis • DWM Mini Project</div>'
    '<div class="f-credit">DATASET SOURCE: CARDEKHO</div>'
    '<div class="f-grid">'
    '<div><div class="f-link">Home</div><div class="f-link">Price Predictor</div></div>'
    '<div><div class="f-link">Market Analysis</div><div class="f-link">Car Segmentation</div></div>'
    '<div><div class="f-link">Insights</div></div>'
    '</div>'
    '</div>'
)