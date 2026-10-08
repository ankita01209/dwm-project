import streamlit as st

def inject_global_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,400&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --paper: #F6F1E7;
    --paper-2: #EFE7D6;
    --card: #FFFFFF;
    --ink: #2A2533;
    --plum: #3A3547;
    --plum-2: #4A4559;
    --terracotta: #C97B4F;
    --terracotta-dark: #B4633A;
    --teal: #5E8C8A;
    --line: #E2D8C4;
    --muted: #6B6477;
    
    --radius-card: 16px;
    --radius-btn: 12px;
}

/* Base Typographic & Reset */
* {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    box-sizing: border-box;
}

/* Hide Streamlit Chrome */
header[data-testid="stHeader"] { display: none; }
footer { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

/* Custom Container Constraint (Fix Bug 1: No horizontal overflow) */
div.block-container {
    max-width: 1240px !important;
    padding: 0 32px !important;
    margin: 0 auto !important;
}
@media (max-width: 768px) {
    div.block-container { padding: 0 24px !important; }
}

/* Faint topographic contour line pattern behind hero via app.py, default app background handled by TOML */

/* Road Dividers */
.road-divider {
    height: 3px;
    background: transparent;
    border-bottom: 3px dashed var(--line);
    margin: 72px 0;
}

/* Plate Chips */
.plate-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 10px;
    border: 3px double var(--ink);
    border-radius: 6px;
    color: var(--ink);
    margin-bottom: 16px;
    background-color: transparent;
}

/* Odometer Rollers: Fix Bug 6: No leading zeroes */
.odometer {
    font-family: 'JetBrains Mono', monospace;
    background: var(--ink);
    color: var(--paper);
    padding: 8px 12px;
    border-radius: 6px;
    display: inline-block;
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.4);
}
.odometer.tera { color: var(--terracotta); }

/* Typography Scale */
h1, h2, h3, h4, h5, h6, .fraunces {
    font-family: 'Fraunces', serif !important;
    color: var(--ink) !important;
    margin: 0;
}
.hero-head {
    font-size: clamp(48px, 6vw, 80px);
    line-height: 1.02;
    letter-spacing: -0.02em;
    margin-bottom: 24px;
}
.hero-italic {
    font-style: italic;
    color: var(--terracotta-dark) !important;
}
.section-title {
    font-size: 40px;
    line-height: 1.1;
    margin-bottom: 16px;
}
.card-title {
    font-size: 20px;
}
.body-text, p {
    font-size: 16px;
    line-height: 1.6;
    color: var(--ink);
}
.label-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--muted);
}

/* Streamlit Button Overrides */
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background-color: var(--terracotta) !important;
    color: var(--ink) !important; /* Dark text on terracotta */
    border-radius: var(--radius-btn) !important;
    border: none !important;
    padding: 12px 24px;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(58,53,71,0.07);
}
div.stButton > button[kind="primary"] * { color: var(--ink) !important; }
div[data-testid="stFormSubmitButton"] > button * { color: var(--ink) !important; }

div.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: var(--ink) !important;
    border-radius: var(--radius-btn) !important;
    border: 1px solid var(--ink) !important;
    padding: 12px 24px;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    transition: background 0.2s ease;
}
div.stButton > button[kind="secondary"]:hover {
    background-color: rgba(42,37,51,0.05) !important;
}
div.stButton > button[kind="secondary"] * { color: var(--ink) !important; }

/* Clean Forms */
.stTextInput>div>div>input {
    border-color: var(--line) !important;
}

/* Animations (respecting prefers-reduced-motion) */
@media (prefers-reduced-motion: no-preference) {
    .animate-up {
        animation: fadeRise 0.5s ease-out forwards;
    }
}
@keyframes fadeRise {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Universal Spacing */
.spacer-1x { height: 8px; }
.spacer-2x { height: 16px; }
.spacer-3x { height: 24px; }
.spacer-4x { height: 32px; }
.spacer-9x { height: 72px; }
.spacer-12x { height: 96px; }

/* Sticky Flex Nav (Fix Bug 2) */
.nav-container {
    position: sticky;
    top: 0;
    background: rgba(246, 241, 231, 0.95);
    backdrop-filter: blur(8px);
    z-index: 999;
    padding: 16px 32px;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nav-logo-group {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Fraunces', serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--ink);
    white-space: nowrap;
}
.nav-links-row {
    display: flex;
    align-items: center;
    gap: 24px;
}
.nav-item {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 500;
    color: var(--ink);
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
    padding: 4px 2px;
    position: relative;
}
.nav-item.active {
    color: var(--terracotta-dark);
}
.nav-item.active::after {
    content: '';
    position: absolute;
    bottom: -4px; left: 0; right: 0;
    height: 2px;
    background: var(--terracotta);
}
@media (max-width: 900px) {
    .nav-links-row {
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 4px;
        display: -webkit-box;
    }
}
</style>
    """, unsafe_allow_html=True)
