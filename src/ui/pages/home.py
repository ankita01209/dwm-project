import streamlit as st
import os
import base64
import numpy as np
from src.ui.components import render_plate_chip, render_odometer, render_road_divider, workflow_map

def get_base64(filepath):
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def render_home():
    """Renders the Home View / Landing Page"""
    st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)

    # ── Hero CSS (single-line, no indentation issues) ──────────────────────
    st.markdown(
        "<style>"
        ".hero-search-card{background:var(--card);border-radius:20px;padding:24px;"
        "box-shadow:0 12px 32px rgba(58,53,71,.08);margin-top:-30px;position:relative;"
        "z-index:10;border:1px solid var(--line)}"
        ".hero-tabs{display:flex;gap:24px;border-bottom:2px solid rgba(226,216,196,.3);margin-bottom:24px}"
        ".hero-tab{font-weight:700;color:var(--muted);padding-bottom:12px;cursor:pointer;"
        "text-transform:uppercase;font-size:13px;letter-spacing:.05em}"
        ".hero-tab.active{color:var(--terracotta-dark);border-bottom:2px solid var(--terracotta);margin-bottom:-2px}"
        ".hero-inputs{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:16px;align-items:end}"
        "@media(max-width:900px){.hero-inputs{grid-template-columns:1fr}}"
        "</style>",
        unsafe_allow_html=True,
    )

    h_col1, h_col2 = st.columns([1.1, 1], gap="large")
    with h_col1:
        from src.data_loader import load_data
        import pandas as pd
        df_clean = load_data()
        n_clean = len(df_clean)
        
        from src.ui.components import format_inr
        
        st.markdown("<div class='animate-up'>", unsafe_allow_html=True)
        render_plate_chip("CARWISE / MARKET INTELLIGENCE", style="accent")
        st.markdown(
            "<h1 class='hero-head'>Know what your car<br>"
            "<span class='hero-italic Fraunces' style='font-style:italic;color:var(--terracotta-dark);'>"
            "is really worth.</span></h1>"
            f"<p class='body-text' style='max-width:540px;margin-bottom:32px;'>"
            f"CarWise leverages historical market listings and a Random Forest model "
            f"trained on {format_inr(n_clean)} CarDekho records to estimate used-car street values.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with h_col2:
        # Hanging-tag CSS (single-line) + card HTML via st.html
        st.markdown(
            "<style>"
            ".hanging-tag{background-color:var(--ink);border-radius:24px;padding:24px 32px;"
            "transform:rotate(-3deg);transform-origin:top center;"
            "box-shadow:0 32px 64px rgba(58,53,71,.25);position:relative;margin-top:10px;"
            "transition:transform .5s cubic-bezier(.175,.885,.32,1.275)}"
            ".hanging-tag:hover{transform:rotate(0deg)}"
            ".tag-hole{width:16px;height:16px;border-radius:50%;background:var(--paper);"
            "position:absolute;top:16px;left:50%;margin-left:-8px;"
            "box-shadow:inset 0 2px 4px rgba(0,0,0,.3)}"
            ".tag-string{position:absolute;top:-100px;left:50%;width:2px;height:110px;"
            "background:rgba(58,53,71,.2);margin-left:-1px;z-index:-1}"
            ".tag-car-svg{width:100%;height:auto;margin:16px 0 24px;display:block}"
            ".tag-val{font-family:'Fraunces',serif;font-size:48px;color:var(--paper);"
            "line-height:1;letter-spacing:-2px;text-align:center;margin-bottom:8px}"
            ".tag-sub{font-family:'JetBrains Mono',monospace;font-size:11px;text-align:center;"
            "color:rgba(246,241,231,.5);text-transform:uppercase;letter-spacing:.1em}"
            ".tag-range-bar{height:4px;border-radius:2px;background:rgba(246,241,231,.1);"
            "margin:16px 0;position:relative}"
            ".tag-range-fill{position:absolute;left:20%;right:20%;height:100%;"
            "background:var(--terracotta);border-radius:2px}"
            ".tag-kpi-row{display:flex;justify-content:space-between;"
            "border-top:1px dashed rgba(246,241,231,.15);padding-top:16px;text-align:center}"
            ".tag-kpi-col{flex:1}"
            ".tag-kpi-val{font-family:'Fraunces',serif;font-size:20px;color:var(--paper)}"
            ".tag-kpi-lbl{font-family:'JetBrains Mono',monospace;font-size:10px;"
            "color:rgba(246,241,231,.5);text-transform:uppercase}"
            "</style>",
            unsafe_allow_html=True,
        )
        st.html(
            '<div class="hanging-tag animate-up" style="animation-delay:0.2s;">'
            '<div class="tag-string"></div>'
            '<div class="tag-hole"></div>'
            '<svg viewBox="0 0 200 80" class="tag-car-svg" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M10 58 L22 40 Q28 30 42 28 L82 24 Q96 12 120 12 L146 14 Q160 18 170 36 L188 42 Q194 46 194 56 L194 60 L10 60 Z" fill="none" stroke="#F6F1E7" stroke-width="2"/>'
            '<path d="M92 26 Q102 17 120 17 L142 19 Q150 22 156 34 L92 34 Z" fill="none" stroke="#F6F1E7" stroke-width="1.5"/>'
            '<circle cx="52" cy="62" r="13" fill="none" stroke="#F6F1E7" stroke-width="2"/>'
            '<circle cx="52" cy="62" r="4" fill="#C97B4F"/>'
            '<circle cx="148" cy="62" r="13" fill="none" stroke="#F6F1E7" stroke-width="2"/>'
            '<circle cx="148" cy="62" r="4" fill="#C97B4F"/>'
            '<line x1="20" y1="46" x2="180" y2="46" stroke="#C97B4F" stroke-width="1"/>'
            '</svg>'
            '<div class="tag-sub" style="margin-bottom:8px;">RANDOM FOREST ESTIMATE</div>'
            '<div class="tag-val">&#8377; 4,85,000</div>'
            '<div class="tag-range-bar"><div class="tag-range-fill"></div></div>'
            '<div class="tag-kpi-row">'
            '<div class="tag-kpi-col"><div class="tag-kpi-val">7</div><div class="tag-kpi-lbl">FEATURES</div></div>'
            '<div class="tag-kpi-col"><div class="tag-kpi-val">3</div><div class="tag-kpi-lbl">SEGMENTS</div></div>'
            f'<div class="tag-kpi-col"><div class="tag-kpi-val">{format_inr(n_clean)}</div><div class="tag-kpi-lbl">RECORDS</div></div>'
            '</div>'
            '</div>'
        )

    render_road_divider()

    # ── Market Trust Framework ─────────────────────────────────────────────
    st.markdown("<h2 class='section-title animate-up'>Market Trust Framework</h2>", unsafe_allow_html=True)
    st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)

    trust_cols = st.columns(5)
    
    from src.ml.model_comparison import get_model_comparisons
    df_metrics, test_mean = get_model_comparisons()
    if df_metrics is not None and not df_metrics.empty:
        rf_row = df_metrics[df_metrics['Model'] == 'Random Forest'].iloc[0]
        rf_r2 = rf_row['R2'] * 100
        rf_mae = int(rf_row['MAE'])
        rf_rmse = int(rf_row['RMSE'])
        mae_pct = rf_row['MAE_Percentage']
    else:
        rf_r2 = 0
        rf_mae = 0
        rf_rmse = 0
        mae_pct = 0
        test_mean = 0

    n_dup = len(pd.read_csv("data/car_details.csv")) - n_clean

    metrics = [
        (str(format_inr(n_clean)), "Clean Records"),
        ("8",     "Attributes"),
        (str(n_dup),   "Duplicates Removed"),
        ("7",     "Prediction Features"),
        (f"{rf_r2:.1f}%", "Test-set R²"),
    ]
    for idx, col in enumerate(trust_cols):
        with col:
            st.markdown(f'<div class="animate-up" style="animation-delay:{idx*0.1}s;">', unsafe_allow_html=True)
            render_odometer(metrics[idx][0], is_highlight=(idx == 4))
            st.markdown(f"<div class='label-text' style='margin-top:12px;'>{metrics[idx][1]}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='spacer-4x'></div>", unsafe_allow_html=True)

    # Compare cards — use st.html() to avoid Markdown code-block parsing
    html_str = (
        '<div class="animate-up" style="display:flex;gap:24px;">'
        '<div style="flex:1;background:var(--card);border:1px solid var(--line);'
        'border-radius:var(--radius-card);padding:32px;box-shadow:0 6px 24px rgba(58,53,71,.07);">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-end;">'
        '<div>'
        '<h4 class="fraunces" style="font-size:24px;color:var(--ink);margin-bottom:4px;">Mean Error Limits</h4>'
        '<p class="body-text" style="font-size:14px;margin:0;">Absolute deviation on the test set</p>'
    )
    if test_mean > 0 and np.isfinite(test_mean):
        html_str += f'<p class="body-text" style="font-size:12px;margin:8px 0 0 0;color:rgba(58,53,71,0.6);">≈ {mae_pct:.1f}% of the average listing price (&#8377; {format_inr(int(test_mean))})</p>'
    
    html_str += (
        '</div>'
        '<div style="text-align:right;">'
        '<div class="label-text">TEST-SET MAE</div>'
        f'<div class="fraunces" style="font-size:32px;color:var(--terracotta);line-height:1;">&#8377; {format_inr(rf_mae)}</div>'
        '</div></div>'
    )
    
    if rf_rmse > rf_mae:
        html_str += f'<div class="label-text" style="margin-top:16px;font-size:10px;color:var(--muted);">RMSE (&#8377; {format_inr(rf_rmse)}) is larger than MAE because a few high-priced cars have large errors.</div>'
        
    html_str += (
        '</div>'
        '<div style="flex:1;background:var(--card);border:1px solid var(--line);'
        'border-radius:var(--radius-card);padding:32px;box-shadow:0 6px 24px rgba(58,53,71,.07);">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-end;">'
        '<div>'
        '<h4 class="fraunces" style="font-size:24px;color:var(--ink);margin-bottom:4px;">Variance Capture</h4>'
        '<p class="body-text" style="font-size:14px;margin:0;">Predictability on unseen listings</p>'
        '</div>'
        '<div style="text-align:right;">'
        '<div class="label-text">R&#178; Test Score</div>'
        f'<div class="fraunces" style="font-size:32px;color:var(--teal);line-height:1;">{rf_r2:.1f}%</div>'
        '</div></div></div>'
        '</div>'
    )
    
    html_str += (
        '<div class="label-text animate-up" style="margin-top:16px;font-size:10px;'
        'color:var(--muted);text-align:center;">NOTE: Model excludes geographic inflation and accident history &mdash; factors that limit ceiling accuracy.</div>'
    )
    st.html(html_str)

    render_road_divider()

    # ── Top Manufacturers — compact 4-col grid ────────────────────────────
    st.markdown("<h2 class='section-title animate-up'>Top Manufacturers</h2>", unsafe_allow_html=True)

    from src.data_loader import load_data
    df_home = load_data()
    top_brands = df_home["brand"].value_counts().head(8)

    # Grid CSS (single-line, no indentation)
    st.markdown(
        "<style>"
        ".brand-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}"
        ".brand-card{border:1px solid var(--line);border-radius:12px;background:var(--card);"
        "padding:12px 16px;display:flex;align-items:center;gap:12px;"
        "transition:transform .2s,box-shadow .2s;cursor:pointer}"
        ".brand-card:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(58,53,71,.08);"
        "border-color:var(--terracotta)}"
        ".brand-icon{width:36px;height:36px;border-radius:50%;background:var(--paper-2);"
        "color:var(--terracotta-dark);font-family:'Fraunces',serif;font-size:18px;"
        "display:flex;align-items:center;justify-content:center;flex-shrink:0}"
        ".brand-name{font-weight:600;color:var(--ink);font-size:14px;margin-bottom:1px}"
        ".brand-count{font-size:11px;color:var(--muted)}"
        "@media(max-width:900px){.brand-grid{grid-template-columns:repeat(2,1fr)}}"
        "</style>",
        unsafe_allow_html=True,
    )

    # Build the entire grid as one HTML string → st.html (no Markdown parsing)
    grid_html = '<div class="brand-grid animate-up">'
    for brand, count in top_brands.items():
        initial = str(brand)[0]
        grid_html += (
            f'<div class="brand-card">'
            f'<div class="brand-icon">{initial}</div>'
            f'<div><div class="brand-name">{brand}</div>'
            f'<div class="brand-count">{count} listings</div></div>'
            f'</div>'
        )
    grid_html += "</div>"
    st.html(grid_html)

    st.markdown("<div class='spacer-8x'></div>", unsafe_allow_html=True)

    # ── DWM Workflow ───────────────────────────────────────────────────────
    st.markdown("<h2 class='section-title'>DWM Workflow</h2>", unsafe_allow_html=True)
    workflow_map()

    st.markdown("<div class='spacer-8x'></div>", unsafe_allow_html=True)
