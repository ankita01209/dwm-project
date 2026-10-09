import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def plotly_theme():
    """Returns a unified layout dictionary for all Plotly charts."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif", color="#3A3547"),
        title=dict(font=dict(family="Fraunces, serif", size=22, color="#3A3547")),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(58,53,71,0.08)", 
            zeroline=False, 
            tickfont=dict(color="#3A3547"),
            title=dict(font=dict(size=13, color="rgba(58,53,71,0.7)"), standoff=20),
            automargin=True
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(58,53,71,0.08)", 
            zeroline=False, 
            tickfont=dict(color="#3A3547"),
            title=dict(font=dict(size=13, color="rgba(58,53,71,0.7)"), standoff=20),
            automargin=True
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        hoverlabel=dict(
            bgcolor="#3A3547",
            font_size=13,
            font_family="DM Sans, sans-serif"
        )
    )

def apply_plotly_config(fig):
    fig.update_layout(**plotly_theme())
    # Format currency hovering on any 'price' dimensions automatically
    m = fig.data
    for tr in m:
        if hasattr(tr, 'hovertemplate') and tr.hovertemplate:
            tr.hovertemplate = tr.hovertemplate.replace('%{y}', '₹%{y:,.0f}').replace('%{x}', '₹%{x:,.0f}')
    return fig

def format_inr(number):
    try:
        s, *d = str(number).partition(".")
        r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]]) if len(s) > 3 else s
        return "".join([r] + d)
    except:
        return str(number)

def render_plate_chip(text, style="default"):
    """Renders a car license plate styled textual chip."""
    color = "#3A3547" if style == "default" else "#C97B4F"
    st.markdown(f'<div class="plate-chip" style="color: {color}; border-color: {color};">{text}</div>', unsafe_allow_html=True)

def render_odometer(value, is_highlight=False):
    """Renders a mechanical odometer wrapper around a value."""
    cls = "odometer tera" if is_highlight else "odometer"
    st.markdown(f'<div class="{cls}">{value}</div>', unsafe_allow_html=True)

def render_road_divider():
    st.markdown('<div class="road-divider"></div>', unsafe_allow_html=True)

def workflow_map():
    """Renders the interconnected DWM workflow path."""
    from src.data_loader import load_data
    from src.ml.model_comparison import get_model_comparisons
    import pandas as pd
    n_clean = len(load_data())
    n_dup = len(pd.read_csv("data/car_details.csv")) - n_clean
    df_metrics, _ = get_model_comparisons()
    if df_metrics is not None and not df_metrics.empty:
        rf_r2 = df_metrics[df_metrics['Model'] == 'Random Forest'].iloc[0]['R2'] * 100
    else:
        rf_r2 = 0
        
    html = f"""
    <style>
        .wf-container {{ width: 100%; display: flex; align-items: flex-start; justify-content: space-between; position: relative; margin: 40px 0; overflow-x: auto; padding-bottom: 24px; scrollbar-width: none; }}
        .wf-line {{ position: absolute; top: 18px; left: 0; right: 0; height: 3px; background: repeating-linear-gradient(90deg, var(--line) 0, var(--line) 12px, transparent 12px, transparent 24px); z-index: 1; }}
        .wf-node {{ position: relative; z-index: 2; text-align: center; width: 120px; flex-shrink: 0; cursor: pointer; transition: transform 0.2s; }}
        .wf-node:hover {{ transform: translateY(-4px); }}
        .wf-circle {{ width: 36px; height: 36px; border-radius: 50%; background: var(--ink); color: var(--paper); display: flex; align-items: center; justify-content: center; margin: 0 auto 12px; font-family: 'JetBrains Mono', monospace; font-weight: 700; border: 4px solid var(--paper); box-shadow: 0 4px 12px rgba(58,53,71,0.15); }}
        .wf-node.active .wf-circle {{ background: var(--terracotta); }}
        .wf-title {{ font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 700; color: var(--ink); margin-bottom: 4px; line-height: 1.2; }}
        .wf-desc {{ font-family: 'DM Sans', sans-serif; font-size: 11px; color: var(--muted); line-height: 1.4; padding: 0 4px; opacity: 0; transition: opacity 0.2s; }}
        .wf-node:hover .wf-desc {{ opacity: 1; }}
        @media (max-width: 768px) {{
            .wf-container {{ flex-direction: column; align-items: center; }}
            .wf-line {{ width: 3px; height: 100%; top: 0; left: 50%; margin-left: -1px; background: repeating-linear-gradient(180deg, #EBE2D0 0, #EBE2D0 12px, transparent 12px, transparent 24px); }}
            .wf-node {{ display: flex; width: 100%; text-align: left; margin-bottom: 16px; align-items: flex-start; }}
            .wf-circle {{ margin: 0 16px 0 0; }}
            .wf-title {{ padding-top: 8px; }}
            .wf-desc {{ opacity: 1; padding: 0; }}
        }}
    </style>
    <div class="wf-container animate-up">
        <div class="wf-line"></div>
        <div class="wf-node"><div class="wf-circle">01</div><div class="wf-title">Data<br>Understanding</div><div class="wf-desc">Exploring CarDekho's raw vehicle listings</div></div>
        <div class="wf-node"><div class="wf-circle">02</div><div class="wf-title">Data<br>Cleaning</div><div class="wf-desc">Removed {n_dup} duplicate records</div></div>
        <div class="wf-node"><div class="wf-circle">03</div><div class="wf-title">Preprocessing</div><div class="wf-desc">Derived car_age & logged conversions</div></div>
        <div class="wf-node"><div class="wf-circle">04</div><div class="wf-title">EDA</div><div class="wf-desc">Mapped price relationships to KM</div></div>
        <div class="wf-node"><div class="wf-circle">05</div><div class="wf-title">K-Means<br>Clustering</div><div class="wf-desc">Discovered 3 distinct vehicle profiles</div></div>
        <div class="wf-node"><div class="wf-circle">06</div><div class="wf-title">Regression</div><div class="wf-desc">Trained Random Forest ensemble</div></div>
        <div class="wf-node"><div class="wf-circle">07</div><div class="wf-title">Model<br>Evaluation</div><div class="wf-desc">Attained {rf_r2:.1f}% variance capture</div></div>
        <div class="wf-node active"><div class="wf-circle">08</div><div class="wf-title">Streamlit<br>Dashboard</div><div class="wf-desc">You are here</div></div>
    </div>
    """
    import textwrap
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

@st.cache_data
def get_feature_importances():
    """Loads aggregate base feature importances natively from centralized metrics."""
    import pandas as pd
    import json
    import os
    try:
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'metrics.json')
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        importances = data.get('rf_feature_importance', {})
        df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
        df['Importance'] = df['Importance'] * 100
        df = df.sort_values(by='Importance', ascending=True)
        df['Label'] = df['Feature'].replace({
            'car_age': 'Car age',
            'km_driven': 'Kilometres driven',
            'fuel': 'Fuel type',
            'seller_type': 'Seller type',
            'transmission': 'Transmission',
            'owner': 'Owner history',
            'brand': 'Brand'
        })
        return df
    except Exception:
        return None

def render_feature_importance_chart(df):
    """Renders a standard Plotly horizontal bar chart for feature importances."""
    if df is None or df.empty:
        st.markdown(
            '<div style="background:#EFE7D6;color:#2A2533;padding:16px;'
            'border-radius:12px;font-size:14px;border:1px solid #E2D8C4;'
            'margin-bottom:16px;">'
            'Model-wide feature importance is currently unavailable.'
            '</div>', unsafe_allow_html=True
        )
        return
        
    colors = ['#3A3547'] * (len(df) - 1) + ['#C97B4F']
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Label'],
        orientation='h',
        marker_color=colors,
        text=df['Importance'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        textfont=dict(color='#3A3547', family='DM Sans')
    ))
    fig = apply_plotly_config(fig)
    fig.update_layout(
        height=320, 
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis_visible=False,
        yaxis_title="",
        xaxis_range=[0, df['Importance'].max() * 1.25],
        dragmode=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
