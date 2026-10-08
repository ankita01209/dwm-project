import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_loader import load_model
from src.ui.components import apply_plotly_config

def render_insights():
    st.html(
        f'<div class="animate-up">'
        f'<h1 class="fraunces" style="font-size:56px;margin-bottom:12px;">Analytical Insights</h1>'
        f'<p class="body-text" style="max-width:600px;margin-bottom:40px;">Evaluating raw regressor weights driving valuations across the Random Forest topology.</p>'
        f'</div>'
    )
    
    st.markdown("<h2 class='section-title'>Model Performance</h2>", unsafe_allow_html=True)
    
    from src.ml.model_comparison import get_model_comparisons
    from src.ui.components import format_inr
    
    df_metrics, test_mean = get_model_comparisons()
    
    if df_metrics is not None and not df_metrics.empty:
        # Reformat metrics for display
        df_display = df_metrics[['Model', 'MAE', 'R2', 'Train_MAE']].copy()
        
        # Calculate dynamic text values
        lr_row = df_metrics[df_metrics['Model'] == 'Linear Regression'].iloc[0]
        dt_row = df_metrics[df_metrics['Model'] == 'Decision Tree'].iloc[0]
        rf_row = df_metrics[df_metrics['Model'] == 'Random Forest'].iloc[0]
        
        error_diff = int(lr_row['MAE'] - rf_row['MAE'])
        rf_train_r2 = rf_row['Train_R2'] * 100
        rf_test_r2 = rf_row['R2'] * 100
        dt_train_r2 = dt_row['Train_R2'] * 100
        
        # Format columns symmetrically
        df_display['MAE'] = df_display['MAE'].apply(lambda x: f"&#8377; {format_inr(int(x))}")
        df_display['Train MAE'] = df_display['Train_MAE'].apply(lambda x: f"&#8377; {format_inr(int(x))}")
        df_display['R²'] = df_display['R2'].apply(lambda x: f"{x * 100:.1f}%")
        
        df_display = df_display[['Model', 'MAE', 'R²', 'Train MAE']]
        
        # Generate HTML table
        table_html = (
            '<div class="animate-up" style="background:var(--card); border:1px solid var(--line);'
            'border-radius:var(--radius-card); padding:24px; box-shadow:0 6px 24px rgba(58,53,71,0.07);'
            'margin-bottom:24px;">'
            '<table style="width:100%; text-align:left; border-collapse:collapse;">'
            '<tr style="border-bottom:1px solid var(--line); color:var(--muted); font-family:\'JetBrains Mono\', monospace; font-size:12px;">'
            '<th style="padding:12px 16px;">Model</th>'
            '<th style="padding:12px 16px; text-align:right;">Test MAE</th>'
            '<th style="padding:12px 16px; text-align:right;">Test R²</th>'
            '<th style="padding:12px 16px; text-align:right;">Train MAE</th>'
            '</tr>'
        )
        
        for _, row in df_display.iterrows():
            row_bg = "background:rgba(201,123,79,0.05);" if row['Model'] == 'Random Forest' else ""
            table_html += (
                f'<tr style="border-bottom:1px solid rgba(58,53,71,0.05); {row_bg}">'
                f'<td style="padding:16px; font-weight:600; color:var(--ink);">{row["Model"]}</td>'
                f'<td style="padding:16px; text-align:right; font-family:\'JetBrains Mono\', monospace;">{row["MAE"]}</td>'
                f'<td style="padding:16px; text-align:right; font-family:\'JetBrains Mono\', monospace;">{row["R²"]}</td>'
                f'<td style="padding:16px; text-align:right; font-family:\'JetBrains Mono\', monospace;">{row["Train MAE"]}</td>'
                f'</tr>'
            )
            
        table_html += '</table></div>'
        
        st.html(table_html)
        
        st.html(
            f'<div class="animate-up" style="background:var(--paper-2); border-left:4px solid var(--terracotta);'
            f'padding:24px; margin-bottom:48px; border-radius:0 12px 12px 0;">'
            f'<h4 class="fraunces" style="font-size:20px; color:var(--ink); margin-bottom:8px;">Why Random Forest?</h4>'
            f'<p class="body-text" style="font-size:15px; margin:0; line-height:1.5;">'
            f'Random Forest outperformed simple baselines, lowering expected error by <strong>&#8377; {format_inr(error_diff)}</strong> compared to Linear Regression '
            f'while maintaining strong generalization (Train R² {rf_train_r2:.1f}% vs Test R² {rf_test_r2:.1f}%). '
            f'Decision tree severely overfit (Train R² {dt_train_r2:.1f}%).'
            f'</p></div>'
        )
    else:
        st.warning("Model comparison metrics could not be generated.", icon="⚠️")
    
    st.markdown("<h2 class='section-title'>Feature Dominance</h2>", unsafe_allow_html=True)
    
    from src.ui.components import get_feature_importances, render_feature_importance_chart
    model, _ = load_model()
    df_imp = get_feature_importances(model)
    render_feature_importance_chart(df_imp)
    
    strongest = "Car Age"
    second = "Manufacturer"
    if df_imp is not None and not df_imp.empty:
        # sorted ascending, so last is strongest
        strongest = df_imp.iloc[-1]['Label']
        second = df_imp.iloc[-2]['Label']

    st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Derived Takeaways</h2>", unsafe_allow_html=True)
    
    st.markdown("<style>.takeaway-box { background: rgba(58,53,71,0.03); border-left: 3px solid var(--terracotta); padding: 24px; margin-bottom: 16px; border-radius: 0 12px 12px 0; }</style>", unsafe_allow_html=True)
    st.html(
        f'<div class="takeaway-box">'
        f'<h4 style="font-family:\'Fraunces\',serif;font-size:20px;color:var(--ink);margin-bottom:8px;">Model Drivers</h4>'
        f'<p class="body-text" style="font-size:15px;margin:0;">Based strictly on regressor weights, <strong>{strongest}</strong> dictates valuation the heaviest, followed sequentially by <strong>{second}</strong>.</p>'
        f'</div>'
        f'<div class="takeaway-box">'
        f'<h4 style="font-family:\'Fraunces\',serif;font-size:20px;color:var(--ink);margin-bottom:8px;">Pipeline Mechanics</h4>'
        f'<p class="body-text" style="font-size:15px;margin:0;">Target variables were logged dynamically mapping exponential decays natively solving skew distributions prior to processing.</p>'
        f'</div>'
        f'<div class="takeaway-box" style="border-left-color:var(--teal);">'
        f'<h4 style="font-family:\'Fraunces\',serif;font-size:20px;color:var(--ink);margin-bottom:8px;">Limitations &amp; Next Steps</h4>'
        f'<p class="body-text" style="font-size:15px;margin:0;">The base framework tests firmly excluding granular geographic inflation nodes and specific vehicle accident histories. Future arrays require condition indexing.</p>'
        f'</div>'
    )
    
    st.markdown("<div class='spacer-8x'></div>", unsafe_allow_html=True)
