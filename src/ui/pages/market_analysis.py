import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_loader import load_data
from src.ui.components import apply_plotly_config

def render_market_analysis():
    import textwrap
    st.markdown(textwrap.dedent("""
        <div class="animate-up">
            <h1 class="fraunces" style="font-size: 56px; margin-bottom: 12px;">Market Analysis</h1>
            <p class="body-text" style="max-width: 600px; margin-bottom: 40px;">Explore underlying pricing patterns across the CarDekho dataset through interactive granular constraints.</p>
        </div>
    """), unsafe_allow_html=True)
    
    df = load_data()
    
    # FILTER BAR
    st.markdown("<div style='background:rgba(246,241,231,0.5); padding:24px 32px; border-radius:12px; margin-bottom:32px; border:1px solid rgba(58,53,71,0.06);'>", unsafe_allow_html=True)
    st.markdown("<div class='label-text' style='margin-bottom:16px;'>Global Constraints</div>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_brand = st.multiselect("Manufacturer", options=sorted(df['brand'].unique()), default=[])
    with f2:
        sel_fuel = st.multiselect("Fuel Profile", options=sorted(df['fuel'].unique()), default=[])
    with f3:
        sel_trans = st.multiselect("Transmission", options=sorted(df['transmission'].unique()), default=[])
    
    c_y1, c_y2 = st.columns(2)
    with c_y1:
        min_y, max_y = int(df['year'].min()), int(df['year'].max())
        sel_years = st.slider("Manufacturing Window", min_value=min_y, max_value=max_y, value=(min_y, max_y))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Application of filters
    dff = df.copy()
    if sel_brand: dff = dff[dff['brand'].isin(sel_brand)]
    if sel_fuel: dff = dff[dff['fuel'].isin(sel_fuel)]
    if sel_trans: dff = dff[dff['transmission'].isin(sel_trans)]
    dff = dff[(dff['year'] >= sel_years[0]) & (dff['year'] <= sel_years[1])]
    
    if len(dff) == 0:
        st.warning("No listings parallel your tight constraint structure. Widen your bounds.")
        return

    # KPI ROW
    avg_price = dff['selling_price'].mean()
    med_km = dff['km_driven'].median()
    top_brand = dff['brand'].mode()[0] if not dff['brand'].empty else "N/A"
    
    st.markdown(textwrap.dedent(f"""
    <div style="display:flex; justify-content:space-between; border-top:1px solid rgba(58,53,71,0.08); border-bottom:1px solid rgba(58,53,71,0.08); padding:32px 0; margin-bottom:48px;">
        <div style="flex:1; border-right:1px solid rgba(58,53,71,0.08); padding-right:24px;">
            <div class="fraunces" style="font-size:36px; line-height:1; color:var(--ink); margin-bottom:8px;">{len(dff):,}</div>
            <div class="label-text">Filtered Listings</div>
        </div>
        <div style="flex:1; border-right:1px solid rgba(58,53,71,0.08); padding:0 24px;">
            <div class="fraunces" style="font-size:36px; line-height:1; color:var(--terracotta); margin-bottom:8px;">₹ {int(avg_price):,}</div>
            <div class="label-text">Average Valuation</div>
        </div>
        <div style="flex:1; border-right:1px solid rgba(58,53,71,0.08); padding:0 24px;">
            <div class="fraunces" style="font-size:36px; line-height:1; color:var(--ink); margin-bottom:8px;">{int(med_km):,}</div>
            <div class="label-text">Median KM Driven</div>
        </div>
        <div style="flex:1; padding-left:24px;">
            <div class="fraunces" style="font-size:36px; line-height:1; color:var(--ink); margin-bottom:8px;">{top_brand}</div>
            <div class="label-text">Dominant Brand</div>
        </div>
    </div>
    """), unsafe_allow_html=True)
    
    # 2x2 CHART GRID
    ch_col1, ch_col2 = st.columns(2, gap="large")
    
    with ch_col1:
        st.markdown("<h3 style='font-size:20px; font-weight:600; font-family:Fraunces, serif; margin-bottom:16px;'>Price Distribution Topology</h3>", unsafe_allow_html=True)
        fig1 = px.histogram(dff, x='selling_price', nbins=50, color_discrete_sequence=['#5E8C8A'])
        fig1 = apply_plotly_config(fig1)
        fig1.update_layout(xaxis_title="Listing Price", yaxis_title="Volume", height=320, showlegend=False)
        # Force a rounded rendering style for bins mimicking classified journals
        fig1.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig1, use_container_width=True, theme=None, config={'displayModeBar': False})
        
        st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size:20px; font-weight:600; font-family:Fraunces, serif; margin-bottom:16px;'>Valuation vs Fuel Type</h3>", unsafe_allow_html=True)
        fig3 = px.box(dff, x='fuel', y='selling_price', color_discrete_sequence=['#C97B4F'])
        fig3 = apply_plotly_config(fig3)
        fig3.update_layout(xaxis_title="Fuel Constitution", yaxis_title="Valuation Spread", height=320)
        st.plotly_chart(fig3, use_container_width=True, theme=None, config={'displayModeBar': False})

    with ch_col2:
        st.markdown("<h3 style='font-size:20px; font-weight:600; font-family:Fraunces, serif; margin-bottom:16px;'>Depreciation via Age Vectors</h3>", unsafe_allow_html=True)
        age_df = dff.groupby("car_age")["selling_price"].mean().reset_index()
        fig2 = px.scatter(age_df, x='car_age', y='selling_price', trendline="ols", color_discrete_sequence=['#3A3547'])
        fig2 = apply_plotly_config(fig2)
        fig2.update_traces(marker=dict(size=12, color="#EBE2D0", line=dict(width=2, color="#3A3547")))
        fig2.update_layout(xaxis_title="Car Age (Years Computed against 2020 Pivot)", yaxis_title="Average Capital Value", height=320)
        st.plotly_chart(fig2, use_container_width=True, theme=None, config={'displayModeBar': False})
        
        st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size:20px; font-weight:600; font-family:Fraunces, serif; margin-bottom:16px;'>Top 10 Manufacturers Volume</h3>", unsafe_allow_html=True)
        top_10 = dff['brand'].value_counts().head(10).reset_index()
        top_10.columns = ['Manufacturer', 'Volume']
        fig4 = px.bar(top_10, y='Manufacturer', x='Volume', orientation='h', color_discrete_sequence=['#5E8C8A'])
        fig4 = apply_plotly_config(fig4)
        fig4.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Total Volume Filtered", yaxis_title="", height=320)
        st.plotly_chart(fig4, use_container_width=True, theme=None, config={'displayModeBar': False})
