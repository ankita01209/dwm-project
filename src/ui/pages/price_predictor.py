import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from src.data_loader import load_data, load_model
from src.ui.components import render_plate_chip, render_odometer, render_road_divider, apply_plotly_config, format_inr

def get_empty_state_html():
    return """
    <div style="background: rgba(246,241,231,0.5); border: 2px dashed rgba(58,53,71,0.15); border-radius: 20px; padding: 64px 32px; text-align: center;">
        <svg viewBox="0 0 200 80" style="width: 140px; margin: 0 auto 24px; opacity: 0.3;" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 58 L22 40 Q28 30 42 28 L82 24 Q96 12 120 12 L146 14 Q160 18 170 36 L188 42 Q194 46 194 56 L194 60 L10 60 Z" fill="none" stroke="#3A3547" stroke-width="2"/>
            <path d="M92 26 Q102 17 120 17 L142 19 Q150 22 156 34 L92 34 Z" fill="none" stroke="#3A3547" stroke-width="1.5"/>
            <circle cx="52" cy="62" r="13" fill="none" stroke="#3A3547" stroke-width="2"/>
            <circle cx="148" cy="62" r="13" fill="none" stroke="#3A3547" stroke-width="2"/>
        </svg>
        <h3 style="font-family: 'Fraunces', serif; font-size: 24px; color: var(--ink); margin-bottom: 8px;">Fill in the details for your car</h3>
        <p style="font-size: 14px; color: rgba(58,53,71,0.6);">Provide the brand, age, and usage history on the left to estimate the final selling price using our dataset.</p>
    </div>
    """

def run_prediction_ui(df_full, form_data):
    model, err = load_model()
    if err:
        st.error(f"Cannot load Random Forest model: {err}")
        return

    # Create inference dataframe exactly matching model expectations
    input_df = pd.DataFrame([form_data])
    
    with st.spinner("Analyzing market patterns..."):
        time.sleep(0.4) # Add slight delay for dramatic effect
        raw_output = model.predict(input_df)[0]
        pred_price = raw_output
        
        # Calculate a simulated low/high range from error characteristics (MAE roughly ~144k but proportional to price)
        # Using 10% bounds for a realistic range estimation simulation logic from a single prediction pipeline
        low_bound = pred_price * 0.90
        high_bound = pred_price * 1.10
        
        # Defensive validation guards
        import math
        if not (math.isfinite(pred_price) and math.isfinite(low_bound) and math.isfinite(high_bound)) or pred_price <= 0:
            st.warning("A valid estimate could not be generated for these inputs. Please adjust the details and try again.")
            return
            
        pred_price = int(pred_price)
        low_bound = int(low_bound)
        high_bound = int(high_bound)
    
    st.html(
        f'<div class="animate-up" style="background:#2A2533;border-radius:20px;'
        f'padding:32px;box-shadow:0 12px 32px rgba(0,0,0,.15);color:#F6F1E7;text-align:center;">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'letter-spacing:.1em;color:rgba(246,241,231,.5);margin-bottom:16px;">ESTIMATED SELLING PRICE</div>'
        f'<div style="font-family:\'Fraunces\',serif;font-size:56px;color:#F6F1E7;'
        f'letter-spacing:-2px;line-height:1;margin-bottom:16px;">&#8377; {format_inr(int(pred_price))}</div>'
        f'<div style="font-size:14px;color:rgba(246,241,231,.7);'
        f'display:flex;align-items:center;justify-content:center;gap:12px;">'
        f'Expected market range: '
        f'<span style="background:rgba(201,123,79,.2);color:#C97B4F;'
        f'padding:4px 12px;border-radius:4px;font-weight:600;">&#8377; {format_inr(low_bound)} &ndash; &#8377; {format_inr(high_bound)}</span>'
        f'</div></div>'
    )
    
    st.markdown("<div class='spacer-3x'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<h4 style='font-family:Fraunces, serif; font-size:18px; margin-bottom:16px;'>Model-wide feature importance</h4>", unsafe_allow_html=True)
        from src.ui.components import get_feature_importances, render_feature_importance_chart
        df_imp = get_feature_importances(model)
        render_feature_importance_chart(df_imp)

    with col2:
        st.markdown("<h4 style='font-family:Fraunces, serif; font-size:18px; margin-bottom:16px;'>Top 3 Similar Listings</h4>", unsafe_allow_html=True)
        # Find 3 closest cars by brand, then age, then km
        sim = df_full[df_full['brand'] == form_data['brand']].copy()
        if len(sim) < 3:
            sim = df_full.copy()
        
        sim['age_diff'] = abs(sim['car_age'] - form_data['car_age'])
        sim['km_diff'] = abs(sim['km_driven'] - form_data['km_driven']) / 10000 
        sim['score'] = sim['age_diff']*2 + sim['km_diff']
        top_3 = sim.sort_values(by='score').head(3)
        
        for _, row in top_3.iterrows():
            html_str = (
                f'<div style="border-bottom:1px solid rgba(58,53,71,.08);padding-bottom:12px;margin-bottom:12px;">'
                f'<div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:4px;">'
                f'<div style="flex:1;min-width:0;word-wrap:break-word;"><span style="font-weight:600;font-size:14px;">{row["name"]}</span></div>'
                f'<div style="flex:none;white-space:nowrap;text-align:right;"><span style="font-family:\'Fraunces\',serif;color:var(--terracotta);font-size:16px;">&#8377;&nbsp;{format_inr(int(row["selling_price"]))}</span></div>'
                f'</div>'
                f'<div style="font-size:12px;color:rgba(58,53,71,.6);display:flex;gap:12px;">'
                f'<span>{row["year"]}</span> &bull; <span>{format_inr(row["km_driven"])} KM</span> &bull; <span>{row["fuel"]}</span>'
                f'</div></div>'
            )
            st.html(html_str)


def render_price_predictor():
    df_full = load_data()
    brands = sorted(df_full["brand"].unique().tolist())
    fuels = sorted(df_full["fuel"].unique().tolist())
    seller_types = sorted(df_full["seller_type"].unique().tolist())
    transmissions = sorted(df_full["transmission"].unique().tolist())
    owners = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    
    import textwrap
    st.markdown(textwrap.dedent("""
        <div class="animate-up">
            <h1 class="fraunces" style="font-size: 56px; margin-bottom: 12px;">Appraise a Vehicle</h1>
            <p class="body-text" style="max-width: 600px; margin-bottom: 48px;">Specify the parameters of your used car to evaluate its localized street value.</p>
        </div>
    """), unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1.2], gap="large")
    
    with c_left:
        with st.form("predictor_form"):
            st.markdown("<div class='label-text' style='margin-bottom:24px;'>Vehicle Specification</div>", unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            with r1c1: brand = st.selectbox("Manufacturer", brands, index=None, placeholder="Select brand")
            with r1c2: year = st.number_input("Year of Manufacture", min_value=1990, max_value=2020, value=2015, step=1)
            
            km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=1000000, value=50000, step=1000)
            
            r2c1, r2c2 = st.columns(2)
            with r2c1: fuel = st.selectbox("Fuel Type", fuels, index=None, placeholder="Select")
            with r2c2: transmission = st.selectbox("Transmission", transmissions, index=None, placeholder="Select")
            
            owner = st.selectbox("Previous Owners", owners, index=None, placeholder="Select history")
            seller_type = st.selectbox("Seller Classification", seller_types, index=None, placeholder="Select")
            
            st.markdown("<div class='spacer-2x'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Generate Estimate", type="primary", use_container_width=True)
    
    with c_right:
        if submitted:
            if None in [brand, fuel, transmission, owner, seller_type]:
                st.error("Please ensure all attributes are populated to maintain rigorous estimation accuracy.")
            else:
                form_data = {
                    "brand": brand,
                    "car_age": 2020 - year,
                    "km_driven": km_driven,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner
                }
                run_prediction_ui(df_full, form_data)
        else:
            st.markdown(get_empty_state_html(), unsafe_allow_html=True)
