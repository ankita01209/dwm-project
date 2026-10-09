import pandas as pd
import numpy as np
import streamlit as st
import json
import os
from src.data_loader import load_data

@st.cache_data
def get_model_comparisons():
    """
    Loads comparative metrics inherently directly from centralized unified model/metrics.json preserving absolute source of truth.
    """
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return pd.DataFrame(), 0
        
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df_metrics = pd.DataFrame(data['models'])
    # Rename columns to match existing UI usage safely
    df_metrics = df_metrics.rename(columns={
        'Test_MAE': 'MAE',
        'Test_RMSE': 'RMSE',
        'Test_R2': 'R2'
    })
    
    df = load_data()
    test_mean = df['selling_price'].mean()
    
    # Inject legacy computed metrics securely expected by UI elements 
    df_metrics['MAE_Percentage'] = (df_metrics['MAE'] / test_mean) * 100
    
    return df_metrics, test_mean
