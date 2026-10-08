import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_data
from src.ui.components import render_plate_chip, apply_plotly_config

def render_segmentation():
    import textwrap
    st.markdown(textwrap.dedent("""
        <div class="animate-up">
            <h1 class="fraunces" style="font-size: 56px; margin-bottom: 12px;">Car Segmentation</h1>
            <p class="body-text" style="max-width: 600px; margin-bottom: 40px;">K-Means clustering algorithm maps three distinct vehicle profile archetypes directly from our empirical dataset.</p>
        </div>
    """), unsafe_allow_html=True)
    
    df = load_data()
    features = ['selling_price', 'car_age', 'km_driven']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Calculate centroids to define rules dynamically
    centroids = df.groupby('Cluster')[features].mean().reset_index()
    counts = df.groupby('Cluster').size().reset_index(name='Count')
    stats = pd.merge(centroids, counts, on='Cluster')
    stats = stats.sort_values('selling_price') # sort by price for consistency (Economy, Mid, Premium)
    
    st.markdown("<style>.seg-card-j { background: #fff; border: 1px solid rgba(58,53,71,0.06); border-radius:16px; padding:32px; box-shadow: 0 4px 12px rgba(0,0,0,0.02); } .seg-stat-row { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(58,53,71,0.1); padding: 8px 0; } .seg-stat-row:last-child { border-bottom: none; }</style>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="large")
    cols = [c1, c2, c3]
    
    names = ["Budget Daily Drivers", "Optimal Commuters", "Premium Low-Mileage"]
    desc = [
        "Older vehicles functioning efficiently as low-capital transit options.",
        "The standard sweet-spot mapping moderate age to moderate valuation.",
        "Newer models retaining high valuations with restricted odometer reads."
    ]
    
    # Need to map sorted cluster values back to layout safely
    sorted_clusters = stats.to_dict('records')
    for i in range(3):
        row = sorted_clusters[i]
        with cols[i]:
            pct = (row['Count'] / len(df)) * 100
            import textwrap
            html_content = textwrap.dedent(f"""
            <div class="seg-card-j animate-up" style="animation-delay: {i*0.1}s;">
                <div class="label-text" style="color:var(--terracotta); margin-bottom:12px;">SEGMENT 0{i+1}</div>
                <h3 style="font-family: 'Fraunces', serif; font-size: 26px; color: var(--ink); margin-bottom: 6px; line-height:1.1;">{names[i]}</h3>
                <p style="font-size: 13px; color: var(--muted); margin-bottom: 24px; line-height: 1.4;">{desc[i]}</p>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 32px; color: var(--ink); margin-bottom: 4px;">{pct:.1f}%</div>
                <div class="label-text" style="margin-bottom: 32px;">Market Share</div>
                <div class="seg-stat-row"><span class="label-text">Avg Valuation</span><span style="font-weight:600;">₹ {int(row['selling_price']):,}</span></div>
                <div class="seg-stat-row"><span class="label-text">Avg Car Age</span><span style="font-weight:600;">{row['car_age']:.1f} y</span></div>
                <div class="seg-stat-row"><span class="label-text">Avg Milage</span><span style="font-weight:600;">{int(row['km_driven']):,} KM</span></div>
            </div>
            """)
            st.markdown(html_content, unsafe_allow_html=True)
            
    st.markdown("<div class='spacer-4x'></div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-title'>Cluster Distribution Topography</h2>", unsafe_allow_html=True)
    df['Cluster_Name'] = df['Cluster'].map(lambda x: names[stats.index[stats['Cluster'] == x].tolist()[0]])
    
    fig = px.scatter(df, x="car_age", y="selling_price", color="Cluster_Name", hover_data=["brand", "km_driven"], color_discrete_sequence=['#5E8C8A', '#3A3547', '#C97B4F'])
    fig = apply_plotly_config(fig)
    fig.update_layout(xaxis_title="Vehicle Age", yaxis_title="Depreciation / Valuation Map", height=500, legend_title="Identified Profiles")
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0)))
    
    st.plotly_chart(fig, use_container_width=True, theme=None, config={'displayModeBar': False})
