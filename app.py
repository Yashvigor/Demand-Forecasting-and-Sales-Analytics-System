import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Set wide layout and premium page config
st.set_page_config(
    page_title="Industrial Alloy Demand Planning & Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏭"
)

# Custom CSS for Premium Glassmorphism & Dark Mode styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Header branding */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Custom metric card */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        font-weight: 500;
    }
    
    /* Tabs styling overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
    
    /* Box layout */
    .content-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sales_data_path = os.path.join(base_dir, "data", "sales_data.csv")
predictions_path = os.path.join(base_dir, "data", "output", "forecast_predictions.csv")
comparison_path = os.path.join(base_dir, "data", "output", "model_comparison.csv")

# Ensure files exist, or show installation/pipeline warning
if not os.path.exists(sales_data_path) or not os.path.exists(predictions_path):
    st.error("⚠️ Data files are missing! Please run the pipeline script first to generate predictions and serialize models.")
    st.markdown("""
    To set up and run the pipeline, run the following commands in your terminal:
    ```bash
    # Run the end-to-end forecasting pipeline
    python src/run_pipeline.py
    ```
    """)
    st.stop()

# Load Data
@st.cache_data
def load_all_data():
    raw_df = pd.read_csv(sales_data_path)
    raw_df['Order_Date'] = pd.to_datetime(raw_df['Order_Date'])
    
    fc_df = pd.read_csv(predictions_path)
    fc_df['Order_Date'] = pd.to_datetime(fc_df['Order_Date'])
    
    comp_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None
    
    return raw_df, fc_df, comp_df

df_raw, df_forecast, df_comparison = load_all_data()

# App branding
st.markdown('<div class="main-header">Industrial Alloy Demand Forecast & Sales Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decision support dashboard utilizing Prophet and XGBoost forecasting pipelines</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("🎛️ Interactive Filters")

# Filter by Region
regions = sorted(df_raw['Region'].unique())
selected_regions = st.sidebar.multiselect("Select Regions", options=regions, default=regions)

# Filter by Alloy Type
alloys = sorted(df_raw['Alloy_Type'].unique())
selected_alloys = st.sidebar.multiselect("Select Alloy Categories", options=alloys, default=alloys)

# Filter by Industry
industries = sorted(df_raw['Industry'].unique())
selected_industries = st.sidebar.multiselect("Select Target Industries", options=industries, default=industries)

# Filter by Year
years = sorted(df_raw['Order_Date'].dt.year.unique())
selected_years = st.sidebar.multiselect("Select Years", options=years, default=years)

# Apply filters to datasets
def apply_filters(df):
    f_df = df.copy()
    if selected_regions:
        f_df = f_df[f_df['Region'].isin(selected_regions)]
    if selected_alloys:
        f_df = f_df[f_df['Alloy_Type'].isin(selected_alloys)]
    if selected_years:
        f_df = f_df[f_df['Order_Date'].dt.year.isin(selected_years)]
    # 'Industry' column only exists in raw df and engineered features, not raw forecast output sometimes
    if 'Industry' in f_df.columns and selected_industries:
        f_df = f_df[f_df['Industry'].isin(selected_industries)]
    return f_df

filtered_raw = apply_filters(df_raw)
filtered_fc = apply_filters(df_forecast)

# Compute metrics for KPIs
total_revenue = filtered_raw['Revenue'].sum()
total_qty = filtered_raw['Quantity'].sum()
total_orders = len(filtered_raw)

# Identify top alloy
if not filtered_raw.empty:
    top_alloy_row = filtered_raw.groupby('Alloy_Type')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False).iloc[0]
    top_alloy = top_alloy_row['Alloy_Type']
    top_alloy_vol = top_alloy_row['Quantity']
else:
    top_alloy = "N/A"
    top_alloy_vol = 0

# RENDER KPI METRIC CARDS
st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-title">Total Sales Revenue</div>
        <div class="metric-value">${total_revenue:,.2f}</div>
        <div class="metric-sub">Across selected periods & regions</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Total Quantity Sold</div>
        <div class="metric-value">{int(total_qty):,} kg</div>
        <div class="metric-sub">Industrial alloy products</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Total Purchase Orders</div>
        <div class="metric-value">{total_orders:,}</div>
        <div class="metric-sub">B2B client transactions</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Top Selling Alloy</div>
        <div class="metric-value">{top_alloy.split('(')[0].strip()}</div>
        <div class="metric-sub">Volume: {int(top_alloy_vol):,} kg</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS SETUP -----------------
tab1, tab2 = st.tabs(["📊 Sales Analytics Dashboard", "📈 Demand Forecasting & Models"])

# ================= TAB 1: SALES ANALYTICS =================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("Monthly Sales Volume Trend")
        
        # Monthly aggregation
        filtered_raw['Year_Month'] = filtered_raw['Order_Date'].dt.to_period('M').dt.to_timestamp()
        monthly_agg = filtered_raw.groupby('Year_Month')[['Quantity', 'Revenue']].sum().reset_index()
        
        fig_trend = px.line(
            monthly_agg, x='Year_Month', y='Quantity', markers=True,
            labels={'Quantity': 'Quantity Sold (kg)', 'Year_Month': 'Date'},
            color_discrete_sequence=['#38bdf8']
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#334155')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("Alloy Type Distribution")
        
        # Group by alloy
        alloy_agg = filtered_raw.groupby('Alloy_Type')['Quantity'].sum().reset_index()
        
        fig_pie = px.pie(
            alloy_agg, values='Quantity', names='Alloy_Type', hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("Regional Performance (Revenue)")
        
        region_agg = filtered_raw.groupby('Region')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=True)
        
        fig_region = px.bar(
            region_agg, x='Revenue', y='Region', orientation='h',
            color='Revenue', color_continuous_scale='Blues',
            labels={'Revenue': 'Revenue ($)', 'Region': 'Sales Region'}
        )
        fig_region.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_region, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("Industry Demand Breakdown")
        
        ind_agg = filtered_raw.groupby('Industry')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False)
        
        fig_ind = px.bar(
            ind_agg, x='Industry', y='Quantity',
            color='Industry', color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'Quantity': 'Quantity Sold (kg)', 'Industry': 'Client Industry'}
        )
        fig_ind.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            showlegend=False
        )
        st.plotly_chart(fig_ind, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: DEMAND FORECASTING =================
with tab2:
    st.subheader("🔮 Model Forecast Visualizations & Comparison")
    
    # Selection of Series to plot
    st.markdown("Select a specific series to visualize actual historical data vs models' outputs:")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        f_alloy = st.selectbox("Forecast Alloy Category", options=df_forecast['Alloy_Type'].unique())
    with col_sel2:
        f_region = st.selectbox("Forecast Region", options=df_forecast['Region'].unique())
        
    # Filter forecast df by specific group
    series_df = df_forecast[(df_forecast['Alloy_Type'] == f_alloy) & (df_forecast['Region'] == f_region)].copy()
    series_df = series_df.sort_values(by='Order_Date').reset_index(drop=True)
    
    # 90-day filter for test/future projection
    horizon = st.slider("Select Forecast Horizon (Days out-of-sample)", min_value=30, max_value=90, value=90, step=30)
    
    # Split into history (training + test) and future forecast
    max_history_date = series_df[series_df['Quantity'] > 0]['Order_Date'].max()
    if pd.isna(max_history_date):
        max_history_date = series_df['Order_Date'].max() - pd.Timedelta(days=90)
        
    # Plot forecast
    fig_fc = go.Figure()
    
    # 1. Plot Actual demand
    history_df = series_df[series_df['Order_Date'] <= max_history_date]
    fig_fc.add_trace(go.Scatter(
        x=history_df['Order_Date'], y=history_df['Quantity'],
        mode='lines', name='Actual Quantity (kg)', line=dict(color='#f8fafc', width=1.5)
    ))
    
    # 2. Plot Prophet Forecast
    fig_fc.add_trace(go.Scatter(
        x=series_df['Order_Date'], y=series_df['Prophet_Forecast'],
        mode='lines', name='Prophet Forecast', line=dict(color='#38bdf8', width=2, dash='dash')
    ))
    
    # 3. Plot Prophet Confidence Interval Bounds
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([series_df['Order_Date'], series_df['Order_Date'][::-1]]),
        y=pd.concat([series_df['Prophet_Upper'], series_df['Prophet_Lower'][::-1]]),
        fill='toself', fillcolor='rgba(56, 189, 248, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='Prophet 95% Confidence Interval'
    ))
    
    # 4. Plot XGBoost Forecast (Only for historical test segment + valid feature rows)
    xgb_df = series_df.dropna(subset=['XGBoost_Forecast'])
    fig_fc.add_trace(go.Scatter(
        x=xgb_df['Order_Date'], y=xgb_df['XGBoost_Forecast'],
        mode='lines', name='XGBoost Forecast', line=dict(color='#fb923c', width=2, dash='dot')
    ))
    
    fig_fc.update_layout(
        title=f"Actual vs Model Forecast: {f_alloy.split('(')[0]} in {f_region} Region",
        paper_bgcolor='rgba(30, 41, 59, 0.45)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Model Comparison Metrics display
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📐 Model Performance Evaluation Comparison")
    
    if df_comparison is not None:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        with col_m2:
            st.markdown("""
            **Metrics Interpretation:**
            * **MAE (Mean Absolute Error):** Measures average absolute forecasting error magnitude. Lower values denote higher accuracy.
            * **RMSE (Root Mean Square Error):** Penalizes larger forecast errors more heavily.
            * **MAPE (Mean Absolute Percentage Error):** Represents errors as a percentage of actual B2B client order quantities.
            * **WAPE (Weighted Absolute Percentage Error):** Normalizes percentage errors to reduce bias from low-volume transaction days.
            """)
    else:
        st.warning("Comparison metrics not found. Run the pipeline script to evaluate models.")
    st.markdown('</div>', unsafe_allow_html=True)
