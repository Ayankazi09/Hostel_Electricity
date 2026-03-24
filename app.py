import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hostel Electricity Dashboard", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "hostel_energy.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "electricity_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.csv")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception as e:
        return None, None

model, scaler = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

df = load_data()

TARGET = "electricity_kwh" if not df.empty and "electricity_kwh" in df.columns else ("electricity_units" if not df.empty and "electricity_units" in df.columns else None)

# -----------------------------
# LOAD METRICS
# -----------------------------
if os.path.exists(METRICS_PATH):
    model_results = pd.read_csv(METRICS_PATH)
else:
    model_results = pd.DataFrame()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3555/3555410.png", width=64)
    st.title("⚡ Energy Hub")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home Preview", "📊 Analytics Dashboard", "🔮 AI Prediction"])
    st.markdown("---")
    st.caption("Hostel Energy Analytics v2.0")

if df.empty or TARGET is None:
    st.error("No data found in Database or target column missing. Please run the data generation scripts.")
    st.stop()

# =============================
# 🏠 HOME PREVIEW
# =============================
if page == "🏠 Home Preview":

    st.title("Welcome to the Hostel Electricity Dashboard")
    st.markdown("Monitor, analyze, and predict electricity consumption across all hostels using machine learning models.")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_records = len(df)
    avg_kwh = df[TARGET].mean()
    max_kwh = df[TARGET].max()
    total_hostels = df['hostel_id'].nunique()
    
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Total Hostels", f"{total_hostels}")
    col3.metric("Avg Consumption", f"{avg_kwh:,.0f} kWh")
    col4.metric("Max Consumption", f"{max_kwh:,.0f} kWh")
    
    st.markdown("---")
    
    # Overview Chart
    st.subheader("Global Consumption Trend")
    trend_df = df.groupby(["year", "month"])[TARGET].mean().reset_index()
    # Create a pseudo-date to make it continuous
    trend_df['date'] = pd.to_datetime(trend_df.assign(day=1)[['year', 'month', 'day']])
    trend_df = trend_df.sort_values('date')
    
    fig = px.area(trend_df, x='date', y=TARGET, 
                  title="Average Monthly Electricity Consumption Across All Hostels", 
                  markers=True, color_discrete_sequence=['#00b4d8'])
    fig.update_layout(xaxis_title="Date", yaxis_title="Consumption (kWh)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)


# =============================
# 📊 ANALYTICS DASHBOARD
# =============================
elif page == "📊 Analytics Dashboard":

    st.title("🔍 Advanced Analytics")
    
    st.sidebar.markdown("### Filters")
    # Filters
    all_hostels = sorted(df["hostel_id"].unique())
    all_years = sorted(df["year"].unique())
    
    default_hostels = all_hostels[:3] if len(all_hostels) >= 3 else all_hostels
    selected_hostels = st.sidebar.multiselect("Select Hostels", options=all_hostels, default=default_hostels)
    selected_years = st.sidebar.multiselect("Select Years", options=all_years, default=all_years)
    
    if not selected_hostels or not selected_years:
        st.warning("Please select at least one hostel and one year from the sidebar.")
        st.stop()
        
    filtered = df[df["hostel_id"].isin(selected_hostels) & df["year"].isin(selected_years)].copy()
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends & Comparisons", "🔥 Heatmaps", "🤖 Model Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Monthly Trend per Hostel")
            monthly = filtered.groupby(["month", "hostel_id"])[TARGET].mean().reset_index()
            monthly['month_name'] = monthly['month'].apply(lambda x: calendar.month_abbr[x])
            # Ensure chronological order by sorting month numeral first
            monthly = monthly.sort_values("month")
            
            fig1 = px.line(monthly, x="month_name", y=TARGET, color="hostel_id", markers=True, 
                          title="Average Monthly Consumption by Hostel")
            fig1.update_layout(xaxis_title="Month", yaxis_title="Consumption")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("### Capacity vs Consumption")
            fig2 = px.scatter(filtered, x="num_students", y=TARGET, color="hostel_id", size="hostel_capacity",
                              hover_data=["year", "month"], title="Students vs Electricity Consumption")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Temperature Influence")
        fig3 = px.scatter(filtered, x="avg_temperature", y=TARGET, color="month",
                          title="Temperature vs Electricity Consumption")
        st.plotly_chart(fig3, use_container_width=True)
        
    with tab2:
        st.markdown("### Consumption Heatmap (Months vs Hostels)")
        heatmap_data = filtered.pivot_table(index="hostel_id", columns="month", values=TARGET, aggfunc="mean")
        
        heatmap_cols = [calendar.month_abbr[i] for i in heatmap_data.columns]
        
        fig4 = px.imshow(heatmap_data, labels=dict(x="Month", y="Hostel ID", color="Consumption"), 
                         x=heatmap_cols,
                         y=heatmap_data.index.astype(str), aspect="auto", color_continuous_scale="Viridis", text_auto=".0f")
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.markdown("### Model Evaluation Metrics")
        if model_results.empty:
            st.info("No model metrics available. Please train your models.")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if "RMSE" in model_results.columns and not model_results["RMSE"].isnull().all():
                    fig_rmse = px.bar(model_results, x="Model", y="RMSE", color="Model", title="RMSE Comparison")
                    st.plotly_chart(fig_rmse, use_container_width=True)
                else:
                    st.warning("RMSE contains missing values. Retrain model.")
            with col_m2:
                if "MAPE" in model_results.columns and not model_results["MAPE"].isnull().all():
                    fig_mape = px.bar(model_results, x="Model", y="MAPE", color="Model", title="MAPE Comparison")
                    st.plotly_chart(fig_mape, use_container_width=True)
                else:
                    st.warning("MAPE contains missing values. Retrain model.")

        if model is not None and hasattr(model, "feature_importances_"):
            st.markdown("### Random Forest Feature Importance")
            features = ["hostel_id", "month", "year", "num_students", "hostel_capacity", "avg_temperature", "exam_month", "vacation_month"]
            
            if len(features) == len(model.feature_importances_):
                importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=True)
                
                fig_feat = px.bar(importance_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                st.plotly_chart(fig_feat, use_container_width=True)


# =============================
# 🔮 AI PREDICTION
# =============================
elif page == "🔮 AI Prediction":
    st.title("🔮 Interactive Electricity Prediction")
    st.markdown("Simulate different scenarios to predict electricity consumption dynamically.")
    
    if model is None or scaler is None:
        st.error("Model or Scaler not found! Please train the model first.")
        st.stop()

    avg_per_hostel = df.groupby('hostel_id')[TARGET].mean().to_dict()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Parameters")
        with st.form("predict_form"):
            st.markdown("**Hostel & Time**")
            hostel_list = sorted(df["hostel_id"].unique())
            hostel_id = st.selectbox("Hostel ID", hostel_list, index=0)
            
            month_names = list(calendar.month_name)[1:]
            month_name = st.selectbox("Month", month_names, index=5)
            month = month_names.index(month_name) + 1
            
            current_year = df["year"].max() if not df.empty else 2024
            year = st.number_input("Year", 2020, 2030, int(current_year))
            
            st.markdown("**Occupancy & Weather**")
            hostel_capacity_val = int(df[df['hostel_id'] == hostel_id]['hostel_capacity'].max()) if len(df) > 0 and not df[df['hostel_id'] == hostel_id].empty else 300
            
            num_students = st.slider("Number of Students", 0, int(max(500, hostel_capacity_val * 1.5)), int(hostel_capacity_val * 0.8))
            avg_temperature = st.slider("Avg Temperature (°C)", 10.0, 45.0, 30.0)
            
            st.markdown("**Events**")
            c1, c2 = st.columns(2)
            exam_month = c1.checkbox("Exam Month", value=False)
            vacation_month = c2.checkbox("Vacation Month", value=False)
            
            submit = st.form_submit_button("Run Prediction", use_container_width=True)
            
    with col2:
        st.subheader("Prediction Output")
        
        if num_students > hostel_capacity_val:
            st.warning(f"⚠️ Warning: Number of students ({num_students}) exceeds typical capacity ({hostel_capacity_val}). Prediction may reflect an overcrowded scenario.")
            
        if submit:
            with st.spinner("Analyzing parameters through AI model..."):
                features = np.array([
                    hostel_id, month, year, num_students, hostel_capacity_val, 
                    avg_temperature, int(exam_month), int(vacation_month)
                ]).reshape(1, -1)

                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                
                historical_avg = avg_per_hostel.get(hostel_id, prediction)
                diff = prediction - historical_avg
                diff_pct = (diff / historical_avg) * 100 if historical_avg > 0 else 0
                
                if prediction > historical_avg * 1.1:
                    border_color = "#dc3545" # Red for high
                elif prediction < historical_avg * 0.9:
                    border_color = "#28a745" # Green for low
                else:
                    border_color = "#ffc107" # Yellow for normal
                    
                bg_color = "#1e1e1e" # Dark view
                
                st.markdown(
                    f'''
                    <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; border-left: 8px solid {border_color}; color: white; margin-bottom: 20px;">
                        <h3 style="margin: 0; color: #aaaaaa; font-weight: normal;">Predicted Consumption</h3>
                        <h1 style="margin: 10px 0; font-size: 3.5em; color: {border_color};">{prediction:,.2f} <span style="font-size: 0.4em; color: white;">kWh</span></h1>
                        <p style="margin: 0; font-size: 1.2em;">
                            <b>{abs(diff_pct):.1f}% {"Higher ⬆️" if diff > 0 else "Lower ⬇️"}</b> than the historical average for Hostel {hostel_id}.
                        </p>
                    </div>
                    ''', 
                    unsafe_allow_html=True
                )
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Consumption vs Baseline", 'font': {'size': 20}},
                    delta = {'reference': historical_avg, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
                    gauge = {
                        'axis': {'range': [None, max(prediction, historical_avg) * 1.5], 'tickwidth': 1},
                        'bar': {'color': border_color},
                        'bgcolor': "#f8f9fa",
                        'steps': [
                            {'range': [0, historical_avg], 'color': "#e9ecef"},
                            {'range': [historical_avg, historical_avg * 1.2], 'color': "#ffefc0"},
                            {'range': [historical_avg * 1.2, max(prediction, historical_avg) * 2], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "#495057", 'width': 4},
                            'thickness': 0.75,
                            'value': historical_avg
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("👈 Adjust the parameters on the left and click 'Run Prediction' to see AI-driven estimates.")