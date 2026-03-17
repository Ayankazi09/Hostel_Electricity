import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Hostel Electricity Dashboard", layout="wide")

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
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    return df

df = load_data()

TARGET = "electricity_kwh" if "electricity_kwh" in df.columns else "electricity_units"

# -----------------------------
# LOAD METRICS (SAFE)
# -----------------------------
if os.path.exists(METRICS_PATH):
    model_results = pd.read_csv(METRICS_PATH)
else:
    model_results = pd.DataFrame()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis", "Prediction"])

# =============================
# 🏠 HOME
# =============================
if page == "Home":

    st.title("🏠 Dashboard Overview")

    st.markdown("""
    This system predicts hostel electricity consumption using machine learning.
    It includes data generation, SQL storage, retraining pipeline, and analytics.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Avg Electricity", f"{df[TARGET].mean():.2f}")
    col3.metric("Max Electricity", f"{df[TARGET].max():.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())


# =============================
# 📊 ANALYSIS
# =============================
elif page == "Analysis":

    st.title("📊 Data Analysis & Model Insights")

    # Filters
    hostel = st.selectbox("Select Hostel", sorted(df["hostel_id"].unique()))
    year = st.selectbox("Select Year", sorted(df["year"].unique()))

    filtered = df[(df["hostel_id"] == hostel) & (df["year"] == year)]

    st.subheader("Filtered Data")
    st.dataframe(filtered)

    # -----------------------------
    # CHARTS
    # -----------------------------
    st.subheader("Monthly Electricity Trend")

    monthly = filtered.groupby("month")[TARGET].mean()

    fig1, ax1 = plt.subplots()
    ax1.plot(monthly.index, monthly.values, marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Electricity")
    st.pyplot(fig1)

    st.subheader("Students vs Electricity")

    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered["num_students"], filtered[TARGET])
    ax2.set_xlabel("Students")
    ax2.set_ylabel("Electricity")
    st.pyplot(fig2)

    st.subheader("Temperature vs Electricity")

    fig3, ax3 = plt.subplots()
    ax3.scatter(filtered["avg_temperature"], filtered[TARGET])
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Electricity")
    st.pyplot(fig3)

    # -----------------------------
    # MODEL COMPARISON (SAFE)
    # -----------------------------
    st.subheader("Model Comparison")

    if model_results.empty:
        st.warning("No model metrics found. Run training script first.")
    else:
        st.dataframe(model_results)

        # RMSE
        st.subheader("RMSE Comparison")

        if model_results["RMSE"].isnull().any():
            st.error("RMSE contains missing values. Retrain model.")
        else:
            fig4, ax4 = plt.subplots()
            ax4.bar(model_results["Model"], model_results["RMSE"])
            ax4.set_ylabel("RMSE")
            st.pyplot(fig4)

        # MAPE
        st.subheader("MAPE Comparison")

        if model_results["MAPE"].isnull().any():
            st.error("MAPE contains missing values. Retrain model.")
        else:
            fig5, ax5 = plt.subplots()
            ax5.bar(model_results["Model"], model_results["MAPE"])
            ax5.set_ylabel("MAPE")
            st.pyplot(fig5)

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    if hasattr(model, "feature_importances_"):

        st.subheader("Feature Importance (Random Forest)")

        features = [
            "hostel_id", "month", "year", "num_students",
            "hostel_capacity", "avg_temperature",
            "exam_month", "vacation_month"
        ]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)

        fig6, ax6 = plt.subplots()
        ax6.barh(importance_df["Feature"], importance_df["Importance"])
        ax6.invert_yaxis()
        st.pyplot(fig6)


# =============================
# 🔮 PREDICTION
# =============================
elif page == "Prediction":

    st.title("🔮 Electricity Prediction")

    with st.form("predict_form"):
        hostel_id = st.number_input("Hostel ID", 1, 10, 1)
        month = st.slider("Month", 1, 12, 6)
        year = st.number_input("Year", 2020, 2030, 2025)
        num_students = st.slider("Students", 0, 500, 250)
        hostel_capacity = st.slider("Capacity", 0, 500, 300)
        avg_temperature = st.slider("Temperature", 10.0, 45.0, 30.0)
        exam_month = st.checkbox("Exam Month")
        vacation_month = st.checkbox("Vacation Month")

        submit = st.form_submit_button("Predict")

    if submit:

        if num_students > hostel_capacity:
            st.error("Students cannot exceed capacity")
        else:
            features = np.array([
                hostel_id,
                month,
                year,
                num_students,
                hostel_capacity,
                avg_temperature,
                int(exam_month),
                int(vacation_month)
            ]).reshape(1, -1)

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            st.success(f"Predicted Electricity Consumption: {prediction:.2f}")