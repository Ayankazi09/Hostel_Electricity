import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "hostel_energy.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "electricity_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    return df

df = load_data()

# -----------------------------
# Decide target column safely
# -----------------------------
if "electricity_units" in df.columns:
    TARGET = "electricity_units"
elif "electricity_kwh" in df.columns:
    TARGET = "electricity_kwh"
else:
    st.error("Electricity column not found in database")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("Hostel Electricity Consumption Prediction")

st.markdown(
    """
    This dashboard demonstrates a deployment-ready regression model
    for predicting monthly electricity consumption of university hostels.
    """
)

# -----------------------------
# Data overview
# -----------------------------
st.subheader("Data Overview")
st.dataframe(df.head())

st.subheader("Basic Statistics")
st.write(df.describe())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
st.header("Data Insights")

# 1. Electricity vs Number of Students
st.subheader("Electricity Consumption vs Number of Students")

fig1, ax1 = plt.subplots()
ax1.scatter(df["num_students"], df[TARGET])
ax1.set_xlabel("Number of Students")
ax1.set_ylabel("Electricity Consumption")
st.pyplot(fig1)

# 2. Electricity vs Temperature
st.subheader("Electricity Consumption vs Average Temperature")

fig2, ax2 = plt.subplots()
ax2.scatter(df["avg_temperature"], df[TARGET])
ax2.set_xlabel("Average Temperature (°C)")
ax2.set_ylabel("Electricity Consumption")
st.pyplot(fig2)

# 3. Monthly Electricity Trend
st.subheader("Monthly Average Electricity Consumption")

monthly_avg = df.groupby("month")[TARGET].mean()

fig3, ax3 = plt.subplots()
ax3.plot(monthly_avg.index, monthly_avg.values, marker="o")
ax3.set_xlabel("Month")
ax3.set_ylabel("Average Electricity Consumption")
st.pyplot(fig3)

# -----------------------------
# Model performance
# -----------------------------
st.subheader("Model Performance")
st.markdown(
    """
    **Model:** Linear Regression  
    **RMSE:** ~1280 units  
    **R² Score:** ~0.95  

    Metrics are computed on a held-out test set.
    """
)

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("Make a Prediction")

with st.form("prediction_form"):
    hostel_id = st.number_input("Hostel ID", min_value=1, value=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=7)
    year = st.number_input("Year", min_value=2020, value=2025)
    num_students = st.number_input("Number of Students", min_value=0, value=250)
    hostel_capacity = st.number_input("Hostel Capacity", min_value=0, value=300)
    avg_temperature = st.number_input("Average Temperature (°C)", value=32.0)
    exam_month = st.selectbox("Exam Month", [0, 1])
    vacation_month = st.selectbox("Vacation Month", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    features = np.array([
        hostel_id,
        month,
        year,
        num_students,
        hostel_capacity,
        avg_temperature,
        exam_month,
        vacation_month
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.success(f"Predicted Electricity Consumption: {prediction:.2f}")
