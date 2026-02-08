import sqlite3
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DB_PATH = "hostel_energy.db"
MODEL_DIR = "models"

def load_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        hostel_id,
        month,
        year,
        num_students,
        hostel_capacity,
        avg_temperature,
        exam_month,
        vacation_month,
        electricity_kwh
    FROM hostel_electricity
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df



def train_and_save():
    df = load_data()

    X = df.drop(columns=["electricity_kwh"])
    y = df["electricity_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Final Model Performance")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/electricity_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_and_save()
