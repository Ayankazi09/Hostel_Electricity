import sqlite3
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from preprocessing import scale_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "hostel_energy.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    return df


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, mape, r2


def train_and_save():
    df = load_data()

    y = df["electricity_kwh"]

    X = df.drop(columns=["electricity_kwh"], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            rmse, mae, mape, r2 = evaluate(model, X_test, y_test)
        else:
            model.fit(X_train_scaled, y_train)
            rmse, mae, mape, r2 = evaluate(model, X_test_scaled, y_test)

        results[name] = {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }

        print(f"\n{name}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R2: {r2:.4f}")

    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = results[best_model_name]["model"]

    print(f"\nBest Model Selected: {best_model_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "electricity_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    train_and_save()