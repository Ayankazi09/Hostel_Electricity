import sqlite3
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
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


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    return df


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, mape, r2


# -----------------------------
# Cross Validation Function
# -----------------------------
def cross_validate_model(model, X, y):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()


# -----------------------------
# Train + Save Pipeline
# -----------------------------
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

    # -----------------------------
    # Train + Evaluate
    # -----------------------------
    for name, model in models.items():

        if name == "Random Forest":
            model.fit(X_train, y_train)
            rmse, mae, mape, r2 = evaluate(model, X_test, y_test)
            cv_rmse = cross_validate_model(model, X, y)
        else:
            model.fit(X_train_scaled, y_train)
            rmse, mae, mape, r2 = evaluate(model, X_test_scaled, y_test)
            cv_rmse = cross_validate_model(model, X_train_scaled, y_train)

        results[name] = {
            "model": model,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "CV_RMSE": cv_rmse
        }

        print(f"\n{name}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"CV RMSE: {cv_rmse:.2f}")

    # -----------------------------
    # Convert to DataFrame
    # -----------------------------
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "RMSE": results[name]["RMSE"],
            "MAPE": results[name]["MAPE"],
            "R2": results[name]["R2"],
            "CV_RMSE": results[name]["CV_RMSE"]
        }
        for name in results
    ])

    # -----------------------------
    # Normalize Metrics
    # -----------------------------
    metrics_df["RMSE_norm"] = metrics_df["RMSE"] / metrics_df["RMSE"].max()
    metrics_df["MAPE_norm"] = metrics_df["MAPE"] / metrics_df["MAPE"].max()
    metrics_df["R2_norm"] = 1 - metrics_df["R2"]
    metrics_df["CV_norm"] = metrics_df["CV_RMSE"] / metrics_df["CV_RMSE"].max()

    # -----------------------------
    # Combined Score
    # -----------------------------
    metrics_df["Score"] = (
        0.3 * metrics_df["RMSE_norm"] +
        0.3 * metrics_df["MAPE_norm"] +
        0.2 * metrics_df["R2_norm"] +
        0.2 * metrics_df["CV_norm"]
    )

    # -----------------------------
    # Select Best Model
    # -----------------------------
    best_model_name = metrics_df.loc[metrics_df["Score"].idxmin(), "Model"]
    best_model = results[best_model_name]["model"]

    print(f"\nBest Model Selected: {best_model_name}")

    # -----------------------------
    # Save Model + Scaler
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "electricity_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Model and scaler saved successfully.")

    # -----------------------------
    # Save Metrics
    # -----------------------------
    metrics_path = os.path.join(BASE_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Metrics saved to {metrics_path}")



if __name__ == "__main__":
    train_and_save()