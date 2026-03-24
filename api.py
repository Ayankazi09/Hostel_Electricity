import sqlite3
import pandas as pd
import joblib
import os
import numpy as np
import calendar
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Models and paths
DB_PATH = os.path.join(BASE_DIR, "hostel_energy.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "electricity_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.csv")

# Load ML components safely
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None
    print("Warning: Model or scaler not found.", e)

def get_db_connection():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
async def read_index():
    # Return index.html from static dir
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/kpi")
def get_kpi():
    conn = get_db_connection()
    if not conn:
        return {"error": "Database not found"}
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    if df.empty:
        return {"error": "No data"}
        
    target = "electricity_kwh" if "electricity_kwh" in df.columns else "electricity_units"
    return {
        "totalRecords": len(df),
        "totalHostels": int(df['hostel_id'].nunique()),
        "avgConsumption": float(df[target].mean()),
        "maxConsumption": float(df[target].max())
    }

@app.get("/api/chart-data")
def get_chart_data():
    conn = get_db_connection()
    if not conn:
        return {"error": "DB missing"}
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    
    target = "electricity_kwh" if "electricity_kwh" in df.columns else "electricity_units"
    
    # Process monthly trend
    monthly = df.groupby(["month"])[target].mean().reset_index()
    monthly = monthly.sort_values("month")
    
    return {
        "monthly_trend": {
            "labels": [calendar.month_abbr[m] for m in monthly['month']],
            "values": monthly[target].tolist()
        },
        "hostels": sorted(df['hostel_id'].unique().tolist()),
        "avg_by_hostel": df.groupby('hostel_id')[target].mean().to_dict()
    }

@app.get("/api/model-performance")
def get_model_performance():
    metrics = []
    if os.path.exists(METRICS_PATH):
        try:
            mr = pd.read_csv(METRICS_PATH)
            col_rmse = mr.columns[mr.columns.str.lower().str.contains("rmse")][0] if any(mr.columns.str.lower().str.contains("rmse")) else None
            col_mape = mr.columns[mr.columns.str.lower().str.contains("mape")][0] if any(mr.columns.str.lower().str.contains("mape")) else None
            
            # Simple fallback if columns are directly named RMSE, MAPE
            raw_metrics = mr.to_dict(orient="records")
            for m in raw_metrics:
                metrics.append({
                    "Model": str(m.get("Model", "Unknown")),
                    "RMSE": float(m.get(col_rmse if col_rmse else "RMSE", 0)),
                    "MAPE": float(m.get(col_mape if col_mape else "MAPE", 0))
                })
        except:
            pass
            
    importances = []
    if model is not None:
        features = ["hostel_id", "month", "year", "num_students", "hostel_capacity", "avg_temperature", "exam_month", "vacation_month"]
        if hasattr(model, "feature_importances_") and len(features) == len(model.feature_importances_):
            importances = [{"feature": f, "importance": float(i)} for f, i in zip(features, model.feature_importances_)]
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            if len(features) == len(coefs):
                importances = [{"feature": f, "importance": abs(float(i))} for f, i in zip(features, coefs)]
        
        if importances:
            importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
            
    return {"metrics": metrics, "importances": importances}

@app.get("/api/scatter-data")
def get_scatter_data():
    conn = get_db_connection()
    if not conn:
        return {"error": "DB"}
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    
    target = "electricity_kwh" if "electricity_kwh" in df.columns else "electricity_units"
    if df.empty or target not in df.columns:
        return {"error": "Data"}
    
    # We will sample 500 points randomly for render speed
    if len(df) > 500:
        df = df.sample(500)
    
    return {
        "students": df["num_students"].tolist(),
        "temperature": df["avg_temperature"].tolist(),
        "electricity": df[target].tolist()
    }

@app.get("/api/analysis-extra")
def get_extra_analysis():
    conn = get_db_connection()
    if not conn: return {}
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    
    # 1. Monthly Avg
    monthly = df.groupby('month')['electricity_kwh'].mean().reindex(range(1,13), fill_value=0).tolist()
    
    # 2. Season Split
    exam_sum = float(df[df['exam_month']==1]['electricity_kwh'].sum())
    vac_sum = float(df[df['vacation_month']==1]['electricity_kwh'].sum())
    norm_sum = float(df[(df['exam_month']==0) & (df['vacation_month']==0)]['electricity_kwh'].sum())
    
    # 3. Utilization Scatter (Capacity vs Students)
    df['utilization'] = df['num_students'] / df['hostel_capacity']
    # Filter safely
    df_util = df[df['utilization'] <= 2.0].dropna(subset=['utilization', 'electricity_kwh'])
    if len(df_util) > 300:
        df_util = df_util.sample(300)
        
    # 4. Distribution Histogram
    hist, bins = np.histogram(df['electricity_kwh'], bins=20)
    hist_labels = [f"{int(bins[i]/1000)}k" for i in range(len(hist))]
    
    return {
        "monthly_avg": monthly,
        "season_split": [exam_sum, norm_sum, vac_sum],
        "utilization_x": df_util['utilization'].tolist(),
        "utilization_y": df_util['electricity_kwh'].tolist(),
        "dist_counts": hist.tolist(),
        "dist_labels": hist_labels
    }

@app.get("/api/forecast/{hostel_id}")
def forecast_12_months(hostel_id: int):
    if model is None or scaler is None:
        return {"error": "Model missing"}
    
    year = 2026 # Forecasting absolute future
    predictions = []
    
    # Defaults
    num_s, cap, temp = 250, 300, 30.0
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(num_students), AVG(hostel_capacity), AVG(avg_temperature) FROM hostel_electricity WHERE hostel_id=?", (hostel_id,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                num_s, cap, temp = int(row[0]), int(row[1]), float(row[2])
        except Exception as e:
            pass
        finally:
            conn.close()
            
    for m in range(1, 13):
        # Infer typical Indian/Global academic exam and vacation months loosely
        exam = 1 if m in [4, 5, 11] else 0
        vac = 1 if m in [6, 7] else 0
        features = np.array([hostel_id, m, year, num_s, cap, temp, exam, vac]).reshape(1, -1)
        try:
            scaled = scaler.transform(features)
            pred = model.predict(scaled)[0]
            predictions.append(float(pred))
        except:
            predictions.append(0)
            
    months = [calendar.month_abbr[m] for m in range(1, 13)]
    return {"months": months, "predictions": predictions, "year": year}

@app.get("/api/raw-data")
def get_raw_data():
    conn = get_db_connection()
    if not conn:
        return {"data": []}
    df = pd.read_sql("SELECT * FROM hostel_electricity", conn)
    conn.close()
    return {"data": df.to_dict(orient="records")}

class PredictionRequest(BaseModel):
    hostel_id: int
    month: int
    year: int
    num_students: int
    hostel_capacity: int
    avg_temperature: float
    exam_month: bool
    vacation_month: bool

@app.post("/api/predict")
def predict_electricity(req: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
        
    features = np.array([
        req.hostel_id,
        req.month,
        req.year,
        req.num_students,
        req.hostel_capacity,
        req.avg_temperature,
        int(req.exam_month),
        int(req.vacation_month)
    ]).reshape(1, -1)
    
    try:
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
