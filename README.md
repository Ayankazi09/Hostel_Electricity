# ⚡ VoltEdge Analytics: Hostel Electricity Consumption Hub  

## Overview
This project implements a production-style machine learning pipeline and a fully custom Single Page Application (SPA) dashboard to predict and analyze monthly electricity consumption of university hostels.  

The focus of the project spans both model performance and robust data engineering practices, including data storage, reproducibility, deployment readiness, and model lifecycle management.

The system features:
- **FastAPI Backend:** A lightweight, high-performance Python backend serving API endpoints.
- **Custom SPA Frontend:** A sleek, glassmorphic UI built with HTML, CSS, and vanilla JavaScript, featuring interactive `Chart.js` visualizations.
- **SQLite Data Storage:** Persistent, queryable historical data layer.
- **Serialized ML Models:** Pre-trained predictive models managed via a dedicated Model Hub.

---

## Problem Statement
Universities need to monitor, analyze, and forecast hostel electricity consumption to support budgeting, infrastructure planning, and energy optimization.

This dynamic tool predicts the monthly electricity usage of individual hostels based on precise dynamic parameters such as occupancy, capacity, temperature, and academic calendar indicators (exams/vacations).

---

## Data Pipeline
- Data is persistently stored in a local SQLite database (`hostel_energy.db`).
- A centralized data generation script ensures scalability, reproducibility, and synthetic realism.
- No static CSV files are used for the main historical data layer.

### Features
- `hostel_id`  
- `month` & `year`  
- `num_students` (Occupancy rate)
- `hostel_capacity`  
- `avg_temperature`  
- `exam_month` (Boolean)  
- `vacation_month` (Boolean)  

### Target
- `electricity_kwh` / `electricity_units` (Monthly electricity consumption)

---

## Model & Training Lifecycle
- **Algorithms:** Evaluates multiple regressors (e.g., Random Forest, Linear Regression).
- **Model Hub:** The dashboard features a strictly integrated "Model Hub" tracking historical model accuracy logs (RMSE, MAPE, R², Score) and exposing serialized model artifacts (`.pkl`) for direct download.
- **Maintenance:** 
  - Retraining is handled via standalone Python scripts within the `training/` directory.
  - Updated models seamlessly replace previous artifacts without requiring frontend downtime.

---

## Dashboard Features
The VoltEdge SPA provides:
- **Overview:** Real-time KPI counters and global seasonal consumption trends.
- **Deep Analytics:** Scatter plots, temperature impact analysis, and seasonal data distribution charts.
- **Prediction Studio:** An interactive AI simulator allowing users to tweak occupancy and temperature sliders to instantly forecast load demands, alert on thresholds, and calculate estimated financial costs.
- **Model Hub:** Transparent display of current model matrices, past performance logs, and an artifact download repository.
- **Raw Data Explorer:** Searchable, paginated datatables to audit historical database records.

---

## How to Run Locally

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Initialize Database & Train Models** *(If starting from scratch)*
```bash
python db/ingest_data.py
python training/train_and_save_model.py
```

**3. Launch the Application**
```bash
python api.py
```
*The full dashboard application will now be natively accessible at **http://127.0.0.1:8000***
