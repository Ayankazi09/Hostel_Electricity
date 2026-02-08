# Hostel Electricity Consumption Prediction  
Hackathon 3 – Model Deployment and Maintenance Pipeline

## Overview
This project implements a production-style machine learning pipeline to predict monthly electricity consumption of university hostels.  
The focus of the project is not model performance, but engineering practices such as data storage, reproducibility, deployment readiness, and model lifecycle management.

The system uses:
- SQL-based data storage
- Python scripts instead of notebooks
- Serialized machine learning models
- A Streamlit dashboard for analysis and prediction
- Git for version control and project tracking

---

## Problem Statement
Universities need to monitor and forecast hostel electricity consumption to support budgeting, infrastructure planning, and energy optimization.

This project predicts monthly electricity usage of hostels based on occupancy, capacity, temperature, and academic calendar indicators.

---

## Data
- Data is synthetically generated to resemble realistic university hostel usage.
- No static CSV files are used.
- A data generation script ensures scalability and reproducibility.
- All data is stored in a SQL database.

### Features
- hostel_id  
- month  
- year  
- num_students  
- hostel_capacity  
- avg_temperature  
- exam_month  
- vacation_month  

### Target
- electricity_units (monthly electricity consumption)

---

## Project Structure

---

## Model
- Algorithm: Linear Regression  
- Rationale: Interpretable, stable, and suitable for structured numerical data  
- Evaluation Metrics:
  - RMSE ≈ 1280 units
  - R² ≈ 0.95

The model is trained using SQL-loaded data and evaluated on a held-out test set.

---

## Dashboard
The Streamlit dashboard provides:
- Data preview and descriptive statistics
- Exploratory analysis of electricity consumption patterns
- A prediction interface for new user inputs
- Live inference using the latest trained model

The dashboard always loads the most recent serialized model.

---

## Model Lifecycle and Maintenance
- New data can be generated or ingested periodically
- Retraining is handled via standalone Python scripts
- Updated models replace previous artifacts
- Git commit history preserves development and update timelines

---

## Version Control
- The repository is public
- Commit history reflects incremental development
- Large data files and model binaries are excluded
- Data generation scripts ensure reproducibility

---

## How to Run
```bash
pip install -r requirements.txt
python db/ingest_data.py
python training/train_and_save_model.py
python -m streamlit run app.py
