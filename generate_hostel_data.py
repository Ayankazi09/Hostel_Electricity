from blinker import signal
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import base

def generate_hostel_data(
    num_hostels=3,
    start_year=2023,
    end_year=2025
):
    records = []

    for hostel_id in range(1, num_hostels + 1):
        hostel_capacity = np.random.randint(200, 400)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):

                num_students = np.random.randint(
                    int(0.7 * hostel_capacity),
                    hostel_capacity
                )

                avg_temperature = np.random.uniform(18, 40)
                exam_month = 1 if month in [3, 4, 10, 11] else 0
                vacation_month = 1 if month in [5, 6] else 0

                student_coeff = np.random.uniform(80, 120)
                temp_coeff = np.random.uniform(3, 8)
                exam_coeff = np.random.uniform(300, 700)
                vacation_coeff = np.random.uniform(-1000, -500)

                base = num_students * student_coeff
                temp_effect = avg_temperature * temp_coeff
                exam_effect = exam_coeff if exam_month else 0
                vacation_effect = vacation_coeff if vacation_month else 0

                signal = base + temp_effect + exam_effect + vacation_effect
                noise = np.random.normal(0, 0.1 * signal)

                electricity_kwh = max(signal + noise, 0)

                records.append({
                    "hostel_id": hostel_id,
                    "month": month,
                    "year": year,
                    "num_students": num_students,
                    "hostel_capacity": hostel_capacity,
                    "avg_temperature": round(avg_temperature, 2),
                    "exam_month": exam_month,
                    "vacation_month": vacation_month,
                    "electricity_kwh": round(electricity_kwh, 2),
                    "created_at": datetime.now().isoformat()
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_hostel_data()
    print(df.head())


