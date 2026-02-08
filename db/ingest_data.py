import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generate_hostel_data import generate_hostel_data

DB_PATH = "hostel_energy.db"

def ingest_data():
    df = generate_hostel_data()

    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        name="hostel_electricity",
        con=conn,
        if_exists="append",
        index=False
    )

    conn.close()
    print(f"Inserted {len(df)} rows into hostel_electricity.")

if __name__ == "__main__":
    ingest_data()
