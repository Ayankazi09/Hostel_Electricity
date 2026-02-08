import sqlite3
import sys
import os

# allow import from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generate_hostel_data import generate_hostel_data

DB_PATH = "hostel_energy.db"

def save_generated_data():
    # generate data
    df = generate_hostel_data()

    # connect to sqlite
    conn = sqlite3.connect(DB_PATH)

    # save to database
    df.to_sql(
        name="hostel_electricity",
        con=conn,
        if_exists="append",
        index=False
    )

    conn.close()

    print(f"Saved {len(df)} rows into hostel_electricity table.")

if __name__ == "__main__":
    save_generated_data()
