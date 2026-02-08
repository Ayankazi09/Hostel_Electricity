import sqlite3

DB_PATH = "hostel_energy.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open("sql/schema.sql", "r") as f:
        schema_sql = f.read()

    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()

    print("Database and table created successfully.")

if __name__ == "__main__":
    init_db()
