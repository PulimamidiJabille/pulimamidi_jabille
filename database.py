from pathlib import Path
import json
import sqlite3
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "wine_quality.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables and import source dataset records if the table is empty."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wine_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wine_type TEXT NOT NULL,
                fixed_acidity REAL,
                volatile_acidity REAL,
                citric_acid REAL,
                residual_sugar REAL,
                chlorides REAL,
                free_sulfur_dioxide REAL,
                total_sulfur_dioxide REAL,
                density REAL,
                ph REAL,
                sulphates REAL,
                alcohol REAL,
                quality INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                wine_type TEXT NOT NULL,
                input_values TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                rounded_score INTEGER NOT NULL,
                grade TEXT NOT NULL
            )
        """)

        count = conn.execute("SELECT COUNT(*) FROM wine_records").fetchone()[0]
        if count == 0:
            try:
                from train_model import ensure_dataset, FEATURES, TARGET
                data = ensure_dataset()
                insert_data = pd.DataFrame({
                    "wine_type": data["wine_type"],
                    "fixed_acidity": data["fixed acidity"],
                    "volatile_acidity": data["volatile acidity"],
                    "citric_acid": data["citric acid"],
                    "residual_sugar": data["residual sugar"],
                    "chlorides": data["chlorides"],
                    "free_sulfur_dioxide": data["free sulfur dioxide"],
                    "total_sulfur_dioxide": data["total sulfur dioxide"],
                    "density": data["density"],
                    "ph": data["pH"],
                    "sulphates": data["sulphates"],
                    "alcohol": data["alcohol"],
                    "quality": data[TARGET],
                })
                insert_data.to_sql("wine_records", conn, if_exists="append", index=False)
            except Exception as exc:
                # Keep the database initialized; the app will show dataset/model setup errors separately.
                print(f"Could not import source records into SQLite yet: {exc}")


def save_prediction(wine_type, values, score, rounded_score, grade):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions
            (created_at, wine_type, input_values, predicted_score, rounded_score, grade)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            wine_type,
            json.dumps(values),
            float(score),
            int(rounded_score),
            grade,
        ))


def get_prediction_history():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT id, created_at, wine_type, predicted_score, rounded_score, grade, input_values
            FROM predictions
            ORDER BY id DESC
        """, conn)


def get_dataset_summary():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT wine_type, COUNT(*) AS records, AVG(quality) AS average_quality
            FROM wine_records
            GROUP BY wine_type
        """, conn)
