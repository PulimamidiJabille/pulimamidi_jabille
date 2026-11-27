
from pathlib import Path
import json
import sqlite3
from datetime import datetime

import pandas as pd


# Project directory
ROOT = Path(__file__).resolve().parent

# Database path
DB_PATH = ROOT / "wine_quality.db"


# Connect to SQLite database
def get_connection():
    return sqlite3.connect(DB_PATH)


# Initialize database and import wine datasets
def init_db():

    with get_connection() as conn:

        # Create wine records table
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

        # Create prediction history table
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

        # Check whether dataset records already exist
        count = conn.execute(
            "SELECT COUNT(*) FROM wine_records"
        ).fetchone()[0]

        if count == 0:

            red_file = ROOT / "winequality-red.csv"
            white_file = ROOT / "winequality-white.csv"

            # Check dataset files
            if not red_file.exists():
                raise FileNotFoundError(
                    f"Red wine dataset not found: {red_file}"
                )

            if not white_file.exists():
                raise FileNotFoundError(
                    f"White wine dataset not found: {white_file}"
                )

            # Read CSV files
            red_data = pd.read_csv(red_file, sep=";")
            white_data = pd.read_csv(white_file, sep=";")

            # Identify wine types
            red_data["wine_type"] = "Red"
            white_data["wine_type"] = "White"

            # Combine datasets
            data = pd.concat(
                [red_data, white_data],
                ignore_index=True
            )

            # Prepare database records
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
                "quality": data["quality"],
            })

            # Insert records into SQLite
            insert_data.to_sql(
                "wine_records",
                conn,
                if_exists="append",
                index=False
            )

            print(
                f"Successfully imported {len(insert_data)} wine records."
            )


# Save prediction results
def save_prediction(
    wine_type,
    values,
    score,
    rounded_score,
    grade
):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO predictions
            (
                created_at,
                wine_type,
                input_values,
                predicted_score,
                rounded_score,
                grade
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            wine_type,
            json.dumps(values),
            float(score),
            int(rounded_score),
            grade
        ))


# Retrieve prediction history
def get_prediction_history():

    with get_connection() as conn:

        return pd.read_sql_query("""
            SELECT
                id,
                created_at,
                wine_type,
                predicted_score,
                rounded_score,
                grade,
                input_values
            FROM predictions
            ORDER BY id DESC
        """, conn)


# Get dataset summary
def get_dataset_summary():

    with get_connection() as conn:

        return pd.read_sql_query("""
            SELECT
                wine_type,
                COUNT(*) AS records,
                AVG(quality) AS average_quality
            FROM wine_records
            GROUP BY wine_type
        """, conn)
