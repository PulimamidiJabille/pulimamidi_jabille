from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "wine_quality.db"
MODEL_PATH = MODEL_DIR / "wine_quality_random_forest.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.csv"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "quality"

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def ensure_dataset():
    """Download the public dataset files if they are not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    red_path = DATA_DIR / "winequality-red.csv"
    white_path = DATA_DIR / "winequality-white.csv"

    if not red_path.exists():
        print("Downloading red wine dataset...")
        urlretrieve(RED_URL, red_path)
    if not white_path.exists():
        print("Downloading white wine dataset...")
        urlretrieve(WHITE_URL, white_path)

    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red["wine_type"] = "Red"
    white["wine_type"] = "White"
    data = pd.concat([red, white], ignore_index=True)
    data = data.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return data


def train_or_load_model(data, force_train=False):
    """Train a Random Forest regressor or load the saved model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and METRICS_PATH.exists() and not force_train:
        model = joblib.load(MODEL_PATH)
        metric_table = pd.read_csv(METRICS_PATH)
        metrics = {
            "mae": float(metric_table.loc[0, "mae"]),
            "r2": float(metric_table.loc[0, "r2"]),
        }
        return model, metrics

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    joblib.dump(model, MODEL_PATH)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    return model, metrics


if __name__ == "__main__":
    dataset = ensure_dataset()
    trained_model, evaluation = train_or_load_model(dataset, force_train=True)
    print(f"Dataset rows: {len(dataset)}")
    print(f"Test MAE: {evaluation['mae']:.3f}")
    print(f"Test R²: {evaluation['r2']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
