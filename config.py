from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Delivery_Logistics.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "parcel_delay_model.joblib"
REPORT_DIR = ROOT / "reports"
TARGET = "delayed"
NUMERIC_FEATURES = ["distance_km", "package_weight_kg"]
CATEGORICAL_FEATURES = [
    "delivery_partner", "package_type", "vehicle_type",
    "delivery_mode", "region", "weather_condition"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
