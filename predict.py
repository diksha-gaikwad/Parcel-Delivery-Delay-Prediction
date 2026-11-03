import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURES


def predict_parcel():
    if not MODEL_PATH.exists():
        print("Model not found. First run: python -m src.train_model")
        return
    print("Enter parcel details")
    row = {
        "distance_km": float(input("Distance in km: ")),
        "package_weight_kg": float(input("Package weight in kg: ")),
        "delivery_partner": input("Delivery partner: ").strip(),
        "package_type": input("Package type: ").strip(),
        "vehicle_type": input("Vehicle type: ").strip(),
        "delivery_mode": input("Delivery mode: ").strip(),
        "region": input("Region: ").strip(),
        "weather_condition": input("Weather condition: ").strip()
    }
    if row["distance_km"] <= 0 or row["package_weight_kg"] <= 0:
        print("Distance and package weight must be greater than zero.")
        return
    if any(not str(row[k]).strip() for k in FEATURES if k not in ["distance_km", "package_weight_kg"]):
        print("Please enter a value for every parcel field.")
        return
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([row], columns=FEATURES)
    probability = float(model.predict_proba(sample)[0, 1])
    print(f"\nEstimated delay risk: {probability:.1%}")
    if probability >= 0.5:
        print("Prediction: Higher delay risk. Operations can review this parcel.")
    else:
        print("Prediction: Lower delay risk. Continue normal monitoring.")


if __name__ == "__main__":
    predict_parcel()
