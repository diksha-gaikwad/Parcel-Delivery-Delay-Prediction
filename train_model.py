import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import DATA_PATH, MODEL_PATH, REPORT_DIR, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def convert_target(series):
    values = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0}
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected target values: {unknown}. Expected yes/no.")
    return values.map(mapping).astype(int)


def load_data():
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("Dataset is missing or empty. Check data/Delivery_Logistics.csv.")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=[TARGET]).copy()
    y = convert_target(df[TARGET])
    X = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
    return X, y


def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    return Pipeline([("preprocessor", prep), ("classifier", model)])


def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4)
    }


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        )
    }
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        results[name] = calculate_metrics(pipeline, X_test, y_test)
        fitted[name] = pipeline
        print(f"\n{name}: {results[name]}")

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_PATH)
    report = {
        "selected_model": best_name,
        "comparison": results,
        "validation": "Stratified random 80/20 split",
        "validation_note": "The dataset has no shipment date column, so chronological validation is not available.",
        "target_note": "Confirm whether failed deliveries should count as delayed before operational use.",
        "features": FEATURES
    }
    report_path = REPORT_DIR / "model_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    train()
