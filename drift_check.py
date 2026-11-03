import json
import pandas as pd
from src.config import DATA_PATH, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, REPORT_DIR


def run_drift_check():
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("Dataset is missing or empty.")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    midpoint = len(df) // 2
    if midpoint == 0:
        raise ValueError("Not enough rows to compare two groups.")
    first, second = df.iloc[:midpoint], df.iloc[midpoint:]
    findings = {"method": "First-half versus second-half row comparison", "findings": {},
                "limitation": "No shipment date is available; this is a demonstration, not time-based production drift monitoring."}
    for col in NUMERIC_FEATURES:
        a = pd.to_numeric(first[col], errors="coerce").mean()
        b = pd.to_numeric(second[col], errors="coerce").mean()
        change = None if pd.isna(a) or pd.isna(b) else (abs(b-a) / max(abs(a), 1e-9))
        findings["findings"][col] = {"first_group_mean": None if pd.isna(a) else float(a),
                                     "second_group_mean": None if pd.isna(b) else float(b),
                                     "relative_change": None if change is None else round(float(change), 4),
                                     "flagged": bool(change is not None and change > 0.20)}
    for col in CATEGORICAL_FEATURES:
        p1 = first[col].fillna("Unknown").astype(str).value_counts(normalize=True)
        p2 = second[col].fillna("Unknown").astype(str).value_counts(normalize=True)
        categories = set(p1.index) | set(p2.index)
        max_change = max((abs(float(p1.get(k, 0)) - float(p2.get(k, 0))) for k in categories), default=0)
        findings["findings"][col] = {"maximum_category_share_change": round(max_change, 4),
                                     "flagged": bool(max_change > 0.20)}
    path = REPORT_DIR / "drift_check.json"
    path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))
    print(f"\nSaved drift report: {path}")


if __name__ == "__main__":
    run_drift_check()
