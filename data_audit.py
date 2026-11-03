import pandas as pd
from src.config import DATA_PATH, REPORT_DIR, FEATURES, TARGET


def run_audit():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if DATA_PATH.stat().st_size == 0:
        raise ValueError("The dataset file is empty. Add the actual CSV data and try again.")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    lines = [
        "PARCEL DELIVERY DATA AUDIT",
        "=" * 32,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Exact duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Missing values by column:",
        df.isna().sum().to_string(),
        "",
        "Target distribution:",
        df[TARGET].value_counts(dropna=False).to_string(),
        "",
        "Columns available:",
        ", ".join(df.columns),
        "",
        "Model features:",
        ", ".join(FEATURES),
        "",
        "Excluded outcome/identifier fields: delivery_id, delivery_status, delivery_time_hours, delivery_rating.",
        "The expected_time_hours field is not used because its format and meaning need confirmation."
    ]
    report = "\n".join(lines)
    output = REPORT_DIR / "data_audit_report.txt"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nAudit saved to: {output}")
    return df


if __name__ == "__main__":
    run_audit()
