# Parcel Delivery Delay Prediction (AI/ML)

## Project overview
This project uses machine learning to estimate whether a parcel may be delayed based on information available in the supplied delivery dataset.

## Algorithms
- Logistic Regression
- Random Forest Classifier

The project compares both models using accuracy, precision, recall, F1-score and ROC-AUC. The model with the higher F1-score is saved for prediction.

## Dataset
Place `Delivery_Logistics.csv` inside the `data` folder. The supplied dataset is included in this project package.

Model input features:
- distance_km
- package_weight_kg
- delivery_partner
- package_type
- vehicle_type
- delivery_mode
- region
- weather_condition

The target is `delayed` with yes/no values. Outcome and identifier fields are excluded from the model inputs to reduce data leakage risk.

## Setup (Windows PowerShell)
Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the install command without activating the environment, or use Command Prompt activation.

## Run the project

```powershell
python -m src.data_audit
python -m src.train_model
python -m src.predict
python -m src.drift_check
python -m pytest -q
```

## Generated outputs
- `reports/data_audit_report.txt`
- `reports/model_comparison.json`
- `reports/drift_check.json`
- `models/parcel_delay_model.joblib`

## Important limitations
- The dataset has no shipment date, so the current 80/20 stratified random split is provisional and cannot test future-period performance.
- The `delayed=yes` target includes records whose delivery status is failed. Confirm the intended business definition before operational use.
- The meaning and units of `expected_time_hours` need clarification; it is not used by the model.
- The assessment mentions carrier reference and warehouse capacity files. They are not included in the current dataset and have not been fabricated.
- Drift comparison is a demonstration using the first and second halves of the rows, not true time-based monitoring.

## AI assistance disclosure
AI assistance was used to help draft and review parts of the project code and documentation. The candidate should understand and be able to explain each file and should disclose the actual assistance used during the assessment review.
