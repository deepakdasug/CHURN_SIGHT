# CHURN_SIGHT

CHURN_SIGHT is a customer churn prediction and model experimentation project for the Telco customer churn dataset. It includes a reusable feature-preparation pipeline, baseline and imbalance-aware classifiers, business-cost-based evaluation, and experiment tracking with MLflow.

## Current project stage

The repository currently supports local data preparation and model experimentation. It does not currently include a web dashboard, deployed prediction API, or a database ingestion application.

Implemented capabilities:

- Cleans the Telco churn data, engineers customer features, and creates stratified train/test datasets.
- Trains and evaluates Logistic Regression and Random Forest variants, including class weighting and SMOTE.
- Tunes XGBoost and LightGBM models with stratified cross-validation and selects probability thresholds using configurable false-negative and false-positive costs.
- Logs experiment metrics and model artifacts to MLflow.
- Compares Random Forest variants and produces evaluation plots and a Markdown report.

## Repository layout

```text
CHURN_SIGHT/
├── data/
│   ├── raw/                         # Input dataset (not tracked by Git)
│   └── processed/                   # Generated CSV files (not tracked by Git)
├── src/
│   ├── archive/
│   │   └── clean_data.py            # Earlier, machine-specific pipeline
│   ├── features/
│   │   └── build_features.py        # Current cleaning and feature pipeline
│   └── models/
│       ├── model_utils.py           # Shared data loading and evaluation helpers
│       ├── train_model.py           # Random Forest baseline
│       ├── train_logistic.py        # Class-balanced Logistic Regression
│       ├── train_rf_baseline.py     # Default and class-balanced Random Forests
│       ├── train_rf_smote.py        # Random Forest with SMOTE
│       ├── train_gradient_boosting.py # Tuned XGBoost and LightGBM
│       └── evaluate_models.py       # Random Forest comparison, plots, and report
├── requirements.txt                 # Dependency list (currently empty)
└── README.md
```

## Requirements

- Python 3.10 or newer
- scikit-learn 1.3 or newer (`TargetEncoder` is used by the feature pipeline)
- The Python packages used by the current scripts:

```bash
python -m pip install pandas numpy "scikit-learn>=1.3" imbalanced-learn mlflow xgboost lightgbm matplotlib tabulate
```

The repository's `requirements.txt` is currently empty; the command above installs the dependencies directly.

## Data

Obtain the Telco customer churn dataset separately and place the CSV here:

```text
data/raw/Telco_customer_churn.csv
```

The dataset is not included in this repository. The feature pipeline expects the Telco dataset columns used in `src/features/build_features.py`, including `Churn Value`, `Total Charges`, `Monthly Charges`, `Tenure Months`, service details, contract, payment method, city, and ZIP code.

## Prepare the data

Run commands from the repository root so the relative data paths resolve correctly:

```bash
python src/features/build_features.py
```

The pipeline writes the cleaned dataset and stratified, encoded, scaled train/test splits to `data/processed/`:

- `cleaned_churn.csv`
- `X_train.csv` and `X_test.csv`
- `y_train.csv` and `y_test.csv`

These generated files are ignored by Git.

## Train and evaluate models

Run one of the following from the repository root after preparing the data:

```bash
python src/models/train_model.py
python src/models/train_logistic.py
python src/models/train_rf_baseline.py
python src/models/train_rf_smote.py
python src/models/train_gradient_boosting.py
python src/models/evaluate_models.py
```

`train_gradient_boosting.py` runs grid searches and cross-validation for both XGBoost and LightGBM, so it may take substantially longer than the baseline scripts. `evaluate_models.py` compares default, class-balanced, and SMOTE Random Forests and writes plots and a report under `reports/`.

The scripts use MLflow to record experiments. Most scripts store tracking data in the root-level `mlflow.db` SQLite database; `train_rf_baseline.py` instead stores runs in the `mlruns/` file-based store. To view the SQLite-tracked experiments, start the MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

For the file-based runs from `train_rf_baseline.py`, use:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open the URL printed by MLflow (normally `http://127.0.0.1:5000`).

## Evaluation and assumptions

The model utilities report accuracy, precision, recall, F1, and ROC-AUC where supported. The model comparison and gradient-boosting scripts also calculate business cost using the assumptions `COST_FN = 500` and `COST_FP = 50` in their source. These are example cost weights; adjust them to reflect the intended business use before interpreting cost-based results. Model performance depends on the dataset and environment and is not guaranteed by this repository.

## Notes

- `data/`, `reports/`, local MLflow stores, virtual environments, and environment files are excluded from Git.
- `src/archive/clean_data.py` is an earlier pipeline and contains a machine-specific absolute input path. Use `src/features/build_features.py` for the current portable pipeline.
- This project is an experimentation codebase; it does not currently provide a production deployment workflow.
