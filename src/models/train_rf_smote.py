# src/models/train_rf_smote.py
# -----------------------------------------------------------------------------
# Random Forest with SMOTE (Synthetic Minority Over-sampling Technique).
#
# What it does:
#   1. Applies SMOTE to the TRAINING data only (creates synthetic churners).
#   2. Trains a Random Forest on the resampled training data.
#   3. Evaluates on the ORIGINAL (untouched) test set.
#   4. Logs the run to MLflow (SQLite backend).
#
# Why SMOTE:
#   - Churn data is imbalanced (few churners, many non-churners).
#   - A model trained on raw data tends to ignore the minority class.
#   - SMOTE creates synthetic minority samples by interpolating between
#     existing ones, giving the model more examples to learn from.
#   - Different from class_weight='balanced': SMOTE changes the DATA,
#     class_weight changes the LOSS.
#
# Critical rule:
#   SMOTE must be applied to TRAINING data ONLY.
#   Never touch the test set — evaluating on synthetic data is meaningless.
# -----------------------------------------------------------------------------

import os
import sys
from collections import Counter

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# --- Make src/model_utils.py importable regardless of where we run from ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_utils import load_processed_data, evaluate_and_log


# --- MLflow tracking URI: use SQLite (avoids filesystem backend issues) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")


def run_rf_with_smote(X_train, y_train, X_test, y_test):
    """
    Apply SMOTE to training data only, train Random Forest, evaluate on
    the original (untouched) test set, and log everything to MLflow.

    Args:
        X_train : training features
        y_train : training labels
        X_test  : test features (NEVER resampled)
        y_test  : test labels (NEVER resampled)

    Returns:
        dict: classification metrics from evaluate_and_log()
    """
    with mlflow.start_run(run_name="rf_smote"):

        # ---------------------------------------------------------------------
        # 1. Log hyperparameters for this run
        # ---------------------------------------------------------------------
        mlflow.log_param("model", "RandomForestClassifier + SMOTE")
        mlflow.log_param("class_weight", "None (SMOTE handles imbalance)")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("smote_k_neighbors", 5)
        mlflow.log_param("smote_random_state", 42)

        # ---------------------------------------------------------------------
        # 2. Apply SMOTE to the TRAINING set only
        # ---------------------------------------------------------------------
        print("\n🔄 Applying SMOTE to training data...")
        print(f"   Before SMOTE: {Counter(y_train)}")

        smote = SMOTE(k_neighbors=5, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print(f"   After SMOTE:  {Counter(y_train_res)}")

        # ---------------------------------------------------------------------
        # 3. Train Random Forest on the SMOTE-resampled training data
        #    class_weight=None because SMOTE already balanced the classes.
        # ---------------------------------------------------------------------
        print("\n🚀 Training: rf_smote (class_weight=None, SMOTE-resampled)")

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train_res, y_train_res)

        # ---------------------------------------------------------------------
        # 4. Evaluate on the ORIGINAL test set (never the resampled one)
        # ---------------------------------------------------------------------
        metrics = evaluate_and_log(model, X_test, y_test, "model")

        return metrics


if __name__ == "__main__":
    # Group all runs under a single MLflow experiment
    mlflow.set_experiment("churn-rf-baseline")

    # Load the preprocessed train/test splits
    X_train, X_test, y_train, y_test = load_processed_data()

    # Run SMOTE + Random Forest
    metrics_smote = run_rf_with_smote(X_train, y_train, X_test, y_test)

    print("\n📊 SMOTE run complete.")
    print(metrics_smote)