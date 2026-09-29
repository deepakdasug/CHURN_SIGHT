# src/train_rf_baseline.py
# -----------------------------------------------------------------------------
# Random Forest Baseline training script.
#
# What it does:
#   1. Trains a default Random Forest (baseline).
#   2. Trains a Random Forest with class_weight='balanced' (imbalance-aware).
#   3. Logs both runs to MLflow.
#   4. Prints a side-by-side metric comparison.
#
# Why:
#   - Default RF often ignores the minority class (churners) on imbalanced data.
#   - class_weight='balanced' upweights the minority class so the model
#     pays more attention to catching churners.
#   - We log metrics so improvements can be tracked and compared over time.
# -----------------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_utils import load_processed_data,evaluate_and_log
import os

# Build an absolute path with NO spaces and use forward slashes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns").replace("\\", "/")

# "file:///" + path is the correct URI form for Windows
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")

def run_random_forest(class_weight,run_name,X_train,y_train,X_test,y_test):
    """
    Train a single Random Forest configuration inside an MLflow run.

    Args:
        class_weight : None or 'balanced' — controls how minority class is treated.
        run_name     : Name shown in MLflow UI for this run.
        X_train, y_train, X_test, y_test : data splits.

    Returns:
        dict: Metrics produced by evaluate_and_log().
    """
    with mlflow.start_run(run_name=run_name):
         # ---------------------------------------------------------------------
        # 1. Log hyperparameters so runs can be compared later
        # ---------------------------------------------------------------------

        mlflow.log_param('model','RandomForestClassifier')
        mlflow.log_param('class_weight',str(class_weight))
        mlflow.log_param('n_estimators',300)
        mlflow.log_param('random_state',42)

        # ---------------------------------------------------------------------
        # 2. Build the Random Forest
        #    - n_estimators=300 : enough trees for stable predictions
        #    - random_state=42  : reproducibility
        #    - n_jobs=-1        : use all CPU cores
        #    - class_weight     : 'balanced' adjusts weights inversely to
        #                         class frequency, boosting the minority class
        # ---------------------------------------------------------------------

        model= RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )

        # ---------------------------------------------------------------------
        # 3. Train the model
        # ---------------------------------------------------------------------

        print(f"\n🚀 Training: {run_name} (class_weight={class_weight})")

        model.fit(X_train,y_train)

         # ---------------------------------------------------------------------
        # 4. Evaluate + log metrics & model artifact
        # ---------------------------------------------------------------------

        metrics = evaluate_and_log(model,X_test,y_test,'model')

        return metrics
if __name__=='__main__':
    # -------------------------------------------------------------------------
    # Group all runs under a single MLflow experiment
    # -------------------------------------------------------------------------

    mlflow.set_experiment('churn-rf-baseline')

    # -------------------------------------------------------------------------
    # Load preprocessed data
    # -------------------------------------------------------------------------
    X_train,X_test,y_train,y_test = load_processed_data()

    # -------------------------------------------------------------------------
    # Run 1 — Default Random Forest (baseline, no class weighting)
    # -------------------------------------------------------------------------

    metric_default = run_random_forest(class_weight = None,
    run_name = 'rf_default',
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_test = y_test

    )

    # -------------------------------------------------------------------------
    # Run 2 — Random Forest with class_weight='balanced'
    # -------------------------------------------------------------------------

    metrics_balanced= run_random_forest(
        class_weight='balanced',
        run_name= 'rf_balanced',
        X_train= X_train,
        X_test= X_test,
        y_train= y_train,
        y_test= y_test
    )

    # -------------------------------------------------------------------------
    # Compare the two runs side-by-side
    # -------------------------------------------------------------------------

    comparison= pd.DataFrame(
        {
            'default': metric_default,
            'balanced': metrics_balanced,
        }).T
    
    print("\n📊 Metric comparison:")
    print(comparison.round(4))

    print("\n📈 Change from default → balanced:")
    print((comparison.loc['balanced']-comparison.loc['default']).round(4))

     # -------------------------------------------------------------------------
    # Interpretation hint:
    #   - Recall / F1 for churn should go UP with 'balanced'.
    #   - Accuracy may go DOWN slightly — that's expected and usually fine.
    #   - ROC-AUC is a good tie-breaker when recall improves.
    # -------------------------------------------------------------------------