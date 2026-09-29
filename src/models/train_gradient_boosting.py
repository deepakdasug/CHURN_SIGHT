# =============================================================================
# train_gradient_boosting.py
# Trains XGBoost and LightGBM with class imbalance handling, hyperparameter
# tuning, threshold optimization for business cost, and MLflow tracking.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------------------------------------------------------
# 1. PATH SETUP
# -----------------------------------------------------------------------------
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_DIR)
from model_utils import load_processed_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------
COST_FN = 500   # Cost of missing a churner (False Negative)
COST_FP = 50    # Cost of a wasted retention offer (False Positive)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def evaluate_model(model, X, y, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    """Calculates metrics and business cost for a given model and threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    cost_per_customer = total_cost / len(y) if len(y) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "total_cost": total_cost,
        "cost_per_customer": cost_per_customer,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Find threshold that minimizes total business cost."""
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.5
    best_cost = float("inf")
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
    return best_threshold, best_cost


def log_metrics(prefix, metrics):
    """Log metrics with a prefix to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{key}", value)


# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mlflow.set_experiment("churn-gradient-boosting")

    # --- Step 1: Load Data ---
    print("📂 Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # --- Step 2: Calculate scale_pos_weight for Imbalance ---
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"⚖️ Calculated scale_pos_weight = {scale_pos_weight:.2f}")

    # Cross-validation strategy for both tuning and OOF threshold search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================================
    # XGBoost
    # =========================================================================
    with mlflow.start_run(run_name="xgboost_tuned"):
        print("\n🚀 Training XGBoost with hyperparameter tuning...")

        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        xgb_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }

        xgb_grid = GridSearchCV(
            estimator=xgb_model,
            param_grid=xgb_param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_

        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        for k, v in xgb_grid.best_params_.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_roc_auc", xgb_grid.best_score_)

        # Tune threshold using out-of-fold predictions on training data
        xgb_oof_proba = cross_val_predict(
            best_xgb, X_train, y_train, method='predict_proba', cv=cv, n_jobs=-1
        )[:, 1]
        xgb_best_threshold, xgb_best_oof_cost = find_best_threshold(
            y_train, xgb_oof_proba, COST_FN, COST_FP
        )
        mlflow.log_metric("best_threshold", xgb_best_threshold)
        mlflow.log_metric("best_oof_cost", xgb_best_oof_cost)

        # Evaluate on test set at default 0.5 and at tuned threshold
        xgb_metrics_default = evaluate_model(best_xgb, X_test, y_test, 0.5, COST_FN, COST_FP)
        xgb_metrics_tuned = evaluate_model(best_xgb, X_test, y_test, xgb_best_threshold, COST_FN, COST_FP)

        log_metrics("default", xgb_metrics_default)
        log_metrics("tuned", xgb_metrics_tuned)

        print(f"   ✅ XGBoost default threshold metrics: {xgb_metrics_default}")
        print(f"   ✅ XGBoost tuned threshold metrics: {xgb_metrics_tuned}")

        mlflow.xgboost.log_model(best_xgb, "xgboost_model")

    # =========================================================================
    # LightGBM
    # =========================================================================
    with mlflow.start_run(run_name="lightgbm_tuned"):
        print("\n🚀 Training LightGBM with hyperparameter tuning...")

        lgb_model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        lgb_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 1.0],
        }

        lgb_grid = GridSearchCV(
            estimator=lgb_model,
            param_grid=lgb_param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        lgb_grid.fit(X_train, y_train)
        best_lgb = lgb_grid.best_estimator_

        # Log parameters
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        for k, v in lgb_grid.best_params_.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_roc_auc", lgb_grid.best_score_)

        # Tune threshold using out-of-fold predictions on training data
        lgb_oof_proba = cross_val_predict(
            best_lgb, X_train, y_train, method='predict_proba', cv=cv, n_jobs=-1
        )[:, 1]
        lgb_best_threshold, lgb_best_oof_cost = find_best_threshold(
            y_train, lgb_oof_proba, COST_FN, COST_FP
        )
        mlflow.log_metric("best_threshold", lgb_best_threshold)
        mlflow.log_metric("best_oof_cost", lgb_best_oof_cost)

        # Evaluate on test set at default 0.5 and at tuned threshold
        lgb_metrics_default = evaluate_model(best_lgb, X_test, y_test, 0.5, COST_FN, COST_FP)
        lgb_metrics_tuned = evaluate_model(best_lgb, X_test, y_test, lgb_best_threshold, COST_FN, COST_FP)

        log_metrics("default", lgb_metrics_default)
        log_metrics("tuned", lgb_metrics_tuned)

        print(f"   ✅ LightGBM default threshold metrics: {lgb_metrics_default}")
        print(f"   ✅ LightGBM tuned threshold metrics: {lgb_metrics_tuned}")

        mlflow.lightgbm.log_model(best_lgb, "lightgbm_model")

    print("\n🎉 Done! Check the MLflow UI to compare tuned XGBoost and LightGBM.")