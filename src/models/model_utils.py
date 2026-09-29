import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==============================================================================
# CONFUSION MATRIX & EVALUATION METRICS CONCEPTS
# ==============================================================================
# True Positive (TP): Model predicted Churn (1), and the customer actually Churned (1).
# True Negative (TN): Model predicted Stay (0), and the customer actually Stayed (0).
# False Positive (FP): Type I Error (False Alarm) - Model predicted Churn (1), but customer Stayed (0).
# False Negative (FN): Type II Error (Missed Churner) - Model predicted Stay (0), but customer Churned (1).
#
# Accuracy: Overall correctness. Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision: Correctness of positive churn predictions (minimizes false alarms). Formula: Precision = TP / (TP + FP)
# Recall: Ability to catch all actual churners (minimizes missed cases). Formula: Recall = TP / (TP + FN)
# F1-Score: Harmonic mean balancing Precision and Recall. Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
# ROC-AUC: Measures how well the model ranks actual churners higher than stayers across all thresholds.

def load_processed_data():
    """Loads the clean, scaled data from the processed folder using relative paths."""
    print("Loading processed datasets...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_and_log(model, X_test, y_test, model_name):
    """Calculates all metrics and logs them to the active MLflow run."""
    # 1. Hard predictions (0 or 1)
    y_pred = model.predict(X_test)
    
    # 2. Probability predictions (Check if the model supports it for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        roc_print = f"{roc_auc:.4f}"
    else:
        roc_print = "N/A"

    # 3. Calculate core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Model Evaluated! Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_print}")

    # 4. Log everything to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(model, model_name)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc if hasattr(model, "predict_proba") else None,
    }