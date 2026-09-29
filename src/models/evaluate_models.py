# What it does:
#   1. Trains three RF variants on the same train/test split:
#         - rf_default  : vanilla RF
#         - rf_balanced : RF with class_weight='balanced'
#         - rf_smote    : RF trained on SMOTE-resampled data
#   2. For each: computes confusion matrix, ROC curve, AUC.
#   3. Computes business COST of misclassification:
#         Total Cost = FN * COST_FN + FP * COST_FP
#   4. Picks a winner based on lowest total cost.
#   5. Saves plots and a markdown report to reports/.
#   6. Logs everything to MLflow.
#
# Business cost assumptions (edit if yours differ):
#   COST_FN = 500   (missed churner = lost revenue)
#   COST_FP =  50   (wasted retention offer)
# -----------------------------------------------------------------------------

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# Make the parent folder (src/) importable so we can use model_utils.py.
# This file lives in src/models/, but model_utils.py lives in src/.
# Adding src/ to sys.path lets Python find it.
# -----------------------------------------------------------------------------
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_DIR)

from model_utils import load_processed_data

# --- MLflow tracking URI: SQLite ----------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# --- Where to save plots and the report ---------------------------------------
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Business cost assumptions ------------------------------------------------
COST_FN = 500   # cost of missing a churner (lost revenue)
COST_FP = 50    # cost of a wasted retention offer


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_three_models(X_train, y_train):
    """Train all three RF variants and return them in a dict."""
    print("\n🚀 Training three models for comparison...")
    models = {}

    # ---- 1. rf_default ------------------------------------------------------
    print("\n  [1/3] rf_default (vanilla RF)...")
    rf_default = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_default.fit(X_train, y_train)
    models["rf_default"] = rf_default

    # ---- 2. rf_balanced -----------------------------------------------------
    print("  [2/3] rf_balanced (class_weight='balanced')...")
    rf_balanced = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_balanced.fit(X_train, y_train)
    models["rf_balanced"] = rf_balanced

    # ---- 3. rf_smote --------------------------------------------------------
    print("  [3/3] rf_smote (SMOTE-resampled training data)...")
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf_smote = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf_smote.fit(X_train_res, y_train_res)
    models["rf_smote"] = rf_smote

    return models


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(model, X_test, y_test):
    """Return a dict with confusion matrix entries, recall, precision, AUC, cost."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    total_cost = fn * COST_FN + fp * COST_FP
    cost_per_customer = total_cost / len(y_test)

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "total_cost": total_cost,
        "cost_per_customer": cost_per_customer,
        "fpr": fpr,
        "tpr": tpr,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


# =============================================================================
# PLOTS
# =============================================================================
def plot_roc_curves(all_metrics, save_path):
    """Plot all three ROC curves on one figure."""
    plt.figure(figsize=(8, 6))

    # Each model's curve
    for name, m in all_metrics.items():
        plt.plot(m["fpr"], m["tpr"], label=f"{name} (AUC = {m['roc_auc']:.3f})")

    # Random baseline — drawn ONCE, outside the loop
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves — Random Forest Variants")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   💾 Saved ROC curve → {save_path}")


def plot_confusion_matrices(all_metrics, save_path):
    """Plot all three confusion matrices side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, m) in zip(axes, all_metrics.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nFN cost = ₹{m['fn'] * COST_FN:,}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   💾 Saved confusion matrices → {save_path}")


# =============================================================================
# REPORT
# =============================================================================
def write_report(all_metrics, save_path):
    """Write a markdown summary of the comparison."""

    df = pd.DataFrame(
        {
            name: {
                "Accuracy":           f"{m['accuracy']:.4f}",
                "Precision":          f"{m['precision']:.4f}",
                "Recall":             f"{m['recall']:.4f}",
                "F1":                 f"{m['f1_score']:.4f}",
                "ROC-AUC":            f"{m['roc_auc']:.4f}",
                "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
                "Total Cost (₹)":     f"{m['total_cost']:,}",
                "Cost/Customer (₹)":  f"{m['cost_per_customer']:.2f}",
            }
            for name, m in all_metrics.items()
        }
    ).T

    # Pick winner: lowest total cost, tiebreak by higher recall
    winner = min(all_metrics.items(), key=lambda kv: (kv[1]["total_cost"], -kv[1]["recall"]))

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Model Results — Churn Prediction\n\n")
        f.write(f"**Business cost assumptions:** FN = ₹{COST_FN}, FP = ₹{COST_FP}\n\n")
        f.write("## Model Comparison\n\n")
        f.write(df.to_markdown())
        f.write("\n\n## Winner\n\n")
        f.write(f"**{winner[0]}** — lowest total business cost.\n\n")
        f.write(f"- Recall: {winner[1]['recall']:.4f} "
                f"(catches {winner[1]['recall']*100:.1f}% of churners)\n")
        f.write(f"- Precision: {winner[1]['precision']:.4f}\n")
        f.write(f"- Total cost: ₹{winner[1]['total_cost']:,}\n")
        f.write(f"- Cost per customer: ₹{winner[1]['cost_per_customer']:.2f}\n\n")
        f.write("### Recommendation\n\n")
        f.write("Pick the model with the lowest total cost. Confirm it also has "
                "reasonable precision (not below 0.5), otherwise you're wasting "
                "too many retention offers. If accuracy is close but cost differs "
                "significantly, trust the cost.\n")

    print(f"   💾 Saved report → {save_path}")
    return winner


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    mlflow.set_experiment("churn-rf-baseline")

    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # 2. Train all three models
    models = train_three_models(X_train, y_train)

    # 3. Compute metrics for each
    print("\n📊 Computing metrics...")
    all_metrics = {}
    for name, model in models.items():
        all_metrics[name] = compute_metrics(model, X_test, y_test)

    # 4. Print a side-by-side table
    print("\n" + "=" * 70)
    print(f"{'Model':<15}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}{'Cost':>12}")
    print("-" * 70)
    for name, m in all_metrics.items():
        print(f"{name:<15}"
              f"{m['accuracy']:>8.4f}"
              f"{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}"
              f"{m['f1_score']:>8.4f}"
              f"{m['roc_auc']:>8.4f}"
              f"{m['total_cost']:>12,}")
    print("=" * 70)

    # 5. Plot ROC + confusion matrices
    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    cm_path  = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    plot_roc_curves(all_metrics, roc_path)
    plot_confusion_matrices(all_metrics, cm_path)

    # 6. Write markdown report
    report_path = os.path.join(REPORTS_DIR, "baseline_model_results.md")
    winner_name, winner_metrics = write_report(all_metrics, report_path)

    # 7. Log everything to MLflow
    with mlflow.start_run(run_name="day5_evaluation_deep_dive"):
        mlflow.log_param("cost_fn", COST_FN)
        mlflow.log_param("cost_fp", COST_FP)
        mlflow.log_param("models_compared", list(all_metrics.keys()))
        mlflow.log_param("winner", winner_name)

        for name, m in all_metrics.items():
            for key in ["accuracy", "precision", "recall", "f1_score",
                        "roc_auc", "total_cost", "cost_per_customer"]:
                mlflow.log_metric(f"{name}_{key}", m[key])

        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

    print(f"\n🏆 Winner: {winner_name}")
    print(f"   Recall = {winner_metrics['recall']:.4f} | "
          f"Precision = {winner_metrics['precision']:.4f} | "
          f"Total Cost = {winner_metrics['total_cost']:,}")