# Baseline Model Results — Churn Prediction

**Business cost assumptions:** FN = ₹500, FP = ₹50

## Model Comparison

|             |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   TP |   FP |   FN |   TN |   Total Cost (₹) |   Cost/Customer (₹) |
|:------------|-----------:|------------:|---------:|-------:|----------:|-----:|-----:|-----:|-----:|-----------------:|--------------------:|
| rf_default  |     0.7963 |      0.6408 |   0.5294 | 0.5798 |    0.846  |  198 |  111 |  176 |  924 |           93,550 |               66.39 |
| rf_balanced |     0.7779 |      0.5679 |   0.6818 | 0.6197 |    0.8464 |  255 |  194 |  119 |  841 |           69,200 |               49.11 |
| rf_smote    |     0.7835 |      0.593  |   0.5882 | 0.5906 |    0.8403 |  220 |  151 |  154 |  884 |           84,550 |               60.01 |

## Winner

**rf_balanced** — lowest total business cost.

- Recall: 0.6818 (catches 68.2% of churners)
- Precision: 0.5679
- Total cost: ₹69,200
- Cost per customer: ₹49.11

### Recommendation

Pick the model with the lowest total cost. Confirm it also has reasonable precision (not below 0.5), otherwise you're wasting too many retention offers. If accuracy is close but cost differs significantly, trust the cost.
