import mlflow
from sklearn.linear_model import LogisticRegression

# Import our custom shared tools
from model_utils import load_processed_data, evaluate_and_log

# ==============================================================================
# MODELING CONCEPTS: LOGISTIC REGRESSION & CLASS BALANCING
# ==============================================================================
# Logistic Regression: Models the probability of churn (between 0 and 1) using a sigmoid curve.
# class_weight='balanced': Modifies the grading system during training. It heavily penalizes 
# the model for missing the minority class (Churners), forcing it to pay attention rather 
# than lazily predicting 'Stay' for everyone to artificially inflate its Accuracy score.

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telco_Churn_Prediction")

def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print("Training Balanced Logistic Regression model...")

    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        
        # 1. Train the model with the "Don't Be Lazy" rule applied
        log_model = LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear', random_state=42)
        log_model.fit(X_train, y_train)

        # 2. Evaluate and Log
        evaluate_and_log(log_model, X_test, y_test, "logistic_regression_model")

if __name__ == "__main__":
    main()