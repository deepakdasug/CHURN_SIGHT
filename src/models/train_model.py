import mlflow
from sklearn.ensemble import RandomForestClassifier

# Import our custom shared tools
from model_utils import load_processed_data, evaluate_and_log

# ==============================================================================
# MODELING CONCEPT: RANDOM FOREST
# ==============================================================================
# Random Forest: An ensemble of decorrelated decision trees whose independent outputs are aggregated via majority voting.
# Bagging (Bootstrap Aggregation): Training each tree on a random sample of rows with replacement to reduce variance.
# Feature Subsampling: Restricting each split to a random subset of features to keep individual trees diverse.

# 1. Point MLflow to your local database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# 2. Name your project experiment
mlflow.set_experiment("Telco_Churn_Prediction")

def main():
    # Uses safe relative paths from model_utils.py
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print("Training Baseline Random Forest model...")

    # 3. Start the MLflow recording block
    with mlflow.start_run(run_name="Baseline_Random_Forest"):
        
        # Initialize and train the standard model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # 4. Evaluate and Log
        # This custom function handles calculating Accuracy, Precision, Recall, F1, ROC-AUC, 
        # and logs them all to the MLflow database automatically.
        evaluate_and_log(rf_model, X_test, y_test, "random_forest_model")

if __name__ == "__main__":
    main()