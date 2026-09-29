import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

# Define file paths (using relative paths for portability)
RAW_DATA_PATH = "data/raw/Telco_customer_churn.csv"
PROCESSED_DATA_DIR = "data/processed/"

def load_and_clean_data(filepath):
    """Loads raw data and performs row-wise cleaning based on your custom logic."""
    print("Loading raw data...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    print("Cleaning basic features...")
    
    # First, convert text spaces into actual NaN values
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    
    # THEN fill those NaNs with our math formula
    df['Total Charges'] = df['Total Charges'].fillna(df['Monthly Charges'] * df['Tenure Months'])

    # Binary mappings
    yes_no_mapping = {'Yes': 1, 'No': 0}
    cols_to_change = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
    for c in cols_to_change:
        df[c] = df[c].map(yes_no_mapping)
    
    # Gender and Contract mappings
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['ContractLength'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year': 24})

    # Feature Engineering
    df['AvgMonthlySpend'] = np.where(df['Tenure Months'] == 0, df['Monthly Charges'], df['Total Charges'] / df['Tenure Months'])
    service_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
    df['ServiceCount'] = df[service_cols].isin(['Yes', 1]).sum(axis=1)
    df['HasMultipleLines'] = np.where(df['Multiple Lines'] == 'Yes', 1, 0)
    
    return df

def separate_features_and_target(df):
    """Drops leakage, redundant, and identifier columns."""
    print("Dropping leakage and redundant columns...")
    
    # We added the original text columns here since we already engineered new features from them
    cols_to_drop = [
        'Churn Value', 'Churn Label', 'Churn Reason', 'Churn Score',
        'CustomerID', 'customerID', 'Count', 'Country', 'State', 'Lat Long',
        'Multiple Lines', 'Online Security', 'Online Backup', 
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['Churn Value']
    return X, y

def apply_encoders_and_scalers(X_train, X_test, y_train):
    """Applies OHE, Target Encoding, and Standard Scaling safely."""
    print("Applying Encoders (One-Hot & Target)...")
    
    # 1. One-Hot Encoding (Low Cardinality)
    multi_category_cols = ['Contract', 'Payment Method', 'Internet Service']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    train_ohe = ohe.fit_transform(X_train[multi_category_cols])
    train_ohe_df = pd.DataFrame(train_ohe, columns=ohe.get_feature_names_out(), index=X_train.index)
    
    test_ohe = ohe.transform(X_test[multi_category_cols])
    test_ohe_df = pd.DataFrame(test_ohe, columns=ohe.get_feature_names_out(), index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=multi_category_cols), train_ohe_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=multi_category_cols), test_ohe_df], axis=1)

    # 2. Target Encoding (High Cardinality)
    target_encode_cols = ['City', 'Zip Code']
    t_encoder = TargetEncoder(smooth="auto") 
    
    X_train[target_encode_cols] = t_encoder.fit_transform(X_train[target_encode_cols], y_train)
    X_test[target_encode_cols] = t_encoder.transform(X_test[target_encode_cols])

    print("Applying Standard Scaler...")
    
    # 3. Standard Scaling (Includes original numeric + newly encoded numeric features)
    numeric_cols = [
        'Tenure Months', 'Monthly Charges', 'Total Charges', 'AvgMonthlySpend', 
        'City', 'Zip Code', 'Latitude', 'Longitude', 'ContractLength', 'ServiceCount'
    ]
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test

def main():
    """The main orchestration function that runs the whole pipeline."""
    print("🚀 Starting Data Pipeline...\n")
    
    # 1. Load and clean
    df = load_and_clean_data(RAW_DATA_PATH)
    
    # Save base cleaned data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(f"{PROCESSED_DATA_DIR}/cleaned_churn.csv", index=False)
    
    # 2. Separate X and y
    X, y = separate_features_and_target(df)
    
    # 3. Train/Test Split
    print("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Encode and Scale
    X_train, X_test = apply_encoders_and_scalers(X_train, X_test, y_train)
    
    # 5. Save Final Files
    print("Saving final processed files...")
    X_train.to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)
    
    print("\n✅ Pipeline executed successfully! Data is completely ML-ready.")

if __name__ == "__main__":
    main()