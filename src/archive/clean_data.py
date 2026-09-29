import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder

# ==========================================
# STEP 1-7: DATA LOADING & ROW-WISE CLEANING
# ==========================================
df = pd.read_csv(r"C:\Users\Deepak Kumar\CHURN_SIGHT\data\raw\Telco_customer_churn.csv")
df.columns = df.columns.str.strip()

df['Total Charges'] = df['Total Charges'].fillna(df['Monthly Charges'] * df['Tenure Months'])
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

yes_no_mapping = {'Yes': 1, 'No': 0}
cols_to_change = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
for c in cols_to_change:
    df[c] = df[c].map(yes_no_mapping)

gender_mapping = {'Male': 1, 'Female': 0}
df['Gender'] = df['Gender'].map(gender_mapping)

contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractLength'] = df['Contract'].map(contract_mapping)

df['AvgMonthlySpend'] = np.where(df['Tenure Months'] == 0, df['Monthly Charges'], df['Total Charges'] / df['Tenure Months'])
service_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
df['ServiceCount'] = df[service_cols].isin(['Yes', 1]).sum(axis=1)
df['HasMultipleLines'] = np.where(df['Multiple Lines'] == 'Yes', 1, 0)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_churn.csv", index=False)

# ==========================================
# STEP 8: SEPARATE X AND y (KEEPING CITY & ZIP)
# ==========================================
# Notice 'City' and 'Zip Code' are no longer in this drop list!
cols_to_drop = [
    'Churn Value', 'Churn Label', 'Churn Reason', 'Churn Score',
    'CustomerID', 'customerID', 'Count', 'Country','State','Lat Long'
]
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['Churn Value']

# ==========================================
# STEP 9: TRAIN/TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# STEP 10: ENCODING (NO SCHEMA OR TARGET LEAKAGE)
# ==========================================

# 10A: ONE-HOT ENCODING (For Low Cardinality: < 5 unique values)
multi_category_cols = ['Contract', 'Payment Method', 'Internet Service']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_ohe = ohe.fit_transform(X_train[multi_category_cols])
train_ohe_df = pd.DataFrame(train_ohe, columns=ohe.get_feature_names_out(), index=X_train.index)

test_ohe = ohe.transform(X_test[multi_category_cols])
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe.get_feature_names_out(), index=X_test.index)

X_train = pd.concat([X_train.drop(columns=multi_category_cols), train_ohe_df], axis=1)
X_test = pd.concat([X_test.drop(columns=multi_category_cols), test_ohe_df], axis=1)

# 10B: TARGET ENCODING (For High Cardinality: City & Zip Code)
target_encode_cols = ['City', 'Zip Code']
t_encoder = TargetEncoder(smooth="auto") 

# Crucial: TargetEncoder requires BOTH X_train and y_train to fit!
# It calculates the average churn for each city using only the training data.
X_train[target_encode_cols] = t_encoder.fit_transform(X_train[target_encode_cols], y_train)

# For the test set, it maps the cities to the averages it learned during training.
X_test[target_encode_cols] = t_encoder.transform(X_test[target_encode_cols])

# ==========================================
# STEP 11: SAFE SCALING
# ==========================================
# We now include the newly Target-Encoded columns in our scaling!
numeric_cols = [
    'Tenure Months', 'Monthly Charges', 'Total Charges', 'AvgMonthlySpend', 'Zip Code' ,'City','Latitude', 'Longitude' # These are now continuous numbers, so they get scaled too
]
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ==========================================
# STEP 12: SAVE FINAL DATASETS
# ==========================================
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Pipeline executed with Target Encoding and zero data leakage!")