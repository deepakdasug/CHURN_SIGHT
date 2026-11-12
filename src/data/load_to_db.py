import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import os

# ==========================================
# STEP 1: DATA LOADING
# Read the customer churn CSV file into a pandas DataFrame
# Display first few rows and data structure for verification
# ==========================================

# Define file paths directly
csv_path = "C:\\Users\\Deepak Kumar\\CHURN_SIGHT\\data\\raw\\Telco_customer_churn.csv"
db_path = "C:\\Users\\Deepak Kumar\\CHURN_SIGHT\\data\\processed\\churn.db"

df = pd.read_csv(csv_path)
print(df.head())
print(df.info())  

# ==========================================
# STEP 2: DATABASE SETUP AND DATA IMPORT
# Create SQLite database connection and import DataFrame into 'customers' table
# Uses SQLAlchemy for easy database interaction
# ==========================================

con=sqlite3.connect(db_path)
df.to_sql('customers',con,index=False,if_exists='replace')

print(f"✅ Table 'customers' created with {len(df)} rows")
print(f"📊 Columns: {df.columns.tolist()}")

# ==========================================
# STEP 3: DATA VERIFICATION
# Reconnect to database and verify data was loaded correctly
# Display first 5 rows from the database table
# ==========================================
conn = sqlite3.connect(db_path)
cursor=conn.cursor()
cursor.execute("SELECT * FROM customers LIMIT 5")
rows=cursor.fetchall()
for row in rows:
    print(row,"\n")

# ==========================================
# STEP 4: ADDITIONAL TABLE CREATION
# Enable foreign key support and create related tables for ML predictions and retention actions
# These tables will be used for storing model predictions and customer retention strategies
# ==========================================

# Enable foreign key support (SQLite turns this off by default)
cursor.execute("PRAGMA foreign_keys = ON;")

# Create the predictions table (linked to customers table via foreign key)
# This table will store ML model predictions for each customer
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    churn_probability REAL,
    risk_label TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (CustomerID) ON DELETE CASCADE
);
""")

# Create the retention_actions table (linked to predictions table via foreign key)
# This table will store recommended actions to retain customers at risk of churning
cursor.execute("""
CREATE TABLE IF NOT EXISTS retention_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    action TEXT,
    reason TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);
""")

# Save (commit) the changes to database
conn.commit()
print("✅ Tables 'predictions' and 'retention_actions' created successfully with all foreign keys!")

# ==========================================
# STEP 5: CHURN ANALYTICS
# Perform various SQL queries to analyze churn patterns and customer behavior
# Generate insights on overall churn, contract types, payment methods, and customer profiles
# ==========================================

# ==========================================
# 5.1 OVERALL CHURN SUMMARY
# Calculate total customers, churn rate, and monthly revenue loss
# ==========================================
query_totals = """
SELECT 
    COUNT(*) AS total_customers,
    SUM("Churn Value") AS total_churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN "Churn Value" = 1 THEN "Monthly Charges" ELSE 0 END), 2) AS monthly_revenue_loss
FROM customers;
"""
df_totals = pd.read_sql(query_totals, conn)
print("--- OVERALL CHURN SUMMARY ---")
print(df_totals, "\n")


# ==========================================
# 5.2 CHURN BY CONTRACT TYPE
# Analyze churn rates across different contract types (Month-to-month, One year, Two year)
# Includes churn volume and average monthly charges to identify high-risk segments
# ==========================================
query_contract = """
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM("Churn Value") AS total_churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
"""
df_contract = pd.read_sql(query_contract, conn)
print("--- CHURN BY CONTRACT ---")
print(df_contract, "\n")


# ==========================================
# 5.3 CHURN BY PAYMENT METHOD
# Analyze churn rates across different payment methods
# Helps identify if payment method influences customer retention
# ==========================================
query_payment = """
SELECT 
    "Payment Method",
    COUNT(*) AS total_customers,
    SUM("Churn Value") AS total_churned,
    ROUND(AVG("Churn Value") * 100, 2) AS churn_rate_pct,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charge
FROM customers
GROUP BY "Payment Method"
ORDER BY churn_rate_pct DESC;
"""
df_payment = pd.read_sql(query_payment, conn)
print("--- CHURN BY PAYMENT METHOD ---")
print(df_payment, "\n")


# ==========================================
# 5.4 CUSTOMER PROFILE: CHURNED VS RETAINED
# Compare characteristics between customers who churned vs those who stayed
# Includes monthly charges, tenure, and customer lifetime value (CLTV) analysis
# ==========================================
query_profile = """
SELECT 
    "Churn Label",
    COUNT(*) AS total_customers,
    ROUND(AVG("Monthly Charges"), 2) AS avg_monthly_charges,
    ROUND(AVG("Tenure Months"), 1) AS avg_tenure_months,
    ROUND(AVG("CLTV"), 0) AS avg_lifetime_value
FROM customers
GROUP BY "Churn Label";
"""
df_profile = pd.read_sql(query_profile, conn)
print("--- CHURNED VS RETAINED PROFILE ---")
print(df_profile)

# ==========================================
# STEP 6: EXPORT ANALYTICS RESULTS
# Save analysis results to CSV files for further use in dashboards, reports, or ML pipelines
# ==========================================
df_totals.to_csv("C:\\Users\\Deepak Kumar\\CHURN_SIGHT\\data\\processed\\overall_totals.csv", index=False)
df_payment.to_csv("C:\\Users\\Deepak Kumar\\CHURN_SIGHT\\data\\processed\\churn_by_payment.csv", index=False)
df_profile.to_csv("C:\\Users\\Deepak Kumar\\CHURN_SIGHT\\data\\processed\\customer_profile.csv", index=False)

# Close database connection
conn.close()