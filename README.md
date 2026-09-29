# CHURN_SIGHT

Customer churn analytics system for telecommunications companies to analyze customer data and identify churn patterns.

## Project Overview

CHURN_SIGHT is a data analytics project that helps telecommunications companies:
- **Load and Process Data**: Import customer data from CSV to database
- **Analyze Churn Patterns**: Understand why customers leave through detailed analytics
- **Generate Reports**: Export analytics results for further analysis

### Current Features
- Automated data pipeline from CSV to SQLite database
- Database table creation with proper schema
- Basic churn analytics and reporting
- Export of analytics results to CSV files

## Project Structure

```
CHURN_SIGHT/
├── data/
│   ├── raw/                   # Raw data files
│   │   └── Telco_customer_churn.csv
│   └── processed/             # Processed data and database
│       └── churn.db
├── src/
│   └── data/                  # Data processing scripts
│       └── load_to_db.py
└── README.md                  # This file
```

## Configuration

The project uses exact file paths in `src/data/load_to_db.py`. You need to customize these paths to match your local directory structure:

**Current paths in the script:**
- CSV file: `C:\Users\Deepak Kumar\CHURN_SIGHT\data\raw\Telco_customer_churn.csv`
- Database: `C:\Users\Deepak Kumar\CHURN_SIGHT\data\processed\churn.db`
- Output files: `C:\Users\Deepak Kumar\CHURN_SIGHT\data\processed\`

**To customize for your system:**
1. Open `src/data/load_to_db.py`
2. Find the path variables at the top of the script
3. Replace the paths with your actual file locations
4. Ensure you use double backslashes for Windows paths (e.g., `C:\\Users\\YourName\\Project\\data\\...`)

**Path variables to modify:**
- `csv_path`: Path to the input CSV file
- `db_path`: Path to the SQLite database file
- Output CSV paths for analytics results

## Setup Instructions

### Step 1: Prerequisites

Ensure you have Python 3.14 or higher installed. Check your Python version:
```bash
python --version
```

If you have multiple Python versions, find your Python 3.14 path:
```bash
where python  # On Windows
which python   # On Linux/Mac
```

### Step 2: Install Required Packages

Install all required Python packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap mlflow fastapi uvicorn streamlit sqlalchemy openpyxl xlsxwriter requests pytest docker
```

If you need to use a specific Python interpreter:
```bash
python -m pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap mlflow fastapi uvicorn streamlit sqlalchemy openpyxl xlsxwriter requests pytest docker
```

Replace `python` with your specific Python path if needed (e.g., `python3.14` or full path to your Python executable).

### Step 3: Verify Installation

Test that all packages are installed correctly by running the data loading script:

```bash
cd src/data
python load_to_db.py
```

Or from the project root:
```bash
python src/data/load_to_db.py
```

Expected output:
- Data frame information display
- Database creation confirmation
- Churn analytics summary

### Step 4: Run Data Pipeline

First, ensure the required directories exist:
```bash
mkdir -p data/processed
```

Then navigate to the src/data directory and run the script:

```bash
cd src/data
python load_to_db.py
```

The data pipeline performs the following steps:
1. **Data Loading**: Reads customer data from CSV file
2. **Database Setup**: Creates SQLite database and imports data
3. **Table Creation**: Sets up customers, predictions, and retention_actions tables
4. **Data Verification**: Validates data import with sample queries
5. **Analytics Generation**: Runs churn analysis queries
6. **Results Export**: Saves analytics results to CSV files

### Step 5: Verify Database

Check that the database was created successfully:
```bash
# Navigate to data/processed directory
cd data/processed

# Check for churn.db file (Windows)
dir churn.db

# Or on Linux/Mac
ls churn.db
```

## Usage

### Running the Data Pipeline

Navigate to the src/data directory and run the data loading script:

```bash
cd src/data
python load_to_db.py
```

Or from the project root:
```bash
python src/data/load_to_db.py
```

If you need to use a specific Python interpreter:
```bash
cd src/data
python3 load_to_db.py
```

## Data Schema

### Customers Table
- **CustomerID**: Unique customer identifier
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: Phone, Internet, Streaming services
- **Contract**: Contract type, Payment method, Billing
- **Charges**: Monthly and Total charges
- **Churn Data**: Churn Label, Churn Value, Churn Score, Churn Reason

### Predictions Table
- **customer_id**: Foreign key to customers table
- **churn_probability**: ML model prediction score
- **risk_label**: Risk category (High/Medium/Low)
- **timestamp**: Prediction timestamp

### Retention Actions Table
- **prediction_id**: Foreign key to predictions table
- **action**: Recommended retention action
- **reason**: Justification for the action

## Analytics Overview

The system currently provides the following analytics:

1. **Overall Churn Summary**: Total customers, churn rate, revenue loss
2. **Churn by Contract**: Analysis across contract types
3. **Churn by Payment Method**: Payment method impact on retention
4. **Customer Profiles**: Comparison of churned vs retained customers

### Sample Analytics Output

**Overall Churn Summary:**
- Total Customers: 7,043
- Total Churned: 1,869
- Churn Rate: 26.54%
- Monthly Revenue Loss: $139,130.85

**Churn by Contract:**
- Month-to-month: 42.71% churn rate
- One year: 11.27% churn rate
- Two year: 2.83% churn rate

**Churn by Payment Method:**
- Electronic check: 45.29% churn rate
- Mailed check: 19.11% churn rate
- Bank transfer (automatic): 16.71% churn rate
- Credit card (automatic): 15.24% churn rate

## Troubleshooting

### Python Version Issues
If you encounter module import errors, ensure you're using Python 3.14 or higher:
```bash
python --version
python3 --version
```

### Package Installation Issues
If pip install fails, try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

### Database Connection Issues
Ensure the data/processed directory exists:
```bash
# Windows
mkdir data\processed

# Linux/Mac
mkdir -p data/processed
```

## Future Enhancements

- [ ] Implement feature engineering pipeline
- [ ] Train ML models for churn prediction
- [ ] Build interactive dashboard
- [ ] Create REST API for model deployment
- [ ] Add automated testing
- [ ] Set up MLflow for experiment tracking

## License

This project is for educational and demonstration purposes.