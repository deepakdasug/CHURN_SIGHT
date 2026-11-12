# Telco Customer Churn - Data Dictionary

This document explains all 33 columns found in the `Telco_customer_churn.csv` dataset.

### 1. Customer Location & Demographics
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **CustomerID** | Text | A unique alphanumeric ID for every customer (e.g., 3668-QPYBK). |
| **Count** | Number | A value used in reporting/dashboarding (always equals 1). |
| **Country** | Text | The country of the customer (United States). |
| **State** | Text | The state where the customer lives (California). |
| **City** | Text | The specific city in California where the customer lives. |
| **Zip Code** | Number | The 5-digit zip code of the customer. |
| **Lat Long** | Text | The combined latitude and longitude coordinates. |
| **Latitude** | Decimal | The latitude coordinate of the customer's location. |
| **Longitude** | Decimal | The longitude coordinate of the customer's location. |
| **Gender** | Text | The customer's gender (Male or Female). |
| **Senior Citizen** | Text | Indicates if the customer is 65 or older (Yes, No). |
| **Partner** | Text | Indicates if the customer is married or has a partner (Yes, No). |
| **Dependents** | Text | Indicates if the customer lives with any dependents like children (Yes, No). |

### 2. Account & Billing Information
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Tenure Months** | Number | Total number of months the customer has been with the company. |
| **Contract** | Text | The customer's current contract type (Month-to-month, One year, Two year). |
| **Paperless Billing** | Text | Indicates if the customer uses paperless billing (Yes, No). |
| **Payment Method** | Text | How the customer pays their bill (Electronic check, Mailed check, Bank transfer, Credit card). |
| **Monthly Charges** | Decimal | The amount charged to the customer every month in dollars. |
| **Total Charges** | Decimal | The cumulative total amount charged to the customer over their lifetime. |

### 3. Services Subscribed
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Phone Service** | Text | Indicates if the customer has a home phone service (Yes, No). |
| **Multiple Lines** | Text | Indicates if the customer has multiple phone lines (Yes, No, No phone service). |
| **Internet Service** | Text | The type of internet connection (DSL, Fiber optic, No). |
| **Online Security** | Text | Indicates if the customer pays for additional online security (Yes, No, No internet). |
| **Online Backup** | Text | Indicates if the customer pays for online backup service (Yes, No, No internet). |
| **Device Protection** | Text | Indicates if the customer has an equipment protection plan (Yes, No, No internet). |
| **Tech Support** | Text | Indicates if the customer pays for premium tech support (Yes, No, No internet). |
| **Streaming TV** | Text | Indicates if the customer uses their internet to stream TV (Yes, No, No internet). |
| **Streaming Movies**| Text | Indicates if the customer uses their internet to stream movies (Yes, No, No internet). |

### 4. Churn & Predictive Metrics
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Churn Label** | Text | Text indicator of whether the customer left ("Yes" or "No"). |
| **Churn Value** | Number | Numeric indicator of churn (1 = Left, 0 = Stayed). |
| **Churn Reason** | Text | The specific reason the customer gave for leaving (e.g., "Competitor made better offer"). |
| **Churn Score** | Number | A score (0-100) predicting how likely they are to leave. |
| **CLTV** | Number | Customer Lifetime Value; the predicted total revenue the business will earn from this customer. |
