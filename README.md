# Customer Churn Prediction for E-Commerce

A data analytics and machine learning project that analyzes e-commerce transaction data and predicts customer churn risk.

## Project Overview

The project:
- checks and explores e-commerce transaction data
- analyzes revenue, categories, customers, regions, and product profit
- creates customer-level RFM features
- trains a Logistic Regression model
- evaluates model performance
- calculates churn probability
- groups customers into Low, Medium, and High Risk levels
- provides practical customer-retention actions

## Dataset

The dataset contains transaction-level information such as:

- Order ID
- Order Date
- Customer ID
- Customer Type
- Product
- Category
- Region
- Quantity
- Revenue
- Profit

## Prediction Features

The model uses:

- **Recency** — days since the last purchase
- **Frequency** — number of historical purchases
- **Monetary** — total historical revenue
- **Average Order Value** — average revenue per order

## Model

**Algorithm:** Logistic Regression

A historical cutoff and a later observation period are used so future customer activity is not included in the prediction features.

## Results

- Accuracy: **63.2%**
- Precision: **50.0%**
- Recall: **42.9%**
- F1 Score: **46.2%**
- ROC-AUC: **67.9%**
- Confusion Matrix: `[[36, 12], [16, 12]]`

### Customer Risk Distribution

- Low Risk: **173**
- Medium Risk: **207**
- High Risk: **0**

## Files

- `customer_churn_project.py` — complete Python project code
- `IBM_Masterclass_Final_Project.ipynb` — Google Colab notebook
- `requirements.txt` — required Python libraries
- `PROJECT_REPORT.pdf` — project report
- `customer_level_churn_dataset.csv` — customer-level model results
- `top_20_risk_customers.csv` — customers with the highest predicted churn risk

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run:

```bash
python customer_churn_project.py
```

You can also open the `.ipynb` notebook in Google Colab and run it there.

## Tools Used

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- Google Colab

## Business Use

The model can help prioritize retention efforts by identifying customers with relatively higher predicted churn probability.

The model is a learning and analysis project. Its predictions show statistical patterns and should not be treated as proof of why a customer stops purchasing.
