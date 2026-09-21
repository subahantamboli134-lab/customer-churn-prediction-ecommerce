# IBM SkillsBuild - Data Analytics with AI
# Final Project: Customer Churn Prediction for E-Commerce
# Student: Subahan Raphikso Tamboli

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

SHEET_ID = "1jh8UeSiUokzElxnSYRbD1JAgN-Wqkk4Sqp9axEvNDbw"

RAW_DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    + SHEET_ID
    + "/export?format=csv&gid=0"
)

CLEAN_DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    + SHEET_ID
    + "/export?format=csv&gid=1728593922"
)

raw_df = pd.read_csv(RAW_DATA_URL)
df = pd.read_csv(CLEAN_DATA_URL)

print("Raw data shape:", raw_df.shape)
print("Clean data shape:", df.shape)

print("\nRaw data missing values:")
print(raw_df.isna().sum())

print("\nDuplicate Order_ID values in raw data:")
print(raw_df["Order_ID"].duplicated().sum())

print("\nClean data columns:")
print(df.columns.tolist())

def convert_order_date(series):
    numeric_dates = pd.to_numeric(series, errors="coerce")

    excel_dates = (
        pd.Timestamp("1899-12-30")
        + pd.to_timedelta(numeric_dates, unit="D")
    )

    text_dates = pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=True
    )

    return text_dates.fillna(excel_dates)

df["Order_Date"] = convert_order_date(df["Order_Date"])
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

print("\nDate range:")
print(df["Order_Date"].min(), "to", df["Order_Date"].max())

# Exploratory analysis

df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\nMonthly revenue:")
print(monthly_revenue)

monthly_revenue.plot(
    kind="line",
    marker="o",
    figsize=(8, 4),
    title="Monthly Revenue"
)
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

category_revenue = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by category:")
print(category_revenue)

category_revenue.plot(
    kind="bar",
    figsize=(7, 4),
    title="Revenue by Product Category"
)
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

customer_segments = (
    df[["Customer_ID", "Customer_Type"]]
    .drop_duplicates("Customer_ID")
    ["Customer_Type"]
    .value_counts()
)

print("\nUnique customers by segment:")
print(customer_segments)

customer_segments.plot(
    kind="bar",
    figsize=(6, 4),
    title="Unique Customers by Customer Type"
)
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

regional_revenue = (
    df[df["Region"] != "Unknown Region"]
    .groupby("Region")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by region:")
print(regional_revenue)

regional_revenue.sort_values().plot(
    kind="barh",
    figsize=(7, 4),
    title="Revenue by Region"
)
plt.xlabel("Revenue")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

product_profit = (
    df.dropna(subset=["Product"])
    .groupby("Product")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 products by profit:")
print(product_profit.head(10))

product_profit.head(10).sort_values().plot(
    kind="barh",
    figsize=(7, 5),
    title="Top 10 Named Products by Profit"
)
plt.xlabel("Profit")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# Customer churn prediction

max_date = df["Order_Date"].max()
cutoff_date = max_date - pd.Timedelta(days=180)

history = df[df["Order_Date"] <= cutoff_date].copy()
future = df[
    (df["Order_Date"] > cutoff_date)
    & (df["Order_Date"] <= max_date)
].copy()

print("\nHistorical cutoff:", cutoff_date.date())
print("Future observation ends:", max_date.date())

customer_df = (
    history.groupby("Customer_ID")
    .agg(
        Order_Count=("Order_ID", "count"),
        Total_Revenue=("Revenue", "sum"),
        Avg_Order_Value=("Revenue", "mean"),
        First_Purchase=("Order_Date", "min"),
        Last_Purchase=("Order_Date", "max"),
    )
    .reset_index()
)

customer_df["Recency"] = (
    cutoff_date - customer_df["Last_Purchase"]
).dt.days

customer_df["Frequency"] = customer_df["Order_Count"]
customer_df["Monetary"] = customer_df["Total_Revenue"]

future_buyers = set(
    future["Customer_ID"]
    .dropna()
    .astype(str)
)

customer_df["Churn_Status"] = (
    ~customer_df["Customer_ID"]
    .astype(str)
    .isin(future_buyers)
).astype(int)

print("\nChurn counts:")
print(
    customer_df["Churn_Status"]
    .value_counts()
    .rename(index={0: "Not Churned", 1: "Churned"})
)

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Avg_Order_Value"
]

X = customer_df[features]
y = customer_df["Churn_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

predicted = model.predict(X_test)
probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, zero_division=0)
recall = recall_score(y_test, predicted, zero_division=0)
f1 = f1_score(y_test, predicted, zero_division=0)
roc_auc = roc_auc_score(y_test, probability)

print("\nMODEL RESULTS")
print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))
print("ROC-AUC  :", round(roc_auc, 3))

cm = confusion_matrix(y_test, predicted)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

final_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

final_model.fit(X, y)

customer_df["Predicted_Churn_Status"] = final_model.predict(X)
customer_df["Churn_Probability"] = final_model.predict_proba(X)[:, 1]

def risk_level(prob):
    if prob < 0.40:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"

customer_df["Risk_Level"] = customer_df["Churn_Probability"].apply(risk_level)

risk_table = customer_df[
    [
        "Customer_ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Churn_Probability",
        "Risk_Level"
    ]
].sort_values(
    "Churn_Probability",
    ascending=False
)

print("\nTop 20 customers by predicted churn risk:")
print(risk_table.head(20).to_string(index=False))

print("\nBUSINESS ACTIONS")
print("1. Contact customers with the highest churn probability first.")
print("2. Use re-engagement offers for customers with long purchase gaps.")
print("3. Encourage repeat purchases among low-frequency customers.")
print("4. Measure campaign results and update the model with new data.")

os.makedirs("outputs", exist_ok=True)

customer_df.to_csv(
    "outputs/customer_level_churn_dataset.csv",
    index=False
)

risk_table.to_csv(
    "outputs/customer_risk_table.csv",
    index=False
)

print("\nProject files saved in the outputs folder.")
