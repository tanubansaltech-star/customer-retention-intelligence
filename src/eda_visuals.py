import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "data/processed/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Create output folder for charts
import os
os.makedirs("data/processed/charts", exist_ok=True)


# 1. Overall Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/processed/charts/churn_distribution.png", dpi=300)
plt.show()


# 2. Contract Type vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("data/processed/charts/contract_vs_churn.png", dpi=300)
plt.show()


# 3. Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("data/processed/charts/monthly_charges_vs_churn.png", dpi=300)
plt.show()


# 4. Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig("data/processed/charts/tenure_vs_churn.png", dpi=300)
plt.show()


# 5. Payment Method vs Churn
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Payment Method vs Customer Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("data/processed/charts/payment_method_vs_churn.png", dpi=300)
plt.show()


print("\n===== EDA VISUALIZATION COMPLETE =====")
print("All charts saved in:")
print("data/processed/charts/")