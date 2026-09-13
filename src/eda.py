import pandas as pd

# Cleaned dataset load karo
file_path = "data/processed/customer_churn_cleaned.csv"
df = pd.read_csv(file_path)

print("\n===== DATASET OVERVIEW =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== CHURN COUNT =====")
print(df["Churn"].value_counts())

print("\n===== CHURN PERCENTAGE =====")
churn_percentage = df["Churn"].value_counts(normalize=True) * 100
print(churn_percentage.round(2))

print("\n===== CONTRACT vs CHURN =====")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index").round(3) * 100)

print("\n===== INTERNET SERVICE vs CHURN =====")
print(pd.crosstab(df["InternetService"], df["Churn"], normalize="index").round(3) * 100)

print("\n===== PAYMENT METHOD vs CHURN =====")
print(pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index").round(3) * 100)

print("\n===== AVERAGE MONTHLY CHARGES =====")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

print("\n===== AVERAGE TENURE =====")
print(df.groupby("Churn")["tenure"].mean().round(2))