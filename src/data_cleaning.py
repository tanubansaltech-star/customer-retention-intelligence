import pandas as pd

# -----------------------------
# 1. Load raw dataset
# -----------------------------
input_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(input_path)

print("Raw dataset loaded!")
print("Shape:", df.shape)


# -----------------------------
# 2. Remove unnecessary spaces
# -----------------------------
df.columns = df.columns.str.strip()

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# -----------------------------
# 3. Convert TotalCharges to numeric
# -----------------------------
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# -----------------------------
# 4. Check missing values
# -----------------------------
print("\nMissing values after conversion:")
print(df.isnull().sum())


# -----------------------------
# 5. Remove duplicate customers
# -----------------------------
before = len(df)

df = df.drop_duplicates(subset="customerID")

after = len(df)

print("\nDuplicate customers removed:", before - after)


# -----------------------------
# 6. Remove rows with missing values
# -----------------------------
df = df.dropna()

print("Shape after cleaning:", df.shape)


# -----------------------------
# 7. Save cleaned dataset
# -----------------------------
output_path = "data/processed/customer_churn_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved to:", output_path)