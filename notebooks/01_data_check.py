import pandas as pd

# Dataset ka path
file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Dataset read karo
df = pd.read_csv(file_path)

# Basic information
print("\n--- DATASET LOADED SUCCESSFULLY ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())