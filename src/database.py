import sqlite3
import pandas as pd

# ============================================================
# DATABASE SETUP
# ============================================================

database_path = "data/customer_retention.db"

connection = sqlite3.connect(database_path)

print("SQLite database connected successfully!")


# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

csv_path = "data/processed/customer_churn_features.csv"

df = pd.read_csv(csv_path)

print("Feature-engineered dataset loaded!")
print("Rows:", len(df))


# ============================================================
# SAVE DATA TO DATABASE
# ============================================================

df.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

print("Customer data inserted into SQLite database!")


# ============================================================
# VERIFY DATABASE
# ============================================================

result = pd.read_sql(
    "SELECT COUNT(*) AS total_customers FROM customers",
    connection
)

print("\n===== DATABASE VERIFICATION =====")
print(result)


# ============================================================
# CLOSE CONNECTION
# ============================================================

connection.close()

print("\nDatabase connection closed.")
print("Database setup completed successfully!")