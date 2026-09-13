import pandas as pd

from sklearn.model_selection import train_test_split


# --------------------------------
# 1. Load feature-engineered data
# --------------------------------
input_path = "data/processed/customer_churn_features.csv"

df = pd.read_csv(input_path)

print("Dataset loaded!")
print("Shape:", df.shape)


# --------------------------------
# 2. Separate target from features
# --------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]


print("\n===== TARGET =====")
print("Target column: Churn")
print("Target values:")
print(y.value_counts())


# --------------------------------
# 3. Identify categorical columns
# --------------------------------
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\n===== CATEGORICAL COLUMNS =====")
print(categorical_columns)

print("\n===== NUMERIC COLUMNS =====")
print(numeric_columns)


# --------------------------------
# 4. Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 5. Display split information
# --------------------------------
print("\n===== TRAIN TEST SPLIT =====")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# --------------------------------
# 6. Check churn distribution
# --------------------------------
print("\n===== TRAINING CHURN DISTRIBUTION =====")
print(y_train.value_counts(normalize=True).round(3) * 100)

print("\n===== TESTING CHURN DISTRIBUTION =====")
print(y_test.value_counts(normalize=True).round(3) * 100)


print("\n===== ML PREPARATION COMPLETE =====")