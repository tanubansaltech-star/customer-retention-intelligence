
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------
# 1. Load feature-engineered data
# --------------------------------
input_path = "data/processed/customer_churn_features.csv"

df = pd.read_csv(input_path)

print("Dataset loaded!")
print("Shape:", df.shape)


# --------------------------------
# 2. Separate features and target
# --------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]


# --------------------------------
# 3. Remove Customer ID
# --------------------------------
X = X.drop(columns=["customerID"])


# --------------------------------
# 4. Identify column types
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
# 5. Create preprocessing pipeline
# --------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)


# --------------------------------
# 6. Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 7. Fit preprocessing
# --------------------------------
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------
# 8. Display results
# --------------------------------
print("\n===== PREPROCESSING COMPLETE =====")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)


print("\nData is now ready for Machine Learning!")

