import pandas as pd
import joblib
import shap

# Load trained model
model = joblib.load(
    "models/best_model.pkl"
)

# Load customer data
df = pd.read_csv(
    "data/processed/customer_churn_features.csv"
)

# Separate features
X = df.drop(columns=["Churn", "customerID"])

# Get preprocessing and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# Transform data
X_processed = preprocessor.transform(X)

# Get feature names
feature_names = (
    preprocessor
    .get_feature_names_out()
)

# Create SHAP explainer
explainer = shap.LinearExplainer(
    classifier,
    X_processed
)

# Calculate SHAP values
shap_values = explainer(
    X_processed
)

print("\n===== SHAP EXPLAINABLE AI =====")

print(
    "SHAP analysis completed successfully!"
)

print(
    "Number of customers:",
    len(X)
)

print(
    "Number of features:",
    len(feature_names)
)

print(
    "\nExample customer's top SHAP factors:"
)

example_values = shap_values.values[0]

explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": example_values
})

explanation["Absolute SHAP"] = (
    explanation["SHAP Value"].abs()
)

explanation = explanation.sort_values(
    "Absolute SHAP",
    ascending=False
)

print(
    explanation.head(10).to_string(
        index=False
    )
)
# ============================================================
# SAVE SHAP RESULTS
# ============================================================

explanation.to_csv(
    "data/processed/shap_results.csv",
    index=False
)

print("\nSHAP results saved to:")
print("data/processed/shap_results.csv")