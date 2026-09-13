import pandas as pd

input_path = "data/processed/customer_churn_cleaned.csv"

df = pd.read_csv(input_path)

print("Cleaned dataset loaded!")
print("Original shape:", df.shape)


# ============================================================
# TARGET ENCODING
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# CUSTOMER LIFECYCLE SEGMENT
# ============================================================

def lifecycle_segment(tenure):

    if tenure <= 6:
        return "New"

    elif tenure <= 24:
        return "Growing"

    elif tenure <= 48:
        return "Mature"

    else:
        return "Loyal"


df["LifecycleSegment"] = (
    df["tenure"]
    .apply(lifecycle_segment)
)


# ============================================================
# MONTHLY CHARGE SEGMENT
# ============================================================

def charge_segment(charge):

    if charge < 40:
        return "Low"

    elif charge < 80:
        return "Medium"

    else:
        return "High"


df["MonthlyChargeSegment"] = (
    df["MonthlyCharges"]
    .apply(charge_segment)
)


# ============================================================
# TOTAL SERVICES
# ============================================================

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


df["TotalServices"] = 0


for column in service_columns:

    df["TotalServices"] += (
        df[column] == "Yes"
    ).astype(int)


# ============================================================
# SUPPORT FEATURES
# ============================================================

df["HasTechSupport"] = (
    df["TechSupport"] == "Yes"
).astype(int)


df["HasOnlineSecurity"] = (
    df["OnlineSecurity"] == "Yes"
).astype(int)


# ============================================================
# PAYMENT FEATURES
# ============================================================

df["AutoPayment"] = (
    df["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
).astype(int)


# ============================================================
# CONTRACT FEATURES
# ============================================================

df["LongTermContract"] = (
    df["Contract"].isin([
        "One year",
        "Two year"
    ])
).astype(int)


# ============================================================
# HIGH VALUE CUSTOMER
# ============================================================

df["HighValueCustomer"] = (
    (df["MonthlyCharges"] >= 80) &
    (df["tenure"] >= 12)
).astype(int)


# ============================================================
# RETENTION PRIORITY FEATURES
# ============================================================

# Customer value score
df["CustomerValueScore"] = (
    df["MonthlyCharges"] *
    (1 + df["tenure"] / 12)
)


# Normalize customer value between 0 and 100
min_value = df["CustomerValueScore"].min()
max_value = df["CustomerValueScore"].max()


df["CustomerValueScore"] = (
    (df["CustomerValueScore"] - min_value)
    /
    (max_value - min_value)
    * 100
)


# ============================================================
# RETENTION PRIORITY SCORE
# ============================================================

# At this stage churn probability is not available yet.
# Therefore we create the customer-value component here.
#
# Final priority score will be calculated later in
# the prediction / dashboard layer using:
#
# Churn Probability + Customer Value


# ============================================================
# SAVE FEATURE-ENGINEERED DATA
# ============================================================

output_path = (
    "data/processed/customer_churn_features.csv"
)


df.to_csv(
    output_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\n===== FEATURE ENGINEERING COMPLETE =====")

print("New shape:", df.shape)


print("\nNew features:")

print([
    "LifecycleSegment",
    "MonthlyChargeSegment",
    "TotalServices",
    "HasTechSupport",
    "HasOnlineSecurity",
    "AutoPayment",
    "LongTermContract",
    "HighValueCustomer",
    "CustomerValueScore"
])


print("\nCustomer Value Score created.")

print(
    "Range:",
    round(df["CustomerValueScore"].min(), 2),
    "to",
    round(df["CustomerValueScore"].max(), 2)
)


print("\nSaved to:")

print(output_path)