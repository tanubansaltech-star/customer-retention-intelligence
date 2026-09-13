import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "models/best_model.pkl"
)

print("Customer Retention Intelligence System")
print("--------------------------------------")


# ============================================================
# EXAMPLE CUSTOMER
# ============================================================

customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0,
    "LifecycleSegment": "New",
    "MonthlyChargeSegment": "High",
    "TotalServices": 4,
    "HasTechSupport": 0,
    "HasOnlineSecurity": 0,
    "AutoPayment": 0,
    "LongTermContract": 0,
    "HighValueCustomer": 1
}])


# ============================================================
# LOAD CLEANED DATA
# ============================================================

cleaned_data = pd.read_csv(
    "data/processed/customer_churn_cleaned.csv"
)


# ============================================================
# CUSTOMER VALUE SCORE
# ============================================================

cleaned_data["RawCustomerValue"] = (
    cleaned_data["MonthlyCharges"]
    *
    (1 + cleaned_data["tenure"] / 12)
)


min_value = (
    cleaned_data["RawCustomerValue"].min()
)

max_value = (
    cleaned_data["RawCustomerValue"].max()
)


raw_customer_value = (
    customer["MonthlyCharges"].iloc[0]
    *
    (1 + customer["tenure"].iloc[0] / 12)
)


customer_value_score = (
    (raw_customer_value - min_value)
    /
    (max_value - min_value)
    * 100
)


customer["CustomerValueScore"] = (
    customer_value_score
)


# ============================================================
# CHURN PREDICTION
# ============================================================

probability = (
    model.predict_proba(customer)[0][1]
)


churn_percentage = (
    probability * 100
)


print(
    f"\nChurn Probability: "
    f"{churn_percentage:.2f}%"
)


# ============================================================
# RISK LEVEL
# ============================================================

if churn_percentage >= 70:

    risk = "HIGH"

elif churn_percentage >= 40:

    risk = "MEDIUM"

else:

    risk = "LOW"


print(
    f"Risk Level: {risk}"
)


# ============================================================
# REVENUE AT RISK
# ============================================================

monthly_charges = (
    customer["MonthlyCharges"].iloc[0]
)


revenue_at_risk = (
    monthly_charges * probability
)


print(
    f"Monthly Revenue at Risk: "
    f"₹{revenue_at_risk:.2f}"
)


# ============================================================
# REVENUE RISK SCORE
# ============================================================

max_monthly_charge = (
    cleaned_data["MonthlyCharges"].max()
)


revenue_risk_score = (
    revenue_at_risk
    /
    max_monthly_charge
    * 100
)


revenue_risk_score = min(
    revenue_risk_score,
    100
)


# ============================================================
# RETENTION PRIORITY SCORE
# ============================================================

retention_priority_score = (

    (churn_percentage * 0.50)

    +

    (customer_value_score * 0.30)

    +

    (revenue_risk_score * 0.20)

)


retention_priority_score = min(
    retention_priority_score,
    100
)


print(
    f"Customer Value Score: "
    f"{customer_value_score:.2f}/100"
)


print(
    f"Revenue Risk Score: "
    f"{revenue_risk_score:.2f}/100"
)


print(
    f"Retention Priority Score: "
    f"{retention_priority_score:.2f}/100"
)


# ============================================================
# RETENTION PRIORITY LEVEL
# ============================================================

if retention_priority_score >= 70:

    priority = "CRITICAL"

elif retention_priority_score >= 50:

    priority = "HIGH"

elif retention_priority_score >= 30:

    priority = "MEDIUM"

else:

    priority = "LOW"


print(
    f"Retention Priority: {priority}"
)


# ============================================================
# RETENTION ACTION
# ============================================================

if priority == "CRITICAL":

    action = (
        "Immediate personalized retention intervention recommended."
    )

elif priority == "HIGH":

    action = (
        "Prioritize this customer for a targeted retention campaign."
    )

elif priority == "MEDIUM":

    action = (
        "Monitor the customer and consider a targeted engagement offer."
    )

else:

    action = (
        "Continue normal customer engagement and monitoring."
    )


print(
    f"Recommended Action: {action}"
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n===== ADVANCED RETENTION INTELLIGENCE COMPLETE ====="
)