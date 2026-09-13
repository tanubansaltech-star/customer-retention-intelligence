import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/best_model.pkl")

print("Customer Retention Recommendation Engine")
print("----------------------------------------")


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
    "HighValueCustomer": 1,
    "CustomerValueScore": 12.34
}])


# ============================================================
# CHURN PREDICTION
# ============================================================

probability = model.predict_proba(customer)[0][1]

churn_percentage = probability * 100


print(f"Churn Probability: {churn_percentage:.2f}%")


# ============================================================
# RISK LEVEL
# ============================================================

if churn_percentage >= 70:

    risk = "HIGH"

elif churn_percentage >= 40:

    risk = "MEDIUM"

else:

    risk = "LOW"


print(f"Risk Level: {risk}")


# ============================================================
# CUSTOMER VALUE
# ============================================================

customer_value = customer["CustomerValueScore"].iloc[0]

print(
    f"Customer Value Score: {customer_value:.2f}/100"
)


# ============================================================
# REVENUE AT RISK
# ============================================================

monthly_charges = customer["MonthlyCharges"].iloc[0]

revenue_at_risk = (
    monthly_charges * probability
)

print(
    f"Monthly Revenue at Risk: ₹{revenue_at_risk:.2f}"
)


# ============================================================
# RETENTION PRIORITY SCORE
# ============================================================

retention_priority_score = (
    (probability * 70)
    +
    (customer_value / 100 * 30)
)

retention_priority_score = min(
    retention_priority_score,
    100
)


print(
    f"Retention Priority Score: "
    f"{retention_priority_score:.2f}/100"
)


# ============================================================
# RETENTION PRIORITY
# ============================================================

if retention_priority_score >= 70:

    retention_priority = "HIGH"

elif retention_priority_score >= 40:

    retention_priority = "MEDIUM"

else:

    retention_priority = "LOW"


print(
    f"Retention Priority: {retention_priority}"
)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

recommendations = []


# Contract recommendation
if customer["Contract"].iloc[0] == "Month-to-month":

    recommendations.append({
        "priority": "HIGH",
        "action":
        "Offer a long-term contract with an attractive retention benefit."
    })


# Technical support recommendation
if customer["TechSupport"].iloc[0] == "No":

    recommendations.append({
        "priority": "MEDIUM",
        "action":
        "Offer technical support assistance."
    })


# Security recommendation
if customer["OnlineSecurity"].iloc[0] == "No":

    recommendations.append({
        "priority": "MEDIUM",
        "action":
        "Consider offering an online security add-on."
    })


# Payment recommendation
if customer["PaymentMethod"].iloc[0] == "Electronic check":

    recommendations.append({
        "priority": "MEDIUM",
        "action":
        "Encourage automatic payment methods for easier billing."
    })


# Monthly charge recommendation
if customer["MonthlyCharges"].iloc[0] >= 80:

    recommendations.append({
        "priority": "HIGH",
        "action":
        "Review the customer's monthly charges and consider a personalized offer."
    })


# New customer recommendation
if customer["tenure"].iloc[0] <= 6:

    recommendations.append({
        "priority": "HIGH",
        "action":
        "Provide an early-stage customer engagement offer."
    })


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

print("\n===== RETENTION RECOMMENDATIONS =====")


if recommendations:

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{number}. "
            f"[{recommendation['priority']}] "
            f"{recommendation['action']}"
        )

else:

    print(
        "No specific retention action is required."
    )


# ============================================================
# FINAL RETENTION ACTION
# ============================================================

if retention_priority == "HIGH":

    final_action = (
        "Prioritize this customer for a targeted "
        "retention campaign."
    )

elif retention_priority == "MEDIUM":

    final_action = (
        "Monitor this customer and consider "
        "a personalized retention offer."
    )

else:

    final_action = (
        "Continue regular customer engagement."
    )


print("\n===== FINAL RETENTION ACTION =====")

print(final_action)