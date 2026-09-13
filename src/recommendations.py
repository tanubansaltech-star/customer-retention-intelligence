import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

print("Customer Retention Recommendation Engine")
print("----------------------------------------")

# Example customer
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

# Predict churn probability
probability = model.predict_proba(customer)[0][1]
churn_percentage = probability * 100

print(f"Churn Probability: {churn_percentage:.2f}%")

# Determine risk level
if churn_percentage >= 70:
    risk = "HIGH"
elif churn_percentage >= 40:
    risk = "MEDIUM"
else:
    risk = "LOW"

print(f"Risk Level: {risk}")

# Recommendation engine
recommendations = []
priorities = []
if customer["Contract"].iloc[0] == "Month-to-month":
    recommendations.append(
        "Offer a long-term contract with an attractive retention benefit."
    )
    priorities.append("HIGH")

if customer["TechSupport"].iloc[0] == "No":
    recommendations.append(
        "Offer technical support assistance."
    )
    priorities.append("MEDIUM")

if customer["OnlineSecurity"].iloc[0] == "No":
    recommendations.append(
        "Consider offering an online security add-on."
    )
    priorities.append("MEDIUM")
if customer["PaymentMethod"].iloc[0] == "Electronic check":
    recommendations.append(
        "Encourage automatic payment methods for easier billing."
    )
    priorities.append("MEDIUM")

if customer["MonthlyCharges"].iloc[0] >= 80:
    recommendations.append(
        "Review the customer's monthly charges and consider a personalized offer."
    )
    priorities.append("HIGH")

if customer["tenure"].iloc[0] <= 6:
    recommendations.append(
        "Provide an early-stage customer engagement offer."
    )
    priorities.append("HIGH")


for number, (recommendation, priority) in enumerate(
    zip(recommendations, priorities),
    start=1
):
    print(
        f"{number}. [{priority}] {recommendation}"
    )
