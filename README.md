# Customer Retention Intelligence System

An end-to-end machine learning system designed to identify customers at risk of churn, understand the reasons behind customer loss, estimate revenue at risk, and prioritize retention actions.

## 🎯 Project Objective

Customer churn is a major business problem. This project uses customer data, machine learning, explainable AI, and business intelligence to help organizations:

- Predict customer churn probability
- Identify high-risk customers
- Understand important churn drivers
- Estimate revenue at risk
- Calculate customer value
- Prioritize retention efforts
- Recommend suitable retention actions

## 🚀 Key Features

- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Customer lifecycle segmentation
- Customer value scoring
- Multiple machine learning models
- Churn probability prediction
- Risk-level classification
- Revenue-at-risk analysis
- Retention priority scoring
- Explainable AI using SHAP
- Retention recommendation engine
- Interactive Streamlit dashboard
- SQLite database integration
- High-risk customer prioritization
- Git/GitHub version control

## 🧠 Machine Learning Models

The project compares multiple classification models:

- Logistic Regression
- Random Forest
- XGBoost

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Best Model

**Logistic Regression**

- ROC-AUC: **0.8365**
- Accuracy: **79.39%**
- Precision: **63.38%**
- Recall: **53.21%**
- F1 Score: **57.85%**

## 🔍 Explainable AI

SHAP (SHapley Additive Explanations) is used to understand which customer characteristics have the strongest influence on churn predictions.

This makes the system more useful for business decision-making because it provides not only a prediction, but also an explanation.

## 💰 Revenue Risk & Retention Intelligence

The system goes beyond basic churn prediction by combining:

- Churn probability
- Customer value
- Monthly charges
- Tenure
- Contract information
- Service usage
- Payment behavior

This helps prioritize customers who represent a higher potential business risk.

## 📊 Dashboard

The interactive Streamlit dashboard provides:

- Business KPI cards
- Churn distribution
- Contract-wise churn analysis
- Customer segmentation
- Model performance
- Confusion matrix
- ROC curve
- SHAP analysis
- High-risk customer table
- Customer risk profile
- Retention recommendations
- Advanced retention intelligence

## 🗄️ Database

The project uses **SQLite** to store customer data and demonstrate database integration.

The dashboard retrieves customer information through a Python database loader rather than relying only on direct CSV loading.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting |
| SHAP | Explainable AI |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualization |
| Streamlit | Dashboard |
| SQLite | Database |
| Git | Version control |
| GitHub | Project hosting |

## 📁 Project Structure

```text
customer-retention-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── 01_data_check.py
│
├── src/
│   ├── data_processing.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── recommendations.py
│   ├── database.py
│   └── database_loader.py
│
├── sql/
│   └── schema.sql
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md```