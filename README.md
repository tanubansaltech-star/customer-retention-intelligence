# 🧠 Customer Retention Intelligence System

AI-powered customer churn prediction, risk intelligence and retention analytics platform.

## 📌 Overview

The Customer Retention Intelligence System is an end-to-end machine learning project designed to help businesses identify customers who are likely to churn and prioritize them for targeted retention actions.

The system combines data processing, exploratory analysis, feature engineering, machine learning, explainable AI, customer segmentation and retention intelligence into a single interactive Streamlit dashboard.

## 🎯 Business Problem

Customer churn can lead to significant recurring revenue loss.

This system helps answer:

- Which customers are most likely to churn?
- What factors are driving their churn risk?
- How much monthly revenue is potentially at risk?
- Which customers should the business prioritize?
- What retention action could be considered for each customer?

## 🚀 Key Features

### 🔹 Customer Churn Prediction
Predicts the probability that a customer will churn using machine learning models.

### 🔹 Risk Classification
Customers are classified into:

- 🔴 HIGH Risk
- 🟠 MEDIUM Risk
- 🟢 LOW Risk

### 🔹 Revenue at Risk
Estimates potential monthly revenue exposure using predicted churn probability and monthly customer charges.

### 🔹 Customer Value Score
Calculates a normalized customer value score based on customer tenure and monthly charges.

### 🔹 Retention Priority
Combines:

- Churn probability
- Customer value
- Revenue risk

to calculate a retention priority score.

### 🔹 Explainable AI
Uses SHAP to explain which customer characteristics increase or reduce predicted churn risk.

### 🔹 Retention Recommendations
Generates customer-focused retention recommendations based on risk and customer characteristics.

### 🔹 Interactive Dashboard
Built with Streamlit and includes:

- Business Overview
- Risk Distribution
- Contract Churn Analysis
- Customer Segmentation
- Revenue at Risk
- Model Performance
- Confusion Matrix
- ROC Curve
- SHAP Explainability
- High-Risk Customer Prioritization
- Customer Risk Profile
- Retention Recommendations

## 🔄 Project Pipeline

```text
Customer Data
      ↓
Data Validation
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Churn Probability
      ↓
Risk Classification
      ↓
Revenue at Risk
      ↓
Explainable AI
      ↓
Retention Priority
      ↓
Retention Recommendations
      ↓
Interactive Dashboard