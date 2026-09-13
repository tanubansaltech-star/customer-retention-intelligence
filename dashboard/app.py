import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from database_loader import load_customers_from_database

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM DARK UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(99, 102, 241, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 20%,
                rgba(14, 165, 233, 0.10),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #050914 0%,
                #08111f 45%,
                #0b1220 100%
            );
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }

    .stDivider {
        margin: 42px 0 !important;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }

    p, label, span {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }

    div[data-testid="stText"] {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        color: #ffffff !important;
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #cbd5e1 !important;
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 6px;
    }

    .header-line {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-top: 8px;
        margin-bottom: 28px;
        color: #94a3b8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(20, 30, 48, 0.98),
                rgba(8, 15, 28, 0.98)
            );
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 24px 26px;
        min-height: 135px;
        box-shadow:
            0 14px 38px rgba(0, 0, 0, 0.32),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow:
            0 18px 45px rgba(0, 0, 0, 0.35);
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] div,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] span {
        color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 30px !important;
        font-weight: 800 !important;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #060b16 0%,
                #091221 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* ========================================================
       DATA TABLE
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stDataFrame"] * {
        opacity: 1 !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* ========================================================
       WIDGET LABELS
       ======================================================== */

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       PLOTLY
       ======================================================== */

    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .axis-title,
    .js-plotly-plot .legendtext {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    .js-plotly-plot .g-xtitle text,
    .js-plotly-plot .g-ytitle text {
        fill: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.35);
        background: rgba(30, 41, 59, 0.75);
        color: #ffffff;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: rgba(99, 102, 241, 0.75);
        background: rgba(51, 65, 85, 0.9);
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(15, 23, 42, 0.85);
        border-color: rgba(148, 163, 184, 0.22);
    }

    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
    background:
        linear-gradient(
            145deg,
            rgba(20, 30, 48, 0.95),
            rgba(8, 15, 28, 0.95)
        );
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 22px;
    min-height: 230px;
    height: 100%;
    margin-bottom: 16px;
    box-sizing: border-box;
    box-shadow: 0 12px 30px rgba(0,0,0,0.22);
}
div[data-testid="column"] {
    display: flex;
}

div[data-testid="column"] > div {
    width: 100%;
}

    .recommendation-card {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.96),
                rgba(15, 23, 42, 0.96)
            );
        border-left: 4px solid #6366f1;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    /* ========================================================
       RISK BADGES
       ======================================================== */

    .risk-high {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5 !important;
        font-weight: 800;
    }

    .risk-medium {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(245, 158, 11, 0.16);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fcd34d !important;
        font-weight: 800;
    }

    .risk-low {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #86efac !important;
        font-weight: 800;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #64748b !important;
        font-size: 13px;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("models/best_model.pkl")
except Exception as e:
    st.error("Best model could not be loaded.")
    st.exception(e)
    st.stop()


# ============================================================
# LOAD DATA FROM SQL DATABASE
# ============================================================

try:
    df = load_customers_from_database()
except Exception as e:
    st.error("Customer database could not be loaded.")
    st.exception(e)
    st.stop()


df["Churn"] = df["Churn"].astype(int)


# ============================================================
# GENERATE CHURN PROBABILITY
# ============================================================

model_features = df.drop(
    columns=["Churn", "customerID"]
)

df["ChurnProbability"] = (
    model.predict_proba(model_features)[:, 1]
)

df["ChurnProbabilityPercent"] = (
    df["ChurnProbability"] * 100
)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    percentage = probability * 100

    if percentage >= 70:
        return "HIGH"

    elif percentage >= 40:
        return "MEDIUM"

    else:
        return "LOW"


df["RiskLevel"] = (
    df["ChurnProbability"]
    .apply(get_risk_level)
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🧠 Customer Retention Intelligence
    </div>

    <div class="subtitle">
        AI-powered customer churn prediction, risk intelligence
        and retention analytics
    </div>

    <div class="header-line">
        <span>AI-Powered</span>
        <span>•</span>
        <span>Churn Prediction</span>
        <span>•</span>
        <span>Retention Intelligence</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 Customer Retention")
    st.markdown("### Intelligence System")

    st.divider()

    st.markdown("**System Status**")

    st.success("Model Loaded")
    st.success("Data Loaded")
    st.success("Prediction Engine Ready")

    st.divider()

    st.markdown("### Project Pipeline")

    st.markdown(
        """
        **01** Data Validation  
        **02** Data Cleaning  
        **03** EDA  
        **04** Feature Engineering  
        **05** Machine Learning  
        **06** Risk Prediction  
        **07** Explainable AI  
        **08** Retention Intelligence
        """
    )

    st.divider()

    st.caption(
        "Customer Retention Intelligence System"
    )

    st.caption(
        "Built with Python • ML • SHAP • Streamlit"
    )
    # ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.header("📊 Business Overview")

total_customers = len(df)

high_risk_customers = len(
    df[df["RiskLevel"] == "HIGH"]
)

medium_risk_customers = len(
    df[df["RiskLevel"] == "MEDIUM"]
)

low_risk_customers = len(
    df[df["RiskLevel"] == "LOW"]
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )

with kpi2:
    st.metric(
        "🔴 High Risk",
        f"{high_risk_customers:,}"
    )

with kpi3:
    st.metric(
        "🟠 Medium Risk",
        f"{medium_risk_customers:,}"
    )

with kpi4:
    st.metric(
        "🟢 Low Risk",
        f"{low_risk_customers:,}"
    )


st.divider()


# ============================================================
# REVENUE AT RISK
# ============================================================

st.header("💰 Revenue at Risk")

# Probability-weighted revenue risk
df["RevenueAtRisk"] = (
    df["MonthlyCharges"]
    * df["ChurnProbability"]
)

total_revenue_at_risk = (
    df["RevenueAtRisk"].sum()
)

high_risk_revenue = (
    df.loc[
        df["RiskLevel"] == "HIGH",
        "RevenueAtRisk"
    ].sum()
)

average_customer_risk = (
    df["RevenueAtRisk"].mean()
)

risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
       "💰 Revenue at Risk",
        f"₹{total_revenue_at_risk:,.2f}"
    )


with risk_col2:

    st.metric(
        "🔴 High-Risk Revenue",
        f"₹{high_risk_revenue:,.2f}"
    )


with risk_col3:

    st.metric(
    "👤 Avg. Risk / Customer",
    f"₹{average_customer_risk:,.2f}"
)


st.caption(
    "Revenue at Risk is estimated using each customer's "
    "monthly charges weighted by their predicted churn probability."
)

st.divider()

# ============================================================
# CUSTOMER RISK DISTRIBUTION
# ============================================================

st.header("📊 Customer Risk Distribution")

risk_counts = (
    df["RiskLevel"]
    .value_counts()
)

risk_order = [
    "HIGH",
    "MEDIUM",
    "LOW"
]

risk_values = [
    risk_counts.get(level, 0)
    for level in risk_order
]


fig_risk = go.Figure(
    go.Bar(
        x=risk_order,
        y=risk_values,
        text=risk_values,
        textposition="auto"
    )
)


fig_risk.update_layout(
    title=dict(
        text="Customers by Risk Level",
        font=dict(
            size=22,
            color="white"
        )
    ),
    xaxis_title="Risk Level",
    yaxis_title="Number of Customers",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="white",
        size=14
    ),
    margin=dict(
        l=60,
        r=30,
        t=70,
        b=60
    ),
    hoverlabel=dict(
        bgcolor="#111827",
        font_size=14,
        font_color="white"
    )
)


st.plotly_chart(
    fig_risk,
    width="stretch"
)

st.divider()


# ============================================================
# CONTRACT-WISE CHURN
# ============================================================

st.header("📈 Contract-wise Churn Analysis")


contract_churn = (
    df.groupby("Contract")["Churn"]
    .mean()
    .mul(100)
    .reset_index()
)


contract_churn.columns = [
    "Contract",
    "Churn Rate"
]


contract_churn = (
    contract_churn
    .sort_values(
        "Churn Rate",
        ascending=False
    )
)


fig_contract = go.Figure(
    go.Bar(
        x=contract_churn["Contract"],
        y=contract_churn["Churn Rate"],
        text=contract_churn["Churn Rate"].map(
            lambda x: f"{x:.1f}%"
        ),
        textposition="auto"
    )
)


fig_contract.update_layout(
    title=dict(
        text="Churn Rate by Contract Type",
        font=dict(
            size=22,
            color="white"
        )
    ),
    xaxis_title="Contract Type",
    yaxis_title="Churn Rate (%)",
    yaxis=dict(
        range=[0, 50],
        gridcolor="rgba(148,163,184,0.12)"
    ),
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="white",
        size=14
    ),
    margin=dict(
        l=60,
        r=30,
        t=70,
        b=60
    ),
    hoverlabel=dict(
        bgcolor="#111827",
        font_size=14,
        font_color="white"
    )
)


st.plotly_chart(
    fig_contract,
    width="stretch"
)

st.divider()


# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

st.header("👥 Customer Segment Analysis")


segment_data = (
    df.groupby("LifecycleSegment")
    .agg(
        Customers=("customerID", "count"),
        AverageChurn=("Churn", "mean")
    )
    .reset_index()
)


segment_data["Churn Rate"] = (
    segment_data["AverageChurn"] * 100
)


segment_data["SortOrder"] = (
    segment_data["LifecycleSegment"]
    .map({
        "New": 1,
        "Growing": 2,
        "Mature": 3,
        "Loyal": 4
    })
)


segment_data = (
    segment_data
    .sort_values("SortOrder")
)


fig_segment = go.Figure(
    go.Bar(
        x=segment_data["LifecycleSegment"],
        y=segment_data["Churn Rate"],
        text=segment_data["Churn Rate"].map(
            lambda x: f"{x:.1f}%"
        ),
        textposition="auto"
    )
)


fig_segment.update_layout(
    title=dict(
        text="Churn Rate by Customer Lifecycle",
        font=dict(
            size=22,
            color="white"
        )
    ),
    xaxis_title="Customer Segment",
    yaxis_title="Churn Rate (%)",
    yaxis=dict(
        range=[0, 50],
        gridcolor="rgba(148,163,184,0.12)"
    ),
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="white",
        size=14
    ),
    margin=dict(
        l=60,
        r=30,
        t=70,
        b=60
    ),
    hoverlabel=dict(
        bgcolor="#111827",
        font_size=14,
        font_color="white"
    )
)


st.plotly_chart(
    fig_segment,
    width="stretch"
)


st.dataframe(
    segment_data[
        [
            "LifecycleSegment",
            "Customers",
            "Churn Rate"
        ]
    ].rename(
        columns={
            "LifecycleSegment":
                "Customer Segment"
        }
    ),
    width="stretch",
    hide_index=True
)


st.divider()
# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🤖 Model Performance")

try:

    model_results = pd.read_csv(
        "data/processed/model_performance.csv"
    )

    st.dataframe(
        model_results,
        width="stretch",
        hide_index=True
    )

    fig_model = go.Figure()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for metric in metrics:

        fig_model.add_trace(
            go.Bar(
                name=metric,
                x=model_results["Model"],
                y=model_results[metric]
            )
        )


    fig_model.update_layout(
        title=dict(
            text="Model Comparison",
            font=dict(
                size=22,
                color="white"
            )
        ),
        barmode="group",
        yaxis=dict(
            range=[0, 1],
            title="Score",
            gridcolor="rgba(148,163,184,0.12)"
        ),
        xaxis=dict(
            title="Model"
        ),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            size=14
        ),
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=70
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=14,
            font_color="white"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )


    st.plotly_chart(
        fig_model,
        width="stretch"
    )


    best_model = model_results.loc[
        model_results["ROC-AUC"].idxmax()
    ]


    st.success(
        f"🏆 Best Model: "
        f"{best_model['Model']} • "
        f"ROC-AUC: "
        f"{best_model['ROC-AUC']:.4f}"
    )


except Exception:

    st.warning(
        "Model performance data is not available."
    )


st.divider()


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("🎯 Confusion Matrix")

try:

    confusion_data = pd.read_csv(
        "data/processed/confusion_matrix.csv",
        index_col=0
    )


    fig_cm = go.Figure(
        data=go.Heatmap(
            z=confusion_data.values,
            x=confusion_data.columns,
            y=confusion_data.index,
            text=confusion_data.values,
            texttemplate="%{text}",
            textfont={"size": 20},
            hoverongaps=False
        )
    )


    fig_cm.update_layout(
        title=dict(
            text="Best Model Confusion Matrix",
            font=dict(
                size=22,
                color="white"
            )
        ),
        xaxis=dict(
            title="Prediction",
            tickfont=dict(
                size=13,
                color="white"
            )
        ),
        yaxis=dict(
            title="Actual",
            tickfont=dict(
                size=13,
                color="white"
            )
        ),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            size=14
        ),
        margin=dict(
            l=80,
            r=30,
            t=70,
            b=70
        )
    )


    st.plotly_chart(
        fig_cm,
        width="stretch"
    )


    st.caption(
        "The confusion matrix shows how the best model "
        "classifies actual churn and non-churn customers."
    )


except Exception:

    st.warning(
        "Confusion matrix data is not available."
    )


st.divider()


# ============================================================
# ROC CURVE
# ============================================================

st.header("📈 ROC Curve")

try:

    roc_df = pd.read_csv(
        "data/processed/roc_curve_data.csv"
    )


    fig_roc = go.Figure()


    for model_name in roc_df["Model"].unique():

        model_data = roc_df[
            roc_df["Model"] == model_name
        ]


        auc_value = model_results.loc[
            model_results["Model"] == model_name,
            "ROC-AUC"
        ].iloc[0]


        fig_roc.add_trace(
            go.Scatter(
                x=model_data["FPR"],
                y=model_data["TPR"],
                mode="lines",
                name=(
                    f"{model_name} "
                    f"(AUC = {auc_value:.4f})"
                ),
                hovertemplate=(
                    f"{model_name}"
                    "<br>FPR: %{x:.3f}"
                    "<br>TPR: %{y:.3f}"
                    "<extra></extra>"
                )
            )
        )


    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(
                dash="dash"
            )
        )
    )


    fig_roc.update_layout(
        title=dict(
            text="ROC Curve — Model Comparison",
            font=dict(
                size=22,
                color="white"
            )
        ),
        xaxis=dict(
            title="False Positive Rate",
            gridcolor="rgba(148,163,184,0.12)"
        ),
        yaxis=dict(
            title="True Positive Rate",
            range=[0, 1],
            gridcolor="rgba(148,163,184,0.12)"
        ),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            size=14
        ),
        margin=dict(
            l=70,
            r=30,
            t=70,
            b=70
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=14,
            font_color="white"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )


    st.plotly_chart(
        fig_roc,
        width="stretch"
    )


    st.caption(
        "Higher ROC-AUC indicates better separation "
        "between churn and non-churn customers."
    )


except Exception:

    st.warning(
        "ROC curve data is not available."
    )


st.divider()


# ============================================================
# SHAP EXPLAINABLE AI
# ============================================================

st.header("🧠 SHAP Explainable AI")

try:

    shap_df = pd.read_csv(
        "data/processed/shap_results.csv"
    )


    top_shap = (
        shap_df
        .head(10)
        .copy()
    )


    top_shap["Feature"] = (
        top_shap["Feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
    )


    top_shap["Impact"] = (
        top_shap["SHAP Value"]
        .apply(
            lambda x:
            "🔴 Increases Risk"
            if x > 0
            else
            "🟢 Decreases Risk"
        )
    )


    fig_shap = go.Figure(
        go.Bar(
            x=top_shap["SHAP Value"],
            y=top_shap["Feature"],
            orientation="h",
            text=top_shap["SHAP Value"].round(3),
            textposition="auto"
        )
    )


    fig_shap.update_layout(
        title=dict(
            text="Top Factors Influencing Churn Risk",
            font=dict(
                size=22,
                color="white"
            )
        ),
        xaxis=dict(
            title="SHAP Impact",
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=True,
            zerolinecolor="rgba(148,163,184,0.25)"
        ),
        yaxis=dict(
            title="Feature"
        ),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            size=14
        ),
        margin=dict(
            l=90,
            r=40,
            t=70,
            b=70
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=14,
            font_color="white"
        )
    )


    st.plotly_chart(
        fig_shap,
        width="stretch"
    )


    st.dataframe(
        top_shap[
            [
                "Feature",
                "SHAP Value",
                "Impact"
            ]
        ],
        width="stretch",
        hide_index=True
    )


    st.caption(
        "Positive SHAP values increase the model's "
        "predicted churn risk, while negative values "
        "decrease it."
    )


except Exception:

    st.warning(
        "SHAP results are not available."
    )


st.divider()
# ============================================================
# ADVANCED RETENTION INTELLIGENCE
# ============================================================

st.header("🎯 Advanced Retention Intelligence")

df["RevenueAtRisk"] = (
    df["MonthlyCharges"]
    * df["ChurnProbability"]
)

max_monthly_charge = df["MonthlyCharges"].max()

df["RevenueRiskScore"] = (
    df["RevenueAtRisk"]
    / max_monthly_charge
    * 100
)

df["RevenueRiskScore"] = (
    df["RevenueRiskScore"]
    .clip(0, 100)
)


df["RetentionPriorityScore"] = (
    df["ChurnProbabilityPercent"] * 0.50
    + df["CustomerValueScore"] * 0.30
    + df["RevenueRiskScore"] * 0.20
)

df["RetentionPriorityScore"] = (
    df["RetentionPriorityScore"]
    .clip(0, 100)
)


def get_priority_level(score):

    if score >= 70:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 30:
        return "MEDIUM"

    else:
        return "LOW"


df["RetentionPriority"] = (
    df["RetentionPriorityScore"]
    .apply(get_priority_level)
)


critical_count = len(
    df[df["RetentionPriority"] == "CRITICAL"]
)

high_priority_count = len(
    df[df["RetentionPriority"] == "HIGH"]
)

medium_priority_count = len(
    df[df["RetentionPriority"] == "MEDIUM"]
)

low_priority_count = len(
    df[df["RetentionPriority"] == "LOW"]
)


priority1, priority2, priority3, priority4 = (
    st.columns(4)
)


with priority1:

    st.metric(
        "🔴 Critical",
        f"{critical_count:,}"
    )


with priority2:

    st.metric(
        "🟠 High Priority",
        f"{high_priority_count:,}"
    )


with priority3:

    st.metric(
        "🟡 Medium Priority",
        f"{medium_priority_count:,}"
    )


with priority4:

    st.metric(
        "🟢 Low Priority",
        f"{low_priority_count:,}"
    )


priority_table = (
    df[
        [
            "customerID",
            "ChurnProbabilityPercent",
            "CustomerValueScore",
            "RevenueAtRisk",
            "RetentionPriorityScore",
            "RetentionPriority"
        ]
    ]
    .sort_values(
        "RetentionPriorityScore",
        ascending=False
    )
    .head(25)
    .copy()
)


priority_table = priority_table.rename(
    columns={
        "customerID": "Customer ID",
        "ChurnProbabilityPercent":
            "Churn Probability (%)",
        "CustomerValueScore":
            "Customer Value Score",
        "RevenueAtRisk":
            "Revenue at Risk",
        "RetentionPriorityScore":
            "Priority Score",
        "RetentionPriority":
            "Priority"
    }
)


priority_table[
    "Churn Probability (%)"
] = priority_table[
    "Churn Probability (%)"
].round(2)


priority_table[
    "Customer Value Score"
] = priority_table[
    "Customer Value Score"
].round(2)


priority_table[
    "Revenue at Risk"
] = priority_table[
    "Revenue at Risk"
].round(2)


priority_table[
    "Priority Score"
] = priority_table[
    "Priority Score"
].round(2)


st.dataframe(
    priority_table,
    width="stretch",
    hide_index=True
)


st.caption(
    "Priority Score combines churn risk, customer value, "
    "and estimated revenue at risk to identify customers "
    "that should receive retention attention first."
)
#

st.divider()
#=========================================================
# HIGH-RISK CUSTOMER PRIORITIZATION
#============================================================

st.header("🔥 High-Risk Customer Prioritization")

high_risk = (
    df[df["RiskLevel"] == "HIGH"]
    .copy()
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    search_customer = st.text_input(
        "🔎 Search Customer ID",
        placeholder="Enter Customer ID..."
    )

with col2:
    contract_filter = st.selectbox(
        "📄 Contract Type",
        ["All", "Month-to-month", "One year", "Two year"]
    )

with col3:
    internet_filter = st.selectbox(
        "🌐 Internet Service",
        ["All", "DSL", "Fiber optic", "No"]
    )

if search_customer:
    high_risk = high_risk[
        high_risk["customerID"]
        .astype(str)
        .str.contains(
            search_customer.strip(),
            case=False,
            na=False
        )
    ]
    
    


high_risk["RevenueAtRisk"] = (
    high_risk["MonthlyCharges"]
    * high_risk["ChurnProbability"]
)


high_risk["RevenueRiskScore"] = (
    high_risk["RevenueAtRisk"]
    / max_monthly_charge
    * 100
)


high_risk["RevenueRiskScore"] = (
    high_risk["RevenueRiskScore"]
    .clip(0, 100)
)


high_risk["RetentionPriorityScore"] = (
    high_risk["ChurnProbabilityPercent"] * 0.50
    + high_risk["CustomerValueScore"] * 0.30
    + high_risk["RevenueRiskScore"] * 0.20
)


high_risk["RetentionPriorityScore"] = (
    high_risk["RetentionPriorityScore"]
    .clip(0, 100)
)


high_risk["Priority"] = (
    high_risk["RetentionPriorityScore"]
    .apply(get_priority_level)
)
with col4:
    priority_filter = st.selectbox(
      "🎯 Priority",
        ["All", "HIGH", "MEDIUM", "LOW"]
)
if contract_filter != "All":
    high_risk = high_risk[
        high_risk["Contract"] == contract_filter
    ]


if internet_filter != "All":
    high_risk = high_risk[
        high_risk["InternetService"] == internet_filter
    ]
if priority_filter != "All":
    high_risk = high_risk[
        high_risk["Priority"] == priority_filter
    ]


high_risk_table = (
    high_risk[
        [
            "customerID",
            "tenure",
            "Contract",
            "InternetService",
            "MonthlyCharges",
            "ChurnProbabilityPercent",
            "CustomerValueScore",
            "RevenueAtRisk",
            "RetentionPriorityScore",
            "Priority"
        ]
    ]
    .sort_values(
        "RetentionPriorityScore",
        ascending=False
    )
    .head(25)
    .copy()
)


high_risk_table = (
    high_risk_table.rename(
        columns={
            "customerID": "Customer ID",
            "tenure": "Tenure",
            "Contract": "Contract",
            "InternetService":
                "Internet Service",
            "MonthlyCharges":
                "Monthly Charges",
            "ChurnProbabilityPercent":
                "Churn Probability (%)",
            "CustomerValueScore":
                "Customer Value Score",
            "RevenueAtRisk":
                "Revenue at Risk",
            "RetentionPriorityScore":
                "Priority Score",
            "Priority":
                "Priority"
        }
    )
)


high_risk_table[
    "Monthly Charges"
] = high_risk_table[
    "Monthly Charges"
].round(2)


high_risk_table[
    "Churn Probability (%)"
] = high_risk_table[
    "Churn Probability (%)"
].round(2)


high_risk_table[
    "Customer Value Score"
] = high_risk_table[
    "Customer Value Score"
].round(2)


high_risk_table[
    "Revenue at Risk"
] = high_risk_table[
    "Revenue at Risk"
].round(2)


high_risk_table[
    "Priority Score"
] = high_risk_table[
    "Priority Score"
].round(2)


st.dataframe(
    high_risk_table,
    width="stretch",
    hide_index=True,
    column_config={
    "Monthly Charges": st.column_config.NumberColumn(
        "Monthly Charges",
        format="₹%.2f"
    ),
    "Revenue at Risk": st.column_config.NumberColumn(
        "Revenue at Risk",
        format="₹%.2f"
    ),
    "Customer Value Score": st.column_config.NumberColumn(
        "Customer Value Score",
        format="%.2f"
    ),
    "Priority Score": st.column_config.NumberColumn(
        "Priority Score",
        format="%.2f"
    ),
    "Priority": st.column_config.TextColumn(
        "Priority",
        help="Customer retention priority level"
    )
}
)


st.divider()


# ============================================================
# CUSTOMER RISK PROFILE
# ============================================================

st.header("👤 Customer Risk Profile")


customer_ids = (
    df["customerID"]
    .astype(str)
    .tolist()
)


selected_customer = st.selectbox(
    "Select Customer",
    customer_ids
)


selected_row = df[
    df["customerID"].astype(str)
    == selected_customer
].iloc[0]


profile1, profile2, profile3, profile4 = st.columns(
    [1.2, 1.4, 1.2, 1.0]
)


with profile1:

    st.metric(
        "Customer ID",
        selected_customer
    )


with profile2:

    st.metric(
        "Churn Probability",
        f"{selected_row['ChurnProbabilityPercent']:.2f}%"
    )


with profile3:

    st.metric(
        "Monthly Charges",
        f"₹{selected_row['MonthlyCharges']:.2f}"
    )


with profile4:

    st.metric(
        "Tenure",
        f"{int(selected_row['tenure'])} months"
    )


risk = selected_row["RiskLevel"]




if risk == "HIGH":

    st.error("🔴 HIGH RISK CUSTOMER")

elif risk == "MEDIUM":

    st.warning("🟠 MEDIUM RISK CUSTOMER")

else:

    st.success("🟢 LOW RISK CUSTOMER")

st.divider()

# ============================================================
# CUSTOMER-SPECIFIC SHAP EXPLANATION
# ============================================================

st.header("🔍 Why Is This Customer at Risk?")
st.caption(
    "The factors below show which customer characteristics "
    "are increasing or reducing predicted churn risk."
)


customer_shap = pd.read_csv(
    "data/processed/customer_shap_results.csv"
)


selected_customer_shap = customer_shap[
    customer_shap["customerID"].astype(str)
    == selected_customer
].copy()


# Clean feature names
feature_names_map = {
    "MonthlyCharges": "Monthly Charges",
    "tenure": "Customer Tenure",
    "TotalCharges": "Total Charges",
    "TotalServices": "Number of Services",
    "LongTermContract": "Long-Term Contract",
    "HighValueCustomer": "High-Value Customer",
    "HasTechSupport": "Technical Support",
    "HasOnlineSecurity": "Online Security",
    "AutoPayment": "Automatic Payment",
}


selected_customer_shap["Feature"] = (
    selected_customer_shap["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .str.replace("_", " ", regex=False)
)


# Display top factors

for _, factor in selected_customer_shap.iterrows():

        feature = factor["Feature"]
        feature_key = feature

        feature = feature_names_map.get( 
        feature_key,
        feature
    )

        impact = factor["Impact"]
        shap_value = factor["SHAP Value"]

        if impact == "Increases Risk":
          st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-priority">
                    🔴 Increases Churn Risk
                </div>
                <div class="recommendation-text">
                    <b>{feature}</b>
                    &nbsp; | &nbsp;
                    SHAP Impact: {shap_value:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        else:

         st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-priority">
                    🟢 Reduces Churn Risk
                </div>
                <div class="recommendation-text">
                    <b>{feature}</b>
                    &nbsp; | &nbsp;
                    SHAP Impact: {shap_value:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# RETENTION RECOMMENDATIONS
# ============================================================


st.header("💡 Retention Recommendations")


recommendations = []


if selected_row["Contract"] == "Month-to-month":

    recommendations.append(
        (
            "🔴 High Priority",
            "Offer a long-term contract option "
            "with a suitable retention incentive."
        )
    )


if selected_row["TechSupport"] == "No":

    recommendations.append(
        (
            "🟠 Medium Priority",
            "Consider offering technical support "
            "or a guided support package."
        )
    )


if selected_row["OnlineSecurity"] == "No":

    recommendations.append(
        (
            "🟠 Medium Priority",
            "Promote an online security package "
            "to increase customer value."
        )
    )


if selected_row["PaymentMethod"] == "Electronic check":

    recommendations.append(
        (
            "🟠 Medium Priority",
            "Encourage automatic payment methods "
            "for a smoother billing experience."
        )
    )


if selected_row["MonthlyCharges"] >= 80:

    recommendations.append(
        (
            "🔴 High Priority",
            "Review pricing, plan value and "
            "available retention offers."
        )
    )


if selected_row["tenure"] <= 6:

    recommendations.append(
        (
            "🔴 High Priority",
            "Customer is early in the lifecycle. "
            "Use proactive onboarding and engagement."
        )
    )


if not recommendations:

    recommendations.append(
        (
            "🟢 Standard",
            "Continue normal engagement and "
            "periodic retention monitoring."
        )
    )


for priority, recommendation in recommendations:

    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="recommendation-priority">
                {priority}
            </div>
            <div class="recommendation-text">
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.header("💡 Key Business Insights")


overall_churn = (
    df["Churn"].mean() * 100
)


avg_monthly_churned = (
    df[df["Churn"] == 1]["MonthlyCharges"]
    .mean()
)


avg_monthly_retained = (
    df[df["Churn"] == 0]["MonthlyCharges"]
    .mean()
)


avg_tenure_churned = (
    df[df["Churn"] == 1]["tenure"]
    .mean()
)


avg_tenure_retained = (
    df[df["Churn"] == 0]["tenure"]
    .mean()
)


month_to_month_churn = (
    df[df["Contract"] == "Month-to-month"]["Churn"]
    .mean() * 100
)


two_year_churn = (
    df[df["Contract"] == "Two year"]["Churn"]
    .mean() * 100
)


# ============================================================
# FOUR KEY INSIGHT CARDS
# ============================================================

insight1, insight2, insight3, insight4 = st.columns(4)


with insight1:

    st.markdown(
        f"""
        <div class="info-card">
            <h3>📉 Overall Churn</h3>
            <p style="font-size:28px;">
                <strong>{overall_churn:.2f}%</strong>
            </p>
            <p>
                Overall percentage of customers
                who have churned.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with insight2:

    st.markdown(
        f"""
        <div class="info-card">
            <h3>💰 Monthly Charges</h3>
            <p>
                Churned:
                <strong>₹{avg_monthly_churned:.2f}</strong>
            </p>
            <p>
                Retained:
                <strong>₹{avg_monthly_retained:.2f}</strong>
            </p>
            <p>
                Churned customers have higher
                average monthly charges.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with insight3:

    st.markdown(
        f"""
        <div class="info-card">
            <h3>⏳ Customer Tenure</h3>
            <p>
                Churned:
                <strong>{avg_tenure_churned:.1f} months</strong>
            </p>
            <p>
                Retained:
                <strong>{avg_tenure_retained:.1f} months</strong>
            </p>
            <p>
                Early lifecycle customers need
                stronger engagement.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with insight4:

    st.markdown(
        f"""
        <div class="info-card">
            <h3>📄 Contract Risk</h3>
            <p>
                Month-to-month:
                <strong>{month_to_month_churn:.1f}%</strong>
            </p>
            <p>
                Two-year:
                <strong>{two_year_churn:.1f}%</strong>
            </p>
            <p>
                Long-term contracts show
                significantly lower churn.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()
# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Retention Intelligence System
        • Machine Learning • Explainable AI • Business Analytics
    </div>
    """,
    unsafe_allow_html=True
)