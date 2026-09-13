import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from xgboost import XGBClassifier


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/customer_churn_features.csv"
)

print("Dataset loaded!")
print("Shape:", df.shape)


# ============================================================
# TARGET + FEATURES
# ============================================================

X = df.drop(
    columns=["Churn", "customerID"]
)

y = df["Churn"]


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# PREPROCESSOR
# ============================================================

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


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []

trained_pipelines = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc
    })


    trained_pipelines[
        model_name
    ] = pipeline


# ============================================================
# MODEL PERFORMANCE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL PERFORMANCE
# ============================================================

results_df.to_csv(
    "data/processed/model_performance.csv",
    index=False
)


print(
    "\nModel performance saved to:"
)

print(
    "data/processed/model_performance.csv"
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_row = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]


best_model_name = (
    best_model_row["Model"]
)


best_pipeline = (
    trained_pipelines[
        best_model_name
    ]
)


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best Model:",
    best_model_name
)

print(
    f"ROC-AUC: "
    f"{best_model_row['ROC-AUC']:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_pipeline,
    "models/best_model.pkl"
)


print(
    "\nBest model saved to:"
)

print(
    "models/best_model.pkl"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

y_pred = best_pipeline.predict(
    X_test
)


cm = confusion_matrix(
    y_test,
    y_pred
)


cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual No",
        "Actual Yes"
    ],

    columns=[
        "Predicted No",
        "Predicted Yes"
    ]
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm_df)


cm_df.to_csv(
    "data/processed/confusion_matrix.csv"
)


print(
    "\nConfusion matrix saved to:"
)

print(
    "data/processed/confusion_matrix.csv"
)


# ============================================================
# ROC CURVE DATA
# ============================================================

roc_data = []


for model_name, pipeline in (
    trained_pipelines.items()
):

    probabilities = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )


    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )


    for false_positive_rate, true_positive_rate in zip(
        fpr,
        tpr
    ):

        roc_data.append({

            "Model":
                model_name,

            "FPR":
                false_positive_rate,

            "TPR":
                true_positive_rate
        })


roc_df = pd.DataFrame(
    roc_data
)


roc_df.to_csv(
    "data/processed/roc_curve_data.csv",
    index=False
)


print(
    "\nROC curve data saved to:"
)

print(
    "data/processed/roc_curve_data.csv"
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)