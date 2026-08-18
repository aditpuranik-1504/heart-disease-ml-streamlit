"""
Heart Disease Risk Prediction - ML Model Comparison App
ML Assignment 2 | BITS Pilani WILP M.Tech (AIML/DSE)

Dataset : Framingham Heart Study (10-year CHD risk, binary classification)
Models  : Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

# ----------------------------------------------------------------- constants
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET_COL = "TenYearCHD"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

st.set_page_config(page_title="Heart Disease ML Comparison",
                   page_icon="🫀", layout="wide")


# ----------------------------------------------------------------- helpers
@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_resource
def load_medians():
    return joblib.load(os.path.join(MODEL_DIR, "train_medians.pkl"))


@st.cache_data
def load_default_test_data():
    return pd.read_csv(os.path.join(HERE, "test_data.csv"))


def prepare_data(df: pd.DataFrame):
    """Validate an uploaded CSV, fill missing values with training medians."""
    medians = load_medians()
    expected = list(medians.index)

    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        return None, None, f"Uploaded CSV is missing columns: {missing_cols}"
    if TARGET_COL not in df.columns:
        return None, None, (f"Uploaded CSV must include the '{TARGET_COL}' "
                            "column (true labels) so metrics can be computed.")

    X = df[expected].fillna(medians)
    y = df[TARGET_COL].astype(int)
    return X, y, None


# ----------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Controls")

model_name = st.sidebar.selectbox("Select a classification model",
                                  list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload test data (CSV)")
uploaded = st.sidebar.file_uploader(
    "Upload a CSV with the same columns as the Framingham dataset "
    f"(15 features + '{TARGET_COL}' label).",
    type=["csv"],
)
st.sidebar.caption("No file? The bundled hold-out test set "
                   "(test_data.csv, 848 rows) is used automatically.")

# ----------------------------------------------------------------- main page
st.title("🫀 Heart Disease Risk Prediction — Model Comparison")
st.markdown(
    "Predicting **10-year risk of coronary heart disease (CHD)** using the "
    "**Framingham Heart Study** dataset (4,240 patients, 15 features). "
    "Choose a model on the left, optionally upload your own test CSV, and "
    "explore the evaluation metrics below."
)

# Load data (uploaded file wins over the bundled test set)
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        source = f"uploaded file · {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    raw = load_default_test_data()
    source = "bundled hold-out test set (test_data.csv)"

X, y, err = prepare_data(raw)
if err:
    st.error(err)
    st.stop()

st.info(f"Evaluating **{model_name}** on **{len(X)} rows** — {source}")

# ----------------------------------------------------------------- predict
model = load_model(model_name)
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)

# ----------------------------------------------------------------- metrics
st.subheader("📊 Evaluation Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{acc:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{prec:.4f}")
c4.metric("Recall", f"{rec:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

# ----------------------------------------------------------------- plots
left, right = st.columns(2)

with left:
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No CHD (0)", "CHD (1)"],
                yticklabels=["No CHD (0)", "CHD (1)"], ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    st.pyplot(fig, width="content")

with right:
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    ax2.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", fontsize=8)
    st.pyplot(fig2, width="content")

# ----------------------------------------------------------------- report
st.subheader("📄 Classification Report")
report = classification_report(
    y, y_pred, target_names=["No CHD (0)", "CHD (1)"],
    output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(4),
             width="stretch")

# ----------------------------------------------------------------- compare all
st.markdown("---")
if st.checkbox("🏁 Compare ALL 5 models on this test data"):
    rows = []
    for name in MODEL_FILES:
        m = load_model(name)
        p = m.predict(X)
        pr = m.predict_proba(X)[:, 1]
        rows.append({
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y, p), 4),
            "AUC": round(roc_auc_score(y, pr), 4),
            "Precision": round(precision_score(y, p, zero_division=0), 4),
            "Recall": round(recall_score(y, p, zero_division=0), 4),
            "F1": round(f1_score(y, p, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y, p), 4),
        })
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp, width="stretch")
    st.caption("Highest AUC and MCC indicate the best overall model "
               "on this imbalanced dataset.")

st.markdown("---")
st.caption("ML Assignment 2 · BITS Pilani WILP · M.Tech (AIML/DSE) · "
           "Framingham Heart Study dataset")
