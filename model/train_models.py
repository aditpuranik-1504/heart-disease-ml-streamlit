"""
Train 5 classification models on the Framingham Heart Disease dataset
and save them (plus the preprocessing info) for the Streamlit app.

Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC

Run from the project root:  python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TARGET_COL = "TenYearCHD"

# ---------------------------------------------------------------- load data
df = pd.read_csv(os.path.join(ROOT, "framingham.csv"))
print(f"Raw dataset shape: {df.shape}")

# Fill missing values with the median of each column (computed on full data,
# then re-computed on train only for the saved imputer values below)
feature_cols = [c for c in df.columns if c != TARGET_COL]

X = df[feature_cols]
y = df[TARGET_COL]

# Stratified split so both sets keep the same % of CHD cases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Impute missing values using TRAIN medians only (no data leakage)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"CHD rate  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

# ---------------------------------------------------------------- models
# Scale-sensitive models (LogReg, kNN) get a StandardScaler inside a Pipeline,
# so each saved .pkl file is fully self-contained.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, class_weight="balanced",
        random_state=RANDOM_STATE),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
}

# ---------------------------------------------------------------- train + eval
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(model, os.path.join(HERE, fname))
    print(f"{name:26s} -> {results[name]}")

# ---------------------------------------------------------------- save artifacts
# Imputation medians (the app uses these to fill missing values in uploads)
joblib.dump(train_medians, os.path.join(HERE, "train_medians.pkl"))

# Metrics table for the README / app
with open(os.path.join(HERE, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# Held-out test data for the Streamlit upload feature
test_out = X_test.copy()
test_out[TARGET_COL] = y_test.values
test_out.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with {len(test_out)} rows")
print("Done.")
