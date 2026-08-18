# Heart Disease Risk Prediction — ML Model Comparison

**ML Assignment 2 · BITS Pilani WILP · M.Tech (AIML/DSE) · Machine Learning**

## a. Problem Statement

Coronary heart disease (CHD) is one of the leading causes of death worldwide. Doctors would like an early warning: given a patient's routine health measurements today, what is the chance they develop heart disease within the next 10 years?

This project treats that question as a **binary classification problem**: given 15 health-related features of a patient, predict whether the patient will develop CHD within 10 years (`TenYearCHD` = 1) or not (`TenYearCHD` = 0). Five classification models are trained on the same dataset, evaluated with six metrics each, and compared through an interactive Streamlit web application.

## b. Dataset Description

**Dataset:** Framingham Heart Study dataset (public dataset, available on Kaggle)
**Source:** https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

The data comes from an ongoing cardiovascular study of residents of Framingham, Massachusetts, USA.

- **Instances:** 4,240 patient records (minimum required: 500 ✅)
- **Features:** 15 (minimum required: 12 ✅)
- **Target:** `TenYearCHD` — 1 if the patient developed coronary heart disease within 10 years, 0 otherwise
- **Class balance:** imbalanced — only ~15.2% of patients are positive (CHD = 1)

| Feature | Meaning |
|---|---|
| male | 1 = male, 0 = female |
| age | Age in years |
| education | Education level (1–4) |
| currentSmoker | Is the patient a current smoker? |
| cigsPerDay | Cigarettes smoked per day |
| BPMeds | On blood pressure medication? |
| prevalentStroke | Had a stroke before? |
| prevalentHyp | Has hypertension? |
| diabetes | Has diabetes? |
| totChol | Total cholesterol (mg/dL) |
| sysBP | Systolic blood pressure |
| diaBP | Diastolic blood pressure |
| BMI | Body Mass Index |
| heartRate | Heart rate (beats/min) |
| glucose | Blood glucose (mg/dL) |

**Preprocessing:**

- Stratified 80/20 train–test split (3,392 train / 848 test rows), so both sets keep the same ~15% CHD rate.
- Missing values (a few columns such as `glucose`, `education`, `BPMeds` have gaps) are filled with the **median of the training set only**, to avoid data leakage.
- For scale-sensitive models (Logistic Regression, kNN), a `StandardScaler` is built into the model pipeline.
- The held-out 848-row test set is saved as `test_data.csv` and bundled for the app's upload feature.

## c. GitHub Repository Link

**https://github.com/aditpuranik-1504/heart-disease-ml-streamlit**

**Live Streamlit App:** https://heart-disease-ml-aditi-2026.streamlit.app

Repository structure:

```
heart-disease-ml/
├── app.py                  # Streamlit web application
├── requirements.txt        # Dependencies for deployment
├── README.md               # This file
├── framingham.csv          # Full dataset
├── test_data.csv           # Held-out test data (for app upload)
└── model/
    ├── train_models.py     # Training + evaluation script
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    ├── train_medians.pkl   # Imputation values (train medians)
    └── metrics.json        # Saved evaluation metrics
```

## d. Models Used and Evaluation Metrics

Five classification models were trained on the same dataset. `class_weight="balanced"` was used for Logistic Regression, Decision Tree and Random Forest to handle the 85/15 class imbalance. All metrics below are computed on the held-out test set (848 rows).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6722 | 0.7005 | 0.2541 | 0.5969 | 0.3565 | 0.2118 |
| Decision Tree | 0.6627 | 0.6778 | 0.2392 | 0.5581 | 0.3349 | 0.1799 |
| kNN | 0.8479 | 0.6476 | 0.5000 | 0.0388 | 0.0719 | 0.1058 |
| Naive Bayes | 0.8054 | 0.6755 | 0.2500 | 0.1395 | 0.1791 | 0.0830 |
| Random Forest (Ensemble) | 0.7229 | 0.6784 | 0.2675 | 0.4729 | 0.3417 | 0.1949 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model. Highest AUC (0.7005), F1 (0.3565) and MCC (0.2118). With balanced class weights it catches ~60% of true CHD cases (recall 0.597), which matters most in a medical screening problem. Its lower accuracy is the price of not ignoring the minority class. |
| Decision Tree | Close behind Logistic Regression but slightly worse on every metric (AUC 0.678, MCC 0.180). A single depth-limited tree captures some non-linear splits (age, blood pressure) but is less stable than the ensemble and tends to overfit if grown deeper. |
| kNN | Misleadingly high accuracy (0.8479 — the best!) but almost useless in practice: recall is only 0.039, meaning it misses ~96% of actual CHD patients. With 85% of patients healthy, kNN just predicts "healthy" nearly always. A clear lesson that accuracy alone is a bad metric on imbalanced data. |
| Naive Bayes | Decent AUC (0.6755) but weak recall (0.14) and the lowest MCC (0.083). Its assumption that features are independent is violated here (e.g. sysBP and diaBP are strongly correlated), which hurts its predictions. |
| Random Forest (Ensemble) | Second-best balanced performer (MCC 0.1949, recall 0.473). Averaging 300 trees makes it more robust than the single Decision Tree, and it offers the best accuracy (0.7229) among the models that actually detect CHD cases. |
| **Overall Winner for your dataset?** | **Logistic Regression.** On this imbalanced medical dataset it has the best AUC, F1 and MCC, and by far the most useful recall. Key insight: kNN "wins" on raw accuracy, but AUC/MCC/recall reveal it is the weakest model — which is exactly why we evaluate with six metrics instead of one. |

## Running the App

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**App features:**

- 📂 Upload test data as CSV (or use the bundled `test_data.csv`)
- 🔽 Model selection dropdown (all 5 models)
- 📊 Display of all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- 🔢 Confusion matrix + classification report + ROC curve
- 🏁 One-click comparison table of all models on the same test data

**To retrain the models:**

```bash
python model/train_models.py
```
