# 🏦 Term Deposit Subscription Prediction (Bank Marketing)
### Predicting Customer Response with Classification Models & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC4E20?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
---

*A production-grade classification pipeline predicting bank term deposit subscriptions — featuring four models, SMOTE class balancing, and dual explainability with SHAP and LIME.*
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Key Visualizations](#-key-visualizations)
- [Models & Results](#-models--results)
- [Explainable AI](#-explainable-ai)
- [Key Insights](#-key-insights)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🎯 Overview

Banks invest millions in direct marketing campaigns to sell term deposits. Most calls end in rejection. The question is: **can we predict who will say yes before picking up the phone?**

This project builds a complete ML pipeline that:
1. Analyzes **11,162 customer records** from a Portuguese bank's phone campaigns
2. Engineers and encodes features across demographics, finances, and campaign history
3. Trains **4 classification models** with SMOTE-balanced data
4. Evaluates with Confusion Matrix, F1-Score, ROC-AUC, and Precision-Recall curves
5. Explains predictions using **SHAP** (global + local) and **LIME** — answering not just *what* the model predicts, but *why*

---

## 📊 Dataset

**Source:** [Bank Marketing Dataset — UCI ML Repository (via Kaggle)](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

| Property | Detail |
|----------|--------|
| **Origin** | Direct marketing campaigns of a Portuguese bank |
| **Records** | 11,162 customer interactions |
| **Features** | 16 input features + 1 binary target |
| **Target** | `deposit` — did the customer subscribe? (yes/no) |

### Feature Categories

| Category | Features | Examples |
|----------|----------|---------|
| **Demographics** | 4 | age, job, marital status, education |
| **Financial** | 4 | balance, default, housing loan, personal loan |
| **Campaign** | 5 | contact type, day, month, duration, # contacts |
| **Previous Campaign** | 3 | days since last contact, previous contacts, outcome |

---

## 🔄 Project Pipeline

┌─────────────────────────────────────────────────────────────┐
│ DATA INGESTION │
│ Load CSV → Inspect types, nulls, duplicates │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ EXPLORATORY DATA ANALYSIS │
│ Target balance · Demographics · Financial profiles │
│ Campaign features · Correlation analysis │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSING │
│ Label encoding (binary) · Ordinal encoding (education) │
│ One-Hot encoding (nominal) · Cyclical encoding (month) │
│ Interaction features · Standard scaling │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ CLASS IMBALANCE HANDLING │
│ SMOTE oversampling on training data only │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING │
│ Logistic Regression │ Random Forest │ XGBoost │ LightGBM │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION │
│ Confusion Matrix · F1 · ROC-AUC · PR Curve · 5-Fold CV │
└────────────────────┬────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────┐
│ EXPLAINABLE AI │
│ SHAP (global beeswarm + local waterfall) │
│ LIME (individual prediction explanations) │
│ SHAP vs LIME agreement analysis │
└─────────────────────────────────────────────────────────────┘

---

## 📈 Key Visualizations

The notebook produces **15+ interactive charts**, including:

- **Class distribution** — bar + donut chart of target imbalance
- **Demographic breakdown** — subscription rate by job, marital status, education, age
- **Financial profile** — balance distribution, housing/personal loan impact
- **Campaign analysis** — duration distribution, contact type, previous outcome
- **Correlation heatmap** — full feature-target correlation matrix
- **SMOTE before/after** — class balance comparison
- **Confusion matrices** — 2×2 grid for all 4 models
- **ROC & Precision-Recall curves** — all models overlaid
- **Model comparison bars** — Accuracy, F1, ROC-AUC side by side
- **Feature importance** — tree-based model feature rankings
- **SHAP beeswarm** — global feature impact with direction and magnitude
- **SHAP waterfall** — individual prediction breakdown
- **LIME explanations** — local interpretable rules for 5 customers
- **SHAP vs LIME** — side-by-side comparison of both methods
- **Cross-validation boxplots** — 5-fold F1 score distributions

---

## 🏆 Models & Results

### Four Classification Models

| Model | Type | Key Strengths |
|-------|------|--------------|
| **Logistic Regression** | Linear | Interpretable baseline, fast, regulatory-friendly |
| **Random Forest** | Bagging | Handles non-linearity, robust to overfitting |
| **XGBoost** | Boosting | State-of-the-art tabular performance |
| **LightGBM** | Boosting | Fast training, handles categoricals natively |

> All models trained on SMOTE-balanced data, evaluated on the original unbalanced test set for realistic performance estimates.

---

## 🤖 Explainable AI

This project goes beyond prediction accuracy to answer **why**:

### SHAP (SHapley Additive exPlanations)
- **Global importance** — which features matter most across all predictions
- **Beeswarm plot** — direction + magnitude of every feature for every customer
- **Waterfall plots** — step-by-step breakdown of individual predictions

### LIME (Local Interpretable Model-agnostic Explanations)
- **Individual rules** — intuitive "if-then" explanations per customer
- **Model-agnostic** — works as a second opinion regardless of model type

### Agreement Analysis
- SHAP and LIME top features compared to validate interpretation robustness

---

## 🔍 Key Insights

1. **Call duration** is the top predictor — but is a leakage variable (known only after the call)
2. **Previous campaign success** is the strongest non-leaky signal — past subscribers convert again
3. **Financial health** — higher balance + no loans = higher subscription rate
4. **Demographics** — students and retirees subscribe most; blue-collar workers least
5. **Diminishing returns** — more than 3 contacts per campaign hurts conversion
6. **Timing matters** — March, September–December campaigns perform best

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly (interactive), Matplotlib |
| **Classification** | scikit-learn, XGBoost, LightGBM |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP, LIME |
| **Evaluation** | scikit-learn metrics, Stratified K-Fold CV |

---

## 🚀 Getting Started

### Option 1: Run on Kaggle (Recommended)

1. Open the notebook on [Kaggle]
2. Add the [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
3. Enable **Internet** in Session Options
4. Run all cells sequentially

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/kinzaemann/bank-marketing-prediction.git
cd bank-marketing-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
bank-marketing-prediction/
│
├── bank-marketing-prediction.ipynb   # Main notebook (all code + analysis)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
└── data/                             # Dataset 
    └── bank.csv

If you found this project useful, please ⭐ star this repository!
