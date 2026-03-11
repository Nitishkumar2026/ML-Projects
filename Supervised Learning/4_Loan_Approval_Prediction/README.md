# 💳 Smart Loan Approval Prediction System

> **Multi-model comparative analysis for automated loan risk assessment — Random Forest, Decision Tree & Logistic Regression.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Best Model](https://img.shields.io/badge/Best%20Model-Random%20Forest%2096.5%25-brightgreen)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

---

## 📌 Project Overview

This project builds an **intelligent loan approval prediction system** that determines whether a loan application should be **approved or rejected** based on applicant's financial profile, demographics, credit history, and employment status. 

What makes this project unique: it performs a **full comparative experiment** — training and evaluating **three different ML models** side-by-side, generating ROC curves and accuracy comparisons, and automatically selecting the best-performing model to serve predictions.

---

## 🧠 Multi-Model Training Architecture

```
loan_risk_prediction_dataset.csv
       │
       ▼
Data Preprocessing:
  ├─ Fill missing: Income (median), CreditScore (median)
  ├─ Fill missing: Education, Gender, City, EmploymentType (mode)
  └─ Label Encoding: Gender, Education, City, EmploymentType
       │
       ▼
Train/Test Split (80/20)
       │
       ├──────────────────────────────────────────────────┐
       │                                                  │
       ▼                                                  │
StandardScaler (for Logistic Regression only)            │
       │                                                  │
       ▼                                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                   3 Models Trained in Parallel                   │
├──────────────────┬─────────────────────┬─────────────────────────┤
│ Logistic         │ Decision Tree        │ Random Forest           │
│ Regression       │ Classifier           │ Classifier              │
│ (scaled data)    │                      │ (n_estimators=100)      │
└──────────────────┴─────────────────────┴─────────────────────────┘
       │                   │                        │
       └───────────────────┴────────────────────────┘
                           │
                     Compare Accuracy
                           │
                   Best Model Selected ──► loan_model.pkl (saved)
                                      ──► encoders.pkl (saved)
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **File** | `loan_risk_prediction_dataset.csv` |
| **Task Type** | Binary Classification |
| **Target** | `LoanApproved` (0 = Rejected, 1 = Approved) |
| **Train / Test Split** | 80% / 20% |

### 🔑 Key Features

| Feature | Description |
|---|---|
| `Income` | Applicant's annual income |
| `CreditScore` | Credit score (important risk indicator) |
| `LoanAmount` | Requested loan amount |
| `Age` | Applicant's age |
| `Gender` | Male/Female (encoded) |
| `Education` | Education level (encoded) |
| `EmploymentType` | Full-time, Part-time, etc. (encoded) |
| `City` | Applicant's city (encoded) |

---

## 🤖 Model Comparison Results

| Model | Accuracy | Notes |
|---|---|---|
| 🥇 **Random Forest** | **`96.50%`** | Best performer — high accuracy & generalization |
| 🥈 **Decision Tree** | `92.20%` | Good accuracy but prone to overfitting |
| 🥉 **Logistic Regression** | `86.40%` | Strong baseline, interpretable, uses scaled data |

> [!NOTE]
> The system automatically selects and deploys the **best-performing model** (Random Forest at 96.50%). The saved `loan_model.pkl` always contains the winner from the last training run.

> [!TIP]
> The ROC curve comparison (`output_comparison_roc.png`) is an excellent way to compare model performance beyond just accuracy — especially important in imbalanced loan approval datasets where false negatives (approved bad loans) are costly.

---

## 📉 Visualizations Generated

| # | Output File | Description |
|---|---|---|
| 1 | `output_01_confusion_matrix.png` | True/False Positives & Negatives heatmap |
| 2 | `output_02_feature_importance.png` | Top 10 loan risk factors (Random Forest) |
| 3 | `output_03_roc_curve.png` | ROC curve for the best model |
| 4 | `output_04_distribution.png` | Distribution of applicant features |
| 5 | `output_comparison_accuracy.png` | Side-by-side accuracy comparison (all 3 models) |
| 6 | `output_comparison_roc.png` | Overlapping ROC curves for all 3 models |

---

## 🖥️ Interactive Dashboard (`app.py`)

The Streamlit app is designed for **banking professionals** and allows:
- Input of applicant demographics and financial profile
- **Real-time loan approval prediction** with confidence score
- Visual breakdown of key risk factors

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models (Comparative Analysis)
```bash
python train.py
```
Trains all 3 models, generates comparison charts, and saves the best model.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
4_Loan_Approval_Prediction/
│
├── train.py                              # Multi-model training pipeline
├── app.py                                # Streamlit banking dashboard
├── loan_risk_prediction_dataset.csv      # Applicant loan dataset
├── loan_model.pkl                        # Best trained model (Random Forest)
├── encoders.pkl                          # Label encoders + scaler artifacts
├── requirements.txt                      # Python dependencies
├── Model_Outputs_Record.md               # Model comparison report
├── Model_Comparison_Record.md            # Detailed comparison summary
│
├── output_01_confusion_matrix.png
├── output_02_feature_importance.png
├── output_03_roc_curve.png
├── output_04_distribution.png
├── output_comparison_accuracy.png
└── output_comparison_roc.png
```

---

## 🛠️ Dependencies

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

## 🔗 Part of the Supervised Learning Projects Collection
← [Back to Main Repository](../README.md)
