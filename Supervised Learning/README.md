# 🤖 Supervised Learning Projects Collection

> **A curated collection of 5 end-to-end Machine Learning projects covering Regression, Classification, and NLP — each with trained models, visualizations, and interactive Streamlit dashboards.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboards-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Projects](https://img.shields.io/badge/Projects-5-blueviolet)](.)
[![Status](https://img.shields.io/badge/Status-All%20Completed-brightgreen)](.)

---

## 📂 Projects at a Glance

| # | Project | Type | Algorithm | Best Metric | Dashboard |
|---|---|---|---|---|---|
| 🏠 1 | [House Price Prediction](./1_House_Price_Prediction/) | Regression | Random Forest | R² = `0.5174` | ✅ Streamlit |
| 📧 2 | [Email Spam Detection](./2_Email_Spam_Detection/) | Classification (NLP) | Logistic Regression + TF-IDF | Accuracy = `64.71%` | ✅ Streamlit |
| 🎓 3 | [Student Performance Prediction](./3_Student_Performance_Prediction/) | Regression | Random Forest | R² = `0.8578` | ✅ Streamlit |
| 💳 4 | [Loan Approval Prediction](./4_Loan_Approval_Prediction/) | Classification | Random Forest *(Winner of 3-model comparison)* | Accuracy = `96.50%` | ✅ Streamlit |
| 🛒 5 | [Walmart Sales Forecasting](./5_Walmart_Sales_Prediction/) | Regression | Random Forest | R² = `0.9596` 🔥 | ✅ Streamlit |

---

## 🗂️ Repository Structure

```
Supervised Learning/
│
├── README.md                             ← You are here (Overall Summary)
│
├── 1_House_Price_Prediction/
│   ├── train.py                          # Model training & visualization
│   ├── app.py                            # Streamlit dashboard
│   ├── data.csv                          # King County house sales dataset
│   ├── house_rf_model.pkl                # Trained model
│   ├── requirements.txt
│   ├── Model_Outputs_Record.md
│   └── README.md                         ← Project-specific README
│
├── 2_Email_Spam_Detection/
│   ├── train.py                          # NLP training pipeline
│   ├── app.py                            # Streamlit spam checker
│   ├── email_spam.csv                    # Labeled email dataset
│   ├── spam_model.pkl                    # Trained LR model
│   ├── tfidf_vectorizer.pkl              # Fitted TF-IDF vectorizer
│   ├── requirements.txt
│   ├── Model_Outputs_Record.md
│   └── README.md                         ← Project-specific README
│
├── 3_Student_Performance_Prediction/
│   ├── train.py                          # Model training & visualization
│   ├── app.py                            # Streamlit grade predictor
│   ├── mat2.csv                          # Math performance dataset
│   ├── student_model.pkl                 # Trained model
│   ├── requirements.txt
│   ├── Model_Outputs_Record.md
│   └── README.md                         ← Project-specific README
│
├── 4_Loan_Approval_Prediction/
│   ├── train.py                          # 3-model comparative training
│   ├── app.py                            # Streamlit banking dashboard
│   ├── loan_risk_prediction_dataset.csv  # Applicant loan dataset
│   ├── loan_model.pkl                    # Best model (Random Forest)
│   ├── encoders.pkl                      # Label encoders + scaler
│   ├── requirements.txt
│   ├── Model_Outputs_Record.md
│   └── README.md                         ← Project-specific README
│
└── 5_Walmart_Sales_Prediction/
    ├── train.py                          # Model training & visualization
    ├── app.py                            # Streamlit sales forecaster
    ├── Walmart_Sales.csv                 # Walmart weekly sales dataset
    ├── rf_model.pkl                      # Trained model
    ├── requirements.txt
    ├── Model_Outputs_Record.md
    └── README.md                         ← Project-specific README
```

---

## 🔬 What's Inside Each Project?

Every project in this collection follows a **consistent, production-grade structure**:

| Component | Description |
|---|---|
| `train.py` | Full ML pipeline: data loading → preprocessing → training → evaluation → visualization |
| `app.py` | Interactive Streamlit dashboard for real-time predictions |
| `*.pkl` | Pre-trained model artifacts (ready to use without re-training) |
| `requirements.txt` | Exact Python package dependencies |
| `Model_Outputs_Record.md` | Auto-generated report with performance metrics and charts |
| `README.md` | Detailed project documentation with architecture, features, and usage |

---

## 🧠 ML Techniques Covered

| Technique | Projects Using It |
|---|---|
| **Random Forest Regressor** | House Price, Student Performance, Walmart Sales |
| **Random Forest Classifier** | Loan Approval (best model winner) |
| **Logistic Regression** | Email Spam, Loan Approval (comparative) |
| **Decision Tree Classifier** | Loan Approval (comparative) |
| **TF-IDF Vectorization (NLP)** | Email Spam Detection |
| **Label Encoding** | Loan Approval |
| **Standard Scaling** | Loan Approval (for Logistic Regression) |
| **Feature Engineering (Temporal)** | House Price (year/month), Walmart (year/month/week) |
| **Multi-Model Comparative Training** | Loan Approval (3 models + ROC comparison) |

---

## 📊 Performance Summary

```
Project                      Metric      Score
──────────────────────────────────────────────────────
🏠 House Price Prediction      R²         0.5174
📧 Email Spam Detection         Accuracy   64.71%
🎓 Student Performance          R²         0.8578
💳 Loan Approval                Accuracy   96.50%  ← Best Classifier
🛒 Walmart Sales                R²         0.9596  ← Best Overall 🏆
```

---

## 🚀 Global Quick Start Guide

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Nitishkumar2026/ML-Projects.git
cd ML-Projects/Supervised\ Learning
```

### Step 2 — Navigate to Any Project
```bash
cd 1_House_Price_Prediction   # or any other project folder
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model *(Optional — pre-trained PKL files are included)*
```bash
python train.py
```

### Step 5 — Launch the Interactive Dashboard
```bash
streamlit run app.py
```
> Open your browser at **`http://localhost:8501`**

---

## 🛠️ Global Prerequisites

```bash
# Ensure Python 3.8+ is installed, then install common dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3 | Data loading & manipulation |
| `numpy` | ≥ 1.21 | Numerical operations |
| `scikit-learn` | ≥ 1.0 | ML models & utilities |
| `matplotlib` | ≥ 3.4 | Plotting & visualizations |
| `seaborn` | ≥ 0.11 | Statistical visualizations |
| `streamlit` | ≥ 1.0 | Interactive web dashboards |
| `joblib` | ≥ 1.0 | Model serialization (`.pkl` files) |

---

## 👤 Author

**Nitish Kumar**
- 🎓 AIML Summer Training — Supervised Learning Specialization
- 🔗 [GitHub: Nitishkumar2026](https://github.com/Nitishkumar2026)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute for educational and personal purposes.

---

*Built with ❤️ using Python, Scikit-Learn & Streamlit*
