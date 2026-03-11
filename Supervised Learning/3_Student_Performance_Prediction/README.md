# 🎓 Student Performance Predictor

> **Predicting final semester math grades using social, academic, and behavioral factors via Random Forest Regression.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Student%20Data-lightblue)](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

---

## 📌 Project Overview

This project uses the **UCI Student Performance dataset** to build a regression model that predicts a **student's final mathematics grade (G3)** based on academic history, family background, lifestyle habits, and social factors. The goal is to identify students at academic risk early so that intervention strategies can be implemented in time.

The system is powered by a **Random Forest Regressor** and features an interactive **Streamlit dashboard** for live grade predictions.

---

## 🧠 Model Pipeline

```
mat2.csv (Math Grades Dataset)
       │
       ▼
  Feature Selection ──────────────────────────────────────────────────────────────
  ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',                 │
   'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
       │                                                                          │
       ▼                                                                          │
Train/Test Split (80/20)                                                          │
       │                                                                          │
       ▼                                                                          │
RandomForestRegressor (n_estimators=100)  ──►  student_model.pkl (saved)         │
       │                                                                          │
       ▼                                                                          └── app.py
  Predict G3 (Final Grade)                                                     (Streamlit UI)
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Domain** | Secondary school math performance (Portugal) |
| **File** | `mat2.csv` |
| **Target Variable** | `G3` — Final Math Grade (0–20 scale) |
| **Train / Test Split** | 80% / 20% |

### 🔑 Key Features & Their Meaning

| Feature | Meaning |
|---|---|
| `G1` | First period grade (most predictive!) |
| `G2` | Second period grade (most predictive!) |
| `studytime` | Weekly study time (1=<2h, 4=>10h) |
| `failures` | Number of past class failures |
| `absences` | Number of school absences |
| `Dalc` | Weekday alcohol consumption (1–5) |
| `Walc` | Weekend alcohol consumption (1–5) |
| `famrel` | Quality of family relationships (1–5) |
| `Medu` | Mother's education level |
| `Fedu` | Father's education level |
| `goout` | Going out with friends frequency |
| `health` | Current health status (1–5) |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| `n_estimators` | 100 trees |
| `random_state` | 42 |
| **Saved As** | `student_model.pkl` |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **R² Score** | `0.8578` |
| **RMSE** | ~1.8 grade points |

> [!NOTE]
> An R² of **0.8578** is excellent — the model explains over **85% of the variance** in final grades. The most influential predictors are `G2` (second period grade), `G1` (first period grade), and `absences`, which aligns with real-world expectations.

---

## 📉 Visualizations Generated

| # | Output File | Description |
|---|---|---|
| 1 | `output_screenshot_prediction_accuracy.png` | Regression plot: Actual G3 vs Predicted G3 |
| 2 | `output_screenshot_factors.png` | Feature importance (key factors influencing grades) |
| 3 | `output_screenshot_distribution.png` | Grade frequency distribution across students |
| 4 | `output_screenshot_g2_correlation.png` | Scatter: G2 progress vs Final G3 grade |

---

## 🖥️ Interactive Dashboard (`app.py`)

The Streamlit app allows educators and students to:
- Input a student's profile (study time, family background, past grades)
- Predict the **expected final math grade (G3)**
- Identify at-risk students through visual analysis

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
Generates `student_model.pkl` and all 4 visualization screenshots.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
3_Student_Performance_Prediction/
│
├── train.py                              # Model training & visualization script
├── app.py                                # Streamlit prediction dashboard
├── mat2.csv                              # Math performance dataset (Portugal)
├── por2.csv                              # Portuguese performance dataset (bonus)
├── student_model.pkl                     # Trained Random Forest model
├── requirements.txt                      # Python dependencies
├── Model_Outputs_Record.md               # Detailed model performance report
│
├── output_screenshot_prediction_accuracy.png
├── output_screenshot_factors.png
├── output_screenshot_distribution.png
└── output_screenshot_g2_correlation.png
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
