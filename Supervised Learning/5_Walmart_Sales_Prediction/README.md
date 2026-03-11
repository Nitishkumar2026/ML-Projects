# 🛒 Walmart Weekly Sales Forecasting

> **Forecasting retail store weekly sales using Random Forest with temporal feature engineering — R² = 0.9596.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![R²](https://img.shields.io/badge/R%C2%B2%20Score-0.9596-brightgreen)](https://scikit-learn.org)

---

## 📌 Project Overview

This project forecasts **Walmart's weekly retail sales** across multiple store departments using a **Random Forest Regressor** with engineered temporal features. The model captures complex non-linear relationships between sales and factors like **temperature, fuel price, CPI (Consumer Price Index), unemployment rate, and holiday flags**.

With an exceptional **R² of 0.9596**, this is the highest-performing model in the entire Supervised Learning collection — making it an excellent showcase of Random Forest's power in time-series retail forecasting.

---

## 🧠 Model Pipeline

```
Walmart_Sales.csv
       │
       ▼
Feature Engineering (Temporal):
  ├─ Date → Year, Month, Week (calendar week)
  └─ Drop: 'Date' column (used only for extraction)
       │
       ▼
Feature Matrix (X):
  Store, Dept, IsHoliday, Temperature, Fuel_Price,
  MarkDown1-5, CPI, Unemployment, Year, Month, Week
       │
       ▼
Train/Test Split (80/20, random_state=42)
       │
       ▼
RandomForestRegressor (n_estimators=100) ──► rf_model.pkl (saved)
       │
       ▼
Predict Weekly_Sales ──► app.py (Streamlit Dashboard)
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **File** | `Walmart_Sales.csv` |
| **Source** | Walmart Store Sales (public retail dataset) |
| **Task Type** | Regression (continuous sales value) |
| **Target Variable** | `Weekly_Sales` (in USD) |
| **Date Format** | `DD-MM-YYYY` |
| **Train / Test Split** | 80% / 20% |

### 🔑 Key Features Used

| Feature | Description |
|---|---|
| `Store` | Store number (1–45) |
| `Dept` | Department number within store |
| `IsHoliday` | Whether the week is a special holiday week |
| `Temperature` | Average temperature in the region (°F) |
| `Fuel_Price` | Cost of fuel in the region |
| `CPI` | Consumer Price Index |
| `Unemployment` | Regional unemployment rate |
| `MarkDown1–5` | Anonymous promotional markdown events |
| `Year` | Extracted from Date |
| `Month` | Extracted from Date |
| `Week` | Calendar week number (ISO week) |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| `n_estimators` | 100 trees |
| `random_state` | 42 |
| **Saved As** | `rf_model.pkl` |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **R² Score** | **`0.9596`** 🔥 |

> [!NOTE]
> An R² of **0.9596** means the model explains **~96% of variance** in weekly sales! This is an outstanding result, making this the top-performing model in this collection. The key drivers are `Dept`, `Store`, and `Week` — indicating that department-level and temporal patterns are highly predictive.

> [!TIP]
> The `IsHoliday` feature captures Black Friday (Thanksgiving), Super Bowl, Christmas, and Labor Day weeks — these are critical outlier spikes in Walmart's weekly sales that significantly improve model accuracy when included.

---

## 📉 Visualizations Generated

| # | Output File | Description |
|---|---|---|
| 1 | `output_screenshot_sales_prediction.png` | Actual vs Predicted weekly sales scatter |
| 2 | `output_screenshot_importance.png` | Feature importance: top sales drivers |
| 3 | `output_screenshot_seasonality.png` | Monthly average sales (seasonal trend analysis) |
| 4 | `output_screenshot_temp_impact.png` | Temperature vs Weekly Sales scatter |
| 5 | `feature_importance.png` | Standalone feature importance chart |

---

## 🖥️ Interactive Dashboard (`app.py`)

The Streamlit dashboard allows retail analysts to:
- Select **Store number, Department, and Week**
- Input external factors (fuel price, CPI, temperature, unemployment)
- Get **real-time weekly sales forecast**
- Explore seasonal trends visually

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
Generates `rf_model.pkl` and all visualization screenshots.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
5_Walmart_Sales_Prediction/
│
├── train.py                              # Model training & visualization script
├── app.py                                # Streamlit sales forecasting dashboard
├── Walmart_Sales.csv                     # Walmart weekly sales dataset
├── rf_model.pkl                          # Trained Random Forest model
├── requirements.txt                      # Python dependencies
├── Model_Outputs_Record.md               # Detailed model performance report
│
├── output_screenshot_sales_prediction.png
├── output_screenshot_importance.png
├── output_screenshot_seasonality.png
├── output_screenshot_temp_impact.png
└── feature_importance.png
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
