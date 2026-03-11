# 🏠 House Price Prediction System

> **Predicting residential property prices using Random Forest Regression with real-world data from King County, Washington.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Random%20Forest%20Regressor-green)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

---

## 📌 Project Overview

This project builds an end-to-end supervised machine learning pipeline to predict **house prices** based on property characteristics such as number of bedrooms, bathrooms, square footage, location grade, and more. The system is trained on the **King County House Sales dataset** and deployed as an interactive **Streamlit dashboard** for real-time predictions.

---

## 🧠 Technical Architecture

```
data.csv  ──►  Feature Engineering  ──►  RandomForestRegressor  ──►  house_rf_model.pkl
                 (Date Parsing,              (n_estimators=100)           │
                  Year/Month cols)                                         ▼
                                                                    streamlit app.py
                                                                  (Real-time Prediction)
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **Source** | King County, Washington House Sales |
| **File** | `data.csv` |
| **Total Samples** | ~21,000+ records |
| **Train / Test Split** | 80% / 20% |

### 🔑 Key Features Used
| Feature | Description |
|---|---|
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Interior living space (sq ft) |
| `sqft_lot` | Lot area (sq ft) |
| `floors` | Number of floors |
| `waterfront` | Waterfront property flag |
| `grade` | Overall grade given by King County |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of last renovation |
| `year_sold` | Year of sale (engineered) |
| `month_sold` | Month of sale (engineered) |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| `n_estimators` | 100 trees |
| `random_state` | 42 |
| **Saved As** | `house_rf_model.pkl` |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **R² Score** | `0.5174` |
| **Mean Absolute Error (MAE)** | `$163,325.48` |

> [!NOTE]
> An R² of 0.5174 means the model explains ~52% of variance in house prices. The remaining variance is due to factors like neighborhood reputation, interior design, and micro-location — which aren't fully captured in a tabular dataset. The model is still highly valuable for ballpark price estimation.

---

## 📉 Visualizations Generated

After running `train.py`, the following plots are automatically saved:

| # | Output File | Description |
|---|---|---|
| 1 | `output_screenshot_actual_vs_predicted.png` | Scatter plot: Actual vs Predicted prices |
| 2 | `output_screenshot_residuals.png` | Residual distribution for error analysis |
| 3 | `output_screenshot_feature_importance.png` | Top features ranked by predictive power |
| 4 | `output_screenshot_correlation.png` | Feature correlation heatmap |

---

## 🖥️ Interactive Dashboard (`app.py`)

The Streamlit dashboard allows users to:
- Input property characteristics via sliders and dropdowns
- Get **real-time price predictions** from the trained model
- Explore visualizations of model behavior

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
This will generate the model file (`house_rf_model.pkl`) and all output visualizations.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
1_House_Price_Prediction/
│
├── train.py                              # Model training & visualization script
├── app.py                                # Streamlit interactive dashboard
├── data.csv                              # House sales dataset
├── house_rf_model.pkl                    # Trained Random Forest model
├── requirements.txt                      # Python dependencies
├── Model_Outputs_Record.md               # Detailed model performance report
│
├── output_screenshot_actual_vs_predicted.png
├── output_screenshot_residuals.png
├── output_screenshot_feature_importance.png
└── output_screenshot_correlation.png
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
