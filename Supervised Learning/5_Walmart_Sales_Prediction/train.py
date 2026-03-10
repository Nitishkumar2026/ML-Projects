import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Loading Walmart Sales Dataset...")
df = pd.read_csv("Walmart_Sales.csv")

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

X = df.drop(["Weekly_Sales", "Date"], axis=1)
y = df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Save Model
joblib.dump(model, "rf_model.pkl")

# Generate Visualizations
print("Generating visualizations...")

# 1. Prediction Accuracy
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title("Actual vs Predicted Weekly Sales", fontsize=16)
plt.savefig("output_screenshot_sales_prediction.png")
plt.close()

# 2. Feature Importance
plt.figure(figsize=(12, 6))
importances = model.feature_importances_
sns.barplot(x=importances, y=X.columns, hue=X.columns, palette="viridis", legend=False)
plt.title("Key Factors in Retail Sales", fontsize=16)
plt.savefig("output_screenshot_importance.png")
plt.close()

# 3. Seasonal Trend (Monthly Sales)
plt.figure(figsize=(10, 6))
df.groupby('Month')['Weekly_Sales'].mean().plot(kind='bar', color='gold')
plt.title("Monthly Average Sales - Seasonal Trend", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Average Sales ($)")
plt.savefig("output_screenshot_seasonality.png")
plt.close()

# 4. Temperature Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(1000), x='Temperature', y='Weekly_Sales', alpha=0.4)
plt.title("Impact of Temperature on Sales", fontsize=16)
plt.savefig("output_screenshot_temp_impact.png")
plt.close()

# Save Record
with open("Model_Outputs_Record.md", "w", encoding="utf-8") as f:
    f.write("# 🛒 Walmart Sales Forecasting - Pro Results\n\n")
    f.write(f"**Model Accuracy ($R^2$):** {r2:.4f}\n\n")
    f.write("## Graphical Analysis\n\n")
    f.write("### 1. Forecasting Confidence\n![Sales](output_screenshot_sales_prediction.png)\n\n")
    f.write("### 2. Business Drivers (Feature Importance)\n![Drivers](output_screenshot_importance.png)\n\n")
    f.write("### 3. Seasonality Trends\n![Season](output_screenshot_seasonality.png)\n\n")
    f.write("### 4. External Environment (Temp vs Sales)\n![Temp](output_screenshot_temp_impact.png)\n")

# Update README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write("# 🛒 Walmart Sales Forecasting\n\n")
    f.write("Predicting weekly retail sales across stores using high-performance regression models.\n\n")
    f.write("## 🔥 Capabilities\n")
    f.write("- **Random Forest Model**: Captures non-linear dependencies in retail data.\n")
    f.write("- **Feature Engineering**: Incorporates CPI, Unemployment, and Holiday factors.\n")
    f.write("- **Interactive Dashboard**: Predict sales for any store and date.\n\n")
    f.write("## 🚀 Quick Start\n")
    f.write("1. `python train.py` to generate model.\n")
    f.write("2. `streamlit run app.py` to launch dashboard.\n")

print("Project 5 records updated!")
