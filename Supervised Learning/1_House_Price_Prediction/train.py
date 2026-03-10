import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

print("Loading dataset...")
df = pd.read_csv("data.csv")

# Ensure dataset size is manageable & clean
df = df[df['price'] > 0] # Remove 0 price houses if any

# Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['year_sold'] = df['date'].dt.year
df['month_sold'] = df['date'].dt.month

# Drop columns that need complex NLP/encoding for a baseline model
X = df.drop(['price', 'date', 'street', 'city', 'statezip', 'country'], axis=1)
y = df['price']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save Model
joblib.dump(model, "house_rf_model.pkl")

# Generate 'Screenshots' of Model Results
print("Generating result screenshots...")

# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted House Prices", fontsize=16)
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("output_screenshot_actual_vs_predicted.png")
plt.close()

# 2. Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=X.columns, hue=X.columns, palette="viridis", legend=False)
plt.title("Importance of Features in Predicting Price", fontsize=16)
plt.tight_layout()
plt.savefig("output_screenshot_feature_importance.png")
plt.close()

# 3. Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='orange', edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title("Residual Plot (Errors)", fontsize=16)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("output_screenshot_residuals.png")
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X.join(y).corr(), annot=True, cmap='RdYlGn', fmt='.2f')
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("output_screenshot_correlation.png")
plt.close()

# Update MD Record
with open("Model_Outputs_Record.md", "w", encoding="utf-8") as f:
    f.write("# 🏠 House Price Prediction - Pro Results\n\n")
    f.write(f"**Accuracy ($R^2$):** {r2:.4f}\n")
    f.write(f"**Mean Absolute Error:** ${mae:,.2f}\n\n")
    f.write("## Graphical Analysis\n\n")
    f.write("### 1. Regression Success (Actual vs Predicted)\n")
    f.write("![Actual vs Predicted](output_screenshot_actual_vs_predicted.png)\n\n")
    f.write("### 2. Error Analysis (Residual Plot)\n")
    f.write("![Residuals](output_screenshot_residuals.png)\n\n")
    f.write("### 3. Feature Importance\n")
    f.write("![Feature Importance](output_screenshot_feature_importance.png)\n\n")
    f.write("### 4. Correlation Matrix\n")
    f.write("![Correlation](output_screenshot_correlation.png)\n")

# Update README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write("# 🏠 House Price Prediction System\n\n")
    f.write("A supervised machine learning project to predict residential house prices using advanced regression techniques.\n\n")
    f.write("## 🚀 Features\n")
    f.write("- **Random Forest Regressor** for high-accuracy predictions.\n")
    f.write("- Interactive **Streamlit Dashboard** for real-time price estimation.\n")
    f.write("- Detailed **Data Visualization** (Residuals, Correlation, Importance).\n\n")
    f.write("## 📊 Model Performance\n")
    f.write(f"- $R^2$ Score: `{r2:.4f}`\n")
    f.write(f"- MAE: `${mae:,.2f}`\n\n")
    f.write("## 🛠 Setup\n")
    f.write("1. Install dependencies: `pip install -r requirements.txt`\n")
    f.write("2. Train model: `python train.py`\n")
    f.write("3. Launch app: `streamlit run app.py`\n")

print("Project 1 records updated!")
