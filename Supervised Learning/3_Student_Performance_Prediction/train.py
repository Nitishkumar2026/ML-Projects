import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Loading Student Performance Dataset (Math)...")
df = pd.read_csv("mat2.csv")

cols_to_use = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
X = df[cols_to_use]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save Model
joblib.dump(model, "student_model.pkl")

# Generate Screenshots
print("Generating visualizations...")

# 1. Prediction Plot
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title("Actual vs Predicted Grades (G3)", fontsize=16)
plt.savefig("output_screenshot_prediction_accuracy.png")
plt.close()

# 2. Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=X.columns, hue=X.columns, palette="magma", legend=False)
plt.title("Key Factors Influencing Student Grades", fontsize=16)
plt.savefig("output_screenshot_factors.png")
plt.close()

# 3. Grade Frequency Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y, color='skyblue')
plt.title("Final Grade Distribution (G3)", fontsize=16)
plt.savefig("output_screenshot_distribution.png")
plt.close()

# 4. G2 vs G3 Correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['G2'], y=df['G3'], hue=df['failures'], palette='viridis')
plt.title("Previous Grade (G2) vs Final Grade (G3)", fontsize=16)
plt.savefig("output_screenshot_g2_correlation.png")
plt.close()

# Save Record
with open("Model_Outputs_Record.md", "w", encoding="utf-8") as f:
    f.write("# 🎓 Student Performance - Pro Results\n\n")
    f.write(f"**$R^2$ Score:** {r2:.4f}\n")
    f.write(f"**Error (RMSE):** {rmse:.2f}\n\n")
    f.write("## Graphical Analysis\n\n")
    f.write("### 1. Accuracy Regression\n![Accuracy](output_screenshot_prediction_accuracy.png)\n\n")
    f.write("### 2. Grade Impact Factors\n![Factors](output_screenshot_factors.png)\n\n")
    f.write("### 3. Students Grade Frequency\n![Dist](output_screenshot_distribution.png)\n\n")
    f.write("### 4. G2 vs G3 Progress\n![Correlation](output_screenshot_g2_correlation.png)\n")

# Update README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write("# 🎓 Student Performance Predictor\n\n")
    f.write("A supervised learning regression project to identify academic risk and predict final semester grades.\n\n")
    f.write("## 📌 Key Metrics\n")
    f.write("- **G1/G2 Scores**: Previous exams recorded.\n")
    f.write("- **Social Factors**: Alcohol consumption, family relationship, study time.\n")
    f.write("- **Target**: Final Math Grade (G3).\n\n")
    f.write("## 🚀 Quick Start\n")
    f.write("1. Train: `python train.py`\n")
    f.write("2. Dashboard: `streamlit run app.py`\n")

print("Project 3 records updated!")
