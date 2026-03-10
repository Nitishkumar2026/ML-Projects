import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("Starting Comparative Model Training...")

df = pd.read_csv("loan_risk_prediction_dataset.csv")

# Preprocessing
df['Income'] = df['Income'].fillna(df['Income'].median())
df['CreditScore'] = df['CreditScore'].fillna(df['CreditScore'].median())
df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['EmploymentType'] = df['EmploymentType'].fillna(df['EmploymentType'].mode()[0])

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_city = LabelEncoder()
le_employment = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education'] = le_education.fit_transform(df['Education'])
df['City'] = le_city.fit_transform(df['City'])
df['EmploymentType'] = le_employment.fit_transform(df['EmploymentType'])

X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to train
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
best_accuracy = 0
best_model_name = ""
best_model_obj = None

plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"Training {name}...")
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"Accuracy": acc, "Report": classification_report(y_test, y_pred)}
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model_obj = model
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig("output_03_roc_curve.png") # Overwrite old ROC plot
plt.savefig("output_comparison_roc.png")
plt.close()

# Comparison Bar Chart
plt.figure(figsize=(10, 6))
names = list(results.keys())
accuracies = [results[n]["Accuracy"] * 100 for n in names]
sns.barplot(x=names, y=accuracies, hue=names, palette='viridis', legend=False)
plt.title("Model Accuracy Comparison")
plt.savefig("output_comparison_accuracy.png")
plt.close()

# Generate additional visualizations for the BEST model (Consistency with old app)
if best_model_name == "Random Forest":
    importances = best_model_obj.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title(f"Key Factors ({best_model_name})")
    plt.savefig("output_02_feature_importance.png")
    plt.close()

# Save defaults
joblib.dump(best_model_obj, "loan_model.pkl")
joblib.dump({
    'gender': le_gender, 'education': le_education, 'city': le_city, 
    'employment': le_employment, 'scaler': scaler, 'best_model_name': best_model_name
}, "encoders.pkl")

# Update Record
with open("Model_Outputs_Record.md", "w", encoding="utf-8") as f:
    f.write("# 💳 Loan Approval Comparison Record\n\n")
    f.write(f"**Best Model:** {best_model_name} ({best_accuracy*100:.2f}%)\n\n")
    for name, data in results.items():
        f.write(f"### {name}\n")
        f.write(f"Accuracy: {data['Accuracy']*100:.2f}%\n")
        f.write("```text\n" + data['Report'] + "```\n\n")

print(f"Training complete! Best model '{best_model_name}' saved.")

