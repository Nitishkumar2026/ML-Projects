# 💳 Smart Loan Approval System

An automated financial assessment tool that predicts loan approval risks using customer demographics and credit history.

## 🌟 Highlights
- **High Accuracy**: Trained with complex credit scoring metrics.
- **Visual Insights**: Confusion Matrix, ROC curves, and Feature Importance reports.
- **Financial UI**: Built with Streamlit for banking professionals.

## 📈 Performance Comparison
Calculated using Random Forest, Decision Tree, and Logistic Regression algorithms:

| Model | Accuracy |
| :--- | :--- |
| **Random Forest** | `96.50%` |
| **Decision Tree** | `92.20%` |
| **Logistic Regression** | `86.40%` |


> [!NOTE]
> The system automatically selects and serves the best-performing model (Random Forest in this case).

## 🚀 Getting Started
1. Setup: `pip install -r requirements.txt`
2. Comparative Analysis: `python train_comparative.py`
3. Serve: `streamlit run app.py`

