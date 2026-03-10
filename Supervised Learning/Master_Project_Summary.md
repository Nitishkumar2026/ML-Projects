# 🚀 Supervised Learning Project Master Summary

This workspace contains 5 comprehensive machine learning projects. Each folder is a self-contained environment with data, training scripts, graphical reports, and an interactive dashboard.

| Project Name | ML Type | Key Metric | Graphical Reports | Status |
| :--- | :--- | :--- | :--- | :--- |
| **🏠 1. House Price Prediction** | Regression | $R^2: 0.5174$ | 4 Visuals | ✅ Completed |
| **📧 2. Email Spam Detection** | Classification | $Accuracy: 64.7%$ | 3 Visuals | ✅ Completed |
| **🎓 3. Student Performance** | Regression | $R^2: 0.8578$ | 4 Visuals | ✅ Completed |
| **💳 4. Loan Approval Prediction** | Classification | $Accuracy: 96.5%$ | 4 Visuals | ✅ Completed |
| **🛒 5. Walmart Sales Prediction** | Regression | $R^2: 0.9596$ | 4 Visuals | ✅ Completed |

---

## 📂 Project Navigation & In-depth Records
Click the links below to view detailed performance screenshots and individual READMEs:

1. [House Price Prediction](./1_House_Price_Prediction/Model_Outputs_Record.md) - [Project README](./1_House_Price_Prediction/README.md)
2. [Email Spam Detection](./2_Email_Spam_Detection/Model_Outputs_Record.md) - [Project README](./2_Email_Spam_Detection/README.md)
3. [Student Performance](./3_Student_Performance_Prediction/Model_Outputs_Record.md) - [Project README](./3_Student_Performance_Prediction/README.md)
4. [Loan Approval Prediction](./4_Loan_Approval_Prediction/Model_Outputs_Record.md) - [Project README](./4_Loan_Approval_Prediction/README.md)
5. [Walmart Sales Prediction](./5_Walmart_Sales_Prediction/Model_Outputs_Record.md) - [Project README](./5_Walmart_Sales_Prediction/README.md)

---

## 🛠️ Global Execution Guide

### 💾 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 🧠 2. Train and Generate Reports
Jaise hi aap kisi folder me `train.py` run karenge, wo auto-generate karega:
- Trained `.pkl` model
- `Model_Outputs_Record.md` (Multiple screenshots ke sath)

### 📈 3. Launch Dashboards
```bash
cd [Project_Folder_Name]
streamlit run app.py
```
