import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="wide")

# Load model and encoders
try:
    model = joblib.load("loan_model.pkl")
    encoders_data = joblib.load("encoders.pkl")
    # Handle different possible structures of encoders.pkl
    if isinstance(encoders_data, dict) and 'gender' in encoders_data:
        encoders = encoders_data
        best_model_name = encoders_data.get('best_model_name', 'Unknown')
        scaler = encoders_data.get('scaler', None)
    else:
        encoders = encoders_data
        best_model_name = "Random Forest"
        scaler = None
except:
    st.error("Model files not found. Run train_comparative.py first.")
    st.stop()

st.title("🏦 Smart Loan Approval System")
st.markdown(f"Instantly evaluate loan eligibility using our best precision model: **{best_model_name}**")

col1, col2 = st.columns(2)

with col1:
    st.header("Personal Details")
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    education = st.selectbox("Education", encoders['education'].classes_)
    city = st.selectbox("City", encoders['city'].classes_)
    
    st.header("Professional & Financial")
    employment = st.selectbox("Employment Type", encoders['employment'].classes_)
    experience = st.number_input("Years of Experience", 0, 50, 5)
    income = st.number_input("Annual Income ($)", 5000, 1000000, 50000)

with col2:
    st.header("Loan Details")
    loan_amount = st.number_input("Requested Loan Amount ($)", 1000, 500000, 20000)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    
    st.markdown("---")
    
    if st.button("Evaluate Application", type="primary", use_container_width=True):
        # Prepare data
        input_data = pd.DataFrame([{
            'Age': age,
            'Income': income,
            'LoanAmount': loan_amount,
            'CreditScore': credit_score,
            'YearsExperience': experience,
            'Gender': encoders['gender'].transform([gender])[0],
            'Education': encoders['education'].transform([education])[0],
            'City': encoders['city'].transform([city])[0],
            'EmploymentType': encoders['employment'].transform([employment])[0]
        }])
        
        # Apply scaling if the model is Logistic Regression
        if best_model_name == "Logistic Regression" and scaler:
            input_data_processed = scaler.transform(input_data)
        else:
            input_data_processed = input_data
            
        prediction = model.predict(input_data_processed)[0]
        probability = model.predict_proba(input_data_processed)[0][1]
        
        if prediction == 1:
            st.success(f"✅ **LOAN APPROVED!** (Probability: {probability*100:.2f}%)")
            st.balloons()
        else:
            st.error(f"❌ **LOAN REJECTED!** (Approval Probability: {probability*100:.2f}%)")

st.markdown("---")
st.subheader("System Insights & Model Comparison")
tab1, tab2, tab3, tab4 = st.tabs(["Algorithm Comparison", "ROC Curves", "Feature Impact", "Risk Distribution"])

with tab1:
    try: st.image("output_comparison_accuracy.png", caption="Accuracy Comparison Across Models")
    except: st.write("Comparison image not found.")

with tab2:
    try: st.image("output_comparison_roc.png", caption="ROC Curve Performance")
    except: st.write("ROC image not found.")

with tab3:
    # Feature importance might vary, but we show for the best model if possible
    try: st.image("output_02_feature_importance.png", caption=f"Key Factors for {best_model_name}")
    except: st.write("Feature importance image not found.")

with tab4:
    try: st.image("output_04_distribution.png", caption="Dataset Distribution: Income vs Loan Amount")
    except: st.write("Distribution image not found.")

