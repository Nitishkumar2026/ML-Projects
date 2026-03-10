import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Walmart Sales Prediction", layout="wide")

# Load the trained model
try:
    model = joblib.load("rf_model.pkl")
except FileNotFoundError:
    st.error("Model not found! Please run train.py first.")
    st.stop()

st.title("📈 Walmart Weekly Sales Prediction System")
st.markdown("Predict future weekly sales for a Walmart store using Machine Learning (Random Forest Regressor).")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Features")
    store = st.number_input("Store ID (1-45)", min_value=1, max_value=45, value=1)
    date_input = st.date_input("Date", datetime.date(2012, 11, 1))

    holiday_flag = st.selectbox("Is it a Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temperature = st.number_input("Temperature (°F)", value=68.0, step=1.0)
    fuel_price = st.number_input("Fuel Price ($)", value=3.2, step=0.1)
    cpi = st.number_input("Consumer Price Index (CPI)", value=211.0, step=1.0)
    unemployment = st.number_input("Unemployment Rate (%)", value=7.5, step=0.1)

    # Process date
    year = date_input.year
    month = date_input.month
    week = date_input.isocalendar()[1]

with col2:
    st.subheader("Model Prediction")
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [year],
        'Month': [month],
        'Week': [week]
    })
    
    st.write("Input Data Summary:")
    st.dataframe(input_data, hide_index=True)

    if st.button("Predict Sales", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        st.success(f"🛒 **Predicted Weekly Sales: ${prediction:,.2f}**")
        
        st.info("💡 Note: The prediction is based on historical Walmart data.")

    st.markdown("---")
    st.subheader("Feature Importance")
    try:
        st.image("output_screenshot_importance.png", use_container_width=True)
    except FileNotFoundError:
        st.write("Feature importance graph not found.")
