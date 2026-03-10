import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="House Price Prediction", layout="wide")

try:
    model = joblib.load("house_rf_model.pkl")
except FileNotFoundError:
    st.error("Model not found! Please run train.py first to generate house_rf_model.pkl.")
    st.stop()

st.title("🏠 House Price Prediction System")
st.markdown("Predict the selling price of a house using Machine Learning (Random Forest Regressor).")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Home Details")
    bedrooms = st.number_input("Bedrooms", value=3, step=1)
    bathrooms = st.number_input("Bathrooms", value=2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", value=2000, step=50)
    sqft_lot = st.number_input("Lot Size (sqft)", value=5000, step=100)
    floors = st.number_input("Floors", value=1.0, step=0.5)
    
    st.header("Condition & Year")
    waterfront = st.selectbox("Waterfront View?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
    condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
    
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1990)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=0)

    sqft_above = st.number_input("Square Footage Above Ground", value=1800, step=50)
    sqft_basement = st.number_input("Basement Square Footage", value=200, step=50)

with col2:
    st.subheader("Current Date Info")
    sold_date = st.date_input("Listing Date", datetime.date.today())
    year_sold = sold_date.year
    month_sold = sold_date.month

    st.markdown("---")
    st.subheader("Price Prediction")
    
    # Input matching train.py DataFrame columns order precisely (check train.py columns)
    # The columns dropped were: price, date, street, city, statezip, country.
    # The columns kept:
    # bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, year_sold, month_sold
    
    input_df = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'year_sold': year_sold,
        'month_sold': month_sold
    }])
    
    if st.button("Predict House Price", type="primary", use_container_width=True):
        pred_price = model.predict(input_df)[0]
        st.success(f"🏡 **Predicted Selling Price:** ${pred_price:,.2f}")
        
    st.markdown("---")
    st.subheader("Model Insights")
    try:
        st.image("output_screenshot_feature_importance.png", caption="Feature Importance for Selling Price")
    except:
        st.write("Feature importance graph hidden.")
