import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Student Performance", page_icon="🎓")

try:
    model = joblib.load("student_model.pkl")
except FileNotFoundError:
    st.error("Model not found. Run train.py first.")
    st.stop()

st.title("🎓 Student Performance Predictor")
st.write("Predict the final grade (G3) of a student based on various factors.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic Info")
    g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
    g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], format_func=lambda x: [
        " < 2 hours", "2 to 5 hours", "5 to 10 hours", " > 10 hours"
    ][x-1])
    failures = st.number_input("Past Class Failures", 0, 4, 0)
    absences = st.number_input("School Absences", 0, 100, 0)

with col2:
    st.subheader("Lifestyle & Family")
    medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
    freetime = st.slider("Free Time after School (1-5)", 1, 5, 3)
    goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
    health = st.slider("Current Health Status (1-5)", 1, 5, 5)

if st.button("Predict Final Grade", type="primary", use_container_width=True):
    # Match the features in train.py: ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    # Some columns are set to defaults if not in UI
    input_data = pd.DataFrame([{
        'age': 18, 
        'Medu': medu,
        'Fedu': fedu,
        'traveltime': 1,
        'studytime': studytime,
        'failures': failures,
        'famrel': 4,
        'freetime': freetime,
        'goout': goout,
        'Dalc': 1,
        'Walc': 1,
        'health': health,
        'absences': absences,
        'G1': g1,
        'G2': g2
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"📈 **Predicted Final Grade (G3): {prediction:.2f} / 20**")

st.markdown("---")
st.subheader("Analysis Results")
try:
    st.image("output_screenshot_factors.png", caption="Key factors affecting students")
except:
    st.write("Visual insight not found.")
