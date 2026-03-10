import streamlit as st
import joblib

st.set_page_config(page_title="Spam Detector", page_icon="📧")

try:
    model = joblib.load("spam_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Model/Vectorizer files not found. Run train.py first.")
    st.stop()

st.title("📧 Email Spam Detection System")
st.write("Enter the content of an email below to check if it's Spam or Not.")

email_text = st.text_area("Email Content Here", height=200, placeholder="Dear customer, you won a lottery...")

if st.button("Check Email", type="primary"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize
        vectorized_text = tfidf.transform([email_text])
        # Predict
        prediction = model.predict(vectorized_text)[0]
        
        if prediction.lower() == 'spam':
            st.error(f"🚨 **PREDICTION: SPAM**")
        else:
            st.success(f"✅ **PREDICTION: NOT SPAM / HAM**")

st.markdown("---")
st.subheader("Model Performance Visualization")
try:
    st.image("output_screenshot_confusion_matrix.png", caption="Confusion Matrix of the Classifer")
except:
    st.write("Confusion matrix plot not available.")
