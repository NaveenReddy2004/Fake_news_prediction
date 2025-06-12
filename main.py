import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load(r"C:\Users\navee\Desktop\Fake_News\rf_model.pickel")
vectorizer = joblib.load(r"C:\Users\navee\Desktop\Fake_News\vectorizer.pickel")

# App Title
st.title("📰 Fake News Classifier (Random Forest)")

# User Input
user_input = st.text_area("Paste any news headline or article text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a news statement to predict.")
    else:
        # Transform the input
        input_vector = vectorizer.transform([user_input])

        # Get prediction and ensure it's int
        prediction = int(model.predict(input_vector)[0])

        # Get prediction confidence
        probability = model.predict_proba(input_vector)[0][prediction] * 100

        # Show prediction and confidence
        if prediction == 0:
            st.success(f"🟢 Predicted: **Real News** ({probability:.2f}% confidence)")
        else:
            st.error(f"🔴 Predicted: **Fake News** ({probability:.2f}% confidence)")