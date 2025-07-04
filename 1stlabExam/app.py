import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('wine_quality_model.pkl')
scaler = joblib.load('wine_scaler.pkl')

st.title("Wine Quality Predictor 🍷")

# Input fields for chemical attributes
fields = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
          'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

user_input = []
for field in fields:
    value = st.number_input(f"{field.capitalize()}:", value=0.0, step=0.1)
    user_input.append(value)

# Predict button
if st.button('Predict Quality'):
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    result = "Good Quality 🍾" if prediction == 1 else "Not Good 😕"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
