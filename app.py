import streamlit as st
import pandas as pd
import tensorflow_decision_forests as tfdf

# App title and description
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")
st.markdown("This app uses **TensorFlow Decision Forests** to predict house prices based on a few key features.")

# Input form
with st.form("prediction_form"):
    st.header("Enter House Features:")
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1500)
    GarageCars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
    FullBath = st.slider("Number of Full Bathrooms", 0, 4, 2)
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
    submit = st.form_submit_button("Predict Price")

# Prediction logic
if submit:
    input_data = pd.DataFrame([{
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "FullBath": FullBath,
        "TotalBsmtSF": TotalBsmtSF
    }])

    # Load saved TFDF model from the current directory
    model = tfdf.keras.models.load_model("tfdf_house_price_model")

    # Predict
    prediction = model.predict(input_data)[0][0]

    # Show result
    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")
