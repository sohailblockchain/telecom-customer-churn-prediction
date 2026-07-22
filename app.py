from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/churn_model.joblib")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("Telecom Customer Churn Prediction")
st.write(
    "Enter customer information to estimate whether the customer "
    "is at risk of leaving the telecom service."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.slider("Tenure in Months", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"],
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"],
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"],
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"],
)

tech_support = st.selectbox(
    "Technical Support",
    ["Yes", "No", "No internet service"],
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"],
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"],
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=840.0,
)

customer = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)

if st.button("Predict Churn Risk"):
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    if prediction == 1:
        st.warning(
            "This customer has a relatively high risk of leaving."
        )
    else:
        st.success(
            "This customer is currently predicted to remain."
        )