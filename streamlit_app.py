import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv(
    "PREDICTION_API_URL",
    "http://127.0.0.1:8000/predict",
)

GENDER_OPTIONS = ["Female", "Male"]
YES_NO_OPTIONS = ["No", "Yes"]

MULTIPLE_LINES_OPTIONS = [
    "No",
    "No phone service",
    "Yes",
]

INTERNET_SERVICE_OPTIONS = [
    "DSL",
    "Fiber optic",
    "No",
]

SERVICE_OPTIONS = [
    "No",
    "No internet service",
    "Yes",
]

CONTRACT_OPTIONS = [
    "Month-to-month",
    "One year",
    "Two year",
]

PAYMENT_METHOD_OPTIONS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]


def main():
    st.set_page_config(
        page_title="Churn Prediction",
        layout="centered",
    )

    st.title("Telecom Customer Churn Prediction 🛒")

    st.markdown(
        "### Provide customer details to get churn prediction"
    )

    with st.form("input_form"):

        gender = st.selectbox(
            "Gender",
            GENDER_OPTIONS,
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
        )

        partner = st.selectbox(
            "Partner",
            YES_NO_OPTIONS,
        )

        dependents = st.selectbox(
            "Dependents",
            YES_NO_OPTIONS,
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
        )

        phone_service = st.selectbox(
            "Phone Service",
            YES_NO_OPTIONS,
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            MULTIPLE_LINES_OPTIONS,
        )

        internet_service = st.selectbox(
            "Internet Service",
            INTERNET_SERVICE_OPTIONS,
        )

        online_security = st.selectbox(
            "Online Security",
            SERVICE_OPTIONS,
        )

        online_backup = st.selectbox(
            "Online Backup",
            SERVICE_OPTIONS,
        )

        device_protection = st.selectbox(
            "Device Protection",
            SERVICE_OPTIONS,
        )

        tech_support = st.selectbox(
            "Tech Support",
            SERVICE_OPTIONS,
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            SERVICE_OPTIONS,
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            SERVICE_OPTIONS,
        )

        contract = st.selectbox(
            "Contract",
            CONTRACT_OPTIONS,
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            YES_NO_OPTIONS,
        )

        payment_method = st.selectbox(
            "Payment Method",
            PAYMENT_METHOD_OPTIONS,
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=70.0,
        )

        submitted = st.form_submit_button("Predict")

    if submitted:

        user_input = {
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

        try:
            response = requests.post(
                API_URL,
                json=user_input,
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            prediction = result["Prediction"]
            probability = result["Probability"]

            st.write("### Prediction Result")

            churn_text = "Yes" if prediction == 1 else "No"

            st.write(f"**Churn:** {churn_text}")
            st.write(
                f"**Churn probability:** {probability:.2%}"
            )

            if prediction == 1:
                st.warning("Customer is likely to churn ⚠️")
            else:
                st.success("Customer is unlikely to churn ✅")

        except requests.exceptions.RequestException as error:
            st.error(
                f"Unable to connect to prediction API: {error}"
            )


if __name__ == "__main__":
    main()