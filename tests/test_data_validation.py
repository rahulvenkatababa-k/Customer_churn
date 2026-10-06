import pandas as pd
from src.customer_churn.components.data_validation import DataValidation

def create_valid_dataframe():
    return pd.DataFrame({
        "customerID": ["CUST001", "CUST002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [70.5, 85.2],
            "TotalCharges": ["846.0", "2044.8"],
            "Churn": ["Yes", "No"],
    })

def test_valid_dataframe_passes_schema():
    validator = DataValidation.__new__(DataValidation)

    dataframe = create_valid_dataframe()
    status, failure_cases = validator._schema_validation(dataframe)

    assert status is True
    assert failure_cases is None

def test_invalid_churn_value_fails_schema_validation():
    validator = DataValidation.__new__(DataValidation)

    dataframe = create_valid_dataframe()
    dataframe.loc[0, "Churn"] = "Maybe"

    status, failure_cases = validator._schema_validation(dataframe)

    assert status is False
    assert failure_cases is not None

def test_duplicate_customer_ids_fail_schema_validation():
    validator = DataValidation.__new__(DataValidation)

    dataframe = create_valid_dataframe()
    dataframe.loc[1, "customerID"] = "CUST001"

    status, failure_cases = validator._schema_validation(dataframe)

    assert status is False
    assert failure_cases is not None