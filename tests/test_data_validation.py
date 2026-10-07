import json
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


def test_valid_dataframe_is_saved(tmp_path):
    validatior = DataValidation.__new__(DataValidation)
    validatior.data_validation_config = type(
        "config",
        (),
        {
            "validated_data_dir" : str(tmp_path/"validated"),
            "validated_data_path" : str(tmp_path/"validated"/"customer_churn.csv")
        }
    )()

    dataframe = create_valid_dataframe()
    saved_path = validatior._save_validated_dataframe(dataframe)

    assert saved_path == validatior.data_validation_config.validated_data_path
    saved_dataframe = pd.read_csv(saved_path, dtype={"TotalCharges":str})
    pd.testing.assert_frame_equal(saved_dataframe, dataframe, check_dtype=False)

def test_successful_validation_report_is_generated(tmp_path):
    validator = DataValidation.__new__(DataValidation)

    validator.data_validation_config = type(
        "Config",
        (),
        {
            "data_validation_dir": str(tmp_path / "validation"),
            "validation_report": str(
                tmp_path / "validation" / "validation_report.json"
            ),
        },
    )()

    report_path = validator._generate_validation_report(
        validation_status=True,
        failure_cases=None,
    )

    with open(report_path, "r") as file:
        report = json.load(file)

    assert report["validation_status"] is True
    assert report["failure_cases"] == []


def test_failed_validation_report_contains_failure_cases(tmp_path):
    validator = DataValidation.__new__(DataValidation)

    validator.data_validation_config = type(
        "Config",
        (),
        {
            "data_validation_dir": str(tmp_path / "validation"),
            "validation_report": str(
                tmp_path / "validation" / "validation_report.json"
            ),
        },
    )()

    failure_cases = pd.DataFrame(
        {
            "column": ["Churn"],
            "failure_case": ["Maybe"],
        }
    )

    report_path = validator._generate_validation_report(
        validation_status=False,
        failure_cases=failure_cases,
    )

    with open(report_path, "r") as file:
        report = json.load(file)

    assert report["validation_status"] is False
    assert report["failure_cases"] == failure_cases.to_dict(orient="records")