import pandera.pandas as pa


class CustomerChurnSchema:

    @staticmethod
    def get_schema() -> pa.DataFrameSchema:

        categorical_features = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]

        binary_features = [
            "SeniorCitizen",
        ]

        numeric_features = [
            "tenure",
            "MonthlyCharges",
        ]

        categorical_checks = {
            "gender": ["Male", "Female"],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["Yes", "No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["Yes", "No", "No internet service"],
            "TechSupport": ["Yes", "No", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }

        columns = {
            "customerID": pa.Column(
                str,
                nullable=False,
                unique=True,
            ),

            "TotalCharges": pa.Column(
                str,
                nullable=False,
            ),

            "Churn": pa.Column(
                str,
                nullable=False,
                checks=pa.Check.isin(["Yes", "No"]),
            ),
        }

        for feature in categorical_features:
            columns[feature] = pa.Column(
                str,
                nullable=False,
                checks=pa.Check.isin(categorical_checks[feature]),
            )

        for feature in binary_features:
            columns[feature] = pa.Column(
                int,
                nullable=False,
                checks=pa.Check.isin([0, 1]),
            )

        columns["tenure"] = pa.Column(
            int,
            nullable=False,
            checks=pa.Check.ge(0),
        )

        columns["MonthlyCharges"] = pa.Column(
            float,
            nullable=False,
            checks=pa.Check.ge(0),
        )

        return pa.DataFrameSchema(
            columns=columns,
            strict=True,
        )