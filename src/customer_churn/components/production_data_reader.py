import pandas as pd
from sqlalchemy import text

from src.customer_churn.components.prediction_logger import engine


FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]


def load_production_features(
    model_version : str,
    limit: int = 1000,
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be greater than zero.")

    columns_sql = ", ".join(
        f'"{column}"' if column[0].isupper() else column
        for column in FEATURE_COLUMNS
    )

    query = text(
        f"""
        SELECT {columns_sql}
        FROM (
            SELECT *
            FROM prediction_logs
            WHERE model_version = :model_version
            ORDER BY predicted_at DESC, id DESC
            LIMIT :limit
        ) AS recent_predictions
        ORDER BY predicted_at, id
        """
    )

    with engine.connect() as connection:
        dataframe = pd.read_sql(
        query,
        connection,
        params={
            "limit": limit,
            "model_version": model_version,
        },
    )

    if dataframe.empty:
        raise ValueError(
            "No production prediction records are available."
        )

    return dataframe[FEATURE_COLUMNS]