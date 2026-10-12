import os
import sys
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException


load_dotenv()


DATABASE_URL = os.getenv("POSTGRES_SQL_URI")

if not DATABASE_URL:
    raise RuntimeError(
        "POSTGRES_SQL_URI environment variable is not configured."
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

INSERT_PREDICTION_SQL = text("""
    INSERT INTO prediction_logs (
        predicted_at,
        gender,
        "SeniorCitizen",
        "Partner",
        "Dependents",
        tenure,
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
        prediction,
        probability,
        model_version
    ) VALUES (
        :predicted_at,
        :gender,
        :SeniorCitizen,
        :Partner,
        :Dependents,
        :tenure,
        :PhoneService,
        :MultipleLines,
        :InternetService,
        :OnlineSecurity,
        :OnlineBackup,
        :DeviceProtection,
        :TechSupport,
        :StreamingTV,
        :StreamingMovies,
        :Contract,
        :PaperlessBilling,
        :PaymentMethod,
        :MonthlyCharges,
        :TotalCharges,
        :prediction,
        :probability,
        :model_version
    )
""")


def save_prediction(
    features: dict,
    prediction: int,
    probability: float,
    model_version: str,
) -> None:
    record = {
        **features,
        "predicted_at": datetime.now(timezone.utc),
        "prediction": prediction,
        "probability": probability,
        "model_version": model_version,
    }

    try:
        with engine.begin() as connection:
            connection.execute(INSERT_PREDICTION_SQL, record)

        logger.info("Prediction record saved successfully.")

    except Exception as e:
        logger.exception("Failed to save prediction record.")
        raise CustomerChurnException(e, sys, "prediction logger", "save prediction")

    
