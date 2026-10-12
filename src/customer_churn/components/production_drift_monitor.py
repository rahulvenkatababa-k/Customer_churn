
import os
from pathlib import Path

import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.customer_churn.components.production_data_reader import (
    load_production_features,
)
from src.customer_churn.configuration.configuration import config


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evidently"
    / "production_drift_report.html"
)

MINIMUM_PRODUCTION_ROWS = 100
MODEL_NAME = "customer_churn_model"
MODEL_ALIAS = "champion"


def get_active_model_version() -> str:
    """Resolve the champion alias to its exact MLflow model version."""
    client = MlflowClient(
        tracking_uri=config.mlflow_tracking_uri
    )

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    return str(model_version.version)


def get_reference_data_path(model_version: str) -> Path:
    """Retrieve the training reference path associated with this version."""
    load_dotenv()
    database_url = os.getenv("POSTGRES_SQL_URI")

    if not database_url:
        raise RuntimeError("POSTGRES_SQL_URI is not configured.")

    engine = create_engine(database_url, pool_pre_ping=True)

    query = text("""
        SELECT reference_data_path
        FROM model_monitoring_baselines
        WHERE model_name = :model_name
          AND model_version = :model_version
    """)

    try:
        with engine.connect() as connection:
            reference_path = connection.execute(
                query,
                {
                    "model_name": MODEL_NAME,
                    "model_version": model_version,
                },
            ).scalar_one_or_none()
    finally:
        engine.dispose()

    if not reference_path:
        raise FileNotFoundError(
            f"No monitoring baseline registered for "
            f"{MODEL_NAME} version {model_version}."
        )

    path = Path(reference_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Training reference file does not exist: {path}"
        )

    return path


def run_production_drift_check() -> dict:
    model_version = get_active_model_version()
    reference_path = get_reference_data_path(model_version)

    reference_data = pd.read_csv(reference_path)
    current_data = load_production_features(model_version=model_version)

    if list(reference_data.columns) != list(current_data.columns):
        raise ValueError(
            "Training and production feature schemas do not match."
        )

    if len(current_data) < MINIMUM_PRODUCTION_ROWS:
        result = {
            "status": "insufficient_data",
            "model_version": model_version,
            "reference_rows": len(reference_data),
            "production_rows": len(current_data),
            "minimum_required": MINIMUM_PRODUCTION_ROWS,
        }
        print(result)
        return result

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    result = {
        "status": "completed",
        "model_version": model_version,
        "reference_rows": len(reference_data),
        "production_rows": len(current_data),
        "reference_data_path": str(reference_path),
        "report_path": str(REPORT_PATH),
    }

    print(result)
    return result


if __name__ == "__main__":
    run_production_drift_check()
