
import os

from sqlalchemy import create_engine, text

from src.customer_churn.entity.config_entity import ModelPusherConfig
from src.customer_churn.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.customer_churn.utils.ml_utils import promote_model_to_alias


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def _save_monitoring_baseline(
        self,
        model_version: str,
        reference_data_path: str,
    ) -> None:
        database_url = os.getenv("POSTGRES_SQL_URI")

        if not database_url:
            raise RuntimeError("POSTGRES_SQL_URI is not configured.")

        engine = create_engine(database_url, pool_pre_ping=True)

        query = text("""
            INSERT INTO model_monitoring_baselines (
                model_name,
                model_version,
                reference_data_path
            )
            VALUES (
                :model_name,
                :model_version,
                :reference_data_path
            )
            ON CONFLICT (model_name, model_version)
            DO UPDATE SET
                reference_data_path = EXCLUDED.reference_data_path
        """)

        try:
            with engine.begin() as connection:
                connection.execute(
                    query,
                    {
                        "model_name": self.model_pusher_config.model_name,
                        "model_version": str(model_version),
                        "reference_data_path": os.path.abspath(
                            reference_data_path
                        ),
                    },
                )
        finally:
            engine.dispose()

    def initiate_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:

        model_version = promote_model_to_alias(
            model_uri=model_evaluation_artifact.registered_model_uri,
            model_name=self.model_pusher_config.model_name,
            alias=self.model_pusher_config.model_alias,
        )

        self._save_monitoring_baseline(
            model_version=str(model_version),
            reference_data_path=(
                model_evaluation_artifact.reference_data_path
            ),
        )

        return ModelPusherArtifact(
            model_name=self.model_pusher_config.model_name,
            model_alias=self.model_pusher_config.model_alias,
            model_version=str(model_version),
        )
