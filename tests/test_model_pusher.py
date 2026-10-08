import pytest

from src.customer_churn.components.model_pusher import ModelPusher
from src.customer_churn.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.customer_churn.entity.config_entity import ModelPusherConfig


def test_initiate_model_pusher_returns_artifact(monkeypatch):

    captured = {}

    def mock_promote_model_to_alias(
        model_uri,
        model_name,
        alias,
    ):
        captured["model_uri"] = model_uri
        captured["model_name"] = model_name
        captured["alias"] = alias

        return "3"

    monkeypatch.setattr(
        "src.customer_churn.components.model_pusher.promote_model_to_alias",
        mock_promote_model_to_alias,
    )

    model_evaluation_artifact = ModelEvaluationArtifact(
        evaluation_report_file_path="evaluation_report.json",
        confusion_matrix_file_paths={},
        registered_model_uri="models:/m-test123",
    )

    model_pusher = ModelPusher(
        model_pusher_config=ModelPusherConfig()
    )

    artifact = model_pusher.initiate_model_pusher(
        model_evaluation_artifact=model_evaluation_artifact
    )

    assert isinstance(artifact, ModelPusherArtifact)

    assert artifact.model_name == "customer_churn_model"
    assert artifact.model_alias == "champion"
    assert artifact.model_version == "3"

    assert captured["model_uri"] == "models:/m-test123"
    assert captured["model_name"] == "customer_churn_model"
    assert captured["alias"] == "champion"


def test_initiate_model_pusher_propagates_failure(monkeypatch):

    def mock_promote_model_to_alias(
        model_uri,
        model_name,
        alias,
    ):
        raise RuntimeError("MLflow promotion failed")

    monkeypatch.setattr(
        "src.customer_churn.components.model_pusher.promote_model_to_alias",
        mock_promote_model_to_alias,
    )

    model_evaluation_artifact = ModelEvaluationArtifact(
        evaluation_report_file_path="evaluation_report.json",
        confusion_matrix_file_paths={},
        registered_model_uri="models:/m-test123",
    )

    model_pusher = ModelPusher(
        model_pusher_config=ModelPusherConfig()
    )

    with pytest.raises(Exception):
        model_pusher.initiate_model_pusher(
            model_evaluation_artifact=model_evaluation_artifact
        )