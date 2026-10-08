from src.customer_churn.entity.config_entity import ModelPusherConfig
from src.customer_churn.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from src.customer_churn.utils.ml_utils import promote_model_to_alias


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:

        model_version = promote_model_to_alias(
            model_uri=model_evaluation_artifact.registered_model_uri,
            model_name=self.model_pusher_config.model_name,
            alias=self.model_pusher_config.model_alias,
        )

        return ModelPusherArtifact(
            model_name=self.model_pusher_config.model_name,
            model_alias=self.model_pusher_config.model_alias,
            model_version=model_version,
        )