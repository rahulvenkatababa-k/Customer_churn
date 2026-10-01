import sys

from src.customer_churn.entity.config_entity import ModelEvaluationConfig
from src.customer_churn.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from src.customer_churn.utils.utils import load_object, save_object, read_csv
from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException

class ModelEvaluation:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact,
                 model_evaluation_config: ModelEvaluationConfig,
                 feature_engineering_artifact: FeatureEngineeringArtifact):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_config = model_evaluation_config
        self.feature_engineering_artifact = feature_engineering_artifact

    def _load_data(self):
        try:
            X_test = read_csv(self.feature_engineering_artifact.X_test_path)
            y_test = read_csv(self.feature_engineering_artifact.y_test_path)
            y_test = y_test.squeeze()

            model = load_object(self.model_trainer_artifact.model_path)

            return X_test, y_test, model
        
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Evaluation", "load data") from e

    def _make_predections(self, model, X_test):
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Trainer", "") from e

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Evaluation", "initiate model evaluation") from e