import sys

from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.utils.utils import (read_csv, save_object)
from src.customer_churn.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainerArtifact
from src.customer_churn.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    def __init__(self, feature_engineering_artifact: FeatureEngineeringArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.feature_engineering_artifact = feature_engineering_artifact
        self.model_trainer_config = model_trainer_config

    def _load_data(self):
        try:
            X_train = read_csv(self.feature_engineering_artifact.X_train_path)
            y_train = read_csv(self.feature_engineering_artifact.y_train_path)
            y_train = y_train.squeeze()
            
            return X_train, y_train
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Trainer", "load data") from e

    def _baseline_model(self, X_train, y_train,):
        try:
            lg_reg = LogisticRegression(random_state=self.model_trainer_config.random_state)
            lg_reg.fit(X_train, y_train)

            return lg_reg
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Trainer", "baseline model") from e

    def _save_model(self, model):
        try:
            save_object(model, self.model_trainer_config.model_path)

            return self.model_trainer_config.model_path
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Trainer", "save model") from e

    def initiate_model_trainer(self):
        try:
            X_train, y_train = self._load_data()
            model = self._baseline_model(X_train=X_train, y_train=y_train)
            saved_model = self._save_model(model=model)
            model_trainer_artifact = ModelTrainerArtifact(saved_model)

            return model_trainer_artifact

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Trainer", "initiate model trainer") from e