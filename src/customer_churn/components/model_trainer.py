import os
import sys

from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.utils.utils import read_csv, save_object
from src.customer_churn.entity.artifact_entity import (
    FeatureEngineeringArtifact,
    ModelTrainerArtifact,
)
from src.customer_churn.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class ModelTrainer:
    def __init__(
        self,
        feature_engineering_artifact: FeatureEngineeringArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.feature_engineering_artifact = feature_engineering_artifact
        self.model_trainer_config = model_trainer_config

    def _load_data(self):
        try:
            X_train = read_csv(
                self.feature_engineering_artifact.X_train_path
            )

            y_train = read_csv(
                self.feature_engineering_artifact.y_train_path
            )

            y_train = y_train.squeeze()

            return X_train, y_train

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "load data"
            ) from e

    def _baseline_model(self, X_train, y_train):
        try:
            lg_reg = LogisticRegression(
                random_state=self.model_trainer_config.random_state
            )

            lg_reg.fit(X_train, y_train)

            return lg_reg

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "baseline model"
            ) from e

    def _candidate_models(self, X_train, y_train):
        try:
            models = {}

            # AdaBoost
            ada_boost = AdaBoostClassifier(
                random_state=self.model_trainer_config.random_state
            )

            ada_boost.fit(X_train, y_train)

            models["adaboost"] = ada_boost

            # XGBoost
            xgboost = XGBClassifier(
                random_state=self.model_trainer_config.random_state,
                eval_metric="logloss",
            )

            xgboost.fit(X_train, y_train)

            models["xgboost"] = xgboost

            # CatBoost
            catboost = CatBoostClassifier(
                random_seed=self.model_trainer_config.random_state,
                verbose=False,
            )

            catboost.fit(X_train, y_train)

            models["catboost"] = catboost

            return models

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "candidate models"
            ) from e

    def _save_model(self, model, model_name):
        try:
            model_directory = os.path.dirname(
                self.model_trainer_config.model_path
            )

            model_path = os.path.join(
                model_directory,
                f"{model_name}.pkl"
            )

            save_object(model, model_path)

            return model_path

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", f"save {model_name}"
            ) from e

    def initiate_model_trainer(self):
        try:
            X_train, y_train = self._load_data()

            # Baseline model
            baseline_model = self._baseline_model(
                X_train=X_train,
                y_train=y_train,
            )

            baseline_model_path = self._save_model(
                model=baseline_model,
                model_name="baseline_model",
            )

            # Candidate models
            candidate_models = self._candidate_models(
                X_train=X_train,
                y_train=y_train,
            )

            candidate_model_paths = {}

            for model_name, model in candidate_models.items():

                model_path = self._save_model(
                    model=model,
                    model_name=model_name,
                )

                candidate_model_paths[model_name] = model_path

            model_trainer_artifact = ModelTrainerArtifact(
                baseline_model_path=baseline_model_path,
                candidate_model_paths=candidate_model_paths,
            )

            return model_trainer_artifact

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Model Trainer","initiate model trainer",) from e