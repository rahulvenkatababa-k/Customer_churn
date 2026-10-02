import sys
from src.customer_churn.entity.config_entity import ModelEvaluationConfig
from src.customer_churn.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, FeatureEngineeringArtifact
from src.customer_churn.utils.utils import load_object, read_csv, save_json, save_confusion_matrix
from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score,average_precision_score, confusion_matrix)

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

            y_test = read_csv(self.feature_engineering_artifact.y_test_path).squeeze()

            baseline_model = load_object(self.model_trainer_artifact.baseline_model_path)

            candidate_models = {}

            for model_name, model_path in (self.model_trainer_artifact.candidate_model_paths.items()):
                candidate_models[model_name] = load_object(model_path)

            return X_test, y_test, baseline_model, candidate_models

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e, sys, "Model Evaluation", "load data") from e

    def _make_predictions(self, model, X_test):
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:,1]

            return y_pred, y_pred_proba
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Evaluation", "make predictions") from e

    def _calculate_metrics(self, y_test, y_pred, y_pred_proba):
        try:

            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred)
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred_proba)
            average_precision = average_precision_score(y_true=y_test,y_score=y_pred_proba)

            cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

            report = {
                "accuracy" : float(accuracy),
                "precision" : float(precision),
                "recall" : float(recall),
                "f1" : float(f1),
                "roc_auc" : float(roc_auc),
                "average_precision" : float(average_precision)
            }

            print (report)
            logger.info(report)
            print(cm)
            logger.info(cm)

            return report, cm

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Evaluation", "calculate metrics") from e

    def _save_evaluation_report(self, report) :
        try:
            save_json(obj=report, file_path= self.model_evaluation_config.model_evaluation_report_path, indent = 4)

            return self.model_evaluation_config.model_evaluation_report_path

        except Exception as e:
            logger.error("%s",e)
            raise CustomerChurnException(e,sys, "Model Evaluation" , "save evaluation model") from e

    def _save_confusion_matrix(self, cm):
        try:
            save_confusion_matrix(confusion_matrix=cm,
                                  file_path= self.model_evaluation_config.model_evaluation_confusion_matrix_path,
                                  display_labels=["No Churn", "Churn"],
                                  title="Customer Churn Confusion matrix",
                                  dpi = 300,
                                  bbox_inches = "tight"
                                  )

            return self.model_evaluation_config.model_evaluation_confusion_matrix_path
        
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e, sys, "Model Evaluation", "save confusion matrix") from e
    def initiate_model_evaluation(self):
        try:
            (
                X_test,
                y_test,
                baseline_model,
                candidate_models,
            ) = self._load_data()

            evaluation_report = {}
            confusion_matrices = {}

            # Evaluate baseline
            y_pred, y_pred_proba = self._make_predictions(
                model=baseline_model,
                X_test=X_test,
            )

            report, cm = self._calculate_metrics(
                y_pred_proba=y_pred_proba,
                y_pred=y_pred,
                y_test=y_test,
            )

            evaluation_report["baseline"] = report
            confusion_matrices["baseline"] = cm

            # Evaluate candidate models
            for model_name, model in candidate_models.items():

                y_pred, y_pred_proba = self._make_predictions(
                    model=model,
                    X_test=X_test,
                )

                report, cm = self._calculate_metrics(
                    y_pred_proba=y_pred_proba,
                    y_pred=y_pred,
                    y_test=y_test,
                )

                evaluation_report[model_name] = report
                confusion_matrices[model_name] = cm

            save_report = self._save_evaluation_report(
                report=evaluation_report
            )

            # For now, save the confusion matrix for each model
            save_cm = self._save_confusion_matrix(
                confusion_matrices=confusion_matrices
            )

            return ModelEvaluationArtifact(
                save_report,
                save_cm,
            )

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys, "Model Evaluation", "initiate model evaluation") from e