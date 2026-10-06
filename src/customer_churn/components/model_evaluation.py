import os
import sys
import json
import mlflow
import pandas as pd

from src.customer_churn.utils.ml_utils import (log_metrics, log_parameters)
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

            y_test = read_csv(
                self.feature_engineering_artifact.y_test_path
            ).squeeze()

            X_val = read_csv(
            self.feature_engineering_artifact.X_val_path
        )
            y_val = read_csv(
                self.feature_engineering_artifact.y_val_path
            ).squeeze()

            models = {
            "baseline": load_object(
                self.model_trainer_artifact.baseline_model_path
            )
        }


            for name, path in (
                self.model_trainer_artifact.candidate_model_paths.items()
            ):
                models[name] = load_object(path)

            # Load experiment results
            with open(
                self.model_trainer_artifact.experiment_results_path,
                "r",
                encoding='utf-8'
            ) as file:
                experiment_results = json.load(file)


            for name, result in experiment_results.items():
                models[name] = load_object(result["model_path"])

            return (
                X_val,
                y_val,
                X_test,
                y_test,
                models
            )

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

    def _calculate_metrics(self, y_test, y_pred, y_pred_proba, model_name):
        try:

            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
            recall = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
            f1 = f1_score(y_true=y_test, y_pred=y_pred,zero_division=0)
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

            logger.info("Model: %s", model_name)
            logger.info("Evaluation Metrics: %s", report)
            logger.info("Confusion Matrix:\n%s", cm)

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

    def _save_confusion_matrix(self, confusion_matrices):
        try:
            confusion_matrix_file_path = {}

            for model_name, confusion_matrix in confusion_matrices.items():
                file_path = os.path.join(self.model_evaluation_config.model_evaluation_confusion_matrix_dir,
                                         f"{model_name}.png")
                
                save_confusion_matrix(confusion_matrix=confusion_matrix,
                                    file_path= file_path,
                                    display_labels=["No Churn", "Churn"],
                                    title=f"Customer Churn {model_name} Confusion matrix",
                                    dpi = 300,
                                    bbox_inches = "tight"
                                    )
                confusion_matrix_file_path[model_name] = file_path

            return confusion_matrix_file_path
        
        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e, sys, "Model Evaluation", "save confusion matrix") from e

    
    def _create_model_summary(self, validation_report: dict) -> pd.DataFrame:
        """Create a validation metrics table and rank models by F1."""

        rows = []

        for model_name, metrics in validation_report.items():
            precision = metrics["precision"]
            recall = metrics["recall"]

            # F2 gives recall more weight than precision.
            denominator = (4 * precision) + recall
            f2_score = (
                (5 * precision * recall) / denominator
                if denominator > 0
                else 0.0
            )

            rows.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": precision,
                "recall": recall,
                "f1": metrics["f1"],
                "f2": f2_score,
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            })

        summary = pd.DataFrame(rows)

        summary = summary.sort_values(
            by=["f1", "recall", "average_precision"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        summary.insert(0, "rank", range(1, len(summary) + 1))

        return summary

    def initiate_model_evaluation(self):
        try:
            X_val, y_val, X_test, y_test, models = self._load_data()

            validation_report = {}
            confusion_matrices = {}

            # Evaluate every model on validation data.
            for model_name, model in models.items():
                y_pred, y_pred_proba = self._make_predictions(
                    model=model,
                    X_test=X_val,
                )

                report, cm = self._calculate_metrics(
                    y_pred_proba=y_pred_proba,
                    y_pred=y_pred,
                    y_test=y_val,
                    model_name=model_name,
                )

                validation_report[model_name] = report
                confusion_matrices[f"validation_{model_name}"] = cm

            
            # Generate and save the validation model comparison report.
            summary = self._create_model_summary(validation_report)

            report_path = self.model_evaluation_config.model_evaluation_report_path
            summary_path = os.path.splitext(report_path)[0] + "_validation_summary.csv"

            os.makedirs(os.path.dirname(summary_path), exist_ok=True)

            summary.to_csv(summary_path, index=False)

            logger.info(
                "\nVALIDATION MODEL RANKING\n%s",
                summary.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                ),
            )

            logger.info(
                "Validation summary saved to: %s",
                summary_path,
            )


            # Select using validation F1, then recall, then average precision.
            selected_model_name = max(
                validation_report,
                key=lambda name: (
                    validation_report[name]["f1"],
                    validation_report[name]["recall"],
                    validation_report[name]["average_precision"],
                ),
            )

            selected_model = models[selected_model_name]


            log_parameters({
                "selected_model": selected_model_name,
                "selection_metric" : "f1"
            })
            log_metrics({
                f"validation_{metric}" : float(value)
                for metric, value in validation_report[selected_model_name].items()
            })

            logger.info(
                "Selected model: %s | Validation F1: %.4f",
                selected_model_name,
                validation_report[selected_model_name]["f1"],
            )

            # Evaluate only the selected model on the held-out test set.
            y_pred, y_pred_proba = self._make_predictions(
                model=selected_model,
                X_test=X_test,
            )

            test_report, test_cm = self._calculate_metrics(
                y_pred_proba=y_pred_proba,
                y_pred=y_pred,
                y_test=y_test,
                model_name=f"final_test_{selected_model_name}",
            )
            log_metrics({
                    f"test_{metric}": float(value)
                    for metric, value in test_report.items()
                })

            confusion_matrices[
                f"final_test_{selected_model_name}"
            ] = test_cm

            evaluation_report = {
                "validation": validation_report,
                "selection": {
                    "selected_model": selected_model_name,
                    "selection_metric": "f1",
                    "tie_breakers": ["recall", "average_precision"],
                },
                "test": {
                    "model": selected_model_name,
                    "metrics": test_report,
                },
            }

            save_report = self._save_evaluation_report(
                report=evaluation_report
            )

            save_cm = self._save_confusion_matrix(
                confusion_matrices=confusion_matrices
            )

            return ModelEvaluationArtifact(
                save_report,
                save_cm,
            )

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Evaluation", "initiate model evaluation"
            ) from e