import json

import numpy as np
import pandas as pd
import pytest

from src.customer_churn.components.model_evaluation import ModelEvaluation
from src.customer_churn.exception.exception import CustomerChurnException


def make_evaluator(tmp_path):
    """Create an evaluator with temporary output paths."""

    class Config:
        model_evaluation_report_path = str(
            tmp_path / "evaluation" / "report.json"
        )
        model_evaluation_confusion_matrix_dir = str(
            tmp_path / "evaluation" / "confusion_matrices"
        )

    evaluator = ModelEvaluation(
        model_trainer_artifact=None,
        model_evaluation_config=Config(),
        feature_engineering_artifact=None,
    )
    return evaluator


def test_make_predictions_returns_labels_and_probabilities(tmp_path):
    evaluator = make_evaluator(tmp_path)

    class FakeModel:
        def predict(self, X):
            return np.array([0, 1, 1])

        def predict_proba(self, X):
            return np.array([
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
            ])

    predictions, probabilities = evaluator._make_predictions(
        FakeModel(), pd.DataFrame({"feature": [1, 2, 3]})
    )

    np.testing.assert_array_equal(predictions, [0, 1, 1])
    np.testing.assert_allclose(probabilities, [0.1, 0.8, 0.7])


def test_make_predictions_wraps_model_failure(tmp_path):
    evaluator = make_evaluator(tmp_path)

    class BrokenModel:
        def predict(self, X):
            raise ValueError("Synthetic prediction failure")

    with pytest.raises(CustomerChurnException):
        evaluator._make_predictions(
            BrokenModel(), pd.DataFrame({"feature": [1]})
        )


def test_calculate_metrics_returns_expected_metrics_and_confusion_matrix(
    tmp_path,
):
    evaluator = make_evaluator(tmp_path)

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.7, 0.8, 0.9])

    report, matrix = evaluator._calculate_metrics(
        y_test=y_true,
        y_pred=y_pred,
        y_pred_proba=y_proba,
        model_name="test_model",
    )

    assert report["accuracy"] == pytest.approx(0.75)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["recall"] == pytest.approx(1.0)
    assert report["f1"] == pytest.approx(0.8)
    assert 0 <= report["roc_auc"] <= 1
    assert 0 <= report["average_precision"] <= 1

    np.testing.assert_array_equal(
        matrix,
        np.array([[1, 1], [0, 2]]),
    )
    assert all(isinstance(value, float) for value in report.values())


def test_calculate_metrics_wraps_invalid_input(tmp_path):
    evaluator = make_evaluator(tmp_path)

    with pytest.raises(CustomerChurnException):
        evaluator._calculate_metrics(
            y_test=[0, 1],
            y_pred=[0],
            y_pred_proba=[0.1, 0.9],
            model_name="invalid_model",
        )


def test_create_model_summary_ranks_by_f1_then_recall(tmp_path):
    evaluator = make_evaluator(tmp_path)

    validation_report = {
        "model_a": {
            "accuracy": 0.8, "precision": 0.7, "recall": 0.6,
            "f1": 0.65, "roc_auc": 0.8, "average_precision": 0.7,
        },
        "model_b": {
            "accuracy": 0.75, "precision": 0.6, "recall": 0.8,
            "f1": 0.65, "roc_auc": 0.82, "average_precision": 0.75,
        },
        "model_c": {
            "accuracy": 0.7, "precision": 0.5, "recall": 0.5,
            "f1": 0.5, "roc_auc": 0.7, "average_precision": 0.6,
        },
    }

    summary = evaluator._create_model_summary(validation_report)

    assert summary["model"].tolist() == [
        "model_b", "model_a", "model_c"
    ]
    assert summary["rank"].tolist() == [1, 2, 3]
    assert "f2" in summary.columns


def test_create_model_summary_handles_zero_precision_and_recall(tmp_path):
    evaluator = make_evaluator(tmp_path)

    report = {
        "zero_model": {
            "accuracy": 0.5, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "roc_auc": 0.5, "average_precision": 0.5,
        }
    }

    summary = evaluator._create_model_summary(report)

    assert summary.loc[0, "f2"] == 0.0
    assert summary.loc[0, "rank"] == 1


def test_save_evaluation_report_creates_json(tmp_path):
    evaluator = make_evaluator(tmp_path)
    report = {
        "selection": {"selected_model": "baseline"},
        "test": {"metrics": {"f1": 0.75}},
    }

    path = evaluator._save_evaluation_report(report)

    with open(path, encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


def test_save_evaluation_report_wraps_save_failure(tmp_path, monkeypatch):
    evaluator = make_evaluator(tmp_path)

    def fail_save(*args, **kwargs):
        raise OSError("Synthetic report save failure")

    monkeypatch.setattr(
        "src.customer_churn.components.model_evaluation.save_json",
        fail_save,
    )

    with pytest.raises(CustomerChurnException):
        evaluator._save_evaluation_report({"test": {}})


def test_save_confusion_matrix_saves_each_model(tmp_path, monkeypatch):
    evaluator = make_evaluator(tmp_path)
    saved = []

    def fake_save_confusion_matrix(**kwargs):
        saved.append(kwargs)

    monkeypatch.setattr(
        "src.customer_churn.components.model_evaluation.save_confusion_matrix",
        fake_save_confusion_matrix,
    )

    matrices = {
        "validation_baseline": np.array([[3, 1], [2, 4]]),
        "final_test_baseline": np.array([[4, 1], [1, 4]]),
    }

    paths = evaluator._save_confusion_matrix(matrices)

    assert set(paths) == set(matrices)
    assert len(saved) == 2
    assert all(path.endswith(".png") for path in paths.values())
    assert all(
        item["display_labels"] == ["No Churn", "Churn"]
        for item in saved
    )


def test_save_confusion_matrix_wraps_failure(tmp_path, monkeypatch):
    evaluator = make_evaluator(tmp_path)

    def fail_save(**kwargs):
        raise OSError("Synthetic image save failure")

    monkeypatch.setattr(
        "src.customer_churn.components.model_evaluation.save_confusion_matrix",
        fail_save,
    )

    with pytest.raises(CustomerChurnException):
        evaluator._save_confusion_matrix({
            "baseline": np.array([[1, 0], [0, 1]])
        })


def test_initiate_model_evaluation_selects_best_model(
    tmp_path, monkeypatch
):
    import pandas as pd

    from src.customer_churn.entity.artifact_entity import (
        ModelEvaluationArtifact,
    )

    evaluator = make_evaluator(tmp_path)

    models = {
        "model_a": object(),
        "model_b": object(),
    }

    X_val = pd.DataFrame({"feature": [1, 2]})
    y_val = pd.Series([0, 1])
    X_test = pd.DataFrame({"feature": [3, 4]})
    y_test = pd.Series([0, 1])

    monkeypatch.setattr(
        evaluator,
        "_load_data",
        lambda: (X_val, y_val, X_test, y_test, models),
    )

    prediction_calls = []

    def fake_predictions(model, X_test):
        prediction_calls.append((model, X_test))

        if X_test is X_val:
            if model is models["model_a"]:
                return [0, 1], [0.2, 0.8]
            return [0, 0], [0.3, 0.4]

        return [0, 1], [0.1, 0.9]

    monkeypatch.setattr(
        evaluator, "_make_predictions", fake_predictions
    )

def fake_metrics(
    y_pred_proba, y_pred, y_test, model_name
):
    if list(y_pred) == [0, 1]:
        report = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "average_precision": 1.0,
        }
        cm = [[1, 0], [0, 1]]
    else:
        report = {
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "average_precision": 0.5,
        }
        cm = [[1, 0], [1, 0]]

    return report, cm

    monkeypatch.setattr(
        evaluator, "_calculate_metrics", fake_metrics
    )
    monkeypatch.setattr(
        evaluator, "_save_evaluation_report", lambda report: None
    )
    monkeypatch.setattr(
        evaluator, "_save_confusion_matrix", lambda matrices: None
    )

    logged_parameters = {}
    logged_metrics = {}

    monkeypatch.setattr(
        "src.customer_churn.components.model_evaluation.log_parameters",
        lambda params: logged_parameters.update(params),
    )
    monkeypatch.setattr(
        "src.customer_churn.components.model_evaluation.log_metrics",
        lambda metrics: logged_metrics.update(metrics),
    )

    artifact = evaluator.initiate_model_evaluation()

    assert isinstance(artifact, ModelEvaluationArtifact)
    assert logged_parameters["selected_model"] == "model_a"
    assert logged_parameters["selection_metric"] == "f1"
    assert logged_metrics["test_f1"] == 1.0

    # Both candidates use validation data; only the winner uses test data.
    assert sum(X is X_val for _, X in prediction_calls) == 2
    assert sum(X is X_test for _, X in prediction_calls) == 1
    assert prediction_calls[-1][0] is models["model_a"]
