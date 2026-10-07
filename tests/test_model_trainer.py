import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from src.customer_churn.components.model_trainer import ModelTrainer
from src.customer_churn.entity.artifact_entity import FeatureEngineeringArtifact
from src.customer_churn.exception.exception import CustomerChurnException


@pytest.fixture
def trainer(tmp_path):
    """Create a trainer with temporary feature dataset paths."""
    X_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"

    X_train = pd.DataFrame({
        "tenure": [1, 3, 6, 12, 18, 24, 30, 36],
        "MonthlyCharges": [30.0, 45.0, 55.0, 60.0, 75.0, 80.0, 90.0, 95.0],
        "TotalCharges": [30.0, 135.0, 330.0, 720.0, 1350.0, 1920.0, 2700.0, 3420.0],
        "SeniorCitizen": [0, 0, 1, 0, 1, 0, 1, 0],
        "Contract": [
            "Month-to-month", "One year", "Two year", "Month-to-month",
            "One year", "Two year", "Month-to-month", "One year",
        ],
        "InternetService": [
            "DSL", "Fiber optic", "No", "DSL",
            "Fiber optic", "No", "DSL", "Fiber optic",
        ],
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="Churn")

    X_train.to_csv(X_train_path, index=False)
    y_train.to_frame().to_csv(y_train_path, index=False)

    artifact = FeatureEngineeringArtifact(
        X_train_path=str(X_train_path),
        y_train_path=str(y_train_path),
        X_val_path=str(tmp_path / "X_val.csv"),
        y_val_path=str(tmp_path / "y_val.csv"),
        X_test_path=str(tmp_path / "X_test.csv"),
        y_test_path=str(tmp_path / "y_test.csv"),
    )

    # ModelTrainerConfig is not needed for these focused tests.
    class Config:
        random_state = 42
        model_path = str(tmp_path / "models" / "model.pkl")
        model_trainer_dir = str(tmp_path / "model_trainer")

    return ModelTrainer(artifact, Config()), X_train, y_train


def test_create_preprocessor_has_expected_transformers(trainer):
    model_trainer, X_train, _ = trainer

    preprocessor = model_trainer._create_preprocessor(X_train)

    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = {
        name for name, _, _ in preprocessor.transformers
    }
    assert transformer_names == {"numerical", "binary", "categorical"}

    transformer_map = {
        name: (transformer, columns)
        for name, transformer, columns in preprocessor.transformers
    }

    assert transformer_map["numerical"][1] == [
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    assert transformer_map["binary"][1] == ["SeniorCitizen"]
    assert transformer_map["binary"][0] == "passthrough"
    assert transformer_map["categorical"][1] == [
        "Contract", "InternetService"
    ]
    assert isinstance(transformer_map["numerical"][0], StandardScaler)
    assert isinstance(
        transformer_map["categorical"][0], OneHotEncoder
    )


def test_preprocessor_transforms_features(trainer):
    model_trainer, X_train, _ = trainer

    preprocessor = model_trainer._create_preprocessor(X_train)
    transformed = preprocessor.fit_transform(X_train)

    # 3 numerical + 1 binary + 3 Contract categories + 3 InternetService categories
    assert transformed.shape == (len(X_train), 10)
    assert np.isfinite(transformed).all()


def test_preprocessor_ignores_unseen_categories(trainer):
    model_trainer, X_train, _ = trainer

    preprocessor = model_trainer._create_preprocessor(X_train)
    preprocessor.fit(X_train)

    new_row = X_train.iloc[[0]].copy()
    new_row["Contract"] = "Unexpected contract"

    transformed = preprocessor.transform(new_row)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_load_data_returns_features_and_target(trainer):
    model_trainer, X_train, y_train = trainer

    loaded_X, loaded_y = model_trainer._load_data()

    pd.testing.assert_frame_equal(loaded_X, X_train)
    pd.testing.assert_series_equal(loaded_y, y_train)


def test_load_data_wraps_read_failure(trainer, monkeypatch):
    model_trainer, _, _ = trainer

    def fail_read_csv(path):
        raise OSError("Unable to read dataset")

    monkeypatch.setattr(
        "src.customer_churn.components.model_trainer.read_csv",
        fail_read_csv,
    )

    with pytest.raises(CustomerChurnException):
        model_trainer._load_data()


def test_baseline_model_is_fitted_pipeline(trainer):
    model_trainer, X_train, y_train = trainer

    model = model_trainer._baseline_model(X_train, y_train)

    assert isinstance(model, SklearnPipeline)
    assert isinstance(model.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(model.named_steps["model"], LogisticRegression)
    assert hasattr(model.named_steps["model"], "coef_")

    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert set(predictions).issubset({0, 1})


def test_baseline_model_wraps_training_failure(trainer, monkeypatch):
    model_trainer, X_train, y_train = trainer

    def fail_fit(*args, **kwargs):
        raise ValueError("Synthetic training failure")

    monkeypatch.setattr(
        "src.customer_churn.components.model_trainer.Pipeline.fit",
        fail_fit,
    )

    with pytest.raises(CustomerChurnException):
        model_trainer._baseline_model(X_train, y_train)


def test_calculate_sample_weights_balances_classes(trainer):
    model_trainer, _, y_train = trainer

    imbalanced_y = pd.Series([0, 0, 0, 0, 1, 1])

    weights = model_trainer._calculate_sample_weights(imbalanced_y)

    assert len(weights) == len(imbalanced_y)
    assert np.all(weights > 0)
    assert weights[imbalanced_y == 0].sum() == pytest.approx(
        weights[imbalanced_y == 1].sum()
    )


def test_calculate_sample_weights_wraps_failure(trainer, monkeypatch):
    model_trainer, _, y_train = trainer

    def fail_weights(*args, **kwargs):
        raise ValueError("Synthetic weighting failure")

    monkeypatch.setattr(
        "src.customer_churn.components.model_trainer.compute_sample_weight",
        fail_weights,
    )

    with pytest.raises(CustomerChurnException):
        model_trainer._calculate_sample_weights(y_train)
