import pandas as pd
import pytest

from src.customer_churn.components.feature_engineering import FeatureEngineering
from src.customer_churn.entity.artifact_entity import (
    DataTransformationArtifact,
    FeatureEngineeringArtifact,
)


@pytest.fixture
def valid_dataframe():
    """Create a representative customer churn dataset."""
    return pd.DataFrame(
        {
            "customerID": [f"CUST{i:03d}" for i in range(100)],
            "gender": ["Female", "Male"] * 50,
            "SeniorCitizen": [0, 1] * 50,
            "Partner": ["Yes", "No"] * 50,
            "Dependents": ["No", "Yes"] * 50,
            "tenure": [i % 72 for i in range(100)],
            "PhoneService": ["Yes", "No"] * 50,
            "MultipleLines": ["No", "Yes"] * 50,
            "InternetService": ["DSL", "Fiber optic"] * 50,
            "OnlineSecurity": ["Yes", "No"] * 50,
            "OnlineBackup": ["No", "Yes"] * 50,
            "DeviceProtection": ["Yes", "No"] * 50,
            "TechSupport": ["No", "Yes"] * 50,
            "StreamingTV": ["Yes", "No"] * 50,
            "StreamingMovies": ["No", "Yes"] * 50,
            "Contract": ["Month-to-month", "One year"] * 50,
            "PaperlessBilling": ["Yes", "No"] * 50,
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
            ] * 50,
            "MonthlyCharges": [50.0 + i for i in range(100)],
            "TotalCharges": [100.0 + i for i in range(100)],
            "Churn": ["Yes", "No"] * 50,
        }
    )


@pytest.fixture
def feature_engineering(tmp_path):
    """Create a feature engineering component with temporary paths."""
    transformed_path = tmp_path / "transformed.csv"
    output_dir = tmp_path / "engineered"
    output_dir.mkdir()

    paths = {
        "X_train_path": output_dir / "X_train.csv",
        "y_train_path": output_dir / "y_train.csv",
        "X_val_path": output_dir / "X_val.csv",
        "y_val_path": output_dir / "y_val.csv",
        "X_test_path": output_dir / "X_test.csv",
        "y_test_path": output_dir / "y_test.csv",
    }

    config = type("TestFeatureEngineeringConfig", (), {})()
    for name, path in paths.items():
        setattr(config, name, str(path))

    config.random_state = 42

    component = FeatureEngineering(
        data_transformation_artifact=DataTransformationArtifact(
            transformed_data_path=str(transformed_path)
        ),
        feature_engineering_config=config,
    )

    return component, transformed_path, paths


def test_prepare_features_and_target_separates_target_and_drops_id(
    feature_engineering, valid_dataframe
):
    component, _, _ = feature_engineering

    X, y = component._prepare_features_and_target(valid_dataframe)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert len(X) == len(y) == len(valid_dataframe)
    assert set(y.unique()) == {0, 1}
    assert y.dtype == "int64"


@pytest.mark.parametrize(
    "invalid_target",
    [
        None,
        "Unknown",
        "",
    ],
)
def test_prepare_features_and_target_rejects_invalid_target(
    feature_engineering, valid_dataframe, invalid_target
):
    component, _, _ = feature_engineering
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "Churn"] = invalid_target

    with pytest.raises(Exception):
        component._prepare_features_and_target(dataframe)


def test_prepare_features_and_target_rejects_missing_required_column(
    feature_engineering, valid_dataframe
):
    component, _, _ = feature_engineering
    dataframe = valid_dataframe.drop(columns=["tenure"])

    with pytest.raises(Exception):
        component._prepare_features_and_target(dataframe)


def test_prepare_features_and_target_rejects_missing_target(
    feature_engineering, valid_dataframe
):
    component, _, _ = feature_engineering
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "Churn"] = None

    with pytest.raises(Exception):
        component._prepare_features_and_target(dataframe)


def test_split_train_validation_test_preserves_rows_and_stratification(
    feature_engineering, valid_dataframe
):
    component, _, _ = feature_engineering
    X, y = component._prepare_features_and_target(valid_dataframe)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        component._split_train_test(X, y)
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    for target in (y_train, y_val, y_test):
        assert set(target.unique()) == {0, 1}


def test_save_feature_datasets_creates_all_six_files(
    feature_engineering, valid_dataframe
):
    component, _, paths = feature_engineering
    X, y = component._prepare_features_and_target(valid_dataframe)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        component._split_train_test(X, y)
    )

    component._save_feature_datasets(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    for path in paths.values():
        assert path.exists()

    saved_X_train = pd.read_csv(paths["X_train_path"])
    saved_y_train = pd.read_csv(paths["y_train_path"])

    assert len(saved_X_train) == len(X_train)
    assert list(saved_y_train.columns) == ["Churn"]
    assert set(saved_y_train["Churn"].unique()) == {0, 1}


def test_initiate_feature_engineering_returns_artifact(
    feature_engineering, valid_dataframe
):
    component, transformed_path, paths = feature_engineering
    valid_dataframe.to_csv(transformed_path, index=False)

    artifact = component.initiate_feature_engineering()

    assert isinstance(artifact, FeatureEngineeringArtifact)

    assert artifact.X_train_path == str(paths["X_train_path"])
    assert artifact.y_train_path == str(paths["y_train_path"])
    assert artifact.X_val_path == str(paths["X_val_path"])
    assert artifact.y_val_path == str(paths["y_val_path"])
    assert artifact.X_test_path == str(paths["X_test_path"])
    assert artifact.y_test_path == str(paths["y_test_path"])

    for path in paths.values():
        assert path.exists()
