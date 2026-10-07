import pandas as pd
import pytest

from src.customer_churn.components.data_transformation import DataTransformation
from src.customer_churn.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.customer_churn.exception.exception import CustomerChurnException


@pytest.fixture
def transformation_config(tmp_path):
    transformed_dir = tmp_path / "transformed"

    return type(
        "Config",
        (),
        {
            "transformed_data_dir": str(transformed_dir),
            "transformed_data_path": str(
                transformed_dir / "customer_churn.csv"
            ),
        },
    )()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "customerID": ["CUST001", "CUST002", "CUST003"],
            "tenure": [0, 12, 24],
            "MonthlyCharges": [70.5, 85.2, 90.0],
            "TotalCharges": [" ", "1022.4", "2160.0"],
            "Churn": ["Yes", "No", "No"],
        }
    )


@pytest.fixture
def transformation(transformation_config, tmp_path):
    validation_artifact = DataValidationArtifact(
        validated_status=True,
        validated_data_path=str(tmp_path / "validated.csv"),
        validation_report_path=str(tmp_path / "validation_report.json"),
    )

    return DataTransformation(
        data_transformation_config=transformation_config,
        data_validation_artifact=validation_artifact,
    )


def test_load_data_returns_validated_dataframe(
    transformation,
    sample_dataframe,
    tmp_path,
):
    input_path = tmp_path / "validated.csv"
    sample_dataframe.to_csv(input_path, index=False)

    transformation.data_validation_artifact.validated_data_path = str(
        input_path
    )

    result = transformation._load_data()

    assert len(result) == 3
    assert "TotalCharges" in result.columns


def test_load_data_fails_when_validated_path_is_none(
    transformation,
):
    transformation.data_validation_artifact.validated_data_path = None

    with pytest.raises(CustomerChurnException):
        transformation._load_data()


def test_transform_data_converts_total_charges_to_numeric(
    transformation,
    sample_dataframe,
):
    result = transformation._transform_data(sample_dataframe)

    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
    assert result["TotalCharges"].tolist() == [0.0, 1022.4, 2160.0]


def test_transform_data_does_not_modify_original_dataframe(
    transformation,
    sample_dataframe,
):
    original = sample_dataframe.copy()

    transformation._transform_data(sample_dataframe)

    pd.testing.assert_frame_equal(sample_dataframe, original)


def test_transform_data_preserves_other_columns(
    transformation,
    sample_dataframe,
):
    result = transformation._transform_data(sample_dataframe)

    for column in ["customerID", "tenure", "MonthlyCharges", "Churn"]:
        pd.testing.assert_series_equal(
            result[column],
            sample_dataframe[column],
        )


def test_save_transformed_data_creates_csv(
    transformation,
    transformation_config,
    sample_dataframe,
):
    output_path = transformation._save_transformed_data(sample_dataframe)

    assert output_path == transformation_config.transformed_data_path

    saved_dataframe = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(
        saved_dataframe,
        sample_dataframe,
        check_dtype=False,
    )


def test_initiate_data_transformation_returns_artifact(
    transformation,
    sample_dataframe,
):
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        monkeypatch.setattr(
            transformation,
            "_load_data",
            lambda: sample_dataframe,
        )

        artifact = transformation.initiate_data_transformation()

    assert isinstance(artifact, DataTransformationArtifact)
    assert artifact.transformed_data_path
    assert pd.read_csv(artifact.transformed_data_path).shape == (
        sample_dataframe.shape
    )
