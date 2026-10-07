import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.entity.artifact_entity import DataIngestionArtifact
from src.customer_churn.exception.exception import CustomerChurnException


@pytest.fixture
def ingestion_config(tmp_path):
    ingested_dir = tmp_path / "ingested"
    versioned_dir = tmp_path / "versioned"

    return type(
        "Config",
        (),
        {
            "ingested_data_dir" : str(ingested_dir),
            "versioned_data_dir" : str(versioned_dir),
            "versioned_data_path" : str(
                versioned_dir/"customer_churn.csv"
            )
        }
    )

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "customerID": ["CUST001", "CUST002"],
            "tenure": [12, 24],
            "MonthlyCharges": [70.5, 85.2],
            "Churn": ["Yes", "No"],
        }
    )

@pytest.fixture
def ingestion(ingestion_config):
    return DataIngestion(ingestion_config)


def test_get_latest_table_selects_highest_version(ingestion):
    engine = MagicMock()
    connection = MagicMock()
    result = MagicMock()

    engine.connect.return_value.__enter__.return_value = connection
    connection.execute.return_value = result
    result.scalars.return_value.all.return_value = [
        "customer_churn_v1",
        "customer_churn_v3",
        "customer_churn_v2",
    ]

    table_name, version = ingestion._get_latest_table(engine)

    assert table_name == "customer_churn_v3"
    assert version == 3


def test_get_latest_table_fails_when_no_tables_exist(ingestion):
    engine = MagicMock()
    connection = MagicMock()
    result = MagicMock()

    engine.connect.return_value.__enter__.return_value = connection
    connection.execute.return_value = result
    result.scalars.return_value.all.return_value = []

    with pytest.raises(CustomerChurnException):
        ingestion._get_latest_table(engine)


def test_read_data_returns_dataframe(ingestion, sample_dataframe):
    with patch(
        "src.customer_churn.components.data_ingestion.pd.read_sql",
        return_value=sample_dataframe,
    ) as mock_read_sql:
        result = ingestion._read_data(
            MagicMock(),
            "customer_churn_v2",
        )

    pd.testing.assert_frame_equal(result, sample_dataframe)
    mock_read_sql.assert_called_once()


def test_read_data_wraps_database_errors(ingestion):
    with patch(
        "src.customer_churn.components.data_ingestion.pd.read_sql",
        side_effect=Exception("Database read failed"),
    ):
        with pytest.raises(CustomerChurnException):
            ingestion._read_data(MagicMock(), "customer_churn_v2")


def test_validate_dataframe_accepts_valid_data(
    ingestion,
    sample_dataframe,
):
    assert ingestion._validate_dataframe(
        sample_dataframe,
        "customer_churn_v1",
    ) is None


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[0, 1]),
        pd.DataFrame([[1, 2]], columns=["tenure", "tenure"]),
    ],
    ids=[
        "empty-dataframe",
        "no-columns",
        "duplicate-columns",
    ],
)
def test_validate_dataframe_rejects_invalid_data(
    ingestion,
    dataframe,
):
    with pytest.raises(CustomerChurnException):
        ingestion._validate_dataframe(
            dataframe,
            "customer_churn_v1",
        )


def test_load_data_returns_data_and_version(
    ingestion,
    sample_dataframe,
):
    with (
        patch.object(
            ingestion,
            "_get_engine",
            return_value=MagicMock(),
        ),
        patch.object(
            ingestion,
            "_get_latest_table",
            return_value=("customer_churn_v4", 4),
        ),
        patch.object(
            ingestion,
            "_read_data",
            return_value=sample_dataframe,
        ),
        patch.object(ingestion, "_validate_dataframe") as mock_validate,
    ):
        dataframe, version = ingestion.load_data()

    pd.testing.assert_frame_equal(dataframe, sample_dataframe)
    assert version == 4
    mock_validate.assert_called_once_with(
        sample_dataframe,
        "customer_churn_v4",
    )


def test_load_data_disposes_engine(
    ingestion,
    sample_dataframe,
):
    engine = MagicMock()

    with (
        patch.object(ingestion, "_get_engine", return_value=engine),
        patch.object(
            ingestion,
            "_get_latest_table",
            return_value=("customer_churn_v2", 2),
        ),
        patch.object(
            ingestion,
            "_read_data",
            return_value=sample_dataframe,
        ),
        patch.object(ingestion, "_validate_dataframe"),
    ):
        ingestion.load_data()

    engine.dispose.assert_called_once()


def test_initiate_data_ingestion_saves_both_snapshots(
    ingestion,
    ingestion_config,
    sample_dataframe,
):
    with patch.object(
        ingestion,
        "load_data",
        return_value=(sample_dataframe, 3),
    ):
        artifact = ingestion.initiate_data_ingestion()

    assert isinstance(artifact, DataIngestionArtifact)
    assert artifact.dataset_version == 3

    timestamped_path = (
        ingestion_config.ingested_data_dir
        + "\\customer_churn_v3.csv"
    )
    versioned_path = ingestion_config.versioned_data_path

    assert pd.read_csv(artifact.ingested_data_path).equals(
        sample_dataframe
    )
    assert pd.read_csv(versioned_path).equals(sample_dataframe)
    assert artifact.ingested_data_path.endswith(
        "customer_churn_v3.csv"
    )
