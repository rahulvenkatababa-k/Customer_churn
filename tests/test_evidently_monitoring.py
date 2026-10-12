
import pandas as pd
import pytest

from evidently_scripts.evidently_monitoring import generate_drift_report


@pytest.fixture
def sample_datasets(tmp_path):
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"

    reference = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48],
        "MonthlyCharges": [30.0, 50.0, 65.0, 80.0, 95.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
        ],
    })

    current = pd.DataFrame({
        "tenure": [2, 14, 26, 38, 50],
        "MonthlyCharges": [32.0, 52.0, 67.0, 82.0, 97.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
        ],
    })

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    return reference_path, current_path


def test_generate_drift_report_creates_html(
    sample_datasets, tmp_path
):
    reference_path, current_path = sample_datasets
    report_path = tmp_path / "reports" / "drift.html"

    generate_drift_report(
        reference_path=reference_path,
        current_path=current_path,
        report_path=report_path,
    )

    assert report_path.is_file()
    assert report_path.stat().st_size > 0


def test_generate_drift_report_rejects_missing_reference(
    sample_datasets, tmp_path
):
    _, current_path = sample_datasets

    with pytest.raises(FileNotFoundError, match="Reference dataset not found"):
        generate_drift_report(
            reference_path=tmp_path / "missing.csv",
            current_path=current_path,
            report_path=tmp_path / "report.html",
        )


def test_generate_drift_report_rejects_missing_current(
    sample_datasets, tmp_path
):
    reference_path, _ = sample_datasets

    with pytest.raises(FileNotFoundError, match="Current dataset not found"):
        generate_drift_report(
            reference_path=reference_path,
            current_path=tmp_path / "missing.csv",
            report_path=tmp_path / "report.html",
        )


def test_generate_drift_report_rejects_mismatched_columns(
    sample_datasets, tmp_path
):
    reference_path, current_path = sample_datasets

    current = pd.read_csv(current_path)
    current = current.rename(columns={"tenure": "customer_tenure"})
    current.to_csv(current_path, index=False)

    with pytest.raises(ValueError, match="identical columns"):
        generate_drift_report(
            reference_path=reference_path,
            current_path=current_path,
            report_path=tmp_path / "report.html",
        )


@pytest.mark.parametrize("empty_dataset", ["reference", "current"])
def test_generate_drift_report_rejects_empty_dataset(
    sample_datasets, tmp_path, empty_dataset
):
    reference_path, current_path = sample_datasets
    empty_path = tmp_path / "empty.csv"

    pd.DataFrame(columns=["tenure", "MonthlyCharges", "Contract"]).to_csv(
        empty_path, index=False
    )

    if empty_dataset == "reference":
        reference_path = empty_path
    else:
        current_path = empty_path

    with pytest.raises(ValueError, match="must not be empty"):
        generate_drift_report(
            reference_path=reference_path,
            current_path=current_path,
            report_path=tmp_path / "report.html",
        )
