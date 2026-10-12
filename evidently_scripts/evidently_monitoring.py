
import argparse
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

artifact_runs = sorted(
    (
        path
        for path in ARTIFACTS_DIR.iterdir()
        if path.is_dir()
        and (path / "feature_engineering" / "engineered" / "X_train.csv").is_file()
        and (path / "feature_engineering" / "engineered" / "X_test.csv").is_file()
    ),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not artifact_runs:
    raise FileNotFoundError(
        "No artifact run containing engineered X_train.csv and X_test.csv was found."
    )

DEFAULT_ARTIFACT_DIR = (
    artifact_runs[0] / "feature_engineering" / "engineered"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evidently"
    / "data_drift_report.html"
)


def generate_drift_report(
    reference_path: Path,
    current_path: Path,
    report_path: Path,
) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

    if not current_path.is_file():
        raise FileNotFoundError(f"Current dataset not found: {current_path}")

    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    if list(reference_data.columns) != list(current_data.columns):
        raise ValueError(
            "Reference and current datasets must have identical columns "
            "in the same order."
        )

    if reference_data.empty or current_data.empty:
        raise ValueError("Reference and current datasets must not be empty.")

    report = Report(metrics=[DataDriftPreset()])

    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    print(f"Reference dataset: {reference_data.shape}")
    print(f"Current dataset:   {current_data.shape}")
    print(f"Report saved to:   {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an Evidently data drift report."
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "X_train.csv",
        help="Reference dataset CSV path.",
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "X_test.csv",
        help="Current dataset CSV path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output HTML report path.",
    )

    args = parser.parse_args()

    generate_drift_report(
        reference_path=args.reference,
        current_path=args.current,
        report_path=args.output,
    )


if __name__ == "__main__":
    main()
