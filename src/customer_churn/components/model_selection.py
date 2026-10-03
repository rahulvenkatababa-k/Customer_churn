import json
import pandas as pd


EVALUATION_REPORT_PATH = (
    r"artifacts\03_10_2026_21_07_20"
    r"\model_evaluation\evaluation_report.json"
)


def load_evaluation_report(path: str) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def create_model_summary(report: dict) -> pd.DataFrame:
    rows = []

    for model_name, metrics in report.items():
        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    report = load_evaluation_report(EVALUATION_REPORT_PATH)

    summary = create_model_summary(report)

    print("\nMODEL EVALUATION SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))