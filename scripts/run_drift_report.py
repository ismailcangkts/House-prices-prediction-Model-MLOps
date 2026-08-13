from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


DEFAULT_REFERENCE_PATH = Path("data/processed/reference.csv")
DEFAULT_CURRENT_PATH = Path("data/production/batch_001.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/drift/first_drift_report.json")
DEFAULT_HTML_OUTPUT_PATH = Path("artifacts/drift/first_drift_report.html")
DEFAULT_WARNING_THRESHOLD = 0.25
DEFAULT_CRITICAL_THRESHOLD = 0.45
DEFAULT_CRITICAL_FEATURE_WARNING_COUNT = 1
DEFAULT_CRITICAL_FEATURE_CRITICAL_COUNT = 4

CRITICAL_FEATURES = [
    "OverallQual",
    "GarageArea",
    "GarageCars",
    "GrLivArea",
    "TotalBsmtSF",
    "BsmtQual",
    "1stFlrSF",
    "ExterQual",
]

FEATURE_WEIGHTS = {
    "OverallQual": 1.000,
    "GarageArea": 0.988,
    "GarageCars": 0.967,
    "GrLivArea": 0.950,
    "TotalBsmtSF": 0.925,
    "BsmtQual": 0.909,
    "1stFlrSF": 0.906,
    "ExterQual": 0.803,
    "Neighborhood": 0.742,
    "TotRmsAbvGrd": 0.732,
    "FullBath": 0.725,
    "KitchenQual": 0.702,
    "YearBuilt": 0.696,
    "GarageFinish": 0.628,
    "OpenPorchSF": 0.609,
    "FireplaceQu": 0.587,
    "Fireplaces": 0.584,
    "YearRemodAdd": 0.573,
    "BsmtExposure": 0.549,
    "BsmtFinSF1": 0.544,
    "LotArea": 0.523,
    "GarageType": 0.491,
    "2ndFlrSF": 0.407,
    "HalfBath": 0.400,
    "LotFrontage": 0.378,
    "MSZoning": 0.356,
    "HouseStyle": 0.342,
    "GarageYrBlt": 0.254,
    "OverallCond": 0.249,
    "MSSubClass": 0.004,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Evidently data drift report for reference and current data."
    )
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--html-output-path", type=Path, default=DEFAULT_HTML_OUTPUT_PATH)
    parser.add_argument(
        "--target-column",
        default="SalePrice",
        help="Column to exclude when it is present in only the reference dataset.",
    )
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=DEFAULT_WARNING_THRESHOLD,
        help="Weighted drift score threshold for DATASET_DRIFT_WARNING.",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=DEFAULT_CRITICAL_THRESHOLD,
        help="Weighted drift score threshold for CRITICAL_DATASET_DRIFT.",
    )
    parser.add_argument(
        "--critical-feature-warning-count",
        type=int,
        default=DEFAULT_CRITICAL_FEATURE_WARNING_COUNT,
        help="Critical feature drift count that triggers DATASET_DRIFT_WARNING.",
    )
    parser.add_argument(
        "--critical-feature-critical-count",
        type=int,
        default=DEFAULT_CRITICAL_FEATURE_CRITICAL_COUNT,
        help="Critical feature drift count that triggers CRITICAL_DATASET_DRIFT.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_drift_data(
    reference_path: Path,
    current_path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    shared_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns and column != target_column
    ]
    if not shared_columns:
        raise ValueError("Reference and current datasets do not have shared feature columns.")

    return reference_df[shared_columns], current_df[shared_columns], shared_columns


def run_evidently_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
):
    report = Report([DataDriftPreset()])
    return report.run(reference_df, current_df)


def is_drift_detected(metric: dict[str, Any]) -> bool:
    config = metric.get("config", {})
    value = metric.get("value")
    method = str(config.get("method", "")).lower()
    threshold = config.get("threshold")

    if threshold is None or value is None:
        return False

    try:
        score = float(value)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return False

    if "p_value" in method:
        return score < threshold_value

    return score > threshold_value


def normalize_drift_result(
    report_dict: dict[str, Any],
    feature_columns: list[str],
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    critical_feature_warning_count: int = DEFAULT_CRITICAL_FEATURE_WARNING_COUNT,
    critical_feature_critical_count: int = DEFAULT_CRITICAL_FEATURE_CRITICAL_COUNT,
) -> dict[str, Any]:
    metrics = report_dict.get("metrics", [])
    drift_count_metric = next(
        (
            metric
            for metric in metrics
            if str(metric.get("metric_name", "")).startswith("DriftedColumnsCount")
        ),
        {},
    )
    drift_count_value = drift_count_metric.get("value", {})

    columns: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        config = metric.get("config", {})
        column = config.get("column")
        if not column:
            continue

        drift_detected = is_drift_detected(metric)
        columns[column] = {
            "drift_detected": drift_detected,
            "drift_score": metric.get("value"),
            "stattest_name": config.get("method"),
            "threshold": config.get("threshold"),
        }

    drifted_columns = [
        column for column, result in columns.items() if result["drift_detected"]
    ]
    drifted_columns_count = int(
        drift_count_value.get("count", len(drifted_columns))
    )
    share_of_drifted_columns = float(
        drift_count_value.get(
            "share",
            drifted_columns_count / len(feature_columns),
        )
    )
    feature_weights = {
        column: FEATURE_WEIGHTS.get(column, 0.0) for column in feature_columns
    }
    total_weight = sum(feature_weights.values())
    drifted_weight = sum(feature_weights[column] for column in drifted_columns)
    weighted_drift_score = drifted_weight / total_weight if total_weight else 0.0
    drifted_critical_features = [
        column for column in CRITICAL_FEATURES if column in drifted_columns
    ]
    critical_feature_drift_count = len(drifted_critical_features)

    if (
        weighted_drift_score >= critical_threshold
        or critical_feature_drift_count >= critical_feature_critical_count
    ):
        alert_status = "CRITICAL_DATASET_DRIFT"
    elif (
        weighted_drift_score >= warning_threshold
        or critical_feature_drift_count >= critical_feature_warning_count
    ):
        alert_status = "DATASET_DRIFT_WARNING"
    else:
        alert_status = "OK"

    return {
        "dataset_drift": alert_status != "OK",
        "alert_status": alert_status,
        "evidently_dataset_drift_by_share": share_of_drifted_columns >= 0.5,
        "drifted_columns_count": drifted_columns_count,
        "total_columns_count": len(feature_columns),
        "share_of_drifted_columns": share_of_drifted_columns,
        "weighted_drift_score": weighted_drift_score,
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "critical_features": CRITICAL_FEATURES,
        "drifted_critical_features": drifted_critical_features,
        "critical_feature_drift_count": critical_feature_drift_count,
        "critical_feature_warning_count": critical_feature_warning_count,
        "critical_feature_critical_count": critical_feature_critical_count,
        "feature_weights": feature_weights,
        "drifted_columns": drifted_columns,
        "columns": columns,
    }


def create_drift_report(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    current_path: Path = DEFAULT_CURRENT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    html_output_path: Path = DEFAULT_HTML_OUTPUT_PATH,
    target_column: str = "SalePrice",
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    critical_feature_warning_count: int = DEFAULT_CRITICAL_FEATURE_WARNING_COUNT,
    critical_feature_critical_count: int = DEFAULT_CRITICAL_FEATURE_CRITICAL_COUNT,
) -> dict[str, Any]:
    reference_df, current_df, feature_columns = load_drift_data(
        reference_path=reference_path,
        current_path=current_path,
        target_column=target_column,
    )
    snapshot = run_evidently_drift_report(reference_df, current_df)
    report_dict = snapshot.dict()
    summary = normalize_drift_result(
        report_dict=report_dict,
        feature_columns=feature_columns,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        critical_feature_warning_count=critical_feature_warning_count,
        critical_feature_critical_count=critical_feature_critical_count,
    )

    output = {
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "feature_columns": feature_columns,
        "summary": summary,
        "evidently_report": report_dict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, default=json_default),
        encoding="utf-8",
    )

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_output_path))

    return output


def main() -> None:
    args = parse_args()
    output = create_drift_report(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_path=args.output_path,
        html_output_path=args.html_output_path,
        target_column=args.target_column,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold,
        critical_feature_warning_count=args.critical_feature_warning_count,
        critical_feature_critical_count=args.critical_feature_critical_count,
    )
    summary = output["summary"]

    print(f"Drift JSON saved to: {args.output_path}")
    print(f"Drift HTML saved to: {args.html_output_path}")
    print(f"Alert status: {summary['alert_status']}")
    print(f"Dataset drift: {summary['dataset_drift']}")
    print(f"Weighted drift score: {summary['weighted_drift_score']:.2%}")
    print(
        "Drifted columns: "
        f"{summary['drifted_columns_count']}/{summary['total_columns_count']} "
        f"({summary['share_of_drifted_columns']:.2%})"
    )
    print(
        "Drifted critical features: "
        + (", ".join(summary["drifted_critical_features"]) or "None")
    )
    print("Drifted column names:", ", ".join(summary["drifted_columns"]) or "None")


if __name__ == "__main__":
    main()
