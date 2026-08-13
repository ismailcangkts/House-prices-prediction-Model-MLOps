from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline import run_training_pipeline


DEFAULT_DRIFT_REPORT_PATH = Path("artifacts/drift/first_drift_report.json")
RETRAINING_STATUSES = {"CRITICAL_DATASET_DRIFT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger model retraining when a drift report reaches critical status."
    )
    parser.add_argument("--drift-report-path", type=Path, default=DEFAULT_DRIFT_REPORT_PATH)
    parser.add_argument(
        "--retrain-on-warning",
        action="store_true",
        help="Also retrain when alert_status is DATASET_DRIFT_WARNING.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the retraining decision without running the training pipeline.",
    )
    return parser.parse_args()


def load_drift_summary(drift_report_path: Path) -> dict[str, Any]:
    drift_report = json.loads(drift_report_path.read_text(encoding="utf-8"))
    if "summary" not in drift_report:
        raise ValueError(f"{drift_report_path} does not contain a summary section.")
    return drift_report["summary"]


def should_retrain(summary: dict[str, Any], retrain_on_warning: bool = False) -> bool:
    retraining_statuses = set(RETRAINING_STATUSES)
    if retrain_on_warning:
        retraining_statuses.add("DATASET_DRIFT_WARNING")
    return summary.get("alert_status") in retraining_statuses


def build_retraining_metadata(
    drift_report_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trigger": "data_drift",
        "reason": summary["alert_status"],
        "drift_report_path": str(drift_report_path),
        "weighted_drift_score": float(summary.get("weighted_drift_score", 0.0)),
        "drifted_columns_count": int(summary.get("drifted_columns_count", 0)),
        "critical_feature_drift_count": int(
            summary.get("critical_feature_drift_count", 0)
        ),
        "drifted_critical_features": summary.get("drifted_critical_features", []),
    }


def retrain_if_needed(
    drift_report_path: Path = DEFAULT_DRIFT_REPORT_PATH,
    retrain_on_warning: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    summary = load_drift_summary(drift_report_path)
    metadata = build_retraining_metadata(drift_report_path, summary)
    decision = {
        "drift_report_path": str(drift_report_path),
        "alert_status": summary.get("alert_status"),
        "weighted_drift_score": summary.get("weighted_drift_score"),
        "drifted_columns_count": summary.get("drifted_columns_count"),
        "drifted_critical_features": summary.get("drifted_critical_features", []),
        "retraining_triggered": False,
        "dry_run": dry_run,
    }

    if not should_retrain(summary, retrain_on_warning=retrain_on_warning):
        decision["reason"] = "alert_status_does_not_require_retraining"
        return decision

    decision["retraining_triggered"] = True
    decision["reason"] = "data_drift_threshold_reached"
    if dry_run:
        return decision

    training_output = run_training_pipeline(retraining_metadata=metadata)
    decision["best_metrics"] = training_output["best_metrics"]
    return decision


def main() -> None:
    args = parse_args()
    decision = retrain_if_needed(
        drift_report_path=args.drift_report_path,
        retrain_on_warning=args.retrain_on_warning,
        dry_run=args.dry_run,
    )

    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
