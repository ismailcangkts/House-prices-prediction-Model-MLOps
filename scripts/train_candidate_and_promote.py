from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline import (
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_REGISTERED_MODEL_NAME,
    run_training_pipeline,
)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None


DEFAULT_CANDIDATE_ALIAS = "candidate"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_OUTPUT_PATH = Path("artifacts/promotion_decision.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/register a candidate model, compare it with champion, "
            "and promote it when validation metrics improve."
        )
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--metric-name", default="rmse")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Required relative improvement. Example: 0.01 means candidate must be one percent better.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Compare the existing candidate alias without training a new candidate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not update the champion alias; only write the decision.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def require_mlflow() -> None:
    if mlflow is None or MlflowClient is None:
        raise ImportError("mlflow is required for candidate/champion comparison.")


def get_model_version_by_alias(client, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None


def get_run_metric(client, run_id: str, metric_name: str) -> float:
    run = client.get_run(run_id)
    if metric_name not in run.data.metrics:
        raise ValueError(f"Metric '{metric_name}' not found in run {run_id}.")
    return float(run.data.metrics[metric_name])


def build_version_info(client, model_version, metric_name: str) -> dict[str, Any] | None:
    if model_version is None:
        return None

    metric_value = get_run_metric(client, model_version.run_id, metric_name)
    return {
        "version": str(model_version.version),
        "run_id": model_version.run_id,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }


def should_promote(
    candidate_metric: float,
    champion_metric: float | None,
    min_improvement: float,
) -> tuple[bool, str, float | None]:
    if champion_metric is None:
        return True, "no_champion_alias_found", None

    required_metric = champion_metric * (1 - min_improvement)
    if candidate_metric <= required_metric:
        return True, "candidate_metric_meets_promotion_requirement", required_metric

    return False, "candidate_metric_did_not_improve_enough", required_metric


def set_promotion_tags(
    client,
    model_name: str,
    candidate_version: str,
    decision: dict[str, Any],
) -> None:
    tags = {
        "promotion_status": decision["promotion_status"],
        "promotion_reason": decision["reason"],
        "candidate_metric": str(decision["candidate"]["metric_value"]),
        "champion_metric": str(
            decision["champion"]["metric_value"] if decision["champion"] else "None"
        ),
        "metric_name": decision["metric_name"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=candidate_version,
            key=key,
            value=value,
        )
    client.set_tag(decision["candidate"]["run_id"], "promotion_status", decision["promotion_status"])
    client.set_tag(decision["candidate"]["run_id"], "promotion_reason", decision["reason"])


def write_decision(output_path: Path, decision: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")


def train_candidate_and_promote(
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    candidate_alias: str = DEFAULT_CANDIDATE_ALIAS,
    champion_alias: str = DEFAULT_CHAMPION_ALIAS,
    metric_name: str = "rmse",
    min_improvement: float = 0.0,
    skip_training: bool = False,
    dry_run: bool = False,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    require_mlflow()
    mlflow.set_tracking_uri(tracking_uri)

    training_output = None
    if not skip_training:
        training_output = run_training_pipeline(
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
            model_alias=candidate_alias,
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    candidate_version = get_model_version_by_alias(
        client,
        registered_model_name,
        candidate_alias,
    )
    if candidate_version is None:
        raise ValueError(f"Candidate alias not found: {registered_model_name}@{candidate_alias}")

    champion_version = get_model_version_by_alias(
        client,
        registered_model_name,
        champion_alias,
    )

    candidate_info = build_version_info(client, candidate_version, metric_name)
    champion_info = build_version_info(client, champion_version, metric_name)
    promote, reason, required_metric = should_promote(
        candidate_metric=candidate_info["metric_value"],
        champion_metric=champion_info["metric_value"] if champion_info else None,
        min_improvement=min_improvement,
    )

    decision = {
        "registered_model_name": registered_model_name,
        "candidate_alias": candidate_alias,
        "champion_alias": champion_alias,
        "metric_name": metric_name,
        "min_improvement": min_improvement,
        "required_candidate_metric": required_metric,
        "candidate": candidate_info,
        "champion": champion_info,
        "promotion_status": "promoted" if promote else "rejected",
        "reason": reason,
        "dry_run": dry_run,
        "training_best_metrics": training_output["best_metrics"] if training_output else None,
    }

    if not dry_run:
        if promote:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=champion_alias,
                version=candidate_info["version"],
            )

        set_promotion_tags(
            client=client,
            model_name=registered_model_name,
            candidate_version=candidate_info["version"],
            decision=decision,
        )

    final_champion_version = get_model_version_by_alias(
        client,
        registered_model_name,
        champion_alias,
    )
    decision["final_champion_version"] = (
        str(final_champion_version.version) if final_champion_version else None
    )
    write_decision(output_path, decision)
    return decision


def main() -> None:
    args = parse_args()
    decision = train_candidate_and_promote(
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
        champion_alias=args.champion_alias,
        metric_name=args.metric_name,
        min_improvement=args.min_improvement,
        skip_training=args.skip_training,
        dry_run=args.dry_run,
        output_path=args.output_path,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
