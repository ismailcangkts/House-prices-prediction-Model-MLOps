from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Joblib modelindeki custom transformer siniflarinin yuklenebilmesi icin gerekli.
import pipeline_machine_learning  # noqa: F401, E402


TRACKING_URI = "sqlite:///storage/mlflow.db"
EXPERIMENT_NAME = "house_price_prediction"
PRESENTATION_BATCH = "mlops_slide_v1"
MODEL_PATH = Path("artifacts/models/best_model.joblib")


RUNS = [
    {
        "name": "Baseline | Ridge",
        "age_minutes": 210,
        "duration_seconds": 18,
        "dataset": "train",
        "params": {
            "stage": "baseline",
            "model_name": "Ridge Regression",
            "alpha": "0.1",
            "feature_count": "30",
            "target_transform": "log1p",
        },
        "metrics": {
            "mae": 19765.12,
            "rmse": 30218.42,
            "r2": 0.842,
        },
    },
    {
        "name": "Tuning | Ridge alpha=1",
        "age_minutes": 180,
        "duration_seconds": 24,
        "dataset": "train",
        "params": {
            "stage": "hyperparameter_tuning",
            "model_name": "Ridge Regression",
            "alpha": "1.0",
            "feature_count": "30",
            "target_transform": "log1p",
        },
        "metrics": {
            "mae": 18730.44,
            "rmse": 28412.66,
            "r2": 0.865,
        },
    },
    {
        "name": "Tuning | Ridge alpha=5",
        "age_minutes": 150,
        "duration_seconds": 27,
        "dataset": "train",
        "params": {
            "stage": "hyperparameter_tuning",
            "model_name": "Ridge Regression",
            "alpha": "5.0",
            "feature_count": "30",
            "target_transform": "log1p",
        },
        "metrics": {
            "mae": 17842.55,
            "rmse": 27081.30,
            "r2": 0.883,
        },
    },
    {
        "name": "Tuning | Ridge alpha=10",
        "age_minutes": 120,
        "duration_seconds": 31,
        "dataset": "train",
        "params": {
            "stage": "hyperparameter_tuning",
            "model_name": "Ridge Regression",
            "alpha": "10.0",
            "feature_count": "30",
            "target_transform": "log1p",
            "scaler": "StandardScaler",
        },
        "metrics": {
            "mae": 17383.893654,
            "rmse": 26330.884888,
            "mae_ratio_percent": 9.659143,
            "rmse_ratio_percent": 14.630427,
            "r2": 0.894,
        },
    },
    {
        "name": "Candidate | Ridge v3",
        "age_minutes": 90,
        "duration_seconds": 36,
        "dataset": "validation",
        "log_model": True,
        "params": {
            "stage": "candidate",
            "model_name": "Ridge Regression",
            "model_version": "3",
            "alpha": "10.0",
            "feature_count": "30",
            "registry_alias": "candidate",
        },
        "metrics": {
            "validation_mae": 17383.893654,
            "validation_rmse": 26330.884888,
            "test_mae": 17236.03,
            "test_rmse": 26494.28,
            "quality_gate_passed": 1.0,
        },
    },
    {
        "name": "Promotion Gate | Passed",
        "age_minutes": 60,
        "duration_seconds": 7,
        "dataset": "validation",
        "params": {
            "stage": "promotion_gate",
            "metric_name": "rmse",
            "candidate_version": "3",
            "champion_version": "3",
            "promotion_status": "promoted",
        },
        "metrics": {
            "candidate_rmse": 26330.884888,
            "champion_rmse": 26330.884888,
            "required_improvement": 0.0,
            "promotion_passed": 1.0,
        },
    },
    {
        "name": "Champion | Ridge v3",
        "age_minutes": 30,
        "duration_seconds": 12,
        "dataset": "validation",
        "log_model": True,
        "params": {
            "stage": "production",
            "model_name": "Ridge Regression",
            "model_version": "3",
            "registry_alias": "champion",
            "serving_mode": "streamlit",
        },
        "metrics": {
            "mae": 17383.893654,
            "rmse": 26330.884888,
            "average_latency_ms": 17.29,
            "promotion_passed": 1.0,
        },
    },
    {
        "name": "Monitoring | Drift Warning",
        "age_minutes": 5,
        "duration_seconds": 9,
        "dataset": "drift",
        "params": {
            "stage": "monitoring",
            "monitor": "Evidently",
            "alert_status": "DATASET_DRIFT_WARNING",
            "reference_rows": "145",
            "production_rows": "145",
        },
        "metrics": {
            "weighted_drift_score": 0.3772607,
            "drifted_columns_count": 10.0,
            "critical_feature_drift_count": 2.0,
            "dataset_drift_detected": 1.0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed clearly tagged synthetic MLflow runs for a presentation screenshot."
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    return parser.parse_args()


def build_datasets() -> dict[str, list]:
    train_df = pd.read_csv("data/processed/train.csv")
    validation_df = pd.read_csv("data/processed/validation.csv")
    reference_df = pd.read_csv("data/processed/reference.csv")
    production_df = pd.read_csv("data/production/batch_001.csv")

    return {
        "train": [
            mlflow.data.from_pandas(
                train_df,
                source="data/processed/train.csv",
                name="ames_train_v1",
                targets="SalePrice",
            )
        ],
        "validation": [
            mlflow.data.from_pandas(
                validation_df,
                source="data/processed/validation.csv",
                name="ames_validation_v1",
                targets="SalePrice",
            )
        ],
        "drift": [
            mlflow.data.from_pandas(
                reference_df,
                source="data/processed/reference.csv",
                name="ames_reference_v1",
                targets="SalePrice",
            ),
            mlflow.data.from_pandas(
                production_df,
                source="data/production/batch_001.csv",
                name="ames_production_batch_001",
            ),
        ],
    }


def existing_demo_runs(client: MlflowClient, experiment_id: str):
    return client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.presentation_batch = '{PRESENTATION_BATCH}'",
        max_results=100,
    )


def create_presentation_runs(tracking_uri: str, experiment_name: str) -> list[str]:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    client = MlflowClient(tracking_uri=tracking_uri)
    existing = existing_demo_runs(client, experiment_id)
    if existing:
        print(
            f"Presentation batch already exists: {PRESENTATION_BATCH} "
            f"({len(existing)} runs)"
        )
        return [run.info.run_id for run in existing]

    client.set_experiment_tag(
        experiment_id,
        "mlflow.note.content",
        "Ames Housing MLOps lifecycle: training, candidate validation, "
        "champion promotion and drift monitoring.",
    )

    datasets = build_datasets()
    ridge_model = joblib.load(MODEL_PATH)
    now = datetime.now(timezone.utc)
    run_ids = []

    for spec in RUNS:
        start_at = now - timedelta(minutes=spec["age_minutes"])
        start_ms = int(start_at.timestamp() * 1000)
        end_ms = start_ms + int(spec["duration_seconds"] * 1000)
        tags = [
            RunTag("mlflow.runName", spec["name"]),
            RunTag("mlflow.source.name", "scripts/mlops_lifecycle_pipeline.py"),
            RunTag("mlflow.source.type", "LOCAL"),
            RunTag("presentation_demo", "true"),
            RunTag("presentation_batch", PRESENTATION_BATCH),
            RunTag("do_not_use_for_model_selection", "true"),
            RunTag(
                "mlflow.note.content",
                "Sunum ekran görüntüsü için sentetik olarak oluşturulmuş demo kaydı. "
                "Production model seçimi için kullanılmaz.",
            ),
        ]
        run = client.create_run(
            experiment_id=experiment_id,
            start_time=start_ms,
            tags={tag.key: tag.value for tag in tags},
        )
        run_id = run.info.run_id
        run_ids.append(run_id)

        client.log_batch(
            run_id=run_id,
            metrics=[
                Metric(key, float(value), start_ms, 0)
                for key, value in spec["metrics"].items()
            ],
            params=[Param(key, str(value)) for key, value in spec["params"].items()],
            tags=[],
        )

        with mlflow.start_run(run_id=run_id):
            for dataset in datasets[spec["dataset"]]:
                context = "monitoring" if spec["dataset"] == "drift" else spec["dataset"]
                mlflow.log_input(dataset, context=context)
            if spec.get("log_model"):
                mlflow.sklearn.log_model(ridge_model, artifact_path="ridge_pipeline")

        client.set_terminated(run_id, status="FINISHED", end_time=end_ms)
        print(f"Created: {spec['name']} ({run_id})")

    return run_ids


def main() -> None:
    args = parse_args()
    run_ids = create_presentation_runs(args.tracking_uri, args.experiment_name)
    print(f"Presentation runs ready: {len(run_ids)}")


if __name__ == "__main__":
    main()
