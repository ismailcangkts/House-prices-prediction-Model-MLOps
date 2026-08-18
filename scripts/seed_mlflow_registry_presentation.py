from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Joblib modelindeki custom transformer siniflarinin yuklenebilmesi icin gerekli.
import pipeline_machine_learning  # noqa: F401, E402


TRACKING_URI = "sqlite:///storage/mlflow.db"
EXPERIMENT_NAME = "house_price_prediction"
REGISTERED_MODEL_NAME = "HousePriceModel"
MODEL_PATH = Path("artifacts/models/best_model.joblib")
PRESENTATION_BATCH = "registry_slide_v1"


REGISTRY_STAGES = [
    {
        "run_name": "Registry | Baseline Archived",
        "artifact_name": "baseline_pipeline",
        "alias": "baseline",
        "lifecycle": "ARCHIVED",
        "decision": "Superseded by a lower-RMSE candidate",
        "rmse": 30218.42,
        "mae": 19765.12,
        "age_minutes": 80,
    },
    {
        "run_name": "Registry | Candidate Rejected",
        "artifact_name": "rejected_candidate",
        "alias": "rejected",
        "lifecycle": "REJECTED",
        "decision": "Quality gate failed on validation RMSE",
        "rmse": 28412.66,
        "mae": 18730.44,
        "age_minutes": 60,
    },
    {
        "run_name": "Registry | Candidate Approved",
        "artifact_name": "approved_candidate",
        "alias": "candidate",
        "lifecycle": "APPROVED",
        "decision": "Validation and test quality gates passed",
        "rmse": 26330.884888,
        "mae": 17383.893654,
        "age_minutes": 40,
    },
    {
        "run_name": "Registry | Champion Production",
        "artifact_name": "production_champion",
        "alias": "champion",
        "lifecycle": "PRODUCTION",
        "decision": "Promoted after candidate/champion comparison",
        "rmse": 26330.884888,
        "mae": 17383.893654,
        "age_minutes": 20,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clearly tagged synthetic Registry history for a slide."
    )
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    return parser.parse_args()


def get_existing_versions(client: MlflowClient, model_name: str):
    return [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.tags.get("presentation_registry_batch") == PRESENTATION_BATCH
    ]


def seed_registry(tracking_uri: str, experiment_name: str, model_name: str) -> list[str]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    existing = get_existing_versions(client, model_name)
    if existing:
        print(f"Registry presentation batch already exists ({len(existing)} versions).")
        return [str(version.version) for version in existing]

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    client.update_registered_model(
        name=model_name,
        description=(
            "Production-ready Ridge pipeline for Ames house-price prediction. "
            "Versions move through candidate validation and champion promotion gates."
        ),
    )
    for key, value in {
        "task": "house_price_regression",
        "framework": "scikit-learn",
        "owner": "MLOps Team",
        "deployment": "Streamlit",
        "monitoring": "Evidently",
        "presentation_demo": "true",
    }.items():
        client.set_registered_model_tag(model_name, key, value)

    model = joblib.load(MODEL_PATH)
    now = datetime.now(timezone.utc)
    versions = []

    for spec in REGISTRY_STAGES:
        start_time = int(
            (now - timedelta(minutes=spec["age_minutes"])).timestamp() * 1000
        )
        run = client.create_run(
            experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id,
            start_time=start_time,
            tags={
                "mlflow.runName": spec["run_name"],
                "mlflow.source.name": "scripts/registry_promotion_pipeline.py",
                "mlflow.source.type": "LOCAL",
                "presentation_demo": "true",
                "presentation_registry_batch": PRESENTATION_BATCH,
                "do_not_use_for_model_selection": "true",
                "lifecycle_status": spec["lifecycle"],
                "mlflow.note.content": (
                    "Sunum ekran görüntüsü için sentetik Registry yaşam döngüsü kaydı."
                ),
            },
        )

        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_params(
                {
                    "model_name": "Ridge Regression",
                    "alpha": 10.0,
                    "registry_stage": spec["lifecycle"],
                    "promotion_decision": spec["decision"],
                    "feature_count": 30,
                }
            )
            mlflow.log_metrics(
                {
                    "rmse": spec["rmse"],
                    "mae": spec["mae"],
                    "validation_rmse": spec["rmse"],
                    "validation_mae": spec["mae"],
                    "quality_gate_passed": float(
                        spec["lifecycle"] in {"APPROVED", "PRODUCTION"}
                    ),
                }
            )
            mlflow.sklearn.log_model(model, artifact_path=spec["artifact_name"])

        client.set_terminated(
            run.info.run_id,
            status="FINISHED",
            end_time=start_time + 12_000,
        )
        registered = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{spec['artifact_name']}",
            name=model_name,
        )
        version = str(registered.version)
        versions.append(version)

        client.set_registered_model_alias(model_name, spec["alias"], version)
        version_tags = {
            "lifecycle_status": spec["lifecycle"],
            "validation_status": (
                "PASSED"
                if spec["lifecycle"] in {"APPROVED", "PRODUCTION"}
                else "FAILED" if spec["lifecycle"] == "REJECTED" else "SUPERSEDED"
            ),
            "promotion_decision": spec["decision"],
            "validation_rmse": str(spec["rmse"]),
            "validation_mae": str(spec["mae"]),
            "presentation_demo": "true",
            "presentation_registry_batch": PRESENTATION_BATCH,
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(model_name, version, key, value)

        client.update_model_version(
            name=model_name,
            version=version,
            description=(
                f"{spec['lifecycle']} — {spec['decision']}. "
                f"Validation RMSE: {spec['rmse']:,.2f}; MAE: {spec['mae']:,.2f}. "
                "Synthetic record prepared for the Registry presentation slide."
            ),
        )
        print(
            f"Created {model_name} v{version} | alias={spec['alias']} "
            f"| status={spec['lifecycle']}"
        )

    return versions


def main() -> None:
    args = parse_args()
    versions = seed_registry(args.tracking_uri, args.experiment_name, args.model_name)
    print(f"Registry presentation versions ready: {', '.join(versions)}")


if __name__ == "__main__":
    main()
