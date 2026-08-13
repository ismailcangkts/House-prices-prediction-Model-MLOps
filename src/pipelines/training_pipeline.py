from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import joblib
import pandas as pd

from pipeline_machine_learning import build_model_candidates, evaluate_model

try:
    import mlflow
except ImportError:
    mlflow = None


TARGET = "SalePrice"
DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_VALIDATION_PATH = Path("data/processed/validation.csv")
DEFAULT_MODEL_PATH = Path("artifacts/models/best_model.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/model_metrics.csv")
DEFAULT_BEST_METRICS_PATH = Path("artifacts/best_model_metrics.json")
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT = "house_price_prediction"
DEFAULT_MLFLOW_BACKEND_STORE_URI = "sqlite:///storage/mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_ROOT = "artifacts/mlflow"
DEFAULT_REGISTERED_MODEL_NAME = "HousePriceModel"
DEFAULT_MODEL_ALIAS = "champion"
MLFLOW_SERVER_LOG_PATH = Path("artifacts/mlflow_server.log")


def load_labeled_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} must contain the target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_model_params(model_pipeline) -> dict:
    target_model = model_pipeline.named_steps["model"]
    regressor = target_model.regressor
    params = regressor.get_params()
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def is_http_tracking_uri(tracking_uri: str) -> bool:
    return urlparse(tracking_uri).scheme in {"http", "https"}


def is_mlflow_server_available(tracking_uri: str, timeout_seconds: float = 1.5) -> bool:
    request = Request(tracking_uri, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except URLError:
        return False
    except TimeoutError:
        return False


def start_local_mlflow_server(
    tracking_uri: str,
    backend_store_uri: str,
    artifact_root: str,
    startup_timeout_seconds: int = 15,
) -> bool:
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.hostname not in {"127.0.0.1", "localhost"}:
        print(f"MLflow server auto-start skipped for non-local URI: {tracking_uri}")
        return False

    port = parsed_uri.port or 5000
    host = parsed_uri.hostname
    MLFLOW_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = MLFLOW_SERVER_LOG_PATH.open("a", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        artifact_root,
        "--host",
        host,
        "--port",
        str(port),
    ]
    subprocess.Popen(command, stdout=log_file, stderr=log_file)

    for _ in range(startup_timeout_seconds):
        if is_mlflow_server_available(tracking_uri):
            print(f"MLflow server started at {tracking_uri}")
            return True
        time.sleep(1)

    print(f"MLflow server could not be reached. See log: {MLFLOW_SERVER_LOG_PATH}")
    return False


def ensure_mlflow_tracking_available(
    tracking_uri: str,
    backend_store_uri: str,
    artifact_root: str,
    auto_start_server: bool,
) -> bool:
    if not is_http_tracking_uri(tracking_uri):
        return True

    if is_mlflow_server_available(tracking_uri):
        return True

    if not auto_start_server:
        print(f"MLflow server is not available at {tracking_uri}.")
        return False

    print(f"MLflow server is not available at {tracking_uri}. Starting it...")
    return start_local_mlflow_server(
        tracking_uri=tracking_uri,
        backend_store_uri=backend_store_uri,
        artifact_root=artifact_root,
    )


def log_training_run_to_mlflow(
    result: dict,
    train_path: Path,
    validation_path: Path,
    model_path: Path,
    metrics_path: Path,
    best_metrics_path: Path,
    train_rows: int,
    validation_rows: int,
    feature_count: int,
    tracking_uri: str,
    experiment_name: str,
    backend_store_uri: str,
    artifact_root: str,
    auto_start_server: bool,
    registered_model_name: str,
    model_alias: str,
    retraining_metadata: dict | None = None,
) -> dict | None:
    if mlflow is None:
        print("MLflow skipped: mlflow package is not installed in this Python environment.")
        return None

    if not ensure_mlflow_tracking_available(
        tracking_uri=tracking_uri,
        backend_store_uri=backend_store_uri,
        artifact_root=artifact_root,
        auto_start_server=auto_start_server,
    ):
        print("MLflow skipped: tracking server is not available.")
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=result["model"]) as run:
        mlflow.log_param("model_name", result["model"])
        mlflow.log_param("target", TARGET)
        mlflow.log_param("train_path", str(train_path))
        mlflow.log_param("validation_path", str(validation_path))
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("validation_rows", validation_rows)
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_params(get_model_params(result["pipeline"]))

        mlflow.log_metric("mae", result["mae"])
        mlflow.log_metric("rmse", result["rmse"])
        mlflow.log_metric("mae_ratio_percent", result["mae_ratio_%"])
        mlflow.log_metric("rmse_ratio_percent", result["rmse_ratio_%"])

        if retraining_metadata:
            mlflow.set_tag("retraining_trigger", retraining_metadata["trigger"])
            mlflow.set_tag("retraining_reason", retraining_metadata["reason"])
            mlflow.set_tag("drift_report_path", retraining_metadata["drift_report_path"])
            mlflow.log_param(
                "drifted_critical_features",
                ",".join(retraining_metadata["drifted_critical_features"]) or "None",
            )
            mlflow.log_metric(
                "drift_weighted_score",
                retraining_metadata["weighted_drift_score"],
            )
            mlflow.log_metric(
                "drifted_columns_count",
                retraining_metadata["drifted_columns_count"],
            )
            mlflow.log_metric(
                "critical_feature_drift_count",
                retraining_metadata["critical_feature_drift_count"],
            )
            drift_report_path = Path(retraining_metadata["drift_report_path"])
            if drift_report_path.exists():
                mlflow.log_artifact(str(drift_report_path), artifact_path="drift")

        mlflow.sklearn.log_model(
            sk_model=result["pipeline"],
            artifact_path="model",
        )
        mlflow.log_artifact(str(model_path), artifact_path="model_file")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(best_metrics_path), artifact_path="metrics")

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=registered_model.name,
            alias=model_alias,
            version=registered_model.version,
        )

        return {
            "run_id": run.info.run_id,
            "registered_model_name": registered_model.name,
            "registered_model_version": registered_model.version,
            "registered_model_alias": model_alias,
            "model_uri": model_uri,
            "alias_model_uri": f"models:/{registered_model.name}@{model_alias}",
        }


def run_training_pipeline(
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    best_metrics_path: Path = DEFAULT_BEST_METRICS_PATH,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    backend_store_uri: str = DEFAULT_MLFLOW_BACKEND_STORE_URI,
    artifact_root: str = DEFAULT_MLFLOW_ARTIFACT_ROOT,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    enable_mlflow: bool = True,
    auto_start_mlflow_server: bool = True,
    retraining_metadata: dict | None = None,
) -> dict:
    X_train, y_train = load_labeled_dataset(train_path)
    X_valid, y_valid = load_labeled_dataset(validation_path)

    results = []
    for model_name, model_pipeline in build_model_candidates().items():
        results.append(
            evaluate_model(
                model_name,
                model_pipeline,
                X_train,
                X_valid,
                y_train,
                y_valid,
            )
        )

    best_result = min(results, key=lambda result: result["rmse"])
    best_pipeline = best_result["pipeline"]

    metrics_df = (
        pd.DataFrame(results)
        .drop(columns=["pipeline"])
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)
    metrics_df.to_csv(metrics_path, index=False)

    best_metrics = {
        "model": best_result["model"],
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "mae_ratio_%": best_result["mae_ratio_%"],
        "rmse_ratio_%": best_result["rmse_ratio_%"],
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "feature_count": X_train.shape[1],
        "model_path": str(model_path),
    }
    best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    mlflow_result = None
    if enable_mlflow:
        mlflow_result = log_training_run_to_mlflow(
            result=best_result,
            train_path=train_path,
            validation_path=validation_path,
            model_path=model_path,
            metrics_path=metrics_path,
            best_metrics_path=best_metrics_path,
            train_rows=len(X_train),
            validation_rows=len(X_valid),
            feature_count=X_train.shape[1],
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            backend_store_uri=backend_store_uri,
            artifact_root=artifact_root,
            auto_start_server=auto_start_mlflow_server,
            registered_model_name=registered_model_name,
            model_alias=model_alias,
            retraining_metadata=retraining_metadata,
        )
        if mlflow_result:
            best_metrics.update(
                {
                    "mlflow_run_id": mlflow_result["run_id"],
                    "registered_model_name": mlflow_result["registered_model_name"],
                    "registered_model_version": mlflow_result["registered_model_version"],
                    "registered_model_alias": mlflow_result["registered_model_alias"],
                    "mlflow_model_uri": mlflow_result["model_uri"],
                    "mlflow_alias_model_uri": mlflow_result["alias_model_uri"],
                }
            )
        best_metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    return {
        "best_metrics": best_metrics,
        "metrics": metrics_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the house price model.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--best-metrics-path", type=Path, default=DEFAULT_BEST_METRICS_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-backend-store-uri", default=DEFAULT_MLFLOW_BACKEND_STORE_URI)
    parser.add_argument("--mlflow-artifact-root", default=DEFAULT_MLFLOW_ARTIFACT_ROOT)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--no-auto-start-mlflow", action="store_true")
    parser.add_argument("--retraining-trigger", default=None)
    parser.add_argument("--retraining-reason", default=None)
    parser.add_argument("--drift-report-path", type=Path, default=None)
    parser.add_argument("--weighted-drift-score", type=float, default=None)
    parser.add_argument("--drifted-columns-count", type=int, default=None)
    parser.add_argument("--critical-feature-drift-count", type=int, default=None)
    parser.add_argument("--drifted-critical-features", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retraining_metadata = None
    if args.retraining_trigger and args.retraining_reason and args.drift_report_path:
        retraining_metadata = {
            "trigger": args.retraining_trigger,
            "reason": args.retraining_reason,
            "drift_report_path": str(args.drift_report_path),
            "weighted_drift_score": args.weighted_drift_score or 0.0,
            "drifted_columns_count": args.drifted_columns_count or 0,
            "critical_feature_drift_count": args.critical_feature_drift_count or 0,
            "drifted_critical_features": [
                feature
                for feature in args.drifted_critical_features.split(",")
                if feature
            ],
        }

    output = run_training_pipeline(
        train_path=args.train_path,
        validation_path=args.validation_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        best_metrics_path=args.best_metrics_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        backend_store_uri=args.mlflow_backend_store_uri,
        artifact_root=args.mlflow_artifact_root,
        registered_model_name=args.registered_model_name,
        model_alias=args.model_alias,
        enable_mlflow=not args.disable_mlflow,
        auto_start_mlflow_server=not args.no_auto_start_mlflow,
        retraining_metadata=retraining_metadata,
    )

    print("Training completed.")
    print(output["metrics"].round(2).to_string(index=False))
    print(f"Best model saved to: {output['best_metrics']['model_path']}")
    if output["best_metrics"].get("mlflow_run_id"):
        print(f"MLflow run id: {output['best_metrics']['mlflow_run_id']}")
    if output["best_metrics"].get("registered_model_name"):
        print(
            "Registered model: "
            f"{output['best_metrics']['registered_model_name']} "
            f"v{output['best_metrics']['registered_model_version']}"
        )
        print(
            "Model alias: "
            f"{output['best_metrics']['registered_model_alias']} -> "
            f"v{output['best_metrics']['registered_model_version']}"
        )


if __name__ == "__main__":
    main()
