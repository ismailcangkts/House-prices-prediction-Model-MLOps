from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from pipeline_machine_learning import build_model_candidates, evaluate_model


TARGET = "SalePrice"
DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_VALIDATION_PATH = Path("data/processed/validation.csv")
DEFAULT_MODEL_PATH = Path("artifacts/models/best_model.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/model_metrics.csv")
DEFAULT_BEST_METRICS_PATH = Path("artifacts/best_model_metrics.json")


def load_labeled_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} must contain the target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def run_training_pipeline(
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    best_metrics_path: Path = DEFAULT_BEST_METRICS_PATH,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_training_pipeline(
        train_path=args.train_path,
        validation_path=args.validation_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        best_metrics_path=args.best_metrics_path,
    )

    print("Training completed.")
    print(output["metrics"].round(2).to_string(index=False))
    print(f"Best model saved to: {output['best_metrics']['model_path']}")


if __name__ == "__main__":
    main()
