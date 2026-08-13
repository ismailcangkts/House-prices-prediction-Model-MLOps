from pathlib import Path
import sys
import time

# Load the environment's numerical stack first, then expose the temporary
# CatBoost installation without replacing the project's pandas/numpy imports.
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CATBOOST_TEMP = Path("/private/tmp/codex_catboost_pkg")
sys.path.insert(0, str(CATBOOST_TEMP))
from catboost import CatBoostRegressor
sys.path.remove(str(CATBOOST_TEMP))

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pipeline_machine_learning import build_model_pipeline


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


train = pd.read_csv(ROOT / "data/processed/train.csv")
valid = pd.read_csv(ROOT / "data/processed/validation.csv")
test = pd.read_csv(ROOT / "data/processed/test.csv")

X_train = train.drop(columns="SalePrice")
y_train = train["SalePrice"]
X_valid = valid.drop(columns="SalePrice")
y_valid = valid["SalePrice"]
X_test = test.drop(columns="SalePrice")
y_test = test["SalePrice"]

models = {
    "Ridge Regression": build_model_pipeline(
        Ridge(alpha=10.0),
        scale_features=True,
    ),
    "Random Forest": build_model_pipeline(
        RandomForestRegressor(
            n_estimators=1000,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
    ),
    "Gradient Boosting": build_model_pipeline(
        GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    ),
    "CatBoost": build_model_pipeline(
        CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    ),
}

rows = []
for name, pipeline in models.items():
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    valid_pred = pipeline.predict(X_valid)
    test_pred = pipeline.predict(X_test)
    elapsed = time.perf_counter() - started

    train_metrics = metrics(y_train, train_pred)
    valid_metrics = metrics(y_valid, valid_pred)
    test_metrics = metrics(y_test, test_pred)
    rows.append(
        {
            "model": name,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "validation_mae": valid_metrics["mae"],
            "validation_rmse": valid_metrics["rmse"],
            "validation_r2": valid_metrics["r2"],
            "validation_mae_percent": valid_metrics["mae"] / y_valid.mean() * 100,
            "validation_rmse_percent": valid_metrics["rmse"] / y_valid.mean() * 100,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_mae_percent": test_metrics["mae"] / y_test.mean() * 100,
            "test_rmse_percent": test_metrics["rmse"] / y_test.mean() * 100,
            "fit_and_predict_seconds": elapsed,
        }
    )
    print(
        f"{name}: valid MAE={valid_metrics['mae']:.2f}, "
        f"valid RMSE={valid_metrics['rmse']:.2f}, "
        f"valid R2={valid_metrics['r2']:.4f}, "
        f"test MAE={test_metrics['mae']:.2f}, "
        f"test RMSE={test_metrics['rmse']:.2f}, "
        f"test R2={test_metrics['r2']:.4f}, seconds={elapsed:.2f}",
        flush=True,
    )

result = pd.DataFrame(rows).sort_values("validation_rmse").reset_index(drop=True)
output = Path(__file__).resolve().parent / "tables" / "requested_model_comparison.csv"
result.to_csv(output, index=False)
print("\nRESULTS")
print(result.round(2).to_string(index=False))
print(f"\nSaved: {output}")
