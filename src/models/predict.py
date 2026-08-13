from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import mlflow
except ImportError:
    mlflow = None


DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_URI = "models:/HousePriceModel@champion"
DEFAULT_INPUT_PATH = Path("data/production/batch_001.csv")
DEFAULT_OUTPUT_PATH = Path("data/predictions/batch_001_predictions.csv")


def load_model_from_registry(
    model_uri: str = DEFAULT_MODEL_URI,
    tracking_uri: str = DEFAULT_TRACKING_URI,
):
    if mlflow is None:
        raise ImportError(
            "mlflow is required to load a model from the registry. "
            "Use the project virtual environment."
        )

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)


def predict_dataframe(
    input_df: pd.DataFrame,
    model_uri: str = DEFAULT_MODEL_URI,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    model = load_model_from_registry(model_uri=model_uri, tracking_uri=tracking_uri)
    predictions = model.predict(input_df)

    output_df = input_df.copy()
    output_df["prediction"] = predictions
    return output_df


def predict_file(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_uri: str = DEFAULT_MODEL_URI,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    input_df = pd.read_csv(input_path)
    output_df = predict_dataframe(
        input_df=input_df,
        model_uri=model_uri,
        tracking_uri=tracking_uri,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df
