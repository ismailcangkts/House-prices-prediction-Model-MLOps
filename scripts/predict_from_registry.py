from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import (
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_URI,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRACKING_URI,
    predict_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the champion model from MLflow Registry and predict a batch."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_file(
        input_path=args.input_path,
        output_path=args.output_path,
        model_uri=args.model_uri,
        tracking_uri=args.tracking_uri,
    )

    print(f"Predictions saved to: {args.output_path}")
    print(f"Rows predicted: {len(predictions)}")
    print(predictions[["prediction"]].head().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
