from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_REFERENCE_PATH = Path("data/processed/reference.csv")
DEFAULT_OUTPUT_PATH = Path("data/production/batch_no_drift.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a normal production batch from the reference distribution "
            "for a no-drift sanity check."
        )
    )
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-column", default="SalePrice")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def create_normal_production_batch(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    target_column: str = "SalePrice",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    reference_df = pd.read_csv(reference_path)
    feature_df = reference_df.drop(columns=[target_column], errors="ignore")

    if sample_size is None:
        batch_df = feature_df.sample(frac=1.0, random_state=random_state)
    else:
        batch_df = feature_df.sample(
            n=sample_size,
            replace=sample_size > len(feature_df),
            random_state=random_state,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(output_path, index=False)
    return batch_df


def main() -> None:
    args = parse_args()
    batch_df = create_normal_production_batch(
        reference_path=args.reference_path,
        output_path=args.output_path,
        target_column=args.target_column,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(f"Normal production batch saved to: {args.output_path}")
    print(f"Rows: {len(batch_df)}")
    print(f"Columns: {len(batch_df.columns)}")


if __name__ == "__main__":
    main()
