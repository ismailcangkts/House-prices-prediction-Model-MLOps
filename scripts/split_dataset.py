from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SELECTED_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "Neighborhood",
    "YearBuilt",
    "YearRemodAdd",
    "OverallCond",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "1stFlrSF",
    "2ndFlrSF",
    "GarageCars",
    "GarageArea",
    "GarageType",
    "GarageFinish",
    "GarageYrBlt",
    "FullBath",
    "HalfBath",
    "TotRmsAbvGrd",
    "KitchenQual",
    "ExterQual",
    "BsmtQual",
    "BsmtExposure",
    "Fireplaces",
    "FireplaceQu",
    "LotArea",
    "LotFrontage",
    "MSSubClass",
    "MSZoning",
    "HouseStyle",
    "OpenPorchSF",
]

TARGET = "SalePrice"

SPLITS = {
    "initial_train": 0.55,
    "validation": 0.15,
    "test": 0.10,
    "reference": 0.10,
    "production": 0.10,
}


def calculate_split_sizes(total_rows: int) -> dict[str, int]:
    raw_sizes = {name: total_rows * ratio for name, ratio in SPLITS.items()}
    sizes = {name: int(size) for name, size in raw_sizes.items()}

    remaining = total_rows - sum(sizes.values())
    by_remainder = sorted(
        raw_sizes,
        key=lambda name: raw_sizes[name] - sizes[name],
        reverse=True,
    )

    for name in by_remainder[:remaining]:
        sizes[name] += 1

    return sizes


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = SELECTED_FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in input dataset: {missing}")

    return df[required_columns].copy()


def split_dataset(input_path: Path, output_dir: Path, random_state: int) -> None:
    df = pd.read_csv(input_path)
    selected_df = select_model_columns(df)
    shuffled = selected_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    processed_dir = output_dir / "processed"
    production_dir = output_dir / "production"
    targets_dir = output_dir / "targets"

    processed_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    selected_df.to_csv(processed_dir / "selected_features.csv", index=False)
    sizes = calculate_split_sizes(len(shuffled))

    start = 0
    manifest_rows = []
    for split_name, size in sizes.items():
        end = start + size
        split_df = shuffled.iloc[start:end]
        if split_name == "initial_train":
            output_path = processed_dir / "train.csv"
            split_df.to_csv(output_path, index=False)
        elif split_name == "production":
            output_path = production_dir / "batch_001.csv"
            target_path = targets_dir / "batch_001_targets.csv"
            split_df.drop(columns=[TARGET]).to_csv(output_path, index=False)
            split_df[[TARGET]].to_csv(target_path, index=False)
        else:
            output_path = processed_dir / f"{split_name}.csv"
            split_df.to_csv(output_path, index=False)

        manifest_rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "ratio": SPLITS[split_name],
                "path": str(output_path),
            }
        )
        start = end

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the cleaned Ames housing dataset for MLOps workflows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cleaned_train.csv"),
        help="Path to the cleaned source dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base directory where processed, production, and target files will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used for deterministic shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(args.input, args.output_dir, args.random_state)


if __name__ == "__main__":
    main()
