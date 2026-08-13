from __future__ import annotations

import os
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("artifacts/.matplotlib_cache")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/processed/reference.csv")
OUTPUT_PATH = Path("artifacts/feature_saleprice_correlations.png")
TARGET_COLUMN = "SalePrice"


def calculate_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    features = [column for column in df.columns if column != TARGET_COLUMN]
    rows = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            correlation = df[feature].corr(df[TARGET_COLUMN])
            rows.append(
                {
                    "feature": feature,
                    "correlation": correlation,
                    "method": "pearson",
                }
            )
            continue

        dummies = pd.get_dummies(df[feature], prefix=feature, dummy_na=True)
        dummies = dummies.loc[:, dummies.nunique(dropna=False) > 1]
        if dummies.empty:
            rows.append(
                {
                    "feature": feature,
                    "correlation": 0.0,
                    "method": "one-hot no varying category",
                }
            )
            continue

        dummy_correlations = dummies.apply(lambda column: column.corr(df[TARGET_COLUMN]))
        dummy_correlations = dummy_correlations.dropna()
        if dummy_correlations.empty:
            rows.append(
                {
                    "feature": feature,
                    "correlation": 0.0,
                    "method": "one-hot no valid correlation",
                }
            )
            continue

        strongest_dummy = dummy_correlations.abs().idxmax()
        rows.append(
            {
                "feature": feature,
                "correlation": dummy_correlations[strongest_dummy],
                "method": f"one-hot strongest: {strongest_dummy}",
            }
        )

    result = pd.DataFrame(rows)
    result["abs_correlation"] = result["correlation"].abs()
    return result.sort_values("abs_correlation", ascending=True)


def plot_correlations(correlations: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 9))
    colors = correlations["correlation"].apply(
        lambda value: "#2ca25f" if value >= 0 else "#de2d26"
    )
    plt.barh(correlations["feature"], correlations["correlation"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Correlation with SalePrice")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    correlations = calculate_feature_correlations(df)
    plot_correlations(correlations)

    print(f"Correlation plot saved to: {OUTPUT_PATH}")
    print(
        correlations.sort_values("abs_correlation", ascending=False)
        .drop(columns="abs_correlation")
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
