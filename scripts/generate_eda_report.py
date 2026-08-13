from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


MATPLOTLIB_CACHE_DIR = Path("artifacts/.matplotlib_cache")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/processed/reference.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/eda")
TARGET_COLUMN = "SalePrice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an EDA report for the house price data.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_saleprice_distribution(df: pd.DataFrame, target_column: str, output_dir: Path) -> str:
    path = output_dir / "saleprice_distribution.png"
    plt.figure(figsize=(9, 5))
    plt.hist(df[target_column], bins=24, color="#376996", edgecolor="white")
    plt.title("SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def save_missing_values_plot(df: pd.DataFrame, output_dir: Path) -> str:
    path = output_dir / "missing_values.png"
    missing = df.isna().mean().sort_values(ascending=True)
    missing = missing[missing > 0]

    plt.figure(figsize=(9, max(3, len(missing) * 0.35)))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    else:
        plt.barh(missing.index, missing.values * 100, color="#8e5572")
        plt.xlabel("Missing Values (%)")
        plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def save_top_numeric_correlations(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
) -> tuple[str, pd.DataFrame]:
    numeric_df = df.select_dtypes(include="number")
    correlations = (
        numeric_df.corr(numeric_only=True)[target_column]
        .drop(labels=[target_column])
        .dropna()
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(15)
        .sort_values()
    )

    path = output_dir / "top_numeric_correlations.png"
    plt.figure(figsize=(9, 6))
    colors = ["#2ca25f" if value >= 0 else "#de2d26" for value in correlations]
    plt.barh(correlations.index, correlations.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top Numeric Correlations with SalePrice")
    plt.xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

    table = correlations.sort_values(key=lambda series: series.abs(), ascending=False)
    return path.name, table.reset_index().rename(columns={"index": "feature", target_column: "correlation"})


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> str:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(numeric_only=True)
    path = output_dir / "numeric_correlation_heatmap.png"

    plt.figure(figsize=(11, 9))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path.name


def calculate_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    target = df[target_column]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "target": {
            "min": float(target.min()),
            "median": float(target.median()),
            "mean": float(target.mean()),
            "max": float(target.max()),
            "std": float(target.std()),
        },
    }


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_html(index=False, classes="table", border=0)


def build_html_report(
    summary: dict[str, Any],
    images: dict[str, str],
    missing_table: pd.DataFrame,
    numeric_describe: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    top_correlations: pd.DataFrame,
) -> str:
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>House Price EDA Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 28px; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric .label {{ color: #627d98; font-size: 13px; }}
    .metric .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; border-radius: 8px; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>House Price EDA Report</h1>
  <section class="grid">
    <div class="metric"><div class="label">Rows</div><div class="value">{summary["rows"]}</div></div>
    <div class="metric"><div class="label">Columns</div><div class="value">{summary["columns"]}</div></div>
    <div class="metric"><div class="label">Missing Values</div><div class="value">{summary["total_missing_values"]}</div></div>
    <div class="metric"><div class="label">Duplicate Rows</div><div class="value">{summary["duplicate_rows"]}</div></div>
  </section>

  <section>
    <h2>Target Summary</h2>
    <div class="grid">
      <div class="metric"><div class="label">Min SalePrice</div><div class="value">${summary["target"]["min"]:,.0f}</div></div>
      <div class="metric"><div class="label">Median SalePrice</div><div class="value">${summary["target"]["median"]:,.0f}</div></div>
      <div class="metric"><div class="label">Mean SalePrice</div><div class="value">${summary["target"]["mean"]:,.0f}</div></div>
      <div class="metric"><div class="label">Max SalePrice</div><div class="value">${summary["target"]["max"]:,.0f}</div></div>
    </div>
  </section>

  <section class="charts">
    <div><h2>SalePrice Distribution</h2><img src="{images["saleprice_distribution"]}"></div>
    <div><h2>Missing Values</h2><img src="{images["missing_values"]}"></div>
    <div><h2>Top Numeric Correlations</h2><img src="{images["top_numeric_correlations"]}"></div>
    <div><h2>Numeric Correlation Heatmap</h2><img src="{images["correlation_heatmap"]}"></div>
  </section>

  <section>
    <h2>Top Correlations with SalePrice</h2>
    {dataframe_to_html_table(top_correlations.round(4), 20)}
  </section>

  <section>
    <h2>Missing Values Table</h2>
    {dataframe_to_html_table(missing_table, 40)}
  </section>

  <section>
    <h2>Numeric Summary</h2>
    {numeric_describe.round(2).to_html(classes="table", border=0)}
  </section>

  <section>
    <h2>Categorical Summary</h2>
    {dataframe_to_html_table(categorical_summary, 40)}
  </section>
</body>
</html>
"""


def generate_eda_report(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    if target_column not in df.columns:
        raise ValueError(f"{input_path} must contain target column: {target_column}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = calculate_summary(df, target_column)
    missing_table = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_table["missing_percent"] = (missing_table["missing_count"] / len(df) * 100).round(2)
    missing_table = missing_table.sort_values("missing_count", ascending=False)

    categorical_summary = pd.DataFrame(
        [
            {
                "column": column,
                "unique_values": df[column].nunique(dropna=True),
                "top_value": df[column].mode(dropna=True).iloc[0] if not df[column].mode(dropna=True).empty else None,
                "top_count": int(df[column].value_counts(dropna=True).iloc[0]) if not df[column].value_counts(dropna=True).empty else 0,
            }
            for column in df.select_dtypes(exclude="number").columns
        ]
    )

    images = {}
    images["saleprice_distribution"] = save_saleprice_distribution(df, target_column, output_dir)
    images["missing_values"] = save_missing_values_plot(df, output_dir)
    images["top_numeric_correlations"], top_correlations = save_top_numeric_correlations(
        df,
        target_column,
        output_dir,
    )
    images["correlation_heatmap"] = save_correlation_heatmap(df, output_dir)

    numeric_describe = df.select_dtypes(include="number").describe().T
    html = build_html_report(
        summary=summary,
        images=images,
        missing_table=missing_table,
        numeric_describe=numeric_describe,
        categorical_summary=categorical_summary,
        top_correlations=top_correlations,
    )
    html_path = output_dir / "eda_report.html"
    json_path = output_dir / "eda_summary.json"
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")

    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    output = generate_eda_report(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
    )
    print(f"EDA HTML report saved to: {output['html_path']}")
    print(f"EDA JSON summary saved to: {output['json_path']}")
    print(json.dumps(output["summary"], indent=2, default=json_default))


if __name__ == "__main__":
    main()
