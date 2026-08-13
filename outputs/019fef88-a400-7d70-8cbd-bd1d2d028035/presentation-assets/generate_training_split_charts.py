from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT = BASE / "graphics"
results = pd.read_csv(BASE / "tables" / "requested_model_comparison.csv").set_index("model")

order = ["Ridge Regression", "Random Forest", "Gradient Boosting", "CatBoost"]
labels = ["Ridge", "Random Forest", "Gradient Boosting", "CatBoost"]
results = results.loc[order]

INK = "#292625"
MUTED = "#716B68"
TRAIN = "#9CB9C9"
VALID = "#6D63A8"
TEST = "#D67A45"
GRID = "#E8E4E0"


def chart(metric, title, subtitle, filename, value_format, ylim=None):
    fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.96, top=0.75, bottom=0.18)
    fig.text(0.07, 0.91, title, fontsize=22, fontweight="bold", color=INK)
    fig.text(0.07, 0.845, subtitle, fontsize=11, color=MUTED)

    x = np.arange(len(labels))
    width = 0.23
    series = [
        ("Train", results[f"train_{metric}"].to_numpy(), TRAIN, -width),
        ("Validation", results[f"validation_{metric}"].to_numpy(), VALID, 0),
        ("Test", results[f"test_{metric}"].to_numpy(), TEST, width),
    ]
    for name, values, color, offset in series:
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    value_format(value), ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xticks(x, labels, fontsize=10, color=INK)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymax = max(results[f"train_{metric}"].max(), results[f"validation_{metric}"].max(), results[f"test_{metric}"].max())
        ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=10)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight", pad_inches=0.12, dpi=180)
    plt.close(fig)


chart(
    "rmse",
    "RMSE · TRAIN vs VALIDATION vs TEST",
    "Same preprocessing pipeline and fixed data splits · lower is better",
    "29_model_rmse_splits",
    lambda value: f"{value / 1000:.1f}K",
)

chart(
    "mae",
    "MAE · TRAIN vs VALIDATION vs TEST",
    "Average absolute prediction error in dollars · lower is better",
    "30_model_mae_splits",
    lambda value: f"{value / 1000:.1f}K",
)

chart(
    "r2",
    "R² · TRAIN vs VALIDATION vs TEST",
    "Regression equivalent of an accuracy-style fit score · higher is better",
    "31_model_r2_splits",
    lambda value: f"{value:.3f}",
    ylim=(0.78, 1.025),
)

print("generated train/validation/test charts")
