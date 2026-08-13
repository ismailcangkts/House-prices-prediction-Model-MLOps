from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

CATBOOST_TEMP = Path("/private/tmp/codex_catboost_pkg")
sys.path.insert(0, str(CATBOOST_TEMP))
from catboost import CatBoostRegressor
sys.path.remove(str(CATBOOST_TEMP))

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
OUT = BASE / "graphics"
sys.path.insert(0, str(ROOT))
from pipeline_machine_learning import build_model_pipeline


INK = "#292625"
MUTED = "#716B68"
TRAIN_COLOR = "#447B9B"
VALID_COLOR = "#D67A45"
GRID = "#E8E4E0"

train = pd.read_csv(ROOT / "data/processed/train.csv")
valid = pd.read_csv(ROOT / "data/processed/validation.csv")
X_all = train.drop(columns="SalePrice")
y_all = train["SalePrice"]
X_valid = valid.drop(columns="SalePrice")
y_valid = valid["SalePrice"]

rng = np.random.default_rng(42)
shuffled_indices = rng.permutation(len(train))
fractions = np.array([0.20, 0.40, 0.60, 0.80, 1.00])


def model_factories():
    return {
        "Ridge Regression": lambda: build_model_pipeline(
            Ridge(alpha=10.0), scale_features=True
        ),
        "Random Forest": lambda: build_model_pipeline(
            RandomForestRegressor(
                n_estimators=1000, max_depth=20, random_state=42, n_jobs=-1
            )
        ),
        "Gradient Boosting": lambda: build_model_pipeline(
            GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=42
            )
        ),
        "CatBoost": lambda: build_model_pipeline(
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
for model_name, factory in model_factories().items():
    for fraction in fractions:
        n_samples = int(round(len(train) * fraction))
        subset = np.sort(shuffled_indices[:n_samples])
        X_sub = X_all.iloc[subset]
        y_sub = y_all.iloc[subset]

        model = factory()
        model.fit(X_sub, y_sub)
        train_pred = model.predict(X_sub)
        valid_pred = model.predict(X_valid)
        train_rmse = mean_squared_error(y_sub, train_pred) ** 0.5
        valid_rmse = mean_squared_error(y_valid, valid_pred) ** 0.5
        rows.append(
            {
                "model": model_name,
                "train_fraction": fraction,
                "train_samples": n_samples,
                "train_rmse": train_rmse,
                "validation_rmse": valid_rmse,
            }
        )
        print(
            f"{model_name} · {fraction:.0%}: train={train_rmse:.2f}, "
            f"validation={valid_rmse:.2f}",
            flush=True,
        )

curves = pd.DataFrame(rows)
curves.to_csv(BASE / "tables" / "model_learning_curves.csv", index=False)

global_max = curves[["train_rmse", "validation_rmse"]].to_numpy().max() / 1000 * 1.12
filenames = {
    "Ridge Regression": "32_learning_curve_ridge",
    "Random Forest": "33_learning_curve_random_forest",
    "Gradient Boosting": "34_learning_curve_gradient_boosting",
    "CatBoost": "35_learning_curve_catboost",
}


def draw_curve(ax, frame, model_name, compact=False):
    x = frame["train_samples"].to_numpy()
    train_y = frame["train_rmse"].to_numpy() / 1000
    valid_y = frame["validation_rmse"].to_numpy() / 1000
    ax.plot(x, train_y, color=TRAIN_COLOR, marker="o", linewidth=2.5,
            markersize=6, label="Train RMSE")
    ax.plot(x, valid_y, color=VALID_COLOR, marker="o", linewidth=2.5,
            markersize=6, label="Validation RMSE")
    ax.fill_between(x, train_y, valid_y, color=VALID_COLOR, alpha=0.08)
    ax.set_ylim(0, global_max)
    ax.set_xticks(x, [f"{int(f * 100)}%" for f in frame["train_fraction"]])
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(model_name, loc="left", fontsize=14 if compact else 18,
                 color=INK, fontweight="bold")
    ax.set_xlabel("Training data used", fontsize=9.5, color=MUTED)
    ax.set_ylabel("RMSE ($K)", fontsize=9.5, color=MUTED)
    gap = valid_y[-1] - train_y[-1]
    ax.text(x[-1], valid_y[-1] + global_max * 0.025,
            f"Validation {valid_y[-1]:.1f}K", ha="right", va="bottom",
            fontsize=8.5, color=VALID_COLOR, fontweight="bold")
    ax.text(0.98, 0.05, f"Final gap: {gap:.1f}K", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=MUTED,
            fontweight="bold")


for model_name, filename in filenames.items():
    frame = curves[curves["model"] == model_name].sort_values("train_fraction")
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.95, top=0.76, bottom=0.17)
    fig.text(0.08, 0.91, "LEARNING CURVE", fontsize=21, color=INK, fontweight="bold")
    fig.text(0.08, 0.845, "Training size vs RMSE · fixed validation set",
             fontsize=10.5, color=MUTED)
    draw_curve(ax, frame, model_name)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9.5)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight",
                    pad_inches=0.12, dpi=180)
    plt.close(fig)

# One-slide overview.
fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), dpi=180, facecolor="white")
fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.10, hspace=0.38, wspace=0.25)
fig.text(0.06, 0.93, "MODEL LEARNING CURVES", fontsize=22, color=INK, fontweight="bold")
fig.text(0.06, 0.88, "More training data should reduce validation error and narrow the generalization gap.",
         fontsize=10.5, color=MUTED)
for ax, model_name in zip(axes.flat, filenames):
    frame = curves[curves["model"] == model_name].sort_values("train_fraction")
    draw_curve(ax, frame, model_name, compact=True)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.96, 0.935), fontsize=9.5)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"36_learning_curves_overview.{ext}", bbox_inches="tight",
                pad_inches=0.12, dpi=180)
plt.close(fig)

print("saved learning curves")
