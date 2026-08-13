from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


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
indices = rng.permutation(len(train))
fractions = np.array([0.20, 0.40, 0.60, 0.80, 1.00])
rows = []

for fraction in fractions:
    n_samples = int(round(len(train) * fraction))
    subset = np.sort(indices[:n_samples])
    X_sub = X_all.iloc[subset]
    y_sub = y_all.iloc[subset]

    model = build_model_pipeline(Ridge(alpha=10.0), scale_features=True)
    model.fit(X_sub, y_sub)
    train_pred = model.predict(X_sub)
    valid_pred = model.predict(X_valid)

    rows.append({
        "train_fraction": fraction,
        "train_samples": n_samples,
        "train_r2": r2_score(y_sub, train_pred),
        "validation_r2": r2_score(y_valid, valid_pred),
        "train_mse": mean_squared_error(y_sub, train_pred),
        "validation_mse": mean_squared_error(y_valid, valid_pred),
    })

history = pd.DataFrame(rows)
history.to_csv(BASE / "tables" / "ridge_training_history.csv", index=False)
x = history["train_samples"].to_numpy()
tick_labels = [f"{int(value * 100)}%" for value in history["train_fraction"]]


def base_figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180, facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.95, top=0.76, bottom=0.17)
    fig.text(0.08, 0.91, title, fontsize=21, color=INK, fontweight="bold")
    fig.text(0.08, 0.845, subtitle, fontsize=10.5, color=MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Training data used", fontsize=10, color=MUTED)
    return fig, ax


def save(fig, filename):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{filename}.{ext}", bbox_inches="tight",
                    pad_inches=0.12, dpi=180)
    plt.close(fig)


# Accuracy-like curve: R².
fig, ax = base_figure(
    "RIDGE · R² LEARNING CURVE",
    "Accuracy-like regression score · higher is better",
)
train_r2 = history["train_r2"].to_numpy()
valid_r2 = history["validation_r2"].to_numpy()
ax.plot(x, train_r2, marker="o", linewidth=2.7, markersize=7,
        color=TRAIN_COLOR, label="Train R²")
ax.plot(x, valid_r2, marker="o", linewidth=2.7, markersize=7,
        color=VALID_COLOR, label="Validation R²")
ax.fill_between(x, train_r2, valid_r2, color=VALID_COLOR, alpha=0.08)
lower = min(train_r2.min(), valid_r2.min()) - 0.04
ax.set_ylim(max(0, lower), min(1.01, max(train_r2.max(), valid_r2.max()) + 0.04))
ax.set_ylabel("R² score", fontsize=10, color=MUTED)
ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=9.5)
ax.text(x[-1], train_r2[-1] - 0.012, f"Train {train_r2[-1]:.3f}",
        ha="right", va="top", color=TRAIN_COLOR, fontsize=9, fontweight="bold")
ax.text(x[-1], valid_r2[-1] + 0.010, f"Validation {valid_r2[-1]:.3f}",
        ha="right", va="bottom", color=VALID_COLOR, fontsize=9, fontweight="bold")
save(fig, "37_ridge_r2_accuracy_curve")


# Loss curve: dollar-scale prediction MSE, shown in million dollars squared.
fig, ax = base_figure(
    "RIDGE · MSE LOSS CURVE",
    "Prediction loss by training size · lower is better",
)
train_loss = history["train_mse"].to_numpy() / 1_000_000
valid_loss = history["validation_mse"].to_numpy() / 1_000_000
ax.plot(x, train_loss, marker="o", linewidth=2.7, markersize=7,
        color=TRAIN_COLOR, label="Train MSE")
ax.plot(x, valid_loss, marker="o", linewidth=2.7, markersize=7,
        color=VALID_COLOR, label="Validation MSE")
ax.fill_between(x, train_loss, valid_loss, color=VALID_COLOR, alpha=0.08)
ax.set_ylim(0, max(train_loss.max(), valid_loss.max()) * 1.15)
ax.set_ylabel("MSE (million $²)", fontsize=10, color=MUTED)
ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9.5)
ax.text(x[-1], train_loss[-1] + 25, f"Train {train_loss[-1]:.0f}",
        ha="right", va="bottom", color=TRAIN_COLOR, fontsize=9, fontweight="bold")
ax.text(x[-1], valid_loss[-1] - 25, f"Validation {valid_loss[-1]:.0f}",
        ha="right", va="top", color=VALID_COLOR, fontsize=9, fontweight="bold")
save(fig, "38_ridge_mse_loss_curve")


# One combined slide visual.
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), dpi=180, facecolor="white")
fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.17, wspace=0.25)
fig.text(0.06, 0.91, "RIDGE TRAINING METRICS", fontsize=21, color=INK, fontweight="bold")
fig.text(0.06, 0.845, "Train and validation behavior as the training set grows",
         fontsize=10.5, color=MUTED)

axes[0].plot(x, train_r2, marker="o", linewidth=2.5, color=TRAIN_COLOR, label="Train")
axes[0].plot(x, valid_r2, marker="o", linewidth=2.5, color=VALID_COLOR, label="Validation")
axes[0].set_title("R² · higher is better", loc="left", fontsize=13, fontweight="bold", color=INK)
axes[0].set_ylabel("R² score", color=MUTED)
axes[0].set_ylim(max(0, lower), min(1.01, max(train_r2.max(), valid_r2.max()) + 0.04))

axes[1].plot(x, train_loss, marker="o", linewidth=2.5, color=TRAIN_COLOR, label="Train")
axes[1].plot(x, valid_loss, marker="o", linewidth=2.5, color=VALID_COLOR, label="Validation")
axes[1].set_title("MSE loss · lower is better", loc="left", fontsize=13, fontweight="bold", color=INK)
axes[1].set_ylabel("MSE (million $²)", color=MUTED)
axes[1].set_ylim(0, max(train_loss.max(), valid_loss.max()) * 1.15)

for ax in axes:
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Training data used", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right",
           bbox_to_anchor=(0.96, 0.925), fontsize=9.5)
save(fig, "39_ridge_accuracy_loss_combined")

print(history.round(4).to_string(index=False))
