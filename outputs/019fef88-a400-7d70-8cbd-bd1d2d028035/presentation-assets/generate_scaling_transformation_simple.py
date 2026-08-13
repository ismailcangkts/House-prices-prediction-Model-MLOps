from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "graphics"

INK = "#292625"
MUTED = "#716B68"
PURPLE = "#6D63A8"
GREEN = "#27836D"
ORANGE = "#D67A45"
BLUE = "#447B9B"
GRID = "#E8E4E0"

train = pd.read_csv(ROOT / "data/processed/train.csv")


def clean_axis(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.tick_params(axis="both", colors=MUTED, labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.12, dpi=180)
    plt.close(fig)


# SCALING — one plot, one common standardized axis.
features = ["YearBuilt", "GarageArea", "TotalBsmtSF", "GrLivArea", "LotArea"]
scaled = StandardScaler().fit_transform(train[features].astype(float))

fig, ax = plt.subplots(figsize=(11.6, 6.2), dpi=180, facecolor="white")
fig.subplots_adjust(left=0.18, right=0.95, top=0.75, bottom=0.18)
fig.text(0.075, 0.91, "STANDARD SCALING", fontsize=22, fontweight="bold", color=INK)
fig.text(0.075, 0.845, "All selected features are shown on one comparable scale · fitted on train only",
         fontsize=11, color=MUTED)

bp = ax.boxplot(
    [scaled[:, i] for i in range(len(features))],
    vert=False,
    tick_labels=features,
    patch_artist=True,
    showfliers=False,
    widths=0.56,
    medianprops={"color": "white", "linewidth": 2},
    whiskerprops={"color": PURPLE, "linewidth": 1.5},
    capprops={"color": PURPLE, "linewidth": 1.5},
)
colors = [BLUE, GREEN, PURPLE, GREEN, ORANGE]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
    patch.set_alpha(0.92)

ax.axvspan(-1, 1, color=GREEN, alpha=0.07)
ax.axvline(0, color=INK, linewidth=1.5)
ax.set_xlim(-3.2, 4.2)
ax.set_xlabel("Standardized value (z-score)", fontsize=10.5, color=MUTED)
ax.text(0.5, 1.03, "mean = 0   ·   standard deviation = 1", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, color=GREEN, fontweight="bold")
clean_axis(ax)
fig.text(0.50, 0.065, "Scaling changes the unit—not the ordering or information carried by a feature.",
         ha="center", fontsize=10, color=MUTED)
save(fig, "25_scaling_standardscaler")


# TRANSFORMATION — one simple chart quantifying the skewness reduction.
y = train["SalePrice"].astype(float)
before = float(y.skew())
after = float(pd.Series(np.log1p(y)).skew())
reduction = (1 - abs(after) / abs(before)) * 100

fig, ax = plt.subplots(figsize=(11.6, 6.2), dpi=180, facecolor="white")
fig.subplots_adjust(left=0.22, right=0.93, top=0.75, bottom=0.20)
fig.text(0.075, 0.91, "LOG1P TARGET TRANSFORMATION", fontsize=22, fontweight="bold", color=INK)
fig.text(0.075, 0.845, "SalePrice transformation used by TransformedTargetRegressor",
         fontsize=11, color=MUTED)

labels = ["Original SalePrice", "log1p(SalePrice)"]
values = [before, after]
bars = ax.barh(labels, values, color=[ORANGE, GREEN], height=0.52)
ax.invert_yaxis()
ax.set_xlim(0, 2.05)
ax.set_xlabel("Skewness · closer to 0 is more symmetric", fontsize=10.5, color=MUTED)
ax.axvline(0, color=INK, linewidth=1.2)
for bar, value, color in zip(bars, values, [ORANGE, GREEN]):
    ax.text(value + 0.045, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
            ha="left", va="center", fontsize=15, color=color, fontweight="bold")
ax.text(0.98, 0.05, f"{reduction:.0f}% lower skewness", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13, color=GREEN, fontweight="bold")
clean_axis(ax)
fig.text(0.50, 0.065, "The right tail is compressed during training; predictions return to dollars with expm1.",
         ha="center", fontsize=10, color=MUTED)
save(fig, "26_transformation_saleprice_log1p")

print("generated simplified scaling and transformation charts")
