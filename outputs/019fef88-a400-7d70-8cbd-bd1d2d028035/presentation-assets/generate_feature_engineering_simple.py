from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "graphics"

INK = "#292625"
MUTED = "#716B68"
PURPLE = "#6D63A8"
PURPLE_PALE = "#F1EFF8"
GREEN = "#27836D"
GREEN_PALE = "#E8F3EF"
ORANGE = "#D67A45"
LINE = "#DDD8D4"

df = pd.read_csv(ROOT / "cleaned_train.csv")
lot_area = df["LotArea"].dropna().to_numpy()
lot_area_log = np.log1p(lot_area)

fig = plt.figure(figsize=(7.5, 8.7), dpi=180, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.5)
ax.set_ylim(0, 8.7)
ax.axis("off")

ax.text(3.75, 8.28, "RAW  →  ENGINEERED",
        ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
ax.text(3.75, 7.92, "New features express density, capacity and age",
        ha="center", va="center", fontsize=10, color=MUTED)


def transformation_row(y, raw, operation, engineered, color):
    ax.add_patch(FancyBboxPatch(
        (0.35, y), 2.25, 1.05,
        boxstyle="round,pad=0.03,rounding_size=0.13",
        facecolor="#F7F5F3", edgecolor=LINE, linewidth=1.2,
    ))
    ax.text(1.475, y + 0.53, raw, ha="center", va="center",
            fontsize=10.5, color=INK, fontweight="semibold", linespacing=1.3)

    ax.add_patch(FancyBboxPatch(
        (2.86, y + 0.22), 0.70, 0.61,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=color, edgecolor="none",
    ))
    ax.text(3.21, y + 0.525, operation, ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")

    ax.add_patch(FancyArrowPatch(
        (3.59, y + 0.53), (3.96, y + 0.53),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.8, color=color,
    ))

    ax.add_patch(FancyBboxPatch(
        (4.00, y), 3.15, 1.05,
        boxstyle="round,pad=0.03,rounding_size=0.13",
        facecolor=PURPLE_PALE, edgecolor=PURPLE, linewidth=1.3,
    ))
    ax.text(5.575, y + 0.53, engineered, ha="center", va="center",
            fontsize=11, color=PURPLE, fontweight="bold")


transformation_row(6.45, "GrLivArea\nTotRmsAbvGrd", "÷", "LivingAreaPerRoom", GREEN)
transformation_row(5.05, "GarageArea\nGarageCars", "÷", "GarageAreaPerCar", GREEN)
transformation_row(3.65, "YearBuilt", "2010−", "HouseAge", ORANGE)

ax.plot([0.35, 7.15], [3.20, 3.20], color=LINE, linewidth=1)
ax.text(0.40, 2.87, "LOG1P TRANSFORMATION", ha="left", va="center",
        fontsize=12.5, color=INK, fontweight="bold")
ax.text(7.10, 2.87, "Example: LotArea", ha="right", va="center",
        fontsize=9.5, color=MUTED)

before = fig.add_axes([0.08, 0.075, 0.35, 0.205])
after = fig.add_axes([0.57, 0.075, 0.35, 0.205])

before.hist(lot_area, bins=32, color=ORANGE, alpha=0.92, edgecolor="white", linewidth=0.4)
after.hist(lot_area_log, bins=32, color=GREEN, alpha=0.92, edgecolor="white", linewidth=0.4)

for axis, title, subtitle, color in [
    (before, "BEFORE", "right-skewed", ORANGE),
    (after, "AFTER  log1p(x)", "more balanced", GREEN),
]:
    axis.set_title(title, loc="left", fontsize=10.5, color=color, fontweight="bold", pad=5)
    axis.text(0.0, 0.88, subtitle, transform=axis.transAxes, ha="left", va="top",
              fontsize=8.5, color=MUTED)
    axis.set_yticks([])
    axis.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=0)
    axis.grid(axis="y", color="#EEEAE7", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)

fig.text(0.50, 0.018, "Goal: represent the same information in a form the model can learn more easily.",
         ha="center", va="bottom", fontsize=9.2, color=MUTED)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"24_feature_engineering_simple.{ext}", bbox_inches="tight",
                pad_inches=0.10, dpi=180)
plt.close(fig)
