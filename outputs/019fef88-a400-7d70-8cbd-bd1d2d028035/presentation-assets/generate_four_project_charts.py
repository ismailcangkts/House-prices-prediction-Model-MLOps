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
RED = "#D95C55"
BLUE = "#447B9B"
GRID = "#E8E4E0"


train = pd.read_csv(ROOT / "data/processed/train.csv")
selected = pd.read_csv(ROOT / "data/processed/selected_features.csv")


def style_axis(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def heading(fig, title, subtitle):
    fig.text(0.075, 0.93, title, fontsize=22, fontweight="bold", color=INK, ha="left")
    fig.text(0.075, 0.875, subtitle, fontsize=11, color=MUTED, ha="left")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.12, dpi=180)
    plt.close(fig)


# 1 — StandardScaler: scale dispersion before/after on the actual training split.
scale_features = ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt"]
raw = train[scale_features].astype(float)
scaled = StandardScaler().fit_transform(raw)
raw_std = raw.std(ddof=0).sort_values()
scaled_std = pd.Series(np.std(scaled, axis=0, ddof=0), index=scale_features).loc[raw_std.index]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6), dpi=180, facecolor="white")
heading(fig, "STANDARD SCALING", "StandardScaler is fitted on the training data for linear models.")
fig.subplots_adjust(left=0.16, right=0.95, top=0.78, bottom=0.16, wspace=0.36)

axes[0].barh(raw_std.index, raw_std.values, color=[ORANGE, BLUE, GREEN, PURPLE, RED])
axes[0].set_xscale("log")
axes[0].set_title("BEFORE · different scales", loc="left", fontsize=13, color=INK, fontweight="bold")
axes[0].set_xlabel("Standard deviation in original units · log scale", color=MUTED, fontsize=9)
for y, value in enumerate(raw_std.values):
    axes[0].text(value * 1.07, y, f"{value:,.0f}", va="center", fontsize=9, color=INK, fontweight="bold")
style_axis(axes[0])

axes[1].barh(scaled_std.index, scaled_std.values, color=GREEN)
axes[1].set_xlim(0, 1.18)
axes[1].set_title("AFTER · comparable scale", loc="left", fontsize=13, color=INK, fontweight="bold")
axes[1].set_xlabel("Standard deviation after scaling", color=MUTED, fontsize=9)
for y, value in enumerate(scaled_std.values):
    axes[1].text(value + 0.025, y, f"{value:.1f}", va="center", fontsize=9, color=GREEN, fontweight="bold")
style_axis(axes[1])
fig.text(0.5, 0.055, "Result: mean ≈ 0 and standard deviation = 1; no feature dominates only because of its unit.",
         ha="center", color=MUTED, fontsize=10)
save(fig, "25_scaling_standardscaler")


# 2 — TransformedTargetRegressor: log1p during training, expm1 on predictions.
y = train["SalePrice"].astype(float)
y_log = np.log1p(y)
fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6), dpi=180, facecolor="white")
heading(fig, "TARGET TRANSFORMATION", "The pipeline models log1p(SalePrice) and returns predictions with expm1.")
fig.subplots_adjust(left=0.08, right=0.96, top=0.78, bottom=0.16, wspace=0.20)

axes[0].hist(y, bins=34, color=ORANGE, alpha=0.93, edgecolor="white", linewidth=0.5)
axes[0].set_title(f"BEFORE · skewness {y.skew():.2f}", loc="left", fontsize=13, color=ORANGE, fontweight="bold")
axes[0].set_xlabel("SalePrice ($)", color=MUTED, fontsize=9)
axes[0].set_ylabel("Number of houses", color=MUTED, fontsize=9)
style_axis(axes[0], "y")

axes[1].hist(y_log, bins=34, color=GREEN, alpha=0.93, edgecolor="white", linewidth=0.5)
axes[1].set_title(f"AFTER log1p · skewness {pd.Series(y_log).skew():.2f}", loc="left", fontsize=13, color=GREEN, fontweight="bold")
axes[1].set_xlabel("log1p(SalePrice)", color=MUTED, fontsize=9)
style_axis(axes[1], "y")
fig.text(0.5, 0.055, "Result: the right tail is compressed, reducing the influence of very expensive houses during training.",
         ha="center", color=MUTED, fontsize=10)
save(fig, "26_transformation_saleprice_log1p")


# 3 — Outlier analysis: GrLivArea upper IQR threshold used in the project EDA.
q1, q3 = selected["GrLivArea"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
outlier_mask = selected["GrLivArea"] > upper

fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=180, facecolor="white")
heading(fig, "OUTLIER ANALYSIS", f"GrLivArea upper threshold = {upper:,.0f} ft² · 1.5×IQR rule · {int(outlier_mask.sum())} observations flagged")
fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.15)
ax.scatter(selected.loc[~outlier_mask, "GrLivArea"], selected.loc[~outlier_mask, "SalePrice"],
           s=23, alpha=0.48, color=GREEN, edgecolors="none", label="Normal range")
ax.scatter(selected.loc[outlier_mask, "GrLivArea"], selected.loc[outlier_mask, "SalePrice"],
           s=58, alpha=0.92, color=ORANGE, edgecolors="white", linewidths=0.6, label="IQR outlier")
ax.axvline(upper, color=RED, linewidth=2, linestyle="--", label=f"Upper IQR limit: {upper:,.0f} ft²")
ax.set_xlabel("GrLivArea · above-ground living area (ft²)", color=MUTED, fontsize=10)
ax.set_ylabel("SalePrice ($)", color=MUTED, fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=9)
style_axis(ax, "both")
fig.text(0.5, 0.055, "Flagging is diagnostic: observations are not removed automatically without domain justification.",
         ha="center", color=MUTED, fontsize=10)
save(fig, "27_outlier_grlivarea_iqr")


# 4 — Pearson correlation with SalePrice using project-selected numeric features.
numeric = selected.select_dtypes(include="number")
corr = numeric.corr(numeric_only=True)["SalePrice"].drop("SalePrice")
top = corr.reindex(corr.abs().sort_values(ascending=False).head(10).index).sort_values()

fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=180, facecolor="white")
heading(fig, "CORRELATION ANALYSIS", "Top numeric Pearson correlations with SalePrice in selected_features.csv.")
fig.subplots_adjust(left=0.19, right=0.94, top=0.78, bottom=0.14)
bar_colors = [GREEN if value >= 0 else RED for value in top.values]
ax.barh(top.index, top.values, color=bar_colors)
ax.axvline(0, color=INK, linewidth=1)
for y_idx, value in enumerate(top.values):
    ax.text(value + (0.015 if value >= 0 else -0.015), y_idx, f"{value:.2f}",
            va="center", ha="left" if value >= 0 else "right", fontsize=9.5,
            color=INK, fontweight="bold")
ax.set_xlim(min(-0.15, top.min() - 0.08), max(0.88, top.max() + 0.08))
ax.set_xlabel("Pearson correlation with SalePrice", color=MUTED, fontsize=10)
style_axis(ax)
fig.text(0.5, 0.045, "Correlation measures linear association—not causality.",
         ha="center", color=MUTED, fontsize=10)
save(fig, "28_correlation_saleprice_top10")

print("generated 4 chart pairs")
