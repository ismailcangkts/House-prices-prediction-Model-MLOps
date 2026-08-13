from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parent / "graphics"

features = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu", "GarageType"]
percent = [99.52, 96.30, 93.77, 80.75, 59.73, 47.26, 5.55]
actions = [
    "HasPool = 0/1 · original column removed",
    "HasMiscFeature = 0/1 · original column removed",
    "NaN → NoAlley",
    "NaN → NoFence · split into Type + Quality",
    "NaN → NoMasVnr · MasVnrArea → 0",
    "NaN → NoFireplace",
    "NaN → NoGarage",
]
colors = ["#6D63A8", "#6D63A8", "#2B806B", "#D67A45", "#2B806B", "#2B806B", "#2B806B"]

fig, ax = plt.subplots(figsize=(14.4, 7.0), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = list(range(len(features)))
bars = ax.barh(y, percent, height=0.62, color=colors, edgecolor="none")
ax.invert_yaxis()
ax.set_xlim(0, 154)
ax.set_yticks(y, features, fontsize=13, fontweight="bold", color="#272321")
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10, color="#716B68")
ax.grid(axis="x", color="#E8E4E0", linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

for bar, value, action, color in zip(bars, percent, actions, colors):
    cy = bar.get_y() + bar.get_height() / 2
    ax.text(value - 1.2 if value > 14 else value + 1.2, cy, f"{value:.2f}%",
            ha="right" if value > 14 else "left", va="center",
            fontsize=10.5, fontweight="bold", color="white" if value > 14 else color)
    box = FancyBboxPatch((103, cy - 0.25), 48, 0.50,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor="#F5F3F1", edgecolor="#E1DCD8", linewidth=0.8)
    ax.add_patch(box)
    ax.text(105, cy, action, va="center", ha="left", fontsize=9.5,
            color="#302C2A", fontweight="semibold")

ax.text(0.0, 1.105, "MISSING VALUES → CLEANING DECISIONS", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=20, fontweight="bold", color="#22201F")
ax.text(0.0, 1.055,
        "Missing values were handled according to domain meaning—not with one generic imputation rule.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5, color="#68625F")
ax.text(0.65, 1.005, "CLEANING DECISION", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=10, fontweight="bold", color="#6D63A8")
ax.text(0.0, -0.10,
        "Rule of thumb: absence of a physical component was encoded explicitly; rare-detail columns were compressed into binary indicators.",
        transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#68625F")

plt.subplots_adjust(left=0.12, right=0.98, top=0.79, bottom=0.15)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"21_missingness_cleaning_actions.{ext}", bbox_inches="tight", pad_inches=0.15, dpi=180)
plt.close(fig)
