from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "graphics"
INK = "#292625"
MUTED = "#716B68"
RED = "#D95C55"
RED_PALE = "#FBEDEC"
GREEN = "#27836D"
GREEN_PALE = "#E8F3EF"
PURPLE = "#6D63A8"
LINE = "#D8D3CF"


fig, ax = plt.subplots(figsize=(14.2, 6.0), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 14.2)
ax.set_ylim(0, 6.0)
ax.axis("off")


def box(x, y, w, h, title, subtitle, face, edge, title_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        facecolor=face, edgecolor=edge, linewidth=1.5,
    ))
    ax.text(x + w / 2, y + h * 0.61, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center",
            fontsize=9.5, color=MUTED)


def arrow(x1, y1, x2, y2, color=LINE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.8, color=color))


# Wrong flow
ax.add_patch(Circle((0.55, 4.55), 0.26, facecolor=RED, edgecolor="none"))
ax.text(0.55, 4.55, "×", ha="center", va="center", fontsize=20,
        color="white", fontweight="bold")
ax.text(0.98, 4.55, "LEAKAGE", ha="left", va="center", fontsize=13,
        color=RED, fontweight="bold")

box(2.45, 3.85, 2.40, 1.35, "FULL DATA", "train + validation", "#F6F4F2", LINE)
box(5.75, 3.85, 2.65, 1.35, "FIT PREPROCESSING", "median · encoder · scaler", RED_PALE, RED, RED)
box(9.30, 3.85, 2.40, 1.35, "SPLIT", "validation already seen", "#F6F4F2", LINE)
arrow(4.85, 4.52, 5.75, 4.52, RED)
arrow(8.40, 4.52, 9.30, 4.52, RED)

# Correct flow
ax.add_patch(Circle((0.55, 1.65), 0.26, facecolor=GREEN, edgecolor="none"))
ax.text(0.55, 1.65, "✓", ha="center", va="center", fontsize=17,
        color="white", fontweight="bold")
ax.text(0.98, 1.65, "LEAKAGE-SAFE", ha="left", va="center", fontsize=13,
        color=GREEN, fontweight="bold")

box(2.45, 0.95, 2.40, 1.40, "SPLIT FIRST", "train | validation | test", "#F6F4F2", LINE)
box(5.75, 0.95, 2.65, 1.40, "FIT ON TRAIN", "learn parameters once", GREEN_PALE, GREEN, GREEN)
box(9.30, 0.95, 2.40, 1.40, "TRANSFORM ONLY", "validation + test", "#F1EFF8", PURPLE, PURPLE)
arrow(4.85, 1.65, 5.75, 1.65, GREEN)
arrow(8.40, 1.65, 9.30, 1.65, GREEN)

ax.text(12.95, 4.52, "Biased\nscore", ha="center", va="center",
        fontsize=11, color=RED, fontweight="bold")
arrow(11.70, 4.52, 12.50, 4.52, RED)
ax.text(12.95, 1.65, "Honest\nevaluation", ha="center", va="center",
        fontsize=11, color=GREEN, fontweight="bold")
arrow(11.70, 1.65, 12.50, 1.65, GREEN)

ax.text(7.1, 5.68, "The split must happen before preprocessing learns anything.",
        ha="center", va="center", fontsize=15, color=INK, fontweight="bold")
ax.text(7.1, 0.28, "Rule: fit on train; transform validation and test with the same learned parameters.",
        ha="center", va="center", fontsize=10.5, color=MUTED)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"22_data_leakage_simple.{ext}", bbox_inches="tight",
                pad_inches=0.10, dpi=180)
plt.close(fig)
