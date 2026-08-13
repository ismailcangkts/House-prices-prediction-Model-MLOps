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
PURPLE_PALE = "#F1EFF8"
LINE = "#D8D3CF"

fig, ax = plt.subplots(figsize=(14.0, 5.7), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.7)
ax.axis("off")


def card(x, y, w, h, eyebrow, title, body, face, edge, accent):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=face, edgecolor=edge, linewidth=1.6,
    ))
    ax.add_patch(Circle((x + 0.42, y + h - 0.46), 0.18, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.42, y + h - 0.46, eyebrow, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.text(x + 0.73, y + h - 0.46, title, ha="left", va="center",
            fontsize=15, color=accent, fontweight="bold")
    ax.text(x + 0.35, y + h - 1.13, body, ha="left", va="top",
            fontsize=11, color=INK, linespacing=1.45)


def arrow(x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=2.2, color=color))


card(
    0.45, 1.20, 3.70, 3.45,
    "!", "DATA LEAKAGE",
    "Validation / test information\nreaches preprocessing or training.\n\nResult: unrealistically good scores",
    RED_PALE, RED, RED,
)

card(
    5.15, 1.20, 3.70, 3.45,
    "✓", "PIPELINE",
    "1  Split the data\n2  Fit preprocessing on train\n3  Transform validation / test",
    GREEN_PALE, GREEN, GREEN,
)

card(
    9.85, 1.20, 3.70, 3.45,
    "→", "PURPOSE",
    "Fair evaluation\nConsistent preprocessing\nReproducible predictions",
    PURPLE_PALE, PURPLE, PURPLE,
)

arrow(4.30, 2.93, 5.00, 2.93, RED)
ax.text(4.65, 3.25, "PREVENT", ha="center", va="center", fontsize=8.5,
        color=MUTED, fontweight="bold")
arrow(9.00, 2.93, 9.70, 2.93, GREEN)
ax.text(9.35, 3.25, "ENSURE", ha="center", va="center", fontsize=8.5,
        color=MUTED, fontweight="bold")

ax.text(7.0, 5.25, "PROBLEM  →  CONTROL  →  OUTCOME",
        ha="center", va="center", fontsize=15, color=INK, fontweight="bold")
ax.text(7.0, 0.52,
        "Fit on train only  ·  Apply the same learned transformations everywhere",
        ha="center", va="center", fontsize=11, color=MUTED)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"23_leakage_pipeline_purpose.{ext}", bbox_inches="tight",
                pad_inches=0.12, dpi=180)
plt.close(fig)
