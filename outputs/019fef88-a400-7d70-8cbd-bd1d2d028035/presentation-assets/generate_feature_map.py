from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent / "graphics"
INK = "#22201F"
MUTED = "#66615F"
PURPLE = "#6D63A8"
GREEN = "#2B806B"
ORANGE = "#D67A45"
BLUE = "#447B9B"
GOLD = "#D2A943"
PALE = "#F4F2FA"


fig, ax = plt.subplots(figsize=(10.5, 9.2), dpi=180)
fig.patch.set_alpha(0)
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 9.2)
ax.axis("off")

ax.text(5.25, 8.82, "WHAT DESCRIBES A HOUSE?", ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
ax.text(5.25, 8.48, "Examples from 82 predictor variables", ha="center", va="center", fontsize=10.5, color=MUTED)

# Central house
ax.add_patch(Polygon([[3.65, 4.9], [5.25, 6.25], [6.85, 4.9]], closed=True, facecolor=PURPLE, edgecolor="none"))
ax.add_patch(Rectangle((3.92, 2.75), 2.66, 2.35, facecolor="#EEEAF8", edgecolor=PURPLE, linewidth=2.2))
ax.add_patch(Rectangle((4.92, 2.75), 0.72, 1.40, facecolor=PURPLE, edgecolor="none"))
ax.add_patch(Rectangle((4.18, 4.15), 0.58, 0.58, facecolor="white", edgecolor=PURPLE, linewidth=1.4))
ax.add_patch(Rectangle((5.75, 4.15), 0.58, 0.58, facecolor="white", edgecolor=PURPLE, linewidth=1.4))
ax.text(5.25, 5.28, "HOUSE", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
ax.add_patch(FancyBboxPatch((4.15, 1.92), 2.20, 0.53, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=INK, edgecolor="none"))
ax.text(5.25, 2.18, "TARGET  ·  SalePrice", ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")

groups = [
    (0.25, 6.45, 3.15, 1.18, "LOCATION", "Neighborhood  ·  MSZoning", GREEN),
    (7.10, 6.45, 3.15, 1.18, "SIZE & AREA", "GrLivArea  ·  LotArea\nTotalBsmtSF", BLUE),
    (0.25, 4.45, 3.15, 1.18, "QUALITY", "OverallQual  ·  KitchenQual\nExterQual", ORANGE),
    (7.10, 4.45, 3.15, 1.18, "AGE", "YearBuilt  ·  YearRemodAdd", GOLD),
    (0.25, 2.45, 3.15, 1.18, "STRUCTURE", "HouseStyle  ·  BldgType\nRoofStyle", PURPLE),
    (7.10, 2.45, 3.15, 1.18, "AMENITIES", "GarageCars  ·  GarageArea\nFireplaces", GREEN),
]

for x, y, w, h, label, examples, color in groups:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="white", edgecolor="#DDD9E8", linewidth=1.4))
    ax.add_patch(Rectangle((x, y), 0.12, h, facecolor=color, edgecolor="none"))
    ax.text(x + 0.28, y + 0.78, label, fontsize=10, color=color, fontweight="bold", va="center")
    ax.text(x + 0.28, y + 0.35, examples, fontsize=9, color=INK, va="center", linespacing=1.35)

# Connectors
connections = [
    ((3.40, 7.04), (4.25, 5.62), GREEN),
    ((7.10, 7.04), (6.25, 5.62), BLUE),
    ((3.40, 5.04), (3.92, 4.52), ORANGE),
    ((7.10, 5.04), (6.58, 4.52), GOLD),
    ((3.40, 3.04), (4.15, 3.30), PURPLE),
    ((7.10, 3.04), (6.35, 3.30), GREEN),
]
for (x1, y1), (x2, y2), color in connections:
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.8, alpha=0.9)
    ax.scatter([x2], [y2], s=28, color=color, zorder=5)

ax.text(5.25, 0.92, "Numeric features measure magnitude · Categorical features describe type and quality", ha="center", va="center", fontsize=9.5, color=MUTED)
ax.text(5.25, 0.52, "40 NUMERIC PREDICTORS   ·   42 CATEGORICAL PREDICTORS   ·   1 TARGET", ha="center", va="center", fontsize=10.5, color=PURPLE, fontweight="bold")

for ext in ("png", "svg"):
    fig.savefig(OUT / f"20_dataset_feature_map.{ext}", transparent=True, bbox_inches="tight", pad_inches=0.08, dpi=180)
plt.close(fig)
