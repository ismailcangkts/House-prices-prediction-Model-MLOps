from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GRAPHICS = OUT / "graphics"
TABLES = OUT / "tables"
sys.path.insert(0, str(ROOT))

from pipeline_machine_learning import build_model_pipeline  # noqa: E402


PAPER = "#F4F0E8"
INK = "#152022"
MUTED = "#66706C"
GREEN = "#1F7A63"
ORANGE = "#D87941"
RED = "#B94F4F"
BLUE = "#3C6E8F"
YELLOW = "#E6C15A"
PALE = "#DCE6DF"
GRID = "#D5D0C6"


def style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 22,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def new_figure(title: str, subtitle: str = "", ncols: int = 1, width_ratios=None):
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(13.333, 7.5),
        dpi=180,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.96, wspace=0.28)
    fig.text(0.06, 0.92, title, fontsize=25, fontweight="bold", color=INK, ha="left")
    if subtitle:
        fig.text(0.06, 0.865, subtitle, fontsize=11.5, color=MUTED, ha="left")
    if ncols == 1:
        axes = [axes]
    return fig, axes


def finish(fig, filename: str, source: str) -> None:
    fig.text(0.06, 0.035, f"Kaynak: {source}", fontsize=7.5, color=MUTED, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(GRAPHICS / f"{filename}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def clean_axis(ax, grid_axis="y") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


def money_k(value: float) -> str:
    return f"${value / 1000:.0f}K"


def calculate_metrics(y_true, pred) -> dict[str, float]:
    safe = np.maximum(pred, 0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "rmsle": float(mean_squared_log_error(y_true, safe) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def benchmark_models(train, validation, test):
    X_train, y_train = train.drop(columns="SalePrice"), train["SalePrice"]
    X_val, y_val = validation.drop(columns="SalePrice"), validation["SalePrice"]
    X_test, y_test = test.drop(columns="SalePrice"), test["SalePrice"]
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    candidates = {
        "Ridge Regression": (Ridge(alpha=10.0), True),
        "Elastic Net": (ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000), True),
        "Random Forest": (
            RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.035,
                max_depth=3,
                loss="huber",
                random_state=42,
            ),
            False,
        ),
    }
    rows = []
    predictions = {}
    for name, (regressor, scale) in candidates.items():
        val_pipe = build_model_pipeline(regressor, scale_features=scale)
        val_pipe.fit(X_train, y_train)
        val_pred = val_pipe.predict(X_val)
        test_pipe = build_model_pipeline(regressor, scale_features=scale)
        test_pipe.fit(X_train_val, y_train_val)
        test_pred = test_pipe.predict(X_test)
        val_metrics = calculate_metrics(y_val, val_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        rows.append(
            {
                "Model": name,
                "Validation MAE": val_metrics["mae"],
                "Validation RMSE": val_metrics["rmse"],
                "Validation RMSLE": val_metrics["rmsle"],
                "Validation R2": val_metrics["r2"],
                "Test MAE": test_metrics["mae"],
                "Test RMSE": test_metrics["rmse"],
                "Test RMSLE": test_metrics["rmsle"],
                "Test R2": test_metrics["r2"],
                "RMSE Genelleme Farkı": test_metrics["rmse"] - val_metrics["rmse"],
            }
        )
        predictions[name] = {"validation": val_pred, "test": test_pred}
    return pd.DataFrame(rows).sort_values("Validation RMSE"), predictions, y_val


def dataset_overview(selected, cleaned):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=180, facecolor=PAPER)
    fig.text(0.06, 0.91, "Veri seti modellemeye hazır; kapsam kontrollü biçimde daraltıldı.", fontsize=25, fontweight="bold")
    fig.text(0.06, 0.85, "Temiz Ames verisi ve seçilen model girdilerinin sunum özeti", fontsize=11.5, color=MUTED)
    cards = [
        (len(cleaned), "Temiz gözlem", "cleaned_train.csv", GREEN),
        (cleaned.shape[1], "Temiz kolon", "hedef dahil", BLUE),
        (selected.shape[1] - 1, "Model feature'ı", "20 sayısal + 10 kategorik", ORANGE),
        (int(selected.isna().sum().sum()), "Kalan missing", "model tablosunda", YELLOW),
        (int(selected.duplicated().sum()), "Duplicate", "seçili feature tablosunda", RED),
    ]
    for i, (value, label, note, color) in enumerate(cards):
        x = 0.06 + i * 0.185
        box = FancyBboxPatch((x, 0.48), 0.16, 0.23, boxstyle="round,pad=0.012,rounding_size=0.015", transform=fig.transFigure, facecolor="#EAE5DA", edgecolor="none")
        fig.patches.append(box)
        fig.text(x + 0.02, 0.61, f"{value:,}".replace(",", "."), fontsize=32, fontweight="bold", color=color)
        fig.text(x + 0.02, 0.55, label, fontsize=12, fontweight="bold")
        fig.text(x + 0.02, 0.505, note, fontsize=8.5, color=MUTED)
    fig.text(0.06, 0.30, "Sunum mesajı", fontsize=10, fontweight="bold", color=GREEN)
    fig.text(0.06, 0.22, "“Amaç bütün kolonları kullanmak değil; anlamlı, izlenebilir ve tekrar üretilebilir bir model girdisi kurmak.”", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.035, "Kaynak: cleaned_train.csv · data/processed/selected_features.csv", fontsize=7.5, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(GRAPHICS / f"01_dataset_overview.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def split_chart(manifest):
    fig, (ax,) = new_figure(
        "Veri bölme tasarımı, model seçimi ile üretim izlemeyi birbirinden ayırıyor.",
        "Deterministik shuffle (random_state=42) sonrası 55/15/10/10/10 ayrımı",
    )
    colors = [GREEN, BLUE, ORANGE, YELLOW, INK]
    labels = ["Train", "Validation", "Test", "Reference", "Production"]
    left = 0
    total = manifest["rows"].sum()
    for i, row in manifest.iterrows():
        width = row["rows"] / total * 100
        ax.barh([0], [width], left=left, height=0.44, color=colors[i], edgecolor=PAPER)
        text_color = INK if i == 3 else "white"
        ax.text(left + width / 2, 0.04, f"{labels[i]}\n{row['rows']} satır\n%{row['ratio']*100:.0f}", ha="center", va="center", color=text_color, fontsize=11 if i == 0 else 9, fontweight="bold")
        left += width
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    roles = ["öğrenme", "model seçimi", "bağımsız test", "drift referansı", "gerçek akış"]
    starts = np.cumsum([0] + (manifest["rows"] / total * 100).tolist()[:-1])
    widths = manifest["rows"] / total * 100
    for start, width, role, color in zip(starts, widths, roles, colors):
        ax.text(start + width / 2, -0.39, role, ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    finish(fig, "02_data_split", "data/manifest.csv · scripts/split_dataset.py")


def semantic_cleaning(selected):
    counts = pd.DataFrame(
        {
            "Durum": ["NoFireplace", "NoGarage", "NoBasement", "GarageYrBlt = 0", "Kalan missing"],
            "Adet": [
                (selected["FireplaceQu"] == "Npfireplace").sum(),
                (selected["GarageType"] == "NoGarage").sum(),
                (selected["BsmtQual"] == "NoBasement").sum(),
                (selected["GarageYrBlt"] == 0).sum(),
                selected.isna().sum().sum(),
            ],
        }
    )
    fig, axes = new_figure(
        "Eksik değerleri silmek yerine, fiziksel anlamlarını modele taşıdık.",
        "Raw veri dosyası projede olmadığı için grafik, temizlenmiş veri durumunu ve kodda tanımlı kuralları gösterir.",
        ncols=2,
        width_ratios=[1.0, 1.15],
    )
    ax = axes[0]
    colors = [GREEN, BLUE, ORANGE, MUTED, YELLOW]
    ax.barh(counts["Durum"][::-1], counts["Adet"][::-1], color=colors[::-1])
    for y, value in enumerate(counts["Adet"][::-1]):
        ax.text(value + max(counts["Adet"]) * 0.025, y, f"{int(value)}", va="center", fontweight="bold")
    ax.set_title("Temiz veri içindeki semantik yokluklar", loc="left", fontsize=16)
    clean_axis(ax, "x")
    ax.set_xlabel("Gözlem sayısı")
    ax = axes[1]
    ax.axis("off")
    steps = [
        ("01", "NaN = yokluk", "Garage/Basement/Fireplace → anlamlı kategori veya 0"),
        ("02", "Koşullu imputasyon", "LotFrontage → train mahalle medyanı; fallback global medyan"),
        ("03", "Ordinal encoding", "Po < Fa < TA < Gd < Ex sırası korunur"),
        ("04", "Nadir detay", "PoolQC/MiscFeature → HasPool/HasMiscFeature"),
    ]
    for i, (num, title, detail) in enumerate(steps):
        y = 0.85 - i * 0.22
        ax.add_patch(FancyBboxPatch((0.03, y - 0.06), 0.12, 0.12, boxstyle="round,pad=0.01", facecolor=GREEN if i == 3 else INK, edgecolor="none"))
        ax.text(0.09, y, num, color="white", ha="center", va="center", fontweight="bold")
        ax.text(0.19, y + 0.025, title, fontsize=12, fontweight="bold", va="center")
        ax.text(0.19, y - 0.035, detail, fontsize=9.5, color=MUTED, va="center", wrap=True)
    finish(fig, "03_semantic_cleaning", "pipeline_machine_learning.py · selected_features.csv")
    return counts


def target_distribution(selected):
    target = selected["SalePrice"]
    log_target = np.log1p(target)
    fig, axes = new_figure(
        "SalePrice sağa çarpık; log1p dönüşümü lineer model için daha dengeli hedef üretiyor.",
        f"Skewness: {skew(target):.2f} → {skew(log_target):.2f} · Medyan {money_k(target.median())} · Ortalama {money_k(target.mean())}",
        ncols=2,
    )
    sns.histplot(target, bins=28, color=ORANGE, ax=axes[0], edgecolor=PAPER)
    axes[0].axvline(target.median(), color=INK, linestyle="--", linewidth=2, label=f"Medyan {money_k(target.median())}")
    axes[0].axvline(target.mean(), color=RED, linestyle=":", linewidth=2, label=f"Ortalama {money_k(target.mean())}")
    axes[0].set_title("Ham SalePrice", loc="left", fontsize=16)
    axes[0].set_xlabel("Satış fiyatı ($)")
    axes[0].set_ylabel("Gözlem")
    axes[0].legend(frameon=False, loc="upper right")
    clean_axis(axes[0])
    sns.histplot(log_target, bins=28, color=GREEN, ax=axes[1], edgecolor=PAPER)
    axes[1].set_title("log1p(SalePrice)", loc="left", fontsize=16)
    axes[1].set_xlabel("Log satış fiyatı")
    axes[1].set_ylabel("Gözlem")
    clean_axis(axes[1])
    finish(fig, "04_saleprice_distribution", "data/processed/selected_features.csv")
    summary = pd.DataFrame(
        {
            "Metrik": ["Minimum", "Q1", "Medyan", "Ortalama", "Q3", "Maksimum", "Std", "Skewness", "Kurtosis", "Log Skewness"],
            "Değer": [target.min(), target.quantile(0.25), target.median(), target.mean(), target.quantile(0.75), target.max(), target.std(), skew(target), kurtosis(target), skew(log_target)],
        }
    )
    return summary


def target_boxplot(selected):
    fig, axes = new_figure(
        "Fiyat dağılımının üst kuyruğu uzun; IQR sınırı $340K üzerinde 60 gözlem işaretliyor.",
        "Aykırı değer bayrağı, otomatik silme kararı değildir.",
        ncols=2,
        width_ratios=[1.4, 0.6],
    )
    sns.boxplot(x=selected["SalePrice"], color=ORANGE, ax=axes[0])
    sns.stripplot(x=selected["SalePrice"].sample(300, random_state=42), color=GREEN, size=3, alpha=0.45, ax=axes[0])
    axes[0].set_title("SalePrice boxplot + örnek gözlemler", loc="left", fontsize=16)
    axes[0].set_xlabel("Satış fiyatı ($)")
    axes[0].set_yticks([])
    clean_axis(axes[0], "x")
    axes[1].axis("off")
    q1, q3 = selected["SalePrice"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    stats = [("Q1", q1), ("Medyan", selected["SalePrice"].median()), ("Q3", q3), ("IQR üst sınırı", upper), ("İşaretli gözlem", (selected["SalePrice"] > upper).sum())]
    for i, (label, value) in enumerate(stats):
        y = 0.86 - i * 0.17
        axes[1].text(0.05, y, label, fontsize=10, color=MUTED, fontweight="bold")
        shown = f"{int(value)}" if label == "İşaretli gözlem" else money_k(value)
        axes[1].text(0.95, y, shown, fontsize=18, color=ORANGE if i >= 3 else INK, fontweight="bold", ha="right")
    finish(fig, "05_saleprice_boxplot", "selected_features.csv · 1.5×IQR")


def outlier_scatter(selected):
    q1, q3 = selected["GrLivArea"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    mask = selected["GrLivArea"] > upper
    fig, (ax,) = new_figure(
        "Yaşam alanı fiyatla güçlü ilişkili; fakat büyük evlerde hata riski ve aykırı örnekler artıyor.",
        f"GrLivArea IQR üst sınırı {upper:.0f} ft² · {mask.sum()} gözlem işaretli",
    )
    ax.scatter(selected.loc[~mask, "GrLivArea"], selected.loc[~mask, "SalePrice"], s=24, alpha=0.52, color=GREEN, edgecolors="none", label="Normal aralık")
    ax.scatter(selected.loc[mask, "GrLivArea"], selected.loc[mask, "SalePrice"], s=55, alpha=0.85, color=ORANGE, edgecolors=PAPER, linewidths=0.7, label="IQR outlier")
    ax.axvline(upper, color=RED, linewidth=2, linestyle="--")
    ax.text(upper + 45, selected["SalePrice"].max() * 0.91, f"IQR sınırı\n{upper:.0f} ft²", color=RED, fontweight="bold")
    ax.set_xlabel("GrLivArea — zemin üstü yaşam alanı (ft²)")
    ax.set_ylabel("SalePrice ($)")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    finish(fig, "06_grlivarea_saleprice_outliers", "selected_features.csv · 1.5×IQR")


def outlier_counts(selected):
    columns = ["LotArea", "SalePrice", "TotalBsmtSF", "GrLivArea", "GarageArea", "1stFlrSF", "BsmtFinSF1"]
    rows = []
    for col in columns:
        q1, q3 = selected[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((selected[col] < lower) | (selected[col] > upper)).sum()
        rows.append({"Feature": col, "Alt Sınır": lower, "Üst Sınır": upper, "Outlier Adedi": count, "Outlier Oranı %": count / len(selected) * 100})
    frame = pd.DataFrame(rows).sort_values("Outlier Adedi", ascending=True)
    fig, (ax,) = new_figure(
        "Outlier yoğunluğu feature'a göre değişiyor; LotArea ve SalePrice üst kuyruğu öne çıkıyor.",
        "Karar: bayrakla, model etkisini ölç, yalnız gerekçeli ise sil.",
    )
    colors = [BLUE if f not in {"LotArea", "SalePrice"} else ORANGE for f in frame["Feature"]]
    ax.barh(frame["Feature"], frame["Outlier Adedi"], color=colors)
    for y, (_, row) in enumerate(frame.iterrows()):
        ax.text(row["Outlier Adedi"] + 1.2, y, f"{int(row['Outlier Adedi'])}  (%{row['Outlier Oranı %']:.1f})", va="center", fontweight="bold")
    ax.set_xlabel("IQR ile işaretlenen gözlem sayısı")
    clean_axis(ax, "x")
    finish(fig, "07_outlier_counts", "selected_features.csv · 1.5×IQR")
    return frame.sort_values("Outlier Adedi", ascending=False)


def correlation_charts(selected):
    numeric = selected.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    target_corr = corr["SalePrice"].drop("SalePrice").sort_values(key=lambda s: s.abs(), ascending=False)
    top = target_corr.head(15).sort_values()
    fig, (ax,) = new_figure(
        "Fiyat sinyalinin merkezinde kalite, yaşam alanı, garaj ve bodrum kapasitesi var.",
        "Pearson korelasyonu; nedensellik değildir.",
    )
    colors = [GREEN if v >= 0 else RED for v in top]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color=INK, linewidth=1)
    for y, value in enumerate(top.values):
        ax.text(value + (0.012 if value >= 0 else -0.012), y, f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontweight="bold")
    ax.set_xlim(min(-0.15, top.min() - 0.05), 0.86)
    ax.set_xlabel("SalePrice ile Pearson korelasyonu")
    clean_axis(ax, "x")
    finish(fig, "08_top_correlations", "selected_features.csv")

    heat_cols = ["SalePrice"] + target_corr.head(11).index.tolist()
    fig, (ax,) = new_figure(
        "En güçlü feature'lar kendi aralarında da ilişkili; multicollinearity kontrolü gerekiyor.",
        "Seçili 12 sayısal değişkenin korelasyon matrisi",
    )
    sns.heatmap(selected[heat_cols].corr(), cmap=sns.diverging_palette(20, 150, as_cmap=True), center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.7, linecolor=PAPER, ax=ax, cbar_kws={"shrink": 0.82})
    ax.set_title("")
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    finish(fig, "09_correlation_heatmap", "selected_features.csv")

    pairs = []
    feature_corr = numeric.drop(columns="SalePrice").corr()
    for i, left in enumerate(feature_corr.columns):
        for right in feature_corr.columns[i + 1 :]:
            pairs.append((left, right, feature_corr.loc[left, right]))
    pairs = sorted(pairs, key=lambda row: abs(row[2]), reverse=True)[:12]
    pair_frame = pd.DataFrame(pairs, columns=["Feature 1", "Feature 2", "Korelasyon"])
    return target_corr.reset_index().rename(columns={"index": "Feature", "SalePrice": "Korelasyon"}), pair_frame


def categorical_charts(selected):
    order = selected.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    fig, (ax,) = new_figure(
        "Mahalle, fiyat seviyesini ve fiyat belirsizliğini birlikte değiştiriyor.",
        "Boxplot: çizgi medyanı, kutu orta %50'yi, noktalar uç değerleri gösterir.",
    )
    sns.boxplot(data=selected, x="Neighborhood", y="SalePrice", order=order, color=GREEN, fliersize=2.5, linewidth=1, ax=ax)
    ax.tick_params(axis="x", rotation=55)
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("SalePrice ($)")
    clean_axis(ax)
    finish(fig, "10_neighborhood_boxplot", "selected_features.csv")

    fig, (ax,) = new_figure(
        "OverallQual arttıkça fiyat dağılımı neredeyse monoton biçimde yukarı kayıyor.",
        "Kalite skoru, tek başına en güçlü sayısal fiyat sinyalidir (r≈0.79).",
    )
    sns.boxplot(data=selected, x="OverallQual", y="SalePrice", color=BLUE, fliersize=2.5, ax=ax)
    medians = selected.groupby("OverallQual")["SalePrice"].median()
    for x, value in medians.items():
        ax.text(int(x) - 1, value + 18000, money_k(value), ha="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlabel("OverallQual")
    ax.set_ylabel("SalePrice ($)")
    clean_axis(ax)
    finish(fig, "11_overallqual_boxplot", "selected_features.csv")

    neighborhood = selected.groupby("Neighborhood")["SalePrice"].agg(["count", "mean", "median", "std", "min", "max"]).sort_values("median", ascending=False).reset_index()
    quality = selected.groupby("OverallQual")["SalePrice"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    return neighborhood, quality


def feature_groups(selected):
    groups = {
        "Alan & oda": ["GrLivArea", "TotalBsmtSF", "BsmtFinSF1", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath", "TotRmsAbvGrd", "LotArea", "LotFrontage", "OpenPorchSF", "Fireplaces"],
        "Kalite": ["OverallQual", "OverallCond", "KitchenQual", "ExterQual", "BsmtQual", "BsmtExposure", "FireplaceQu"],
        "Garaj": ["GarageCars", "GarageArea", "GarageType", "GarageFinish"],
        "Konum / tip": ["Neighborhood", "MSSubClass", "MSZoning", "HouseStyle"],
        "Yaş": ["YearBuilt", "YearRemodAdd", "GarageYrBlt"],
    }
    rows = [{"Grup": group, "Feature": feature, "Veri Tipi": str(selected[feature].dtype)} for group, features in groups.items() for feature in features]
    frame = pd.DataFrame(rows)
    counts = frame.groupby("Grup").size().sort_values()
    fig, (ax,) = new_figure(
        "30 feature, beş domain grubuyla fiyatın fiziksel ve piyasa boyutlarını kapsıyor.",
        "Seçim yalnız korelasyona değil; domain anlamı, tekrar ve pipeline maliyetine de dayanır.",
    )
    colors = [MUTED, BLUE, ORANGE, YELLOW, GREEN]
    ax.barh(counts.index, counts.values, color=colors)
    for y, value in enumerate(counts.values):
        ax.text(value + 0.2, y, str(value), va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 13.5)
    ax.set_xlabel("Feature sayısı")
    clean_axis(ax, "x")
    finish(fig, "12_feature_groups", "scripts/split_dataset.py · SELECTED_FEATURES")
    return frame


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=180, facecolor=PAPER)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.12)
    fig.text(0.05, 0.91, "Pipeline sırası, temizleme kararlarını leakage-safe ve tekrar üretilebilir kılıyor.", fontsize=25, fontweight="bold")
    fig.text(0.05, 0.85, "Her dönüşüm tek model nesnesinde saklanır ve validation/test üzerinde yeniden fit edilmez.", fontsize=11.5, color=MUTED)
    ax.axis("off")
    stages = [
        ("01", "30 feature", "X / y ayrımı"),
        ("02", "DomainFeatureEngineer", "semantic NA + türetimler"),
        ("03", "LotFrontage Imputer", "train mahalle medyanı"),
        ("04", "Ordinal + One-hot", "unknown = ignore"),
        ("05", "StandardScaler", "lineer modeller"),
        ("06", "log1p hedef + model", "expm1 ile çıktı"),
    ]
    for i, (num, title, detail) in enumerate(stages):
        x = 0.02 + i * 0.165
        y = 0.46 + (i % 2) * 0.08
        fill = INK if i == 5 else (PALE if i == 2 else "#EAE5DA")
        ax.add_patch(FancyBboxPatch((x, y), 0.145, 0.22, boxstyle="round,pad=0.012", transform=ax.transAxes, facecolor=fill, edgecolor=GRID))
        ax.text(x + 0.02, y + 0.165, num, transform=ax.transAxes, fontsize=10, color=ORANGE if i == 5 else GREEN, fontweight="bold")
        ax.text(x + 0.0725, y + 0.105, title, transform=ax.transAxes, fontsize=9.2, color="white" if i == 5 else INK, fontweight="bold", ha="center", wrap=True)
        ax.text(x + 0.0725, y + 0.045, detail, transform=ax.transAxes, fontsize=7.8, color="#AEBAB7" if i == 5 else MUTED, ha="center", wrap=True)
        if i < len(stages) - 1:
            next_x = 0.02 + (i + 1) * 0.165
            next_y = 0.46 + ((i + 1) % 2) * 0.08
            arrow = FancyArrowPatch((x + 0.145, y + 0.11), (next_x, next_y + 0.11), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14, linewidth=2, color=GREEN if i >= 2 else ORANGE)
            ax.add_patch(arrow)
    ax.add_patch(FancyBboxPatch((0.08, 0.12), 0.37, 0.17, boxstyle="round,pad=0.015", transform=ax.transAxes, facecolor=PALE, edgecolor="none"))
    ax.text(0.10, 0.235, "FIT", transform=ax.transAxes, color=GREEN, fontweight="bold")
    ax.text(0.10, 0.165, "İstatistikler yalnız train'den öğrenilir.", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.55, 0.12), 0.37, 0.17, boxstyle="round,pad=0.015", transform=ax.transAxes, facecolor=INK, edgecolor="none"))
    ax.text(0.57, 0.235, "TRANSFORM", transform=ax.transAxes, color=YELLOW, fontweight="bold")
    ax.text(0.57, 0.165, "Validation/test aynı parametrelerden geçer.", transform=ax.transAxes, fontsize=12, fontweight="bold", color="white")
    fig.text(0.05, 0.035, "Kaynak: pipeline_machine_learning.py · src/pipelines/training_pipeline.py", fontsize=7.5, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(GRAPHICS / f"13_pipeline_workflow.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def feature_engineering_chart(fe):
    labels = {"combination": "Birleştirme", "decomposition": "Ayrıştırma", "transformation": "Dönüştürme", "interaction": "Etkileşim", "indicator": "Indicator", "binning": "Binning", "domain": "Domain"}
    frame = fe.copy()
    frame["Yöntem"] = frame["experiment"].map(labels)
    frame = frame.sort_values("test_rmse_improvement_pct", ascending=True)
    fig, (ax,) = new_figure(
        "Yedi feature grubundan hiçbiri RMSE + MAE kriterini validation ve testte birlikte geçemedi.",
        "Pozitif değer hata azalmasını, negatif değer kötüleşmeyi gösterir.",
    )
    y = np.arange(len(frame))
    height = 0.34
    ax.barh(y + height / 2, frame["validation_rmse_improvement_pct"], height=height, color=BLUE, label="Validation RMSE")
    ax.barh(y - height / 2, frame["test_rmse_improvement_pct"], height=height, color=ORANGE, label="Test RMSE")
    ax.axvline(0, color=INK, linewidth=1.5)
    ax.set_yticks(y, frame["Yöntem"])
    ax.set_xlabel("Baz modele göre RMSE iyileşmesi (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    for i, (_, row) in enumerate(frame.iterrows()):
        ax.text(row["validation_rmse_improvement_pct"] - 0.12 if row["validation_rmse_improvement_pct"] < 0 else row["validation_rmse_improvement_pct"] + 0.12, i + height / 2, f"{row['validation_rmse_improvement_pct']:+.2f}%", ha="right" if row["validation_rmse_improvement_pct"] < 0 else "left", va="center", fontsize=8, color=BLUE, fontweight="bold")
        ax.text(row["test_rmse_improvement_pct"] - 0.12 if row["test_rmse_improvement_pct"] < 0 else row["test_rmse_improvement_pct"] + 0.12, i - height / 2, f"{row['test_rmse_improvement_pct']:+.2f}%", ha="right" if row["test_rmse_improvement_pct"] < 0 else "left", va="center", fontsize=8, color=RED, fontweight="bold")
    clean_axis(ax, "x")
    finish(fig, "14_feature_engineering_comparison", "artifacts/feature_engineering/feature_engineering_summary.csv")


def model_chart(benchmark):
    order = benchmark.sort_values("Validation RMSE")["Model"].tolist()
    frame = benchmark.set_index("Model").loc[order].reset_index()
    fig, axes = new_figure(
        "Gradient Boosting validation'da lider; bağımsız testte lineer modeller daha kararlı.",
        "Aynı preprocessing ve log hedef dönüşümüyle taze benchmark",
        ncols=2,
    )
    y = np.arange(len(frame))
    axes[0].barh(y + 0.18, frame["Validation RMSE"] / 1000, height=0.34, color=BLUE, label="Validation")
    axes[0].barh(y - 0.18, frame["Test RMSE"] / 1000, height=0.34, color=ORANGE, label="Test")
    axes[0].set_yticks(y, frame["Model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE ($K) — düşük daha iyi")
    axes[0].legend(frameon=False)
    clean_axis(axes[0], "x")
    axes[0].set_title("Tahmin hatası", loc="left", fontsize=16)
    colors = [GREEN if model in {"Ridge Regression", "Elastic Net"} else RED for model in frame["Model"]]
    axes[1].barh(frame["Model"], frame["RMSE Genelleme Farkı"] / 1000, color=colors)
    for y_pos, value in enumerate(frame["RMSE Genelleme Farkı"] / 1000):
        axes[1].text(value + 0.08, y_pos, f"+${value:.1f}K", va="center", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test − validation RMSE ($K)")
    axes[1].set_title("Genelleme farkı", loc="left", fontsize=16)
    clean_axis(axes[1], "x")
    finish(fig, "15_model_comparison", "mevcut pipeline ile yerel benchmark")


def residual_chart(validation, predictions):
    actual = validation["SalePrice"].to_numpy()
    pred = predictions["Ridge Regression"]["validation"]
    residual = actual - pred
    frame = pd.DataFrame({"Gerçek Fiyat": actual, "Tahmin": pred, "Residual": residual})
    frame["Fiyat Bandı"] = pd.qcut(frame["Gerçek Fiyat"], q=4, labels=["Alt %25", "Orta-alt", "Orta-üst", "Üst %25"])
    bands = frame.groupby("Fiyat Bandı", observed=True).apply(lambda g: pd.Series({"MAE": np.abs(g["Residual"]).mean(), "Bias": g["Residual"].mean(), "Gözlem": len(g)}), include_groups=False).reset_index()
    fig, axes = new_figure(
        "Ridge hatası pahalı evlerde yoğunlaşıyor; üst fiyat çeyreği ayrı iyileştirme hedefi.",
        "Residual = gerçek fiyat − tahmin; pozitif değer modelin düşük tahmin ettiğini gösterir.",
        ncols=2,
        width_ratios=[1.5, 0.7],
    )
    color = np.where(frame["Gerçek Fiyat"] >= frame["Gerçek Fiyat"].quantile(0.75), ORANGE, GREEN)
    axes[0].scatter(frame["Gerçek Fiyat"], frame["Residual"], c=color, s=25, alpha=0.72, edgecolors="none")
    axes[0].axhline(0, color=INK, linewidth=1.6)
    axes[0].set_xlabel("Gerçek fiyat ($)")
    axes[0].set_ylabel("Residual ($)")
    axes[0].set_title("Gerçek fiyat × residual", loc="left", fontsize=16)
    clean_axis(axes[0])
    colors = [BLUE, BLUE, BLUE, ORANGE]
    axes[1].barh(bands["Fiyat Bandı"], bands["MAE"] / 1000, color=colors)
    for y, value in enumerate(bands["MAE"] / 1000):
        axes[1].text(value + 0.4, y, f"${value:.1f}K", va="center", fontweight="bold", color=RED if y == 3 else INK)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE ($K)")
    axes[1].set_title("Fiyat bandına göre MAE", loc="left", fontsize=16)
    clean_axis(axes[1], "x")
    finish(fig, "16_ridge_residual_analysis", "taze Ridge validation tahminleri")
    return bands


def save_table_png(frame: pd.DataFrame, filename: str, title: str, formats: dict[str, str] | None = None):
    display = frame.copy()
    if formats:
        for col, fmt in formats.items():
            if col in display:
                display[col] = display[col].map(lambda value: fmt.format(value) if pd.notna(value) else "")
    rows = min(len(display), 20)
    display = display.head(rows)
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=180, facecolor=PAPER)
    ax.axis("off")
    fig.text(0.05, 0.92, title, fontsize=24, fontweight="bold")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left", colLoc="left", bbox=[0.03, 0.08, 0.94, 0.72])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PAPER)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EAE5DA" if row % 2 else PAPER)
    fig.savefig(GRAPHICS / f"{filename}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def write_guide() -> None:
    guide = """# Sunumda Kullanılacak Görsel Paketi

## Önerilen anlatım sırası

1. **Problem ve veri kapsamı** — `01_dataset_overview`: 1.451 gözlemi ve 30 feature'lık kontrollü model alanını tanıtın.
2. **Veri bölme disiplini** — `02_data_split`: train/validation/test/reference/production rollerini ve leakage riskini anlatın.
3. **Data cleaning** — `03_semantic_cleaning`: NaN değerlerinin her zaman gerçek eksik olmadığını; bazılarının “garaj yok” gibi fiziksel anlam taşıdığını söyleyin.
4. **Hedef değişken** — `04_saleprice_distribution` ve `05_saleprice_boxplot`: sağa çarpıklığı, log1p kararını ve uzun fiyat kuyruğunu açıklayın.
5. **Outlier analizi** — `06_grlivarea_saleprice_outliers` ve `07_outlier_counts`: outlier bayrağının otomatik silme kararı olmadığını vurgulayın.
6. **Korelasyonlar** — `08_top_correlations` ve `09_correlation_heatmap`: OverallQual ve GrLivArea'nın güçlü olduğunu; korelasyonun nedensellik olmadığını söyleyin.
7. **Kategorik değişkenler** — `10_neighborhood_boxplot` ve `11_overallqual_boxplot`: konum ve kalite etkisini örnekleyin.
8. **Feature seçimi ve pipeline** — `12_feature_groups` ve `13_pipeline_workflow`: 30 feature'ın domain kapsamını ve fit/transform sınırını anlatın.
9. **Feature engineering deneyleri** — `14_feature_engineering_comparison`: yedi grubun validation + test kriterini birlikte geçemediğini gösterin.
10. **Model karşılaştırması** — `15_model_comparison`: Gradient Boosting'in validation lideri olmasına rağmen testte bozulduğunu; lineer modellerin daha kararlı olduğunu anlatın.
11. **Hata analizi** — `16_ridge_residual_analysis`: üst fiyat çeyreğindeki hata yoğunluğunu ve sonraki deneyleri açıklayın.

## Kullanabileceğiniz ana cümleler

- “Veriyi yalnız temizlemedik; eksikliklerin fiziksel anlamını koruduk.”
- “Korelasyonları feature seçiminin başlangıcı olarak kullandık, nihai karar olarak değil.”
- “Outlier'ları kör biçimde silmedik; model etkisini ölçülecek risk olarak işaretledik.”
- “Feature engineering hipotezlerini tek tek test ederek hangi değişikliğin sonucu etkilediğini izole ettik.”
- “Validation skoru tek başına yeterli değil; bağımsız testteki kararlılık model seçiminde belirleyici.”

## Kritik not

Projede ham, temizlik öncesi veri dosyası bulunmadığı için temizlik görseli uydurma “önceki missing sayıları” göstermez. Kodda tanımlanan kuralları ve temizlenmiş verideki sonucu gösterir.
"""
    (OUT / "SUNUM_ANLATIM_REHBERI.md").write_text(guide, encoding="utf-8")


def main() -> None:
    style()
    GRAPHICS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(ROOT / "cleaned_train.csv")
    selected = pd.read_csv(ROOT / "data/processed/selected_features.csv")
    manifest = pd.read_csv(ROOT / "data/manifest.csv")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    validation = pd.read_csv(ROOT / "data/processed/validation.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    fe = pd.read_csv(ROOT / "artifacts/feature_engineering/feature_engineering_summary.csv")

    benchmark, predictions, _ = benchmark_models(train, validation, test)
    dataset_overview(selected, cleaned)
    split_chart(manifest)
    semantic_counts = semantic_cleaning(selected)
    target_summary = target_distribution(selected)
    target_boxplot(selected)
    outlier_scatter(selected)
    outlier_summary = outlier_counts(selected)
    correlation_summary, pair_summary = correlation_charts(selected)
    neighborhood_summary, quality_summary = categorical_charts(selected)
    feature_list = feature_groups(selected)
    pipeline_diagram()
    feature_engineering_chart(fe)
    model_chart(benchmark)
    residual_bands = residual_chart(validation, predictions)

    dataset_summary = pd.DataFrame(
        {
            "Metrik": ["Temiz gözlem", "Temiz kolon", "Seçili feature", "Sayısal feature", "Kategorik feature", "Kalan missing", "Duplicate"],
            "Değer": [len(cleaned), cleaned.shape[1], selected.shape[1] - 1, selected.select_dtypes(include="number").shape[1] - 1, selected.select_dtypes(exclude="number").shape[1], selected.isna().sum().sum(), selected.duplicated().sum()],
        }
    )
    tables = {
        "dataset_summary": dataset_summary,
        "target_summary": target_summary,
        "semantic_cleaning_counts": semantic_counts,
        "outlier_summary": outlier_summary,
        "top_correlations": correlation_summary.head(20),
        "multicollinearity_pairs": pair_summary,
        "neighborhood_summary": neighborhood_summary,
        "overallqual_summary": quality_summary,
        "feature_list": feature_list,
        "feature_engineering_comparison": fe,
        "model_comparison": benchmark,
        "residual_by_price_band": residual_bands,
    }
    for name, frame in tables.items():
        frame.to_csv(TABLES / f"{name}.csv", index=False)

    save_table_png(correlation_summary.head(15), "17_top_correlations_table", "SalePrice ile en güçlü 15 sayısal korelasyon", {"Korelasyon": "{:.3f}"})
    benchmark_table = benchmark[["Model", "Validation MAE", "Validation RMSE", "Test MAE", "Test RMSE", "RMSE Genelleme Farkı"]].rename(columns={"RMSE Genelleme Farkı": "RMSE Farkı"})
    save_table_png(benchmark_table, "18_model_metrics_table", "Model karşılaştırma tablosu — validation ve bağımsız test", {col: "{:,.0f}" for col in benchmark_table.columns if col != "Model"})
    save_table_png(fe[["method", "validation_rmse", "test_rmse", "validation_rmse_improvement_pct", "test_rmse_improvement_pct", "decision"]], "19_feature_engineering_table", "Feature engineering deney sonuçları", {"validation_rmse": "{:,.0f}", "test_rmse": "{:,.0f}", "validation_rmse_improvement_pct": "{:+.2f}%", "test_rmse_improvement_pct": "{:+.2f}%"})

    payload = {name: frame.replace({np.nan: None}).to_dict(orient="records") for name, frame in tables.items()}
    (OUT / "analysis_tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_guide()
    print(json.dumps({"graphics": len(list(GRAPHICS.glob("*.png"))), "svgs": len(list(GRAPHICS.glob("*.svg"))), "tables": len(tables), "model_rows": len(benchmark)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
