from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_machine_learning import (
    ONE_HOT_COLS,
    DomainFeatureEngineer,
    NeighborhoodLotFrontageImputer,
    OrdinalFeatureEncoder,
    make_one_hot_encoder,
)


TARGET = "SalePrice"
METHODS = (
    "combination",
    "decomposition",
    "transformation",
    "interaction",
    "indicator",
    "binning",
    "domain",
)

METHOD_LABELS = {
    "baseline": "Baz model (yeni feature yok)",
    "combination": "Birleştirme",
    "decomposition": "Ayrıştırma",
    "transformation": "Dönüştürme",
    "interaction": "Etkileşim",
    "indicator": "Binary / Indicator",
    "binning": "Binning / Gruplama",
    "domain": "Domain-based",
    "validation_positive_combined": "Validation'da olumlu yöntemlerin birleşimi",
    "all_methods": "Tüm yöntemlerin birleşimi",
    "current_project": "Mevcut proje feature engineering'i",
}

FEATURE_DESCRIPTIONS = {
    "combination": "TotalSF, TotalBathrooms",
    "decomposition": "BuildDecade, BuildYearInDecade, RemodelDecade, RemodelYearInDecade",
    "transformation": (
        "LogLotArea, LogGrLivArea, LogLotFrontage, LogOpenPorchSF, "
        "LogTotalBsmtSF, Log1stFlrSF, Log2ndFlrSF, LogGarageArea"
    ),
    "interaction": "Qual_x_GrLivArea, Qual_x_TotalBsmtSF, Qual_x_GarageCars",
    "indicator": (
        "HasGarage, HasBasement, HasFireplace, HasSecondFloor, "
        "HasOpenPorch, WasRemodeled"
    ),
    "binning": (
        "Qual_Low/Mid/High, Built_Pre1946/1946_1970/1971_1999/2000Plus"
    ),
    "domain": (
        "HouseAge2010, RemodelAge2010, GarageAge2010, LivingAreaPerRoom, "
        "GarageAreaPerCar, FinishedBasementRatio"
    ),
}


class ExperimentalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add only the requested feature groups; never learn from validation/test data."""

    def __init__(self, methods: tuple[str, ...] = ()):
        self.methods = methods

    def fit(self, X, y=None):
        unknown = sorted(set(self.methods) - set(METHODS))
        if unknown:
            raise ValueError(f"Unknown feature engineering methods: {unknown}")
        return self

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0)

    def transform(self, X):
        X = X.copy()

        if "combination" in self.methods:
            X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalBathrooms"] = X["FullBath"] + 0.5 * X["HalfBath"]

        if "decomposition" in self.methods:
            X["BuildDecade"] = X["YearBuilt"] // 10
            X["BuildYearInDecade"] = X["YearBuilt"] % 10
            X["RemodelDecade"] = X["YearRemodAdd"] // 10
            X["RemodelYearInDecade"] = X["YearRemodAdd"] % 10

        if "transformation" in self.methods:
            for column in (
                "LotArea",
                "GrLivArea",
                "LotFrontage",
                "OpenPorchSF",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
                "GarageArea",
            ):
                X[f"Log{column}"] = np.log1p(X[column].clip(lower=0))

        if "interaction" in self.methods:
            X["Qual_x_GrLivArea"] = X["OverallQual"] * X["GrLivArea"]
            X["Qual_x_TotalBsmtSF"] = X["OverallQual"] * X["TotalBsmtSF"]
            X["Qual_x_GarageCars"] = X["OverallQual"] * X["GarageCars"]

        if "indicator" in self.methods:
            X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
            X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
            X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
            X["HasSecondFloor"] = (X["2ndFlrSF"] > 0).astype(int)
            X["HasOpenPorch"] = (X["OpenPorchSF"] > 0).astype(int)
            X["WasRemodeled"] = (X["YearRemodAdd"] > X["YearBuilt"]).astype(int)

        if "binning" in self.methods:
            quality = X["OverallQual"]
            X["Qual_Low"] = (quality <= 4).astype(int)
            X["Qual_Mid"] = quality.between(5, 7).astype(int)
            X["Qual_High"] = (quality >= 8).astype(int)

            year = X["YearBuilt"]
            X["Built_Pre1946"] = (year < 1946).astype(int)
            X["Built_1946_1970"] = year.between(1946, 1970).astype(int)
            X["Built_1971_1999"] = year.between(1971, 1999).astype(int)
            X["Built_2000Plus"] = (year >= 2000).astype(int)

        if "domain" in self.methods:
            # Ames sales in this project end in 2010. YrSold is not among the 30
            # selected model columns, so 2010 is used as a fixed, leakage-free reference.
            X["HouseAge2010"] = (2010 - X["YearBuilt"]).clip(lower=0)
            X["RemodelAge2010"] = (2010 - X["YearRemodAdd"]).clip(lower=0)
            garage_year = X["GarageYrBlt"].fillna(0)
            X["GarageAge2010"] = np.where(
                garage_year > 0,
                np.maximum(2010 - garage_year, 0),
                0,
            )
            X["LivingAreaPerRoom"] = self._safe_ratio(X["GrLivArea"], X["TotRmsAbvGrd"])
            X["GarageAreaPerCar"] = self._safe_ratio(X["GarageArea"], X["GarageCars"])
            X["FinishedBasementRatio"] = self._safe_ratio(
                X["BsmtFinSF1"], X["TotalBsmtSF"]
            )

        return X


def build_pipeline(methods: tuple[str, ...], use_current_project: bool = False) -> Pipeline:
    feature_engineer = (
        DomainFeatureEngineer()
        if use_current_project
        else ExperimentalFeatureEngineer(methods=methods)
    )
    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("lot_frontage_imputer", NeighborhoodLotFrontageImputer()),
            ("ordinal_encoder", OrdinalFeatureEncoder()),
            (
                "one_hot_encoder",
                ColumnTransformer(
                    transformers=[
                        ("one_hot", make_one_hot_encoder(), ONE_HOT_COLS),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=Ridge(alpha=10.0),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    safe_predictions = np.maximum(predictions, 0)
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "rmsle": float(mean_squared_log_error(y_true, safe_predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate(
    experiment: str,
    methods: tuple[str, ...],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    split: str,
) -> dict:
    pipeline = build_pipeline(methods, use_current_project=experiment == "current_project")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_eval)
    metrics = calculate_metrics(y_eval, predictions)
    transformed_feature_count = pipeline.named_steps["preprocessing"].transform(X_train).shape[1]
    features_added = (
        "TotalSF ve TotalBathrooms (kaynak kolonlar düşürülür); HasGarage, "
        "HasBasement, Qual_High, Qual_Low"
        if experiment == "current_project"
        else "; ".join(FEATURE_DESCRIPTIONS[m] for m in methods) or "Yok"
    )
    return {
        "experiment": experiment,
        "method": METHOD_LABELS[experiment],
        "methods": ",".join(methods) or "none",
        "features_added": features_added,
        "split": split,
        "train_rows": len(X_train),
        "evaluation_rows": len(X_eval),
        "transformed_feature_count": transformed_feature_count,
        **metrics,
    }


def add_baseline_deltas(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["rmse_improvement"] = 0.0
    output["rmse_improvement_pct"] = 0.0
    output["mae_improvement"] = 0.0
    output["mae_improvement_pct"] = 0.0
    output["rmsle_improvement"] = 0.0
    output["rmsle_improvement_pct"] = 0.0

    for split, split_rows in output.groupby("split"):
        baseline = split_rows.loc[split_rows["experiment"] == "baseline"].iloc[0]
        mask = output["split"] == split
        output.loc[mask, "rmse_improvement"] = baseline["rmse"] - output.loc[mask, "rmse"]
        output.loc[mask, "rmse_improvement_pct"] = (
            output.loc[mask, "rmse_improvement"] / baseline["rmse"] * 100
        )
        output.loc[mask, "mae_improvement"] = baseline["mae"] - output.loc[mask, "mae"]
        output.loc[mask, "mae_improvement_pct"] = (
            output.loc[mask, "mae_improvement"] / baseline["mae"] * 100
        )
        output.loc[mask, "rmsle_improvement"] = baseline["rmsle"] - output.loc[mask, "rmsle"]
        output.loc[mask, "rmsle_improvement_pct"] = (
            output.loc[mask, "rmsle_improvement"] / baseline["rmsle"] * 100
        )
    return output


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    individual = results[results["experiment"].isin(METHODS)]
    rows = []
    for experiment in METHODS:
        method_rows = individual[individual["experiment"] == experiment].set_index("split")
        validation = method_rows.loc["validation"]
        test = method_rows.loc["test"]
        validation_positive = (
            validation["rmse_improvement"] > 0 and validation["mae_improvement"] > 0
        )
        test_positive = test["rmse_improvement"] > 0 and test["mae_improvement"] > 0
        if validation_positive and test_positive:
            decision = "Olumlu ve testte doğrulandı"
        elif validation_positive:
            decision = "Validation olumlu, test doğrulamadı"
        elif test_positive:
            decision = "Validation olumsuz, test olumlu"
        else:
            decision = "Olumlu sonuç vermedi"
        rows.append(
            {
                "experiment": experiment,
                "method": METHOD_LABELS[experiment],
                "features_added": validation["features_added"],
                "validation_rmse": validation["rmse"],
                "validation_rmse_improvement": validation["rmse_improvement"],
                "validation_rmse_improvement_pct": validation["rmse_improvement_pct"],
                "validation_mae_improvement_pct": validation["mae_improvement_pct"],
                "validation_rmsle_improvement_pct": validation["rmsle_improvement_pct"],
                "test_rmse": test["rmse"],
                "test_rmse_improvement": test["rmse_improvement"],
                "test_rmse_improvement_pct": test["rmse_improvement_pct"],
                "test_mae_improvement_pct": test["mae_improvement_pct"],
                "test_rmsle_improvement_pct": test["rmsle_improvement_pct"],
                "decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["test_rmse_improvement_pct", "validation_rmse_improvement_pct"],
        ascending=False,
    )


def write_markdown_report(
    output_path: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    positive_methods: tuple[str, ...],
) -> None:
    validation_baseline = results.query(
        "split == 'validation' and experiment == 'baseline'"
    ).iloc[0]
    test_baseline = results.query("split == 'test' and experiment == 'baseline'").iloc[0]
    combined_test = results.query(
        "split == 'test' and experiment == 'validation_positive_combined'"
    ).iloc[0]
    current_validation = results.query(
        "split == 'validation' and experiment == 'current_project'"
    ).iloc[0]
    current_test = results.query(
        "split == 'test' and experiment == 'current_project'"
    ).iloc[0]
    domain_test = results.query("split == 'test' and experiment == 'domain'").iloc[0]
    transformation_test = results.query(
        "split == 'test' and experiment == 'transformation'"
    ).iloc[0]
    all_methods_test = results.query(
        "split == 'test' and experiment == 'all_methods'"
    ).iloc[0]

    table_rows = []
    for _, row in summary.iterrows():
        table_rows.append(
            "| {method} | {vr:+.2f}% | {tr:+.2f}% | {vm:+.2f}% | {tm:+.2f}% | {decision} |".format(
                method=row["method"],
                vr=row["validation_rmse_improvement_pct"],
                tr=row["test_rmse_improvement_pct"],
                vm=row["validation_mae_improvement_pct"],
                tm=row["test_mae_improvement_pct"],
                decision=row["decision"],
            )
        )

    confirmed = summary[summary["decision"] == "Olumlu ve testte doğrulandı"]["method"].tolist()
    selected = [METHOD_LABELS[m] for m in positive_methods]
    report = f"""# Feature Engineering Deney Raporu

## Yönetici özeti

- Baz model: Ridge Regression (`alpha=10`), standartlaştırılmış girdiler ve log-dönüşümlü hedef.
- Validation bazı: RMSE **{validation_baseline['rmse']:.2f}**, MAE **{validation_baseline['mae']:.2f}**.
- Bağımsız test bazı: RMSE **{test_baseline['rmse']:.2f}**, MAE **{test_baseline['mae']:.2f}**.
- Hem validation hem testte RMSE ve MAE'yi birlikte iyileştiren yöntemler: **{', '.join(confirmed) or 'Yok'}**.
- Validation sonucuna göre birleşim için seçilenler: **{', '.join(selected) or 'Yok'}**.
- Seçilen yöntemlerin birleşimi testte RMSE'yi **{combined_test['rmse_improvement_pct']:+.2f}%**, MAE'yi **{combined_test['mae_improvement_pct']:+.2f}%** değiştirdi.
- Mevcut proje feature engineering'i baz modele göre validation RMSE'yi **{current_validation['rmse_improvement_pct']:+.2f}%**, test RMSE'yi **{current_test['rmse_improvement_pct']:+.2f}%** değiştirdi.

Pozitif yüzde baz modele göre hata azalmasını, negatif yüzde hata artışını gösterir.

## Metrik bazında kısmi olumlu sonuçlar

- **Domain-based** feature'lar test MAE'yi **{domain_test['mae_improvement_pct']:+.2f}%** ve RMSLE'yi **{domain_test['rmsle_improvement_pct']:+.2f}%** iyileştirdi; ancak RMSE **{domain_test['rmse_improvement_pct']:+.2f}%** kötüleşti. Tipik/tahmini oransal hata azalırken büyük hatalar arttı.
- **Dönüştürme** test MAE'yi **{transformation_test['mae_improvement_pct']:+.2f}%** ve RMSLE'yi **{transformation_test['rmsle_improvement_pct']:+.2f}%** iyileştirdi; buna karşılık RMSE **{transformation_test['rmse_improvement_pct']:+.2f}%** kötüleşti.
- **Tüm yöntemlerin birleşimi** test MAE'de **{all_methods_test['mae_improvement_pct']:+.2f}%**, RMSLE'de **{all_methods_test['rmsle_improvement_pct']:+.2f}%** kazanç sağladı; RMSE'de **{all_methods_test['rmse_improvement_pct']:+.2f}%** kayıp oluşturdu. Projenin ana seçim metriği RMSE olduğu için production adayı sayılmadı.

## Sonuç tablosu

| Yöntem | Validation RMSE | Test RMSE | Validation MAE | Test MAE | Karar |
|---|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

## Deney protokolü

- Projenin mevcut `train.csv` (798), `validation.csv` (218) ve daha önce model seçimi için kullanılmamış `test.csv` (145) ayrımları kullanıldı.
- İlk aşamada her yöntem yalnızca train üzerinde eğitilip validation üzerinde ölçüldü.
- İkinci aşamada train ve validation birleştirildi; her yöntem yeniden eğitilip bağımsız test setinde ölçüldü.
- Her deneyde model, hedef dönüşümü, imputation, ordinal encoding, one-hot encoding ve ölçekleme aynı tutuldu. Değişen tek unsur eklenen feature grubudur.
- “Olumlu ve testte doğrulandı” kararı için hem RMSE'nin hem MAE'nin baz modele göre iki ayrımda da düşmesi şartı uygulandı.

## Denenen feature'lar

"""
    for method in METHODS:
        report += f"- **{METHOD_LABELS[method]}:** {FEATURE_DESCRIPTIONS[method]}\n"

    report += """

## Yorum ve öneri

- Tek bir validation ayrımındaki küçük kazançlar tesadüfi olabilir; üretim kararında bağımsız test doğrulaması esas alınmalıdır.
- Bu deney yalnızca mevcut Ridge modeli için nedensel karşılaştırmadır. Ağaç tabanlı modeller etkileşimleri ve eşikleri kendileri öğrenebildiği için sıralama değişebilir.
- `HouseAge2010` için sabit 2010 referansı kullanıldı; seçili 30 kolonda `YrSold` bulunmuyor. İleride `YrSold` modele geri alınırsa `HouseAge = YrSold - YearBuilt` tercih edilmelidir.
- Mevcut `test_feature_engineering.py` içindeki alt sınıf önce `super().transform()` çağırdığı için kaynak alan kolonları siliniyor; ardından yeniden hesaplanan `TotalSF` ve `TotalBathrooms` sıfıra dönüşüyor. Bu dosyadaki görünen combo iyileşmesi geçerli bir feature-engineering sonucu değildir. Tekrarlanabilir deney için bu raporu üreten `scripts/run_feature_engineering_experiments.py` kullanılmalıdır.
- Üretim pipeline'ına yalnızca bağımsız testte doğrulanan feature gruplarının kontrollü bir aday model olarak alınması önerilir.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature engineering methods fairly.")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument(
        "--validation-path", type=Path, default=Path("data/processed/validation.csv")
    )
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/feature_engineering")
    )
    return parser.parse_args()


def load_xy(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def main() -> None:
    args = parse_args()
    X_train, y_train = load_xy(args.train_path)
    X_validation, y_validation = load_xy(args.validation_path)
    X_test, y_test = load_xy(args.test_path)

    experiment_methods = {"baseline": ()}
    experiment_methods.update({method: (method,) for method in METHODS})
    experiment_methods["current_project"] = ()

    results = []
    for experiment, methods in experiment_methods.items():
        results.append(
            evaluate(
                experiment,
                methods,
                X_train,
                y_train,
                X_validation,
                y_validation,
                split="validation",
            )
        )

    validation_results = add_baseline_deltas(pd.DataFrame(results))
    positive_methods = tuple(
        validation_results.loc[
            (validation_results["experiment"].isin(METHODS))
            & (validation_results["rmse_improvement"] > 0)
            & (validation_results["mae_improvement"] > 0),
            "experiment",
        ].tolist()
    )

    X_train_validation = pd.concat([X_train, X_validation], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], ignore_index=True)
    test_experiments = dict(experiment_methods)
    test_experiments["validation_positive_combined"] = positive_methods
    test_experiments["all_methods"] = METHODS

    test_results = []
    for experiment, methods in test_experiments.items():
        test_results.append(
            evaluate(
                experiment,
                methods,
                X_train_validation,
                y_train_validation,
                X_test,
                y_test,
                split="test",
            )
        )

    test_results_df = add_baseline_deltas(pd.DataFrame(test_results))
    results_df = pd.concat([validation_results, test_results_df], ignore_index=True)
    summary_df = build_summary(results_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "feature_engineering_results.csv"
    summary_path = args.output_dir / "feature_engineering_summary.csv"
    json_path = args.output_dir / "feature_engineering_results.json"
    report_path = args.output_dir / "feature_engineering_report_tr.md"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    payload = {
        "protocol": {
            "model": "Ridge Regression",
            "alpha": 10.0,
            "target_transform": "log1p/expm1",
            "train_rows": len(X_train),
            "validation_rows": len(X_validation),
            "test_rows": len(X_test),
            "positive_selection_rule": "RMSE and MAE improve on validation",
            "confirmed_rule": "RMSE and MAE improve on validation and independent test",
        },
        "validation_positive_methods": list(positive_methods),
        "summary": summary_df.to_dict(orient="records"),
        "results": results_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report_path, results_df, summary_df, positive_methods)

    print(summary_df.round(4).to_string(index=False))
    print(f"\nValidation-positive methods: {positive_methods or 'none'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
