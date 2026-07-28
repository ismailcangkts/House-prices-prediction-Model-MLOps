import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# Bu dosya, machine_learning.py dosyasindaki ayni akisi pipeline mantigi ile
# yazar. Amac: preprocessing adimlarini daginik satirlar yerine sirali,
# tekrar kullanilabilir ve train/valid ayrimina daha guvenli bir yapida tutmak.

# Ayni anlam grubundaki kolonlari sabit listelerde tutuyoruz. Boylece kodun
# icinde ayni kolon isimlerini tekrar tekrar yazmak yerine bu listeleri kullaniyoruz.
GARAGE_CATEGORICAL_COLS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]

BASEMENT_CATEGORICAL_COLS = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]

BASEMENT_NUMERIC_COLS = [
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]

ONE_HOT_COLS = [
    "MSZoning",
    "Neighborhood",
    "HouseStyle",
    "GarageType",
]


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ham house-prices kolonlarini anlamli model feature'larina cevirir."""

    def fit(self, X, y=None):
        # Bu transformer veri uzerinden herhangi bir istatistik ogrenmiyor.
        # Sadece sabit kurallari uyguluyor; bu yuzden fit icinde self donmek yeterli.
        return self

    def transform(self, X):
        # Pipeline icinde gelen dataframe'i direkt degistirmiyoruz.
        # copy() ile kopya alirsak onceki adimlara yan etki yapmayiz.
        X = X.copy()

        # Id her satir icin benzersiz bir kimlik. Model icin genelde anlamli
        # bir bilgi tasimadigi icin feature'lardan cikariyoruz.
        if "Id" in X.columns:
            X = X.drop("Id", axis=1)

        # PoolQC cok fazla NaN iceriyor. NaN burada "havuz yok" anlamina
        # gelebilecegi icin kolonu silip HasPool adinda binary feature uretiyoruz.
        if "PoolQC" in X.columns:
            X["HasPool"] = X["PoolQC"].notna().astype(int)
            X = X.drop("PoolQC", axis=1)

        # MiscFeature da benzer sekilde nadir dolu geliyor. Detay kolonunu
        # atip "ekstra ozellik var mi?" bilgisini HasMiscFeature ile tutuyoruz.
        if "MiscFeature" in X.columns:
            X["HasMiscFeature"] = X["MiscFeature"].notna().astype(int)
            X = X.drop("MiscFeature", axis=1)

        # Alley icin NaN degeri eksik olmasindan cok "alley yok" anlamina geliyor.
        if "Alley" in X.columns:
            X["Alley"] = X["Alley"].fillna("NoAlley")

        # Fence tek kolon halinde hem tur hem kalite bilgisi tasiyor.
        # Bunu iki ayri feature'a bolerek modelin bilgiyi daha acik gormesini sagliyoruz.
        if "Fence" in X.columns:
            X["Fence"] = X["Fence"].fillna("NoFence")
            fence_type_map = {
                "NoFence": "NoFence",
                "GdPrv": "Privacy",
                "MnPrv": "Privacy",
                "GdWo": "WoodWire",
                "MnWw": "WoodWire",
            }
            fence_quality_map = {
                "NoFence": "NoFence",
                "GdPrv": "Good",
                "GdWo": "Good",
                "MnPrv": "Minimum",
                "MnWw": "Minimum",
            }
            X["FenceType"] = X["Fence"].map(fence_type_map)
            X["FenceQuality"] = X["Fence"].map(fence_quality_map)
            X = X.drop("Fence", axis=1)

        # Masonry veneer yoksa tipi NoMasVnr, alani da 0 olarak dusunuyoruz.
        if "MasVnrType" in X.columns:
            X["MasVnrType"] = X["MasVnrType"].fillna("NoMasVnr")

        if "MasVnrArea" in X.columns:
            X["MasVnrArea"] = X["MasVnrArea"].fillna(0)

        # FireplaceQu NaN ise genelde somine yok demektir.
        if "FireplaceQu" in X.columns:
            X["FireplaceQu"] = X["FireplaceQu"].fillna("Npfireplace")

        # Garajla ilgili kategorik kolonlardaki NaN'lari garaj yok anlaminda dolduruyoruz.
        available_garage_cols = [col for col in GARAGE_CATEGORICAL_COLS if col in X.columns]
        if available_garage_cols:
            X[available_garage_cols] = X[available_garage_cols].fillna("NoGarage")

        # Garaj yili yoksa 0 veriyoruz. Bu, "garaj yok" bilgisini sayisal tarafta temsil eder.
        if "GarageYrBlt" in X.columns:
            X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)

        # Garaj alani 0'dan buyukse evde garaj var diye binary feature uretiyoruz.
        if "GarageArea" in X.columns:
            X["HasGarage"] = (X["GarageArea"] > 0).astype(int)

        # Bodrum kategorik kolonlarindaki NaN'lari "bodrum yok" olarak dolduruyoruz.
        available_basement_cat_cols = [
            col for col in BASEMENT_CATEGORICAL_COLS if col in X.columns
        ]
        if available_basement_cat_cols:
            X[available_basement_cat_cols] = X[available_basement_cat_cols].fillna(
                "NoBasement"
            )

        # Bodrum sayisal kolonlarindaki NaN'lar icin 0 daha anlamli: bodrum yoksa alan/banyo da yok.
        available_basement_num_cols = [
            col for col in BASEMENT_NUMERIC_COLS if col in X.columns
        ]
        if available_basement_num_cols:
            X[available_basement_num_cols] = X[available_basement_num_cols].fillna(0)

        # Bodrum var/yok bilgisini ayrica binary feature olarak ekliyoruz.
        if "TotalBsmtSF" in X.columns:
            X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)

        return X


class NeighborhoodLotFrontageImputer(BaseEstimator, TransformerMixin):
    """LotFrontage eksiklerini sadece train'den ogrendigi medianlarla doldurur."""

    def fit(self, X, y=None):
        # fit sadece train datasinda calisir. Burada Neighborhood bazli medianlari
        # ogreniyoruz; validation/test datasindan bilgi sizdirmiyoruz.
        self.lot_frontage_by_neighborhood_ = X.groupby("Neighborhood")[
            "LotFrontage"
        ].median()
        self.global_lot_frontage_median_ = X["LotFrontage"].median()
        return self

    def transform(self, X):
        # transform train, validation ve test icin ayni ogrenilmis medianlari uygular.
        X = X.copy()
        neighborhood_medians = X["Neighborhood"].map(
            self.lot_frontage_by_neighborhood_
        )
        X["LotFrontage"] = X["LotFrontage"].fillna(neighborhood_medians)
        X["LotFrontage"] = X["LotFrontage"].fillna(
            self.global_lot_frontage_median_
        )
        return X


class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    """Sirali kategorik feature'lari elle tanimlanan anlamli skorlara cevirir."""

    def fit(self, X, y=None):
        # Mappingler sabit oldugu icin burada ogrenilecek bir sey yok.
        return self

    def transform(self, X):
        X = X.copy()

        # Kalite kolonlarinda siralama anlamli: Po < Fa < TA < Gd < Ex.
        # Bu yuzden one-hot yapmak yerine sayisal siraya ceviriyoruz.
        quality_map = {
            "NoBasement": 0,
            "NoGarage": 0,
            "Npfireplace": 0,
            "Po": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5,
        }
        quality_cols = [
            "ExterQual",
            "ExterCond",
            "BsmtQual",
            "BsmtCond",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
        ]

        for col in quality_cols:
            if col in X.columns:
                X[col] = X[col].map(quality_map)

        # Her ordinal kolonun kendi anlamli sirasi var. Bu sozlukler o sirayi
        # sayisal degerlere ceviriyor.
        mappings = {
            "BsmtExposure": {
                "NoBasement": 0,
                "No": 1,
                "Mn": 2,
                "Av": 3,
                "Gd": 4,
            },
            "BsmtFinType1": {
                "NoBasement": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "BsmtFinType2": {
                "NoBasement": 0,
                "Unf": 1,
                "LwQ": 2,
                "Rec": 3,
                "BLQ": 4,
                "ALQ": 5,
                "GLQ": 6,
            },
            "GarageFinish": {
                "NoGarage": 0,
                "Unf": 1,
                "RFn": 2,
                "Fin": 3,
            },
            "Functional": {
                "Sal": 1,
                "Sev": 2,
                "Maj2": 3,
                "Maj1": 4,
                "Mod": 5,
                "Min2": 6,
                "Min1": 7,
                "Typ": 8,
            },
            "PavedDrive": {
                "N": 0,
                "P": 1,
                "Y": 2,
            },
            "LotShape": {
                "IR3": 1,
                "IR2": 2,
                "IR1": 3,
                "Reg": 4,
            },
            "LandSlope": {
                "Sev": 1,
                "Mod": 2,
                "Gtl": 3,
            },
            "FenceQuality": {
                "NoFence": 0,
                "Minimum": 1,
                "Good": 2,
            },
        }

        for col, mapping in mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)

        return X


def make_one_hot_encoder():
    # scikit-learn surumleri arasinda parametre adi degisti:
    # yeni surum: sparse_output=False, eski surum: sparse=False.
    # Bu yardimci fonksiyon iki surumde de calismasi icin yazildi.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor():
    # Preprocessing pipeline'i burada kuruluyor. Adimlar sirayla calisir:
    # 1) domain feature engineering
    # 2) LotFrontage imputation
    # 3) ordinal encoding
    # 4) one-hot encoding
    return Pipeline(
        steps=[
            ("feature_engineering", DomainFeatureEngineer()),
            ("lot_frontage_imputer", NeighborhoodLotFrontageImputer()),
            ("ordinal_encoder", OrdinalFeatureEncoder()),
            (
                "one_hot_encoder",
                ColumnTransformer(
                    transformers=[
                        # get_dummies yerine OneHotEncoder kullaniyoruz.
                        # handle_unknown="ignore": validation/test'te yeni kategori gelirse hata vermez.
                        ("one_hot", make_one_hot_encoder(), ONE_HOT_COLS),
                    ],
                    # ONE_HOT_COLS disindaki kolonlari oldugu gibi sonraki adima gecirir.
                    remainder="passthrough",
                ),
            ),
        ]
    )


def build_model_pipeline(regressor, scale_features=False):
    steps = [
        # Bu final pipeline'da once X preprocessing'den gecer, sonra model egitilir.
        ("preprocessing", build_preprocessor()),
    ]

    # Lineer regresyon gibi katsayi bazli modeller, feature olcekleri birbirine
    # yakin oldugunda daha dengeli calisir. Random forest icin scaling gerekli degil.
    if scale_features:
        steps.append(("scaler", StandardScaler()))

    # Eski dosyada y_train icin np.log1p, prediction icin np.expm1 elle yapiliyordu.
    # TransformedTargetRegressor bunu modelin etrafina sarar ve otomatik yapar.
    log_target_model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    steps.append(("model", log_target_model))

    return Pipeline(steps=steps)


def build_model_candidates():
    # Karsilastirilacak modelleri tek yerde tutuyoruz. Her model kendi pipeline'ini
    # alir; boylece preprocessing fit islemi her denemede train datasinda yapilir.
    return {
        # Eski Random Forest modelini simdilik devre disi biraktik.
        # Tekrar denemek istersen bu blogun yorum satirlarini kaldirabilirsin.
        # "Random Forest": build_model_pipeline(
        #     RandomForestRegressor(
        #         n_estimators=1000,
        #         random_state=42,
        #         n_jobs=-1,
        #         max_depth=20,
        #     )
        # ),
        # Lineer regresyon modelini simdilik devre disi biraktik.
        # Tekrar denemek istersen bu blogun yorum satirlarini kaldirabilirsin.
        # "Linear Regression (Ridge)": build_model_pipeline(
        #     Ridge(alpha=10.0),
        #     scale_features=True,
        # ),
        "Gradient Boosting": build_model_pipeline(
            GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),
        # XGBoost modelini simdilik devre disi biraktik.
        # Tekrar denemek istersen bu blogun yorum satirlarini kaldirabilirsin.
        # "XGBoost": build_model_pipeline(
        #     XGBRegressor(
        #         n_estimators=500,
        #         learning_rate=0.03,
        #         max_depth=3,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         objective="reg:squarederror",
        #         random_state=42,
        #         n_jobs=-1,
        #     )
        # ),
        # LightGBM modelini simdilik devre disi biraktik.
        # Tekrar denemek istersen bu blogun yorum satirlarini kaldirabilirsin.
        # "LightGBM": build_model_pipeline(
        #     LGBMRegressor(
        #         n_estimators=500,
        #         learning_rate=0.03,
        #         max_depth=3,
        #         num_leaves=15,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         random_state=42,
        #         n_jobs=-1,
        #         verbose=-1,
        #     )
        # ),
        #"CatBoost": build_model_pipeline(
        #    CatBoostRegressor(
        #        iterations=500,
        #        learning_rate=0.03,
        #        depth=4,
        #        loss_function="RMSE",
        #        random_state=42,
        #        verbose=False,
        #    )
        #),
    }


def evaluate_model(name, model_pipeline, X_train, X_valid, y_train, y_valid):
    model_pipeline.fit(X_train, y_train)
    valid_predictions = model_pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid, valid_predictions)
    rmse = mean_squared_error(y_valid, valid_predictions) ** 0.5

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "mae_ratio_%": mae / y_valid.mean() * 100,
        "rmse_ratio_%": rmse / y_valid.mean() * 100,
        "pipeline": model_pipeline,
    }


def export_cleaned_data_for_analysis(X_train, X_valid, y_train, y_valid):
    # Analiz yapmak icin sadece okunabilir temiz veri export ediyoruz.
    # Burada one-hot/ordinal encoding yapmiyoruz; amac data analysis icin
    # feature engineering ve LotFrontage temizligi uygulanmis tablo uretmek.
    export_preprocessor = Pipeline(
        steps=[
            ("feature_engineering", DomainFeatureEngineer()),
            ("lot_frontage_imputer", NeighborhoodLotFrontageImputer()),
        ]
    )

    clean_train = export_preprocessor.fit_transform(X_train)
    # Burada valid icin fit_transform degil transform kullaniyoruz.
    # Boylece LotFrontage medianlari sadece train'den ogrenilmis oluyor.
    clean_valid = export_preprocessor.transform(X_valid)

    clean_df = pd.concat([clean_train, clean_valid]).sort_index()
    clean_df["SalePrice"] = pd.concat([y_train, y_valid]).sort_index()
    clean_df.to_csv("pipeline_cleaned_train.csv", index=False)


def main():
    # 1) Daha once hazirlanan 30 feature'li train/validation dosyalari okunur.
    train_df = pd.read_csv("data/processed/train.csv")
    valid_df = pd.read_csv("data/processed/validation.csv")

    # 2) Target ve feature'lar ayrilir.
    y_train = train_df["SalePrice"]
    X_train = train_df.drop("SalePrice", axis=1)
    y_valid = valid_df["SalePrice"]
    X_valid = valid_df.drop("SalePrice", axis=1)

    # 4) Analiz icin pipeline_cleaned_train.csv olusturulur.
    export_cleaned_data_for_analysis(X_train, X_valid, y_train, y_valid)

    # 5) Ayni preprocessing ile farkli modeller egitilir ve karsilastirilir.
    results = []
    for name, model_pipeline in build_model_candidates().items():
        results.append(
            evaluate_model(name, model_pipeline, X_train, X_valid, y_train, y_valid)
        )

    results_df = pd.DataFrame(results).drop(columns="pipeline")
    results_df = results_df.sort_values("rmse").reset_index(drop=True)
    best_result = min(results, key=lambda result: result["rmse"])
    best_pipeline = best_result["pipeline"]

    # Bu iki satir sadece egitim amacli: one-hot sonrasi kolon sayisini gormek icin.
    transformed_train = best_pipeline.named_steps["preprocessing"].transform(X_train)
    transformed_valid = best_pipeline.named_steps["preprocessing"].transform(X_valid)

    print("X_train transformed shape:", transformed_train.shape)
    print("X_valid transformed shape:", transformed_valid.shape)
    print("y_train shape:", y_train.shape)
    print("y_valid shape:", y_valid.shape)

    # 7) Model performanslari yazdirilir.
    print("\nValidation model comparison:")
    print(results_df.round(2).to_string(index=False))
    print("\nBest model by RMSE:", best_result["model"])
    print("Validation SalePrice Mean:", round(y_valid.mean(), 2))
    print("Validation SalePrice Median:", round(y_valid.median(), 2))


if __name__ == "__main__":
    main()
