import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pipeline_machine_learning import (
    build_model_candidates,
    DomainFeatureEngineer,
    NeighborhoodLotFrontageImputer,
    OrdinalFeatureEncoder,
    make_one_hot_encoder,
    ONE_HOT_COLS
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import warnings
warnings.filterwarnings('ignore')

def build_custom_preprocessor(feature_engineer):
    return Pipeline(
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

def build_custom_pipeline(feature_engineer):
    steps = [
        ("preprocessing", build_custom_preprocessor(feature_engineer)),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=Ridge(alpha=10.0),
            func=np.log1p,
            inverse_func=np.expm1,
        ))
    ]
    return Pipeline(steps=steps)

def evaluate(pipeline, X_train, y_train, X_valid, y_valid):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    return mean_squared_error(y_valid, preds) ** 0.5

def main():
    train_df = pd.read_csv("data/processed/train.csv")
    valid_df = pd.read_csv("data/processed/validation.csv")

    y_train = train_df["SalePrice"]
    X_train = train_df.drop("SalePrice", axis=1)
    y_valid = valid_df["SalePrice"]
    X_valid = valid_df.drop("SalePrice", axis=1)

    print("--- BASELINE ---")
    baseline_rmse = evaluate(build_custom_pipeline(DomainFeatureEngineer()), X_train, y_train, X_valid, y_valid)
    print(f"Baseline RMSE: {baseline_rmse:.2f}")

    print("\n--- EXPERIMENT 7: Combo (TotalSF + TotalBathrooms + Qual_High/Low) ---")
    class FeatEngExp7(DomainFeatureEngineer):
        def transform(self, X):
            X = super().transform(X)
            # 1. TotalSF
            X["TotalSF"] = X.get("TotalBsmtSF", 0) + X.get("1stFlrSF", 0) + X.get("2ndFlrSF", 0)
            X = X.drop(columns=[col for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"] if col in X.columns])
            
            # 2. TotalBathrooms
            X["TotalBathrooms"] = X.get("FullBath", 0) + 0.5 * X.get("HalfBath", 0)
            X = X.drop(columns=[col for col in ["FullBath", "HalfBath"] if col in X.columns])
            
            # 3. Binning Qual
            if "OverallQual" in X.columns:
                X["Qual_High"] = (X["OverallQual"] >= 8).astype(int)
                X["Qual_Low"] = (X["OverallQual"] <= 4).astype(int)
                
            return X
            
    rmse_7 = evaluate(build_custom_pipeline(FeatEngExp7()), X_train, y_train, X_valid, y_valid)
    print(f"Combo RMSE: {rmse_7:.2f} (diff: {rmse_7 - baseline_rmse:.2f})")

if __name__ == '__main__':
    main()
