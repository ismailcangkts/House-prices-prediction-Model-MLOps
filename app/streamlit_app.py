from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
INFERENCE_DB_PATH = PROJECT_ROOT / "storage" / "inference.db"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Joblib ile kaydedilen pipeline icindeki custom transformer siniflarinin
# Streamlit calisirken de bulunabilmesi icin module import'u burada tutuluyor.
try:
    import pipeline_machine_learning as pml

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name in (
            "DomainFeatureEngineer",
            "NeighborhoodLotFrontageImputer",
            "OrdinalFeatureEncoder",
        ):
            setattr(main_module, name, getattr(pml, name))
except Exception:
    pml = None

from src.models.predict import (
    DEFAULT_MODEL_URI,
    DEFAULT_TRACKING_URI,
    load_model_from_registry,
)


FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "Neighborhood",
    "YearBuilt",
    "YearRemodAdd",
    "OverallCond",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "1stFlrSF",
    "2ndFlrSF",
    "GarageCars",
    "GarageArea",
    "GarageType",
    "GarageFinish",
    "GarageYrBlt",
    "FullBath",
    "HalfBath",
    "TotRmsAbvGrd",
    "KitchenQual",
    "ExterQual",
    "BsmtQual",
    "BsmtExposure",
    "Fireplaces",
    "FireplaceQu",
    "LotArea",
    "LotFrontage",
    "MSSubClass",
    "MSZoning",
    "HouseStyle",
    "OpenPorchSF",
]


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource
def load_champion_model():
    return load_model_from_registry(
        model_uri=DEFAULT_MODEL_URI,
        tracking_uri=DEFAULT_TRACKING_URI,
    )


@st.cache_data
def get_champion_model_version() -> str:
    model_name, alias = parse_model_uri(DEFAULT_MODEL_URI)
    client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
    model_version = client.get_model_version_by_alias(model_name, alias)
    return str(model_version.version)


def parse_model_uri(model_uri: str) -> tuple[str, str]:
    prefix = "models:/"
    if not model_uri.startswith(prefix) or "@" not in model_uri:
        raise ValueError(f"Unsupported model URI format: {model_uri}")

    model_name, alias = model_uri.removeprefix(prefix).split("@", maxsplit=1)
    return model_name, alias


def init_inference_db() -> None:
    INFERENCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INFERENCE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_uri TEXT NOT NULL,
                model_version TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                prediction REAL NOT NULL,
                input_json TEXT NOT NULL
            )
            """
        )


def save_inference_log(
    input_df: pd.DataFrame,
    prediction: float,
    model_version: str,
    latency_ms: float,
) -> int:
    init_inference_db()
    created_at = datetime.now(timezone.utc).isoformat()
    input_json = json.dumps(input_df.iloc[0].to_dict(), default=str)

    with sqlite3.connect(INFERENCE_DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO inference_logs (
                created_at,
                model_uri,
                model_version,
                latency_ms,
                prediction,
                input_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                DEFAULT_MODEL_URI,
                model_version,
                latency_ms,
                prediction,
                input_json,
            ),
        )
        return int(cursor.lastrowid)


def options_for(df: pd.DataFrame, column: str) -> list[str]:
    return sorted(df[column].dropna().astype(str).unique().tolist())


def number_input(
    label: str,
    value: float,
    min_value: float,
    max_value: float,
    step=None,
):
    if step is None:
        step = 1 if isinstance(value, int) else 1.0

    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
    )


def main() -> None:
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

    st.title("House Price Prediction")
    st.caption("User prediction form for the model trained on the Ames housing data")

    st.info(f"Model: `{DEFAULT_MODEL_URI}` | Tracking URI: `{DEFAULT_TRACKING_URI}`")
    init_inference_db()

    df = load_reference_data()
    medians = df[FEATURE_COLUMNS].median(numeric_only=True)

    with st.form("prediction_form"):
        st.subheader("Property Details")

        left, middle, right = st.columns(3)

        with left:
            overall_qual = st.slider("Overall Quality", 1, 10, int(medians["OverallQual"]))
            overall_cond = st.slider("Overall Condition", 1, 10, int(medians["OverallCond"]))
            year_built = number_input(
                "Year Built", int(medians["YearBuilt"]), 1870, 2026
            )
            year_remod = number_input(
                "Year Remodeled", int(medians["YearRemodAdd"]), 1870, 2026
            )
            neighborhood = st.selectbox(
                "Neighborhood",
                options_for(df, "Neighborhood"),
                index=options_for(df, "Neighborhood").index("NAmes"),
            )
            zoning = st.selectbox("Zoning", options_for(df, "MSZoning"))
            house_style = st.selectbox("House Style", options_for(df, "HouseStyle"))
            ms_subclass = number_input(
                "Building Class", int(medians["MSSubClass"]), 20, 190
            )

        with middle:
            gr_liv_area = number_input(
                "Above Ground Living Area", int(medians["GrLivArea"]), 300, 6000
            )
            first_flr_sf = number_input(
                "First Floor Area", int(medians["1stFlrSF"]), 300, 5000
            )
            second_flr_sf = number_input(
                "Second Floor Area", int(medians["2ndFlrSF"]), 0, 2500
            )
            total_bsmt_sf = number_input(
                "Total Basement Area", int(medians["TotalBsmtSF"]), 0, 4000
            )
            bsmt_fin_sf1 = number_input(
                "Finished Basement Area", int(medians["BsmtFinSF1"]), 0, 4000
            )
            full_bath = number_input("Full Bathrooms", int(medians["FullBath"]), 0, 5)
            half_bath = number_input("Half Bathrooms", int(medians["HalfBath"]), 0, 3)
            rooms = number_input(
                "Total Rooms Above Ground", int(medians["TotRmsAbvGrd"]), 2, 15
            )

        with right:
            garage_cars = number_input(
                "Garage Car Capacity", int(medians["GarageCars"]), 0, 5
            )
            garage_area = number_input(
                "Garage Area", int(medians["GarageArea"]), 0, 1500
            )
            garage_type = st.selectbox("Garage Type", options_for(df, "GarageType"))
            garage_finish = st.selectbox(
                "Garage Finish", options_for(df, "GarageFinish")
            )
            garage_year = number_input(
                "Garage Year Built", int(medians["GarageYrBlt"]), 0, 2026
            )
            fireplaces = number_input("Fireplaces", int(medians["Fireplaces"]), 0, 4)
            fireplace_qu = st.selectbox(
                "Fireplace Quality", options_for(df, "FireplaceQu")
            )
            open_porch_sf = number_input(
                "Open Porch Area", int(medians["OpenPorchSF"]), 0, 800
            )

        st.subheader("Quality and Lot")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            kitchen_qual = st.selectbox("Kitchen Quality", options_for(df, "KitchenQual"))
        with q2:
            exter_qual = st.selectbox("Exterior Quality", options_for(df, "ExterQual"))
        with q3:
            bsmt_qual = st.selectbox("Basement Quality", options_for(df, "BsmtQual"))
        with q4:
            bsmt_exposure = st.selectbox(
                "Basement Exposure", options_for(df, "BsmtExposure")
            )

        a1, a2 = st.columns(2)
        with a1:
            lot_area = number_input("Lot Area", int(medians["LotArea"]), 1000, 100000)
        with a2:
            lot_frontage = number_input(
                "Lot Frontage", float(medians["LotFrontage"]), 0.0, 350.0
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "OverallQual": overall_qual,
                    "GrLivArea": gr_liv_area,
                    "Neighborhood": neighborhood,
                    "YearBuilt": year_built,
                    "YearRemodAdd": year_remod,
                    "OverallCond": overall_cond,
                    "TotalBsmtSF": total_bsmt_sf,
                    "BsmtFinSF1": bsmt_fin_sf1,
                    "1stFlrSF": first_flr_sf,
                    "2ndFlrSF": second_flr_sf,
                    "GarageCars": garage_cars,
                    "GarageArea": garage_area,
                    "GarageType": garage_type,
                    "GarageFinish": garage_finish,
                    "GarageYrBlt": garage_year,
                    "FullBath": full_bath,
                    "HalfBath": half_bath,
                    "TotRmsAbvGrd": rooms,
                    "KitchenQual": kitchen_qual,
                    "ExterQual": exter_qual,
                    "BsmtQual": bsmt_qual,
                    "BsmtExposure": bsmt_exposure,
                    "Fireplaces": fireplaces,
                    "FireplaceQu": fireplace_qu,
                    "LotArea": lot_area,
                    "LotFrontage": lot_frontage,
                    "MSSubClass": ms_subclass,
                    "MSZoning": zoning,
                    "HouseStyle": house_style,
                    "OpenPorchSF": open_porch_sf,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        try:
            model = load_champion_model()
            model_version = get_champion_model_version()
            start_time = time.perf_counter()
            prediction = float(model.predict(input_df)[0])
            latency_ms = (time.perf_counter() - start_time) * 1000
            inference_id = save_inference_log(
                input_df=input_df,
                prediction=prediction,
                model_version=model_version,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            st.error(
                "Prediction could not be completed. Make sure you are using the "
                "same package versions as the environment where the model was saved, "
                "and that the MLflow tracking server is running."
            )
            st.exception(exc)
            st.stop()

        st.success(f"Predicted sale price: ${prediction:,.0f}")
        st.caption(
            f"Inference ID: `{inference_id}` | Model version: `{model_version}` | "
            f"Latency: `{latency_ms:.2f} ms`"
        )
        with st.expander("Data sent to the model"):
            st.dataframe(input_df, width="stretch")


if __name__ == "__main__":
    main()
