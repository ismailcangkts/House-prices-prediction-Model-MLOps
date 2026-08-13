from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DB_PATH = PROJECT_ROOT / "storage" / "inference.db"
DRIFT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "drift" / "first_drift_report.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import DEFAULT_MODEL_URI, DEFAULT_TRACKING_URI


@st.cache_data(ttl=10)
def load_drift_report() -> dict:
    if not DRIFT_REPORT_PATH.exists():
        return {}

    return json.loads(DRIFT_REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=10)
def load_inference_logs() -> pd.DataFrame:
    if not INFERENCE_DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(INFERENCE_DB_PATH) as conn:
        logs = pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                model_uri,
                model_version,
                latency_ms,
                prediction,
                input_json
            FROM inference_logs
            ORDER BY id ASC
            """,
            conn,
        )

    if logs.empty:
        return logs

    logs["created_at"] = pd.to_datetime(logs["created_at"], errors="coerce")
    logs["prediction"] = pd.to_numeric(logs["prediction"], errors="coerce")
    logs["latency_ms"] = pd.to_numeric(logs["latency_ms"], errors="coerce")
    logs["model_version"] = logs["model_version"].astype(str)
    return logs


def format_price(value: float) -> str:
    return f"${value:,.0f}"


def render_summary(logs: pd.DataFrame) -> None:
    latest = logs.iloc[-1]
    total_predictions = len(logs)
    avg_latency = logs["latency_ms"].mean()
    avg_prediction = logs["prediction"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", f"{total_predictions:,}")
    c2.metric("Latest Model Version", latest["model_version"])
    c3.metric("Average Latency", f"{avg_latency:.2f} ms")
    c4.metric("Average Prediction", format_price(avg_prediction))


def render_drift_status(drift_report: dict) -> None:
    if not drift_report:
        st.info("No drift report found. Run the drift report script first.")
        return

    summary = drift_report["summary"]
    alert_status = summary["alert_status"]
    weighted_score = summary["weighted_drift_score"]
    drifted_columns = summary["drifted_columns_count"]
    total_columns = summary["total_columns_count"]
    drifted_critical_features = summary["drifted_critical_features"]

    if alert_status == "CRITICAL_DATASET_DRIFT":
        st.error("Critical dataset drift detected.")
    elif alert_status == "DATASET_DRIFT_WARNING":
        st.warning("Dataset drift warning detected.")
    else:
        st.success("No dataset drift warning.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drift Status", alert_status)
    c2.metric("Weighted Drift Score", f"{weighted_score:.2%}")
    c3.metric("Drifted Columns", f"{drifted_columns}/{total_columns}")
    c4.metric("Critical Feature Drift", f"{len(drifted_critical_features)}")

    if drifted_critical_features:
        st.caption(
            "Drifted critical features: "
            + ", ".join(drifted_critical_features)
        )

    with st.expander("Drift details"):
        details = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "weight": summary["feature_weights"].get(feature, 0.0),
                    "drift_detected": result["drift_detected"],
                    "drift_score": result["drift_score"],
                    "test": result["stattest_name"],
                    "threshold": result["threshold"],
                }
                for feature, result in summary["columns"].items()
            ]
        ).sort_values(["drift_detected", "weight"], ascending=[False, False])
        st.dataframe(details, width="stretch", hide_index=True)


def render_prediction_trend(logs: pd.DataFrame) -> None:
    chart = (
        alt.Chart(logs)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Time"),
            y=alt.Y("prediction:Q", title="Predicted Sale Price"),
            tooltip=[
                alt.Tooltip("id:Q", title="Inference ID"),
                alt.Tooltip("created_at:T", title="Time"),
                alt.Tooltip("prediction:Q", title="Prediction", format="$,.0f"),
                alt.Tooltip("model_version:N", title="Model Version"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_latency_trend(logs: pd.DataFrame) -> None:
    chart = (
        alt.Chart(logs)
        .mark_line(point=True, color="#d95f02")
        .encode(
            x=alt.X("created_at:T", title="Time"),
            y=alt.Y("latency_ms:Q", title="Latency (ms)"),
            tooltip=[
                alt.Tooltip("id:Q", title="Inference ID"),
                alt.Tooltip("created_at:T", title="Time"),
                alt.Tooltip("latency_ms:Q", title="Latency", format=".2f"),
                alt.Tooltip("model_version:N", title="Model Version"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_version_counts(logs: pd.DataFrame) -> None:
    version_counts = (
        logs.groupby("model_version", as_index=False)
        .size()
        .rename(columns={"size": "prediction_count"})
    )
    chart = (
        alt.Chart(version_counts)
        .mark_bar()
        .encode(
            x=alt.X("model_version:N", title="Model Version"),
            y=alt.Y("prediction_count:Q", title="Prediction Count"),
            tooltip=[
                alt.Tooltip("model_version:N", title="Model Version"),
                alt.Tooltip("prediction_count:Q", title="Prediction Count"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_prediction_distribution(logs: pd.DataFrame) -> None:
    chart = (
        alt.Chart(logs)
        .mark_bar()
        .encode(
            x=alt.X("prediction:Q", bin=alt.Bin(maxbins=20), title="Predicted Sale Price"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_recent_predictions(logs: pd.DataFrame) -> None:
    recent = logs.sort_values("id", ascending=False).head(20).copy()
    recent["created_at"] = recent["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    recent["prediction"] = recent["prediction"].map(format_price)
    recent["latency_ms"] = recent["latency_ms"].map(lambda value: f"{value:.2f}")

    st.dataframe(
        recent[
            [
                "id",
                "created_at",
                "model_version",
                "latency_ms",
                "prediction",
                "model_uri",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("Latest input JSON"):
        st.code(logs.iloc[-1]["input_json"], language="json")


def main() -> None:
    st.set_page_config(page_title="Inference Monitoring", page_icon=":bar_chart:", layout="wide")

    st.title("Inference Monitoring")
    st.caption("Production prediction logs from the SQLite inference store")
    st.info(f"Model: `{DEFAULT_MODEL_URI}` | Tracking URI: `{DEFAULT_TRACKING_URI}`")

    st.subheader("Drift Status")
    render_drift_status(load_drift_report())

    logs = load_inference_logs()
    if logs.empty:
        st.info("No predictions logged yet. Run a prediction from the main page first.")
        return

    render_summary(logs)

    st.subheader("Prediction Trend")
    render_prediction_trend(logs)

    left, right = st.columns(2)
    with left:
        st.subheader("Latency Trend")
        render_latency_trend(logs)
    with right:
        st.subheader("Predictions by Model Version")
        render_version_counts(logs)

    st.subheader("Prediction Distribution")
    render_prediction_distribution(logs)

    st.subheader("Recent Predictions")
    render_recent_predictions(logs)


if __name__ == "__main__":
    main()
