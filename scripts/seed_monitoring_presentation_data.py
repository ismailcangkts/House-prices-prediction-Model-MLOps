from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


REFERENCE_PATH = Path("data/processed/reference.csv")
DATABASE_PATH = Path("storage/inference.db")
MODEL_URI = "models:/HousePriceModel@champion"
TARGET = "SalePrice"
SYNTHETIC_SCENARIO = "monitoring_slide_v1"
DEFAULT_COUNT = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed clearly marked synthetic inference logs for the monitoring slide."
    )
    parser.add_argument("--reference-path", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--database-path", type=Path, default=DATABASE_PATH)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace only records created by this presentation scenario.",
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
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
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(inference_logs)").fetchall()
    }
    if "is_synthetic" not in existing_columns:
        conn.execute(
            "ALTER TABLE inference_logs ADD COLUMN is_synthetic INTEGER NOT NULL DEFAULT 0"
        )
    if "scenario" not in existing_columns:
        conn.execute("ALTER TABLE inference_logs ADD COLUMN scenario TEXT")


def model_version_for(index: int, count: int) -> str:
    progress = index / max(count, 1)
    if progress < 0.20:
        return "2"
    if progress < 0.48:
        return "3"
    if progress < 0.70:
        return "6"
    return "7"


def latency_for(version: str, index: int) -> float:
    base_latency = {
        "2": 36.0,
        "3": 27.0,
        "6": 20.0,
        "7": 14.5,
    }[version]
    variation = 2.4 * math.sin(index * 0.73) + 0.8 * math.cos(index * 0.31)
    return round(max(8.5, base_latency + variation), 2)


def prediction_for(actual_price: float, version: str, index: int) -> float:
    calibration = {
        "2": 0.955,
        "3": 0.975,
        "6": 0.990,
        "7": 1.000,
    }[version]
    deterministic_noise = actual_price * 0.035 * math.sin(index * 1.17)
    prediction = actual_price * calibration + deterministic_noise
    return round(min(max(prediction, 75_000), 520_000), 2)


def build_rows(reference_df: pd.DataFrame, count: int) -> list[tuple]:
    if TARGET not in reference_df.columns:
        raise ValueError(f"Reference dataset must contain {TARGET}.")

    sampled = reference_df.sample(
        n=count,
        replace=count > len(reference_df),
        random_state=42,
    ).reset_index(drop=True)
    end_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    start_time = end_time - timedelta(hours=2 * (count - 1))
    rows = []

    for index, record in sampled.iterrows():
        version = model_version_for(index, count)
        created_at = start_time + timedelta(hours=2 * index)
        actual_price = float(record[TARGET])
        input_payload = record.drop(labels=[TARGET]).to_dict()
        rows.append(
            (
                created_at.isoformat(),
                MODEL_URI,
                version,
                latency_for(version, index),
                prediction_for(actual_price, version, index),
                json.dumps(input_payload, default=str),
                1,
                SYNTHETIC_SCENARIO,
            )
        )

    return rows


def seed_monitoring_data(
    reference_path: Path,
    database_path: Path,
    count: int,
    replace: bool,
) -> int:
    if count <= 0:
        raise ValueError("count must be greater than zero")

    reference_df = pd.read_csv(reference_path)
    rows = build_rows(reference_df, count)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as conn:
        ensure_schema(conn)
        existing_count = conn.execute(
            "SELECT COUNT(*) FROM inference_logs WHERE scenario = ?",
            (SYNTHETIC_SCENARIO,),
        ).fetchone()[0]

        if existing_count and not replace:
            print(
                f"Synthetic monitoring scenario already exists: "
                f"{SYNTHETIC_SCENARIO} ({existing_count} rows)"
            )
            return int(existing_count)

        if replace:
            conn.execute(
                "DELETE FROM inference_logs WHERE scenario = ?",
                (SYNTHETIC_SCENARIO,),
            )

        conn.executemany(
            """
            INSERT INTO inference_logs (
                created_at,
                model_uri,
                model_version,
                latency_ms,
                prediction,
                input_json,
                is_synthetic,
                scenario
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return len(rows)


def main() -> None:
    args = parse_args()
    inserted = seed_monitoring_data(
        reference_path=args.reference_path,
        database_path=args.database_path,
        count=args.count,
        replace=args.replace,
    )
    print(f"Synthetic monitoring rows ready: {inserted}")
    print(f"Scenario: {SYNTHETIC_SCENARIO}")


if __name__ == "__main__":
    main()
