"""Clean and aggregate raw air quality data from MongoDB raw_air collection.

Uses MongoDB aggregation pipelines (server-side) to avoid loading 21M+ docs
into RAM.  Produces monthly and annual means per station per pollutant,
upserts results into ``processed_air_monthly`` / ``processed_air_annual``,
and exports annual aggregates to Parquet.

Idempotent: re-running replaces existing processed documents via upsert.

Usage::

    python -m src.etl.air_clean
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import UpdateOne

from src.config import get_mongo_db, PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

_COL_RAW = "raw_air"
_COL_MONTHLY = "processed_air_monthly"
_COL_ANNUAL = "processed_air_annual"

# Spike thresholds per pollutant (µg/m³ or dB).  Values above these are
# sensor errors, not real readings.  Conservative upper bounds.
_SPIKE_LIMITS: dict[str, float] = {
    "NO2": 500.0,
    "NOx": 1000.0,
    "PM2.5": 500.0,
    "PM10": 1000.0,
    "O3": 400.0,
    "SO2": 500.0,
    "CO": 50000.0,
    "LAeq": 120.0,
    "LA90": 120.0,
    "LA10": 120.0,
    "LAFmax": 130.0,
}


# ---------------------------------------------------------------------------
# MongoDB aggregation approach — no full load into RAM
# ---------------------------------------------------------------------------


def _coerce_fields_stage() -> list[dict]:
    """Return pipeline stages that coerce string fields to proper types.

    Real API docs store value/year/month/lat/lon as strings and pollutant
    in lowercase.  This normalizes everything before filtering/grouping.
    """
    return [
        {
            "$addFields": {
                "value": {"$toDouble": "$value"},
                "year": {"$toInt": {"$toDouble": "$year"}},
                "month": {"$toInt": {"$toDouble": "$month"}},
                "lat": {"$toDouble": "$lat"},
                "lon": {"$toDouble": "$lon"},
                "pollutant": {"$toUpper": "$pollutant"},
            }
        },
    ]


def _monthly_pipeline() -> list[dict]:
    """Build MongoDB aggregation pipeline for monthly means.

    Filters: value must be numeric and >= 0, year and month must exist.
    Groups by (monitor_id, pollutant, year, month) with optional station
    location fields.
    """
    return [
        # Stage 0: coerce string types from API docs
        *_coerce_fields_stage(),
        # Stage 1: filter bad rows
        {
            "$match": {
                "value": {"$gte": 0, "$type": "number"},
                "year": {"$exists": True, "$ne": None},
                "month": {"$exists": True, "$ne": None},
            }
        },
        # Stage 2: group by station/pollutant/year/month
        {
            "$group": {
                "_id": {
                    "monitor_id": "$monitor_id",
                    "pollutant": "$pollutant",
                    "year": "$year",
                    "month": "$month",
                },
                "mean_value": {"$avg": "$value"},
                "min_value": {"$min": "$value"},
                "max_value": {"$max": "$value"},
                "count": {"$sum": 1},
                "station_name": {"$first": "$station_name"},
                "lat": {"$first": "$lat"},
                "lon": {"$first": "$lon"},
            }
        },
        # Stage 3: flatten _id
        {
            "$project": {
                "_id": 0,
                "monitor_id": "$_id.monitor_id",
                "pollutant": "$_id.pollutant",
                "year": "$_id.year",
                "month": "$_id.month",
                "mean_value": 1,
                "min_value": 1,
                "max_value": 1,
                "count": 1,
                "station_name": 1,
                "lat": 1,
                "lon": 1,
            }
        },
    ]


def _annual_pipeline() -> list[dict]:
    """Build MongoDB aggregation pipeline for annual means."""
    return [
        *_coerce_fields_stage(),
        {
            "$match": {
                "value": {"$gte": 0, "$type": "number"},
                "year": {"$exists": True, "$ne": None},
            }
        },
        {
            "$group": {
                "_id": {
                    "monitor_id": "$monitor_id",
                    "pollutant": "$pollutant",
                    "year": "$year",
                },
                "mean_value": {"$avg": "$value"},
                "min_value": {"$min": "$value"},
                "max_value": {"$max": "$value"},
                "count": {"$sum": 1},
                "station_name": {"$first": "$station_name"},
                "lat": {"$first": "$lat"},
                "lon": {"$first": "$lon"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "monitor_id": "$_id.monitor_id",
                "pollutant": "$_id.pollutant",
                "year": "$_id.year",
                "mean_value": 1,
                "min_value": 1,
                "max_value": 1,
                "count": 1,
                "station_name": 1,
                "lat": 1,
                "lon": 1,
            }
        },
    ]


def _run_aggregation(db, pipeline: list[dict], label: str) -> pd.DataFrame:
    """Run a MongoDB aggregation pipeline and return result as DataFrame.

    Args:
        db: PyMongo Database object.
        pipeline: Aggregation pipeline stages.
        label: Label for logging (e.g. "monthly", "annual").

    Returns:
        DataFrame with aggregated rows.
    """
    logger.info("Running %s aggregation pipeline on %s ...", label, _COL_RAW)
    cursor = db[_COL_RAW].aggregate(pipeline, allowDiskUse=True)
    docs = list(cursor)
    logger.info("%s aggregation returned %d rows.", label, len(docs))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # Apply spike limits — drop rows where mean exceeds threshold
    if "pollutant" in df.columns and "mean_value" in df.columns:
        before = len(df)
        mask = pd.Series(True, index=df.index)
        for pollutant, limit in _SPIKE_LIMITS.items():
            pmask = df["pollutant"] == pollutant
            mask &= ~(pmask & (df["mean_value"] > limit))
        df = df[mask].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info("Spike filter removed %d %s rows.", dropped, label)
    return df


def _upsert_monthly(db, df: pd.DataFrame) -> int:
    """Upsert monthly aggregates into processed_air_monthly."""
    if df.empty:
        logger.warning("Monthly DataFrame empty — nothing to upsert.")
        return 0

    col = db[_COL_MONTHLY]
    col.create_index(
        [("monitor_id", 1), ("pollutant", 1), ("year", 1), ("month", 1)],
        unique=True,
        background=True,
    )

    ops = [
        UpdateOne(
            {
                "monitor_id": row["monitor_id"],
                "pollutant": row["pollutant"],
                "year": int(row["year"]),
                "month": int(row["month"]),
            },
            {"$set": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                       for k, v in row.items()}},
            upsert=True,
        )
        for row in df.to_dict("records")
    ]
    result = col.bulk_write(ops, ordered=False)
    total = result.upserted_count + result.modified_count
    logger.info(
        "processed_air_monthly upserted=%d modified=%d (total=%d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


def _upsert_annual(db, df: pd.DataFrame) -> int:
    """Upsert annual aggregates into processed_air_annual."""
    if df.empty:
        logger.warning("Annual DataFrame empty — nothing to upsert.")
        return 0

    col = db[_COL_ANNUAL]
    col.create_index(
        [("monitor_id", 1), ("pollutant", 1), ("year", 1)],
        unique=True,
        background=True,
    )

    ops = [
        UpdateOne(
            {
                "monitor_id": row["monitor_id"],
                "pollutant": row["pollutant"],
                "year": int(row["year"]),
            },
            {"$set": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                       for k, v in row.items()}},
            upsert=True,
        )
        for row in df.to_dict("records")
    ]
    result = col.bulk_write(ops, ordered=False)
    total = result.upserted_count + result.modified_count
    logger.info(
        "processed_air_annual upserted=%d modified=%d (total=%d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


def _export_parquet(df: pd.DataFrame, filename: str) -> None:
    """Export DataFrame to Parquet in processed data directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    export_df = df.copy()
    for col in export_df.select_dtypes(include=["datetimetz", "datetime64"]).columns:
        export_df[col] = export_df[col].astype(str)
    export_df.to_parquet(out_path, index=False)
    logger.info("Exported %d rows to %s", len(export_df), out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run() -> dict[str, pd.DataFrame]:
    """Clean and aggregate raw air quality data using server-side aggregation.

    Uses MongoDB aggregation pipelines instead of loading all documents into
    pandas — handles 20M+ documents without OOM.

    Returns:
        Dict with keys "monthly" and "annual" mapping to aggregated DataFrames.
    """
    db = get_mongo_db()

    raw_count = db[_COL_RAW].estimated_document_count()
    logger.info("raw_air has ~%d documents. Aggregating server-side ...", raw_count)

    if raw_count == 0:
        logger.warning("raw_air empty — skipping ETL.")
        empty = pd.DataFrame()
        return {"monthly": empty, "annual": empty}

    # --- server-side aggregation ---
    df_monthly = _run_aggregation(db, _monthly_pipeline(), "monthly")
    df_annual = _run_aggregation(db, _annual_pipeline(), "annual")

    # --- persist to MongoDB ---
    _upsert_monthly(db, df_monthly)
    _upsert_annual(db, df_annual)

    # --- export Parquet ---
    if not df_annual.empty:
        _export_parquet(df_annual, "air_annual.parquet")

    logger.info(
        "air_clean complete. monthly_rows=%d  annual_rows=%d",
        len(df_monthly),
        len(df_annual),
    )
    return {"monthly": df_monthly, "annual": df_annual}


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run()
    print(
        f"Done — monthly rows: {len(result['monthly'])}, "
        f"annual rows: {len(result['annual'])}"
    )
