"""Clean and aggregate raw air quality data from MongoDB raw_air collection.

Reads every document from ``raw_air``, applies quality filters (negative
readings, impossible spikes), aggregates to monthly and annual means per
station per pollutant, upserts results back into ``processed_air_monthly``
and ``processed_air_annual``, and exports annual aggregates to Parquet.

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

# Columns that must be present after loading raw_air into a DataFrame.
_REQUIRED_COLS = {"monitor_id", "pollutant", "value", "year", "month"}

# MongoDB collection names.
_COL_RAW = "raw_air"
_COL_MONTHLY = "processed_air_monthly"
_COL_ANNUAL = "processed_air_annual"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(db) -> pd.DataFrame:
    """Load all documents from raw_air into a DataFrame.

    Args:
        db: PyMongo Database object.

    Returns:
        DataFrame with one row per raw reading.  May be empty.
    """
    logger.info("Loading documents from %s …", _COL_RAW)
    docs = list(db[_COL_RAW].find({}, {"_id": 0}))
    logger.info("Loaded %d raw_air documents.", len(docs))
    return pd.DataFrame(docs)


def _coerce_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and drop bad rows.

    Steps applied in order:
    1. Coerce ``value`` to numeric; drop rows where coercion fails (NaN).
    2. Drop rows with ``value < 0`` (sensor hardware errors).
    3. Per-pollutant spike removal: drop values above the 99.9th percentile.
    4. Parse ``timestamp`` to datetime and backfill ``year`` / ``month``
       columns from it when they are missing or null.

    Args:
        df: Raw DataFrame loaded from MongoDB.

    Returns:
        Cleaned DataFrame.
    """
    # --- coerce value ---
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["value"])
    logger.info("Dropped %d rows with non-numeric value.", before - len(df))

    # --- drop negative readings ---
    neg_mask = df["value"] < 0
    logger.info("Dropping %d rows with value < 0.", neg_mask.sum())
    df = df[~neg_mask].copy()

    # --- per-pollutant spike removal (99.9th percentile) ---
    rows_before_spike = len(df)
    keep_mask = pd.Series(True, index=df.index)
    for pollutant in df["pollutant"].unique():
        mask = df["pollutant"] == pollutant
        threshold = df.loc[mask, "value"].quantile(0.999)
        spike_mask = mask & (df["value"] > threshold)
        keep_mask &= ~spike_mask
        n_spikes = spike_mask.sum()
        if n_spikes:
            logger.debug(
                "Pollutant %s: removed %d spike readings above %.2f.",
                pollutant,
                n_spikes,
                threshold,
            )
    df = df[keep_mask].copy()
    logger.info(
        "Spike removal: dropped %d rows across all pollutants.",
        rows_before_spike - len(df),
    )

    # --- parse timestamp → year / month ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Fill missing year/month from parsed timestamp.
        if "year" not in df.columns:
            df["year"] = np.nan
        if "month" not in df.columns:
            df["month"] = np.nan
        year_null = df["year"].isna()
        month_null = df["month"].isna()
        df.loc[year_null, "year"] = df.loc[year_null, "timestamp"].dt.year
        df.loc[month_null, "month"] = df.loc[month_null, "timestamp"].dt.month

    # Cast year/month to int where possible (some may still be NaN if timestamp
    # was also null — drop those rows to avoid groupby issues).
    df = df.dropna(subset=["year", "month"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    logger.info("%d rows remain after all cleaning steps.", len(df))
    return df


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly mean/min/max/count per station per pollutant.

    Args:
        df: Cleaned DataFrame from :func:`_coerce_and_filter`.

    Returns:
        Aggregated DataFrame with one row per
        (monitor_id, station_name, lat, lon, pollutant, year, month).
    """
    # Identify which optional location columns are present.
    group_cols = ["monitor_id", "pollutant", "year", "month"]
    for col in ("station_name", "lat", "lon"):
        if col in df.columns:
            group_cols.append(col)

    agg = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(
            mean_value="mean",
            min_value="min",
            max_value="max",
            count="count",
        )
        .reset_index()
    )
    logger.info("Monthly aggregation produced %d rows.", len(agg))
    return agg


def _aggregate_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual mean/min/max/count per station per pollutant.

    Args:
        df: Cleaned DataFrame from :func:`_coerce_and_filter`.

    Returns:
        Aggregated DataFrame with one row per
        (monitor_id, station_name, lat, lon, pollutant, year).
    """
    group_cols = ["monitor_id", "pollutant", "year"]
    for col in ("station_name", "lat", "lon"):
        if col in df.columns:
            group_cols.append(col)

    agg = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(
            mean_value="mean",
            min_value="min",
            max_value="max",
            count="count",
        )
        .reset_index()
    )
    logger.info("Annual aggregation produced %d rows.", len(agg))
    return agg


def _upsert_monthly(db, df: pd.DataFrame) -> int:
    """Upsert monthly aggregates into ``processed_air_monthly``.

    Composite upsert key: ``(monitor_id, pollutant, year, month)``.

    Args:
        db: PyMongo Database object.
        df: Monthly aggregates DataFrame.

    Returns:
        Total upserted + modified count.
    """
    if df.empty:
        logger.warning("Monthly DataFrame is empty — nothing to upsert.")
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
        "processed_air_monthly upserted=%d modified=%d (total in collection: %d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


def _upsert_annual(db, df: pd.DataFrame) -> int:
    """Upsert annual aggregates into ``processed_air_annual``.

    Composite upsert key: ``(monitor_id, pollutant, year)``.

    Args:
        db: PyMongo Database object.
        df: Annual aggregates DataFrame.

    Returns:
        Total upserted + modified count.
    """
    if df.empty:
        logger.warning("Annual DataFrame is empty — nothing to upsert.")
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
        "processed_air_annual upserted=%d modified=%d (total in collection: %d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


def _export_parquet(df: pd.DataFrame, filename: str) -> None:
    """Export a DataFrame to Parquet in the processed data directory.

    Timestamps (if present) are cast to strings before export to avoid
    Arrow type-inference issues with mixed timezone-aware/naive values.

    Args:
        df: DataFrame to export.
        filename: Filename (e.g. ``"air_annual.parquet"``).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename

    export_df = df.copy()
    # Coerce any datetime columns to ISO strings for safe Parquet serialisation.
    for col in export_df.select_dtypes(include=["datetimetz", "datetime64"]).columns:
        export_df[col] = export_df[col].astype(str)

    export_df.to_parquet(out_path, index=False)
    logger.info("Exported %d rows to %s", len(export_df), out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run() -> dict[str, pd.DataFrame]:
    """Clean and aggregate raw air quality data, writing processed collections.

    Orchestration steps:
        1. Load all documents from MongoDB ``raw_air``.
        2. Coerce types, drop bad rows, remove spikes.
        3. Aggregate to monthly and annual granularity.
        4. Upsert results into ``processed_air_monthly`` and
           ``processed_air_annual`` (idempotent).
        5. Export annual aggregates to ``data/processed/air_annual.parquet``.

    Returns:
        A dict with keys ``"monthly"`` and ``"annual"`` mapping to the
        respective aggregated DataFrames.  Both will be empty DataFrames if
        ``raw_air`` contains no usable data.
    """
    db = get_mongo_db()

    # --- load ---
    df_raw = _load_raw(db)
    if df_raw.empty:
        logger.warning("raw_air is empty — skipping ETL. Returning empty DataFrames.")
        empty = pd.DataFrame()
        return {"monthly": empty, "annual": empty}

    # --- check required columns ---
    missing = _REQUIRED_COLS - set(df_raw.columns)
    if missing:
        logger.warning(
            "raw_air documents are missing expected columns: %s. "
            "Will attempt to continue with available data.",
            missing,
        )

    # --- clean ---
    df_clean = _coerce_and_filter(df_raw)
    if df_clean.empty:
        logger.warning("No rows survived cleaning. Returning empty DataFrames.")
        empty = pd.DataFrame()
        return {"monthly": empty, "annual": empty}

    # --- aggregate ---
    df_monthly = _aggregate_monthly(df_clean)
    df_annual = _aggregate_annual(df_clean)

    # --- persist to MongoDB ---
    _upsert_monthly(db, df_monthly)
    _upsert_annual(db, df_annual)

    # --- export Parquet ---
    _export_parquet(df_annual, "air_annual.parquet")

    logger.info(
        "air_clean complete. monthly_rows=%d  annual_rows=%d",
        len(df_monthly),
        len(df_annual),
    )
    return {"monthly": df_monthly, "annual": df_annual}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
