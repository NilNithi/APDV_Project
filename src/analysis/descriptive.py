"""Descriptive statistics for the enriched property dataset."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def _load_enriched() -> pd.DataFrame:
    """Load the enriched property parquet, falling back to the clean interim file.

    Returns:
        DataFrame of enriched (or cleaned) property records.

    Raises:
        FileNotFoundError: If neither parquet file exists.
    """
    enriched_path = PROCESSED_DIR / "property_enriched.parquet"
    interim_path = INTERIM_DIR / "property_clean.parquet"

    if enriched_path.exists():
        logger.info("Loading enriched parquet from %s", enriched_path)
        return pd.read_parquet(enriched_path)

    if interim_path.exists():
        logger.warning(
            "Enriched parquet not found; falling back to %s", interim_path
        )
        return pd.read_parquet(interim_path)

    raise FileNotFoundError(
        f"No enriched parquet at {enriched_path} and no interim at {interim_path}. "
        "Run the ETL pipeline first."
    )


def _ensure_log_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_price column if not already present.

    Args:
        df: Property DataFrame that must contain a ``price`` column.

    Returns:
        DataFrame with ``log_price`` column added (or left unchanged if present).
    """
    if "log_price" not in df.columns and "price" in df.columns:
        df = df.copy()
        df["log_price"] = np.log(df["price"].clip(lower=1))
    return df


def run() -> dict[str, pd.DataFrame]:
    """Compute descriptive statistics for the enriched property dataset.

    Produces four summary tables:
    - ``by_year``: aggregated stats per sale year.
    - ``by_postal_code``: aggregated stats per postal code.
    - ``quintiles``: stats by green-distance quintile.
    - ``overall``: single-row summary of all numeric columns of interest.

    Each table is persisted as a parquet file under ``data/processed/``.

    Returns:
        Dict with keys ``"by_year"``, ``"by_postal_code"``, ``"quintiles"``,
        ``"overall"`` mapping to the corresponding DataFrames.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_enriched()
    df = _ensure_log_price(df)

    logger.info("Loaded %d rows for descriptive analysis", len(df))

    # ------------------------------------------------------------------
    # 1. by_year
    # ------------------------------------------------------------------
    agg_cols: dict[str, object] = {
        "price": ["count", "mean", "median",
                  lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
    }
    optional_means = ["nearest_park_dist_m", "mean_no2_year"]
    group_col_year = "year_of_sale" if "year_of_sale" in df.columns else None

    if group_col_year:
        by_year_parts: list[pd.DataFrame] = []

        grp_year = df.groupby(group_col_year)

        base_year = grp_year["price"].agg(
            count="count",
            mean_price="mean",
            median_price="median",
            q1_price=lambda x: x.quantile(0.25),
            q3_price=lambda x: x.quantile(0.75),
        )
        by_year_parts.append(base_year)

        for col in optional_means:
            if col in df.columns:
                by_year_parts.append(
                    grp_year[col].mean().rename(f"mean_{col}")
                )

        by_year = pd.concat(by_year_parts, axis=1).reset_index()
    else:
        logger.warning("year_of_sale column missing — by_year will be empty")
        by_year = pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. by_postal_code
    # ------------------------------------------------------------------
    pc_col: Optional[str] = None
    for candidate in ("postal_code", "postcode", "eircode"):
        if candidate in df.columns:
            pc_col = candidate
            break

    if pc_col:
        grp_pc = df.groupby(pc_col)
        pc_parts: list[pd.DataFrame] = [
            grp_pc["price"].agg(
                count="count",
                mean_price="mean",
                median_price="median",
                q1_price=lambda x: x.quantile(0.25),
                q3_price=lambda x: x.quantile(0.75),
            )
        ]
        for col in optional_means:
            if col in df.columns:
                pc_parts.append(grp_pc[col].mean().rename(f"mean_{col}"))

        by_postal_code = pd.concat(pc_parts, axis=1).reset_index()
    else:
        logger.warning("No postal code column found — by_postal_code will be empty")
        by_postal_code = pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. quintiles  (by nearest_park_dist_m)
    # ------------------------------------------------------------------
    if "nearest_park_dist_m" in df.columns:
        valid_dist = df["nearest_park_dist_m"].dropna()
        if len(valid_dist) >= 5:
            df = df.copy()
            df["green_quintile"] = pd.qcut(
                df["nearest_park_dist_m"],
                q=5,
                labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"],
                duplicates="drop",
            )
            grp_q = df.groupby("green_quintile", observed=True)
            q_parts: list[pd.DataFrame] = [
                grp_q["price"].agg(
                    count="count",
                    median_price="median",
                    mean_price="mean",
                )
            ]
            if "mean_no2_year" in df.columns:
                q_parts.append(grp_q["mean_no2_year"].mean().rename("mean_no2_year"))

            quintiles = pd.concat(q_parts, axis=1).reset_index()
        else:
            logger.warning(
                "Too few non-null nearest_park_dist_m rows (%d) for quintiles",
                len(valid_dist),
            )
            quintiles = pd.DataFrame()
    else:
        logger.warning("nearest_park_dist_m not present — quintiles will be empty")
        quintiles = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. overall
    # ------------------------------------------------------------------
    stat_cols = [
        "price",
        "log_price",
        "nearest_park_dist_m",
        "green_area_within_500m",
        "mean_no2_year",
        "mean_pm25_year",
    ]
    present_cols = [c for c in stat_cols if c in df.columns]
    overall = (
        df[present_cols]
        .agg(["min", "max", "mean", "median", "std"])
        .T.rename_axis("feature")
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    results: dict[str, pd.DataFrame] = {
        "by_year": by_year,
        "by_postal_code": by_postal_code,
        "quintiles": quintiles,
        "overall": overall,
    }

    for name, table in results.items():
        out_path = PROCESSED_DIR / f"descriptive_{name}.parquet"
        if not table.empty:
            table.to_parquet(out_path, index=False)
            logger.info("Saved %s (%d rows) → %s", name, len(table), out_path)
        else:
            logger.warning("Table '%s' is empty — skipping parquet write", name)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tables = run()
    for k, v in tables.items():
        print(f"\n=== {k} ===")
        print(v.head())
