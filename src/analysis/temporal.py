"""Year-over-year price trends by green-proximity quintile."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.environ.get("GREEN_PREMIUM_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def _load_enriched() -> pd.DataFrame:
    """Load enriched property parquet, falling back to interim clean file.

    Returns:
        Property DataFrame ready for temporal analysis.

    Raises:
        FileNotFoundError: If neither file exists on disk.
    """
    enriched = PROCESSED_DIR / "property_enriched.parquet"
    interim = INTERIM_DIR / "property_clean.parquet"

    if enriched.exists():
        logger.info("Loading %s", enriched)
        return pd.read_parquet(enriched)
    if interim.exists():
        logger.warning("Falling back to %s", interim)
        return pd.read_parquet(interim)

    raise FileNotFoundError(
        f"No enriched parquet at {enriched} and no interim at {interim}."
    )


def run() -> dict[str, Any]:
    """Compute year-over-year price trends segmented by green-proximity quintile.

    Steps:
    1. Load the enriched parquet.
    2. Assign 5 green-proximity quintiles based on ``nearest_park_dist_m``.
    3. Pivot: rows = year_of_sale, columns = quintile, values = median price.
    4. Compute year-over-year percentage change per quintile.
    5. Measure the Q1–Q5 price gap over time via linear regression (gap ~ year).
    6. Save results to ``data/processed/``.

    Returns:
        Dict with keys:
        - ``"pivot"``: DataFrame of median prices (years x quintiles).
        - ``"yoy_change"``: DataFrame of YoY % changes.
        - ``"gap_trend"``: Dict with keys ``slope``, ``p_value``, ``r2``,
          ``interpretation``.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_enriched()

    if "year_of_sale" not in df.columns:
        raise ValueError(
            "year_of_sale column is required for temporal analysis but was not found."
        )

    if "nearest_park_dist_m" not in df.columns:
        raise ValueError(
            "nearest_park_dist_m column is required for quintile analysis but was not found."
        )

    if "price" not in df.columns:
        raise ValueError("price column is required but was not found.")

    # Drop rows missing any key column
    df = df.dropna(subset=["year_of_sale", "nearest_park_dist_m", "price"])
    logger.info("Temporal analysis: %d rows after dropping NaN", len(df))

    # ------------------------------------------------------------------
    # Assign quintiles
    # ------------------------------------------------------------------
    df = df.copy()
    df["green_quintile"] = pd.qcut(
        df["nearest_park_dist_m"],
        q=5,
        labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"],
        duplicates="drop",
    )

    # ------------------------------------------------------------------
    # Pivot: median price per year × quintile
    # ------------------------------------------------------------------
    pivot = (
        df.groupby(["year_of_sale", "green_quintile"], observed=True)["price"]
        .median()
        .unstack(level="green_quintile")
    )
    pivot.index.name = "year_of_sale"
    pivot = pivot.sort_index()

    logger.info(
        "Pivot table: %d years × %d quintiles", len(pivot), len(pivot.columns)
    )

    # ------------------------------------------------------------------
    # Year-over-year % change
    # ------------------------------------------------------------------
    yoy_change = pivot.pct_change() * 100  # percentage points
    yoy_change.index.name = "year_of_sale"

    # ------------------------------------------------------------------
    # Q1–Q5 price gap trend (linear regression: gap ~ year)
    # ------------------------------------------------------------------
    gap_trend: dict[str, Any] = {}
    if "Q1_closest" in pivot.columns and "Q5_farthest" in pivot.columns:
        gap_series = (pivot["Q1_closest"] - pivot["Q5_farthest"]).dropna()
        years = gap_series.index.astype(float).values
        gaps = gap_series.values

        if len(gaps) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                years, gaps
            )
            r2 = r_value**2

            if p_value < 0.05:
                direction = "widening" if slope > 0 else "narrowing"
                interp = (
                    f"The Q1–Q5 price gap is statistically significantly {direction} "
                    f"by €{slope:,.0f}/year (p={p_value:.4f}, R²={r2:.3f})."
                )
            else:
                interp = (
                    f"No statistically significant trend in the Q1–Q5 price gap "
                    f"(slope=€{slope:,.0f}/year, p={p_value:.4f}, R²={r2:.3f})."
                )

            gap_trend = {
                "slope": float(slope),
                "intercept": float(intercept),
                "p_value": float(p_value),
                "r2": float(r2),
                "std_err": float(std_err),
                "n_years": len(gaps),
                "interpretation": interp,
            }
            logger.info(interp)
        else:
            logger.warning(
                "Only %d years with both Q1 and Q5 data — skipping gap regression",
                len(gaps),
            )
            gap_trend = {
                "slope": None,
                "p_value": None,
                "r2": None,
                "interpretation": "Insufficient data for gap trend regression.",
            }
    else:
        available_quintiles = pivot.columns.tolist()
        logger.warning(
            "Q1_closest or Q5_farthest not in pivot columns %s — skipping gap analysis",
            available_quintiles,
        )
        gap_trend = {
            "slope": None,
            "p_value": None,
            "r2": None,
            "interpretation": "Q1 or Q5 quintile column missing.",
        }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    pivot_out = PROCESSED_DIR / "temporal_median_price.parquet"
    pivot.reset_index().to_parquet(pivot_out, index=False)
    logger.info("Saved pivot -> %s", pivot_out)

    yoy_out = PROCESSED_DIR / "temporal_yoy_change.parquet"
    yoy_change.reset_index().to_parquet(yoy_out, index=False)
    logger.info("Saved YoY change -> %s", yoy_out)

    return {
        "pivot": pivot,
        "yoy_change": yoy_change,
        "gap_trend": gap_trend,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run()
    print("\n=== Median price pivot ===")
    print(results["pivot"])
    print("\n=== YoY % change ===")
    print(results["yoy_change"])
    print("\n=== Gap trend ===")
    print(results["gap_trend"]["interpretation"])
