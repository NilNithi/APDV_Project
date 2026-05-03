"""Pearson and Spearman correlation matrices between key features."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

ANALYSIS_COLS = [
    "log_price",
    "nearest_park_dist_m",
    "green_area_within_500m",
    "green_area_within_1000m",
    "mean_no2_year",
    "mean_pm25_year",
    "mean_pm10_year",
    "mean_noise_db_year",
]


def _load_enriched() -> pd.DataFrame:
    """Load enriched property parquet, falling back to interim clean file.

    Returns:
        Property DataFrame ready for correlation analysis.

    Raises:
        FileNotFoundError: If neither file is present on disk.
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


def _pearson_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Pearson r and p-values.

    Args:
        df: DataFrame whose columns are all numeric; no NaNs expected.

    Returns:
        Tuple of (r_matrix, p_matrix) as DataFrames with identical index/columns.
    """
    cols = df.columns.tolist()
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                r_mat.loc[c1, c2] = 1.0
                p_mat.loc[c1, c2] = 0.0
            elif i < j:
                r, p = stats.pearsonr(df[c1], df[c2])
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = r
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = p

    return r_mat, p_mat


def _spearman_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Spearman rho and p-values.

    Args:
        df: DataFrame whose columns are all numeric; no NaNs expected.

    Returns:
        Tuple of (rho_matrix, p_matrix) as DataFrames with identical index/columns.
    """
    cols = df.columns.tolist()
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                r_mat.loc[c1, c2] = 1.0
                p_mat.loc[c1, c2] = 0.0
            elif i < j:
                r, p = stats.spearmanr(df[c1], df[c2])
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = float(r)
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = float(p)

    return r_mat, p_mat


def _pearson_matrix_pairwise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson r using pairwise complete observations.

    Args:
        df: DataFrame with possibly sparse columns.

    Returns:
        Tuple of (r_matrix, p_matrix).
    """
    cols = df.columns.tolist()
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                r_mat.loc[c1, c2] = 1.0
                p_mat.loc[c1, c2] = 0.0
            elif i < j:
                pair = df[[c1, c2]].dropna()
                if len(pair) < 3:
                    continue
                r, p = stats.pearsonr(pair[c1], pair[c2])
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = r
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = p

    return r_mat, p_mat


def _spearman_matrix_pairwise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman rho using pairwise complete observations.

    Args:
        df: DataFrame with possibly sparse columns.

    Returns:
        Tuple of (rho_matrix, p_matrix).
    """
    cols = df.columns.tolist()
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                r_mat.loc[c1, c2] = 1.0
                p_mat.loc[c1, c2] = 0.0
            elif i < j:
                pair = df[[c1, c2]].dropna()
                if len(pair) < 3:
                    continue
                r, p = stats.spearmanr(pair[c1], pair[c2])
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = float(r)
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = float(p)

    return r_mat, p_mat


def run() -> dict[str, pd.DataFrame]:
    """Compute Pearson and Spearman correlation matrices for key features.

    Only columns that exist in the loaded DataFrame are included. Rows with any
    NaN across the selected columns are dropped before computation.

    Returns:
        Dict with keys:
        - ``"pearson"``: Pearson r matrix.
        - ``"pearson_pval"``: Pearson p-value matrix.
        - ``"spearman"``: Spearman rho matrix.
        - ``"spearman_pval"``: Spearman p-value matrix.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_enriched()

    # Add log_price if needed
    if "log_price" not in df.columns and "price" in df.columns:
        df = df.copy()
        df["log_price"] = np.log(df["price"].clip(lower=1))

    # Select only columns present in this dataset
    available = [c for c in ANALYSIS_COLS if c in df.columns]
    if len(available) < 2:
        logger.error(
            "Fewer than 2 analysis columns available (%s) — cannot compute correlations",
            available,
        )
        empty = pd.DataFrame()
        return {
            "pearson": empty,
            "pearson_pval": empty,
            "spearman": empty,
            "spearman_pval": empty,
        }

    subset = df[available]

    # Use pairwise complete observations so sparse AQ columns (23% non-null)
    # don't eliminate all rows from the analysis.
    non_null_counts = subset.notna().sum()
    logger.info(
        "Non-null counts per column: %s",
        non_null_counts.to_dict(),
    )

    # Pairwise Pearson / Spearman — each pair uses only rows non-null in both.
    logger.info(
        "Computing pairwise correlations on %d total rows, %d columns: %s",
        len(subset),
        len(available),
        available,
    )

    pearson_r, pearson_p = _pearson_matrix_pairwise(subset)
    spearman_r, spearman_p = _spearman_matrix_pairwise(subset)

    results: dict[str, pd.DataFrame] = {
        "pearson": pearson_r,
        "pearson_pval": pearson_p,
        "spearman": spearman_r,
        "spearman_pval": spearman_p,
    }

    file_map = {
        "pearson": "correlation_pearson.parquet",
        "pearson_pval": "correlation_pearson_pval.parquet",
        "spearman": "correlation_spearman.parquet",
        "spearman_pval": "correlation_spearman_pval.parquet",
    }

    for key, fname in file_map.items():
        out = PROCESSED_DIR / fname
        results[key].to_parquet(out)
        logger.info("Saved %s -> %s", key, out)

    # Log headline correlations for quick sanity-check
    for col in [c for c in available if c != "log_price"]:
        if "log_price" in pearson_r.columns and col in pearson_r.index:
            r_val = pearson_r.loc["log_price", col]
            p_val = pearson_p.loc["log_price", col]
            logger.info(
                "Pearson log_price ~ %s: r=%.3f  p=%.4f", col, r_val, p_val
            )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    matrices = run()
    for k, v in matrices.items():
        print(f"\n=== {k} ===")
        print(v)
