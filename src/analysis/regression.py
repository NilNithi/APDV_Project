"""OLS regression: log_price ~ green proximity + air quality + controls."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.environ.get("GREEN_PREMIUM_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Percentile thresholds for outlier trimming
_PRICE_LOW_PCT = 1
_PRICE_HIGH_PCT = 99
_PARK_DIST_CAP_PCT = 99


def _load_enriched() -> pd.DataFrame:
    """Load enriched property parquet, falling back to interim clean file.

    Returns:
        Property DataFrame ready for regression.

    Raises:
        FileNotFoundError: If neither file is present.
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


def _extract_coef_df(model: RegressionResultsWrapper) -> pd.DataFrame:
    """Extract coefficient table with 95% confidence intervals.

    Args:
        model: A fitted statsmodels OLS results object.

    Returns:
        DataFrame with columns: coef, std_err, t_stat, p_value, ci_lower, ci_upper.
        Index is the parameter name.
    """
    conf = model.conf_int()
    return pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": conf[0],
            "ci_upper": conf[1],
        }
    )


def _build_formula(df: pd.DataFrame) -> str:
    """Construct the main OLS formula from columns actually present in df.

    Args:
        df: The modelling DataFrame.

    Returns:
        A patsy formula string.
    """
    terms = ["np.log1p(nearest_park_dist_m)"]

    if "green_area_within_500m" in df.columns:
        terms.append("green_area_within_500m")

    if "mean_no2_year" in df.columns and df["mean_no2_year"].notna().any():
        terms.append("mean_no2_year")

    if "mean_pm25_year" in df.columns and df["mean_pm25_year"].notna().any():
        terms.append("mean_pm25_year")

    if "year_of_sale" in df.columns:
        terms.append("C(year_of_sale)")

    if "construction" in df.columns:
        terms.append("C(construction)")

    rhs = " + ".join(terms)
    formula = f"log_price ~ {rhs}"
    logger.info("Main model formula: %s", formula)
    return formula


def _build_interaction_formula(df: pd.DataFrame) -> str:
    """Construct the interaction OLS formula from columns present in df.

    Args:
        df: The modelling DataFrame.

    Returns:
        A patsy formula string including the green x air-quality interaction.
    """
    # Base interaction term — only meaningful if both variables exist
    if "mean_no2_year" in df.columns:
        core = "np.log1p(nearest_park_dist_m) * mean_no2_year"
    else:
        core = "np.log1p(nearest_park_dist_m)"

    terms = [core]

    if "green_area_within_500m" in df.columns:
        terms.append("green_area_within_500m")

    if "year_of_sale" in df.columns:
        terms.append("C(year_of_sale)")

    rhs = " + ".join(terms)
    formula = f"log_price ~ {rhs}"
    logger.info("Interaction model formula: %s", formula)
    return formula


def run() -> dict[str, Any]:
    """Fit main OLS and interaction OLS models; persist results and summary.

    Steps:
    1. Load enriched parquet.
    2. Drop rows missing key modelling columns.
    3. Trim price outliers (1st–99th percentile) and cap park distance.
    4. Fit main model with HC3 robust standard errors.
    5. Fit interaction model (green proximity x NO2).
    6. Save coefficient tables as parquet and OLS summary as plain text.

    Returns:
        Dict with keys:
        - ``"main_model"``: fitted statsmodels RegressionResultsWrapper.
        - ``"interaction_model"``: fitted interaction model.
        - ``"coef_df"``: coefficient DataFrame for the main model.
        - ``"r2"``: adjusted R² of the main model (float).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_enriched()

    # Ensure log_price exists
    if "log_price" not in df.columns:
        if "price" not in df.columns:
            raise ValueError("DataFrame has neither 'log_price' nor 'price' column.")
        df = df.copy()
        df["log_price"] = np.log(df["price"].clip(lower=1))

    # Drop rows missing required columns
    required = ["log_price", "nearest_park_dist_m"]
    if "mean_no2_year" in df.columns:
        required.append("mean_no2_year")
    if "year_of_sale" in df.columns:
        required.append("year_of_sale")

    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Dropped %d rows with NaN in required columns", before - len(df))

    # Trim price outliers
    low = df["log_price"].quantile(_PRICE_LOW_PCT / 100)
    high = df["log_price"].quantile(_PRICE_HIGH_PCT / 100)
    df = df[(df["log_price"] >= low) & (df["log_price"] <= high)]
    logger.info(
        "After price trim [%.0f%% – %.0f%%]: %d rows remain",
        _PRICE_LOW_PCT,
        _PRICE_HIGH_PCT,
        len(df),
    )

    # Cap park distance at 99th percentile to reduce leverage
    if "nearest_park_dist_m" in df.columns:
        cap = df["nearest_park_dist_m"].quantile(_PARK_DIST_CAP_PCT / 100)
        df = df.copy()
        df["nearest_park_dist_m"] = df["nearest_park_dist_m"].clip(upper=cap)
        logger.info("Park distance capped at %.1f m (99th pct)", cap)

    if len(df) < 10:
        raise ValueError(
            f"Too few rows ({len(df)}) after filtering to fit a regression."
        )

    # Coerce nullable extension types to plain numpy dtypes so patsy/statsmodels
    # can introspect them without TypeError.
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], "dtype") and hasattr(df[col].dtype, "numpy_dtype"):
            df[col] = df[col].astype(df[col].dtype.numpy_dtype)
        elif df[col].dtype == "object":
            df[col] = df[col].astype(str)
    if "year_of_sale" in df.columns:
        df["year_of_sale"] = df["year_of_sale"].astype(int)
    if "construction" in df.columns:
        df["construction"] = df["construction"].astype(str).replace("nan", "Unknown").fillna("Unknown")

    # Fit main model
    main_formula = _build_formula(df)
    main_model = smf.ols(main_formula, data=df).fit(cov_type="HC3")

    logger.info(
        "Main OLS — N=%d  R²=%.4f  adj-R²=%.4f",
        int(main_model.nobs),
        main_model.rsquared,
        main_model.rsquared_adj,
    )

    # Log key coefficients
    green_term = "np.log1p(nearest_park_dist_m)"
    for term_key in [green_term, "mean_no2_year", "mean_pm25_year"]:
        if term_key in main_model.params.index:
            coef = main_model.params[term_key]
            pval = main_model.pvalues[term_key]
            ci_lo, ci_hi = main_model.conf_int().loc[term_key]
            logger.info(
                "  %s: coef=%.4f  p=%.4f  CI=[%.4f, %.4f]",
                term_key,
                coef,
                pval,
                ci_lo,
                ci_hi,
            )

    coef_df = _extract_coef_df(main_model)

    # Fit interaction model
    int_formula = _build_interaction_formula(df)
    interaction_model = smf.ols(int_formula, data=df).fit(cov_type="HC3")
    logger.info(
        "Interaction OLS — N=%d  R²=%.4f  adj-R²=%.4f",
        int(interaction_model.nobs),
        interaction_model.rsquared,
        interaction_model.rsquared_adj,
    )

    int_coef_df = _extract_coef_df(interaction_model)

    # Persist
    main_out = PROCESSED_DIR / "regression_results.parquet"
    coef_df.to_parquet(main_out)
    logger.info("Saved main coefficients -> %s", main_out)

    int_out = PROCESSED_DIR / "regression_interaction_results.parquet"
    int_coef_df.to_parquet(int_out)
    logger.info("Saved interaction coefficients -> %s", int_out)

    summary_path = PROCESSED_DIR / "ols_summary.txt"
    summary_path.write_text(
        "=== MAIN MODEL ===\n"
        + str(main_model.summary())
        + "\n\n=== INTERACTION MODEL ===\n"
        + str(interaction_model.summary()),
        encoding="utf-8",
    )
    logger.info("OLS summary text -> %s", summary_path)

    return {
        "main_model": main_model,
        "interaction_model": interaction_model,
        "coef_df": coef_df,
        "r2": float(main_model.rsquared_adj),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(f"\nAdj-R²: {result['r2']:.4f}")
    print(result["coef_df"])
