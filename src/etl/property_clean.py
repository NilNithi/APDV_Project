"""Clean and standardize raw property data from raw.property table.

Reads the raw.property table from PostgreSQL, applies all cleaning/parsing
transformations, and writes the result to data/interim/property_clean.parquet
for downstream geocoding and spatial-join steps.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import get_postgres_engine, PROJECT_ROOT

logger = logging.getLogger(__name__)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Regex for a full Irish Eircode: one letter, two digits, space (optional),
# then four alphanumeric characters.  e.g. "D02 XY45" or "D02XY45".
_RE_EIRCODE = re.compile(r"\b([A-Z]\d{2}\s?[A-Z0-9]{4})\b")

# Regex for a Dublin area code as written in PPR addresses.
# Handles "DUBLIN 1", "DUBLIN 10", "DUBLIN 6W", etc.
_RE_DUBLIN_AREA = re.compile(r"\b(DUBLIN\s+\d{1,2}[W]?)\b")

# Floor-area category strings that cannot be converted to a plain number.
# We'll keep them as-is in a separate column rather than silently drop them.
_FLOOR_AREA_CATEGORIES = frozenset(
    {
        "greater than 125 sq metres",
        "38 sq metres and under",
        "between 38 sq metres and 125 sq metres",
    }
)


def _parse_price(series: pd.Series) -> pd.Series:
    """Strip currency symbols/commas and coerce to float.

    Args:
        series: Raw price column (strings like "€245,000.00").

    Returns:
        Float series; unparseable values become NaN.
    """
    # Remove €, commas, spaces, then cast.
    cleaned = (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_postal_code(
    address_clean: pd.Series, existing_postal: pd.Series
) -> pd.Series:
    """Extract or normalise postal codes from address text.

    Preference order:
    1. Use the existing postal_code column if it has a non-empty value.
    2. Extract a full Eircode from the cleaned address string.
    3. Extract a Dublin area code (e.g. "DUBLIN 2") from the cleaned address.

    Args:
        address_clean: Uppercased, whitespace-normalised address strings.
        existing_postal: The raw postal_code column from the database.

    Returns:
        Series of postal code strings (may be NaN where none found).
    """
    result = existing_postal.str.strip().replace("", np.nan)

    # Fill missing from Eircode pattern in the address.
    eircode_match = address_clean.str.extract(_RE_EIRCODE, expand=False)
    result = result.fillna(eircode_match)

    # Fill remaining gaps with Dublin area code pattern.
    dublin_match = address_clean.str.extract(_RE_DUBLIN_AREA, expand=False)
    # Normalise spacing: "DUBLIN  2" → "DUBLIN 2"
    dublin_match = dublin_match.str.replace(r"\s+", " ", regex=True)
    result = result.fillna(dublin_match)

    return result


def _parse_floor_area(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split floor_area into a numeric column and a category column.

    The PPR floor_area field mixes numeric strings (e.g. "85") with
    category descriptions.  We separate them so downstream models can
    use whichever is appropriate.

    Args:
        series: Raw floor_area strings.

    Returns:
        Tuple of (floor_area_sqm: float Series, floor_area_category: str Series).
        floor_area_sqm is NaN for rows that are category strings.
        floor_area_category is NaN for rows that parse as numbers.
    """
    lowered = series.astype(str).str.lower().str.strip()
    numeric_sqm = pd.to_numeric(lowered, errors="coerce")

    # Category column: keep the original string only where it is a known
    # category (and therefore couldn't be parsed as a number).
    is_category = numeric_sqm.isna() & lowered.isin(_FLOOR_AREA_CATEGORIES)
    category_col = series.where(is_category, other=np.nan)

    return numeric_sqm, category_col


def clean_property_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning and parsing transformations to the raw property DataFrame.

    Transformations applied (in order):
    - date_of_sale → datetime; year_of_sale extracted as Int64
    - price → float; log_price derived; zero/null price rows dropped
    - address_clean → uppercased, whitespace-collapsed string
    - postal_code → extracted/normalised (Eircode or Dublin area)
    - not_full_market_price → bool
    - floor_area → split into floor_area_sqm (float) + floor_area_category (str)
    - Rows with null date_of_sale dropped

    Args:
        df: Raw DataFrame loaded from raw.property (column names match DB schema).

    Returns:
        Cleaned DataFrame ready for geocoding and spatial join.
    """
    df = df.copy()

    # ------------------------------------------------------------------ dates
    df["date_of_sale"] = pd.to_datetime(
        df["date_of_sale"], format="%d/%m/%Y", errors="coerce"
    )
    null_dates = df["date_of_sale"].isna().sum()
    if null_dates:
        logger.warning("Dropping %d rows with unparseable date_of_sale", null_dates)
    df = df[df["date_of_sale"].notna()].copy()

    df["year_of_sale"] = df["date_of_sale"].dt.year.astype("Int64")

    # ----------------------------------------------------------------- prices
    df["price"] = _parse_price(df["price"])

    null_prices = df["price"].isna() | (df["price"] <= 0)
    n_bad = null_prices.sum()
    if n_bad:
        logger.warning(
            "Dropping %d rows with null or non-positive price", n_bad
        )
    df = df[~null_prices].copy()

    # Log-transform is safe here: price > 0 guaranteed by the filter above.
    df["log_price"] = np.log(df["price"])

    # --------------------------------------------------------------- address
    df["address_clean"] = (
        df["address"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ----------------------------------------------------------- postal code
    existing_postal = df.get("postal_code", pd.Series([""] * len(df), index=df.index))
    df["postal_code"] = _extract_postal_code(df["address_clean"], existing_postal)

    # ------------------------------------------------------ not_full_market_price
    # Raw values are the strings "Yes" / "No" (case-insensitive).
    nfmp = df["not_full_market_price"].astype(str).str.strip().str.title()
    df["not_full_market_price"] = nfmp.map({"Yes": True, "No": False})
    # Anything that didn't map cleanly stays NaN → nullable bool via object dtype.

    # ------------------------------------------------------------ floor area
    df["floor_area_sqm"], df["floor_area_category"] = _parse_floor_area(
        df.get("floor_area", pd.Series([np.nan] * len(df), index=df.index))
    )
    # Drop the original mixed-content column now that it is split.
    if "floor_area" in df.columns:
        df = df.drop(columns=["floor_area"])

    logger.debug(
        "Cleaning summary — rows: %d | null postal_code: %d | "
        "null floor_area_sqm: %d",
        len(df),
        df["postal_code"].isna().sum(),
        df["floor_area_sqm"].isna().sum(),
    )

    return df


def run() -> pd.DataFrame:
    """Load raw.property, clean every column, cache to parquet, return DataFrame.

    Returns:
        Cleaned property DataFrame with all engineered columns added.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If the database is unreachable.
        ValueError: If the raw table is empty.
    """
    engine = get_postgres_engine()

    logger.info("Loading raw.property from PostgreSQL…")
    df = pd.read_sql("SELECT * FROM raw.property", engine)
    logger.info("Loaded %d raw property rows", len(df))

    if df.empty:
        raise ValueError(
            "raw.property is empty — run src.ingest.property_ingest first."
        )

    df = clean_property_df(df)
    logger.info("After cleaning: %d rows remain", len(df))

    # Persist to interim so downstream steps can skip the DB round-trip.
    out = INTERIM_DIR / "property_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Cleaned data cached → %s", out)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run()
