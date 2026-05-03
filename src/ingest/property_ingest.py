"""Property price register ingest: downloads PSRA CSVs and loads to PostgreSQL raw.property.

Downloads year-by-year CSVs from the Irish Property Price Register (PPR),
filters to Dublin records only, and bulk-inserts into the raw.property table
using ON CONFLICT DO NOTHING for idempotency.
"""

import logging
import time
import urllib3
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from sqlalchemy import text

from src.config import get_postgres_engine

# Suppress SSL warnings — corporate network may have self-signed CA
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "property"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2015, 2022))  # 2015–2021 — available on data.smartdublin.ie

# Direct download URLs from data.smartdublin.ie (Dublin Residential PPR dataset).
# Source: https://data.gov.ie/dataset/dublin-residential-property-price-register
# Note: 2020 and 2021 contain all counties; filtered to Dublin in load_year().
# 2022-2024 not yet published on data.smartdublin.ie as of project date.
_PPR_YEAR_URLS: dict[int, str] = {
    2015: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/11d2f401-87d6-4252-bb1d-8f2af1ef19db/download/ppr-2015-dublin.csv",
    2016: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/a4d52dde-6749-4d6f-87db-26c7ec87b205/download/ppr-2016-dublin.csv",
    2017: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/b91fc10e-0cf2-42b5-80d6-6a7ba2eae59f/download/ppr-2017-dublin.csv",
    2018: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/612b458e-95f6-46d6-8d73-9d06a6c772e3/download/ppr-2018-dublin.csv",
    2019: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/2d81e403-5cb7-4b1c-b296-2b21b95df5c8/download/ppr-2019-dublin.csv",
    2020: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/1b3ccb47-ed22-460c-af96-f066ce2b3c35/download/ppr-2020.csv",
    2021: "https://data.smartdublin.ie/dataset/b0dd7d39-8eb5-4710-b46c-6a0db49e64af/resource/43209239-0ee7-4c8b-9599-79a54b61dd01/download/ppr-2021.csv",
}

# The PPR CSVs are Windows-1252 encoded; decoding as UTF-8 causes errors.
_ENCODING = "cp1252"

# Column names exactly as they appear in the PPR CSV files.
COLUMN_MAP: dict[str, str] = {
    "Date of Sale (dd/mm/yyyy)": "date_of_sale",
    "Address": "address",
    "Postal Code": "postal_code",
    "County": "county",
    "Price (\u20ac)": "price",  # Unicode for €
    "Not Full Market Price": "not_full_market_price",
    "VAT Exclusive": "vat_exclusive",
    "Description of Property": "construction",
    "Property Size Description": "floor_area",
}

_INSERT_SQL = text(
    """
    INSERT INTO raw.property (
        date_of_sale, address, postal_code, county, price,
        not_full_market_price, vat_exclusive, construction, floor_area,
        source_file
    )
    VALUES (
        :date_of_sale, :address, :postal_code, :county, :price,
        :not_full_market_price, :vat_exclusive, :construction, :floor_area,
        :source_file
    )
    ON CONFLICT (date_of_sale, address, price) DO NOTHING
    """
)

_CHUNK_SIZE = 500  # rows per executemany batch
_REQUEST_TIMEOUT = 30  # seconds
_RETRY_ATTEMPTS = 3
_RETRY_SLEEP = 2  # seconds between retries


def download_year(year: int) -> Optional[Path]:
    """Download the PPR CSV for a single year, caching to RAW_DIR.

    Args:
        year: Calendar year (e.g. 2020) to download.

    Returns:
        Path to the downloaded (or already-cached) CSV file, or None if all
        download attempts fail.
    """
    dest = RAW_DIR / f"PPR-{year}.csv"
    if dest.exists():
        logger.info("Cache hit — skipping download for %d (%s)", year, dest)
        return dest

    url = _PPR_YEAR_URLS.get(year)
    if not url:
        logger.error("No URL configured for year %d", year)
        return None
    logger.info("Downloading PPR %d from %s", year, url)

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True, verify=False)
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
            logger.info("Saved %d bytes -> %s", dest.stat().st_size, dest)
            return dest
        except requests.HTTPError as exc:
            logger.warning(
                "HTTP error on attempt %d/%d for year %d: %s",
                attempt,
                _RETRY_ATTEMPTS,
                year,
                exc,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Request error on attempt %d/%d for year %d: %s",
                attempt,
                _RETRY_ATTEMPTS,
                year,
                exc,
            )

        if attempt < _RETRY_ATTEMPTS:
            logger.debug("Sleeping %ds before retry…", _RETRY_SLEEP)
            time.sleep(_RETRY_SLEEP)

    # Clean up partial file if it was created
    if dest.exists():
        dest.unlink()
    logger.error("All %d download attempts failed for year %d", _RETRY_ATTEMPTS, year)
    return None


def load_year(path: Path, year: int, engine) -> int:
    """Read a PPR CSV, filter to Dublin rows, and insert into raw.property.

    The insert uses ON CONFLICT (date_of_sale, address, price) DO NOTHING so
    that re-running the function is safe and produces no duplicates.

    Args:
        path: Absolute path to the PPR CSV file.
        year: Calendar year the file belongs to (used only for logging).
        engine: SQLAlchemy Engine connected to the green_premium database.

    Returns:
        Number of rows actually inserted (conflicts excluded).
    """
    logger.info("Reading %s", path)
    try:
        df = pd.read_csv(path, encoding=_ENCODING, low_memory=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return 0

    logger.debug("Raw row count for %d: %d", year, len(df))

    # Strip leading/trailing whitespace from all column names before mapping.
    df.columns = [c.strip() for c in df.columns]

    # Filter to Dublin rows only (covers "Co. Dublin", "Dublin", etc.)
    county_col = "County"
    if county_col not in df.columns:
        logger.error(
            "Column '%s' not found in %s. Available: %s",
            county_col,
            path,
            list(df.columns),
        )
        return 0

    dublin_mask = df[county_col].str.contains("Dublin", case=False, na=False)
    df = df[dublin_mask].copy()
    logger.info("Year %d — %d Dublin rows after filter", year, len(df))

    if df.empty:
        return 0

    # Rename to DB column names; drop any source columns not in our map.
    available_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing = set(COLUMN_MAP) - set(available_map)
    if missing:
        logger.warning("Columns missing from %s: %s", path.name, missing)

    df = df.rename(columns=available_map)
    db_cols = list(COLUMN_MAP.values())
    present_cols = [c for c in db_cols if c in df.columns]
    df = df[present_cols].copy()

    # Ensure every DB column exists; fill absent ones with None.
    for col in db_cols:
        if col not in df.columns:
            df[col] = None

    # Coerce all columns to plain Python strings (or None) for psycopg2.
    for col in db_cols:
        df[col] = df[col].where(df[col].notna(), other=None)
        df[col] = df[col].apply(lambda v: str(v).strip() if v is not None else None)

    df["source_file"] = path.name

    records = df.to_dict(orient="records")
    inserted = 0

    with engine.begin() as conn:
        for start in range(0, len(records), _CHUNK_SIZE):
            batch = records[start : start + _CHUNK_SIZE]
            result = conn.execute(_INSERT_SQL, batch)
            inserted += result.rowcount

    logger.info("Year %d — inserted %d rows (conflicts silently skipped)", year, inserted)
    return inserted


def run() -> None:
    """Orchestrate full ingest for all configured years.

    Downloads each year's CSV (skipping cached files), filters to Dublin, and
    loads into raw.property. Logs a grand-total at the end.
    """
    engine = get_postgres_engine()
    total_inserted = 0
    failed_years: list[int] = []

    for year in YEARS:
        path = download_year(year)
        if path is None:
            logger.warning("Skipping year %d — download unavailable", year)
            failed_years.append(year)
            continue

        count = load_year(path, year, engine)
        total_inserted += count

    logger.info(
        "Ingest complete. Total rows inserted: %d. Failed years: %s",
        total_inserted,
        failed_years if failed_years else "none",
    )

    # Verify final row count in the table.
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM raw.property"))
        db_count = result.scalar()
    logger.info("raw.property now contains %d total rows", db_count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run()
