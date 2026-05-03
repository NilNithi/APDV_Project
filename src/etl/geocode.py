"""Geocode property addresses using a three-tier strategy.

Tier 1 — SQLite cache   : instant; avoids any external call for seen addresses.
Tier 2 — Nominatim      : free OSM geocoder; rate-limited to 1 request/second.
Tier 3 — Postcode centroid : hard-coded lat/lon for Dublin area codes; used
                             when Nominatim returns no result.

The cache is stored at data/interim/geocode_cache.db so it persists across
pipeline runs.  Every Nominatim result (success or postcode fallback) is
written back to the cache so the same address is never looked up twice.
"""

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CACHE_DB = PROJECT_ROOT / "data" / "interim" / "geocode_cache.db"

# Approximate centroids for each Dublin postal district (WGS-84, EPSG:4326).
# Used as a last-resort fallback when Nominatim returns nothing.
DUBLIN_CENTROIDS: dict[str, tuple[float, float]] = {
    "Dublin 1": (53.3498, -6.2603),
    "Dublin 2": (53.3382, -6.2591),
    "Dublin 3": (53.3647, -6.2305),
    "Dublin 4": (53.3267, -6.2297),
    "Dublin 5": (53.3826, -6.2072),
    "Dublin 6": (53.3188, -6.2788),
    "Dublin 6W": (53.3139, -6.3091),
    "Dublin 7": (53.3580, -6.2920),
    "Dublin 8": (53.3321, -6.2838),
    "Dublin 9": (53.3831, -6.2503),
    "Dublin 10": (53.3404, -6.3503),
    "Dublin 11": (53.3847, -6.3040),
    "Dublin 12": (53.3233, -6.3188),
    "Dublin 13": (53.3938, -6.1758),
    "Dublin 14": (53.2988, -6.2668),
    "Dublin 15": (53.3939, -6.3780),
    "Dublin 16": (53.2855, -6.2697),
    "Dublin 17": (53.3942, -6.2267),
    "Dublin 18": (53.2762, -6.1876),
    "Dublin 20": (53.3313, -6.3780),
    "Dublin 22": (53.3237, -6.4058),
    "Dublin 24": (53.2987, -6.3855),
}

# Nominatim usage policy: no more than 1 request per second.
_RATE_LIMIT_SECS: float = 1.0
_NOMINATIM_TIMEOUT: int = 10  # seconds per request
_MAX_NOMINATIM_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _init_cache(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite cache, creating the schema if it does not yet exist.

    Args:
        db_path: Filesystem path for the SQLite database file.

    Returns:
        An open sqlite3 Connection with WAL journal mode for concurrent reads.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            address_clean TEXT PRIMARY KEY,
            lat            REAL,
            lon            REAL,
            source         TEXT,
            cached_at      TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    logger.debug("Geocode cache initialised at %s", db_path)
    return conn


def _cache_lookup(
    conn: sqlite3.Connection, address: str
) -> Optional[tuple[float, float, str]]:
    """Return cached coordinates for an address, or None if not cached.

    Args:
        conn: Open SQLite connection to the cache database.
        address: The address_clean string used as the cache key.

    Returns:
        Tuple of (lat, lon, source) if found, else None.
    """
    row = conn.execute(
        "SELECT lat, lon, source FROM cache WHERE address_clean = ?",
        (address,),
    ).fetchone()
    if row is None:
        return None
    lat, lon, source = row
    # A stored NULL (failed lookup) is a valid cache entry — return it so we
    # don't waste another Nominatim call on an address that is un-geocodable.
    return (lat, lon, source)


def _cache_store(
    conn: sqlite3.Connection,
    address: str,
    lat: Optional[float],
    lon: Optional[float],
    source: str,
) -> None:
    """Insert or replace a geocoding result in the cache.

    Args:
        conn: Open SQLite connection to the cache database.
        address: The address_clean string (cache key).
        lat: Latitude, or None for a failed lookup.
        lon: Longitude, or None for a failed lookup.
        source: One of 'nominatim', 'postcode', or 'failed'.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO cache (address_clean, lat, lon, source, cached_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        """,
        (address, lat, lon, source),
    )
    # Caller is responsible for committing in batches.


# ---------------------------------------------------------------------------
# Geocoding tiers
# ---------------------------------------------------------------------------


def _nominatim_geocode(
    geolocator: Nominatim, address: str
) -> Optional[tuple[float, float]]:
    """Query Nominatim for a single address.

    Always sleeps _RATE_LIMIT_SECS *before* the request so callers can
    invoke this in a tight loop without breaching the usage policy.

    Args:
        geolocator: A configured Nominatim instance.
        address: Free-text address string to look up.

    Returns:
        (lat, lon) tuple if Nominatim returned a result, else None.
    """
    time.sleep(_RATE_LIMIT_SECS)

    for attempt in range(1, _MAX_NOMINATIM_RETRIES + 1):
        try:
            location = geolocator.geocode(
                address,
                timeout=_NOMINATIM_TIMEOUT,
                country_codes="ie",  # restrict to Ireland
            )
            if location is not None:
                return (location.latitude, location.longitude)
            return None
        except GeocoderTimedOut:
            logger.warning(
                "Nominatim timed out for address '%s' (attempt %d/%d)",
                address,
                attempt,
                _MAX_NOMINATIM_RETRIES,
            )
            time.sleep(_RATE_LIMIT_SECS * attempt)  # back-off on timeout
        except GeocoderServiceError as exc:
            logger.warning(
                "Nominatim service error for '%s': %s (attempt %d/%d)",
                address,
                exc,
                attempt,
                _MAX_NOMINATIM_RETRIES,
            )
            time.sleep(_RATE_LIMIT_SECS * attempt)

    logger.debug("All Nominatim retries exhausted for '%s'", address)
    return None


def _postcode_centroid(postal_code: str) -> Optional[tuple[float, float]]:
    """Return the hard-coded centroid for a Dublin postal code string.

    Handles both "DUBLIN 2" (from address parsing) and "D02 XY45" (Eircode)
    forms.  For Eircodes only the routing-key prefix is used to map to a
    district (e.g. "D02" -> "Dublin 2").

    Args:
        postal_code: Raw postal code string (may be uppercased).

    Returns:
        (lat, lon) centroid tuple, or None if no mapping exists.
    """
    if not postal_code or not isinstance(postal_code, str):
        return None

    code = postal_code.strip().upper()

    # Normalise "DUBLIN 2" / "DUBLIN 6W" -> title-case key used in the dict.
    if code.startswith("DUBLIN"):
        key = code.title().replace("  ", " ")  # "DUBLIN  2" -> "Dublin 2"
        return DUBLIN_CENTROIDS.get(key)

    # Attempt Eircode routing-key mapping (first 3 chars -> Dublin district).
    # Only a subset of routing keys is covered — expand as needed.
    _eircode_map: dict[str, str] = {
        "D01": "Dublin 1",
        "D02": "Dublin 2",
        "D03": "Dublin 3",
        "D04": "Dublin 4",
        "D05": "Dublin 5",
        "D06": "Dublin 6",
        "D07": "Dublin 7",
        "D08": "Dublin 8",
        "D09": "Dublin 9",
        "D10": "Dublin 10",
        "D11": "Dublin 11",
        "D12": "Dublin 12",
        "D13": "Dublin 13",
        "D14": "Dublin 14",
        "D15": "Dublin 15",
        "D16": "Dublin 16",
        "D17": "Dublin 17",
        "D18": "Dublin 18",
        "D20": "Dublin 20",
        "D22": "Dublin 22",
        "D24": "Dublin 24",
        "D6W": "Dublin 6W",
    }
    routing_key = code[:3]
    district = _eircode_map.get(routing_key)
    if district:
        return DUBLIN_CENTROIDS.get(district)

    return None


def geocode_address(
    address: str,
    postal_code: str,
    geolocator: Nominatim,
    cache_conn: sqlite3.Connection,
) -> tuple[Optional[float], Optional[float], str]:
    """Geocode a single address through the three-tier strategy.

    Tier 1 — SQLite cache lookup (no external call).
    Tier 2 — Nominatim (rate-limited to 1 req/s).
    Tier 3 — Postcode centroid fallback.

    Results from tiers 2 and 3 are written back to the cache so subsequent
    calls for the same address are served instantly from tier 1.

    Args:
        address: Cleaned address string (used as Nominatim query and cache key).
        postal_code: Postal/Eircode string for the centroid fallback.
        geolocator: Configured Nominatim instance.
        cache_conn: Open SQLite connection (caller manages lifecycle).

    Returns:
        Tuple of (lat, lon, source) where source is one of:
        'cache', 'nominatim', 'postcode', 'failed'.
    """
    # Tier 1: cache
    cached = _cache_lookup(cache_conn, address)
    if cached is not None:
        lat, lon, source = cached
        return (lat, lon, "cache")

    # Tier 2: Nominatim — append ", Dublin, Ireland" for better results
    nominatim_query = f"{address}, Dublin, Ireland"
    coords = _nominatim_geocode(geolocator, nominatim_query)
    if coords is not None:
        lat, lon = coords
        _cache_store(cache_conn, address, lat, lon, "nominatim")
        return (lat, lon, "nominatim")

    # Tier 3: postcode centroid
    centroid = _postcode_centroid(postal_code)
    if centroid is not None:
        lat, lon = centroid
        _cache_store(cache_conn, address, lat, lon, "postcode")
        return (lat, lon, "postcode")

    # All tiers failed — cache the failure so we don't retry it.
    _cache_store(cache_conn, address, None, None, "failed")
    logger.debug("Geocoding failed for address: '%s'", address)
    return (None, None, "failed")


# ---------------------------------------------------------------------------
# Batch geocoding
# ---------------------------------------------------------------------------


def geocode_dataframe(df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """Geocode every row in *df*, adding lat, lon, geocode_source columns.

    The DataFrame must contain 'address_clean' and 'postal_code' columns.
    Progress is logged every *batch_size* rows.  The cache is committed after
    every batch so a crash mid-run does not discard earlier work.

    Args:
        df: Cleaned property DataFrame (from property_clean.run()).
        batch_size: Number of rows between cache commits and progress logs.

    Returns:
        The input DataFrame with three new columns appended:
        lat (float64), lon (float64), geocode_source (object/str).

    Raises:
        KeyError: If 'address_clean' or 'postal_code' are missing from df.
    """
    required = {"address_clean", "postal_code"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"geocode_dataframe requires columns {required}; missing: {missing}"
        )

    user_agent = os.getenv("NOMINATIM_USER_AGENT", "green_premium_nci_2026")
    geolocator = Nominatim(user_agent=user_agent)
    cache_conn = _init_cache(CACHE_DB)

    n = len(df)
    lats: list[Optional[float]] = []
    lons: list[Optional[float]] = []
    sources: list[str] = []

    logger.info("Geocoding %d addresses (cache at %s)…", n, CACHE_DB)

    for i, (_, row) in enumerate(df.iterrows()):
        address = row["address_clean"]
        postal_code = row["postal_code"] if pd.notna(row["postal_code"]) else ""

        lat, lon, source = geocode_address(
            address, postal_code, geolocator, cache_conn
        )
        lats.append(lat)
        lons.append(lon)
        sources.append(source)

        # Commit and log progress every batch.
        if (i + 1) % batch_size == 0 or (i + 1) == n:
            cache_conn.commit()
            null_so_far = sum(1 for v in lats if v is None)
            logger.info(
                "Geocoded %d/%d rows | failed so far: %d (%.1f%%)",
                i + 1,
                n,
                null_so_far,
                100.0 * null_so_far / (i + 1),
            )

    cache_conn.close()

    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    df["geocode_source"] = sources

    null_rate = df["lat"].isna().mean() * 100
    logger.info(
        "Geocoding complete — null rate: %.1f%% (%d/%d rows)",
        null_rate,
        df["lat"].isna().sum(),
        n,
    )
    if null_rate > 30:
        logger.warning(
            "Null geocode rate (%.1f%%) exceeds 30%% target — "
            "consider improving Nominatim queries or expanding postcode centroids.",
            null_rate,
        )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run geocoding on the cleaned property DataFrame.

    If *df* is not supplied, reads it from the interim parquet cache written
    by property_clean.run().  Writes the geocoded result to
    data/interim/property_geocoded.parquet.

    Args:
        df: Pre-cleaned property DataFrame, or None to load from cache.

    Returns:
        DataFrame with lat, lon, geocode_source columns added.

    Raises:
        FileNotFoundError: If df is None and the interim parquet does not exist.
    """
    if df is None:
        interim = CACHE_DB.parent / "property_clean.parquet"
        if not interim.exists():
            raise FileNotFoundError(
                f"Interim parquet not found at {interim}. "
                "Run src.etl.property_clean first."
            )
        logger.info("Loading cleaned property data from %s", interim)
        df = pd.read_parquet(interim)

    df = geocode_dataframe(df)

    out = CACHE_DB.parent / "property_geocoded.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Geocoded data cached -> %s", out)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run()
