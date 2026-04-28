"""Air quality ingest — Sonitus API → MongoDB raw_air.

Fetches monitoring station metadata and historical pollutant readings from the
Dublin City Council Sonitus API, caches raw responses to disk, and upserts
normalised documents into the MongoDB ``raw_air`` collection.

Key API details (from OpenAPI spec at data.smartdublin.ie/sonitus-openapi.json):
- All endpoints are POST requests with credentials as QUERY PARAMS (not body/Basic Auth).
- /api/monitors  — params: username, password
- /api/data      — params: username, password, monitor (serial_number), start (Unix ts), end (Unix ts)
- Response rows are WIDE format: one row per timestamp, one column per pollutant.
  We melt these into long-format documents (one per pollutant per timestamp).

Idempotent: re-running produces no duplicates (upsert on monitor_id + timestamp + pollutant).
Cached JSON files on disk prevent redundant HTTP calls.

Usage::

    python -m src.ingest.air_ingest

"""

from __future__ import annotations

import calendar
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import urllib3
from pymongo import UpdateOne
from pymongo.collection import Collection
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_mongo_db

# Suppress SSL warnings — corporate network may have self-signed CA
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://data.smartdublin.ie/sonitus-api"

INGEST_START_YEAR = 2015
INGEST_START_MONTH = 1
INGEST_END_YEAR = 2024
INGEST_END_MONTH = 12

SLEEP_BETWEEN_CALLS: float = 0.5  # seconds — respect rate limits

COLLECTION_NAME = "raw_air"

# Pollutant columns in AQ monitor responses.
_AQ_POLLUTANTS = ("no2", "so2", "pm10", "pm25", "o3", "co")

# Pollutant columns in noise monitor responses.
_NOISE_POLLUTANTS = ("laeq", "lafmax", "la10", "la90", "lceq", "lcfmax", "lceq_1s")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "raw" / "air"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _build_creds() -> dict[str, str]:
    """Return credentials dict for query-param auth.

    Returns:
        Dict with ``username`` and ``password`` keys sourced from env vars.
    """
    return {
        "username": os.getenv("SONITUS_USER", "dublincityapi"),
        "password": os.getenv("SONITUS_PASS", "Xpa5vAQ9ki"),
    }


# ---------------------------------------------------------------------------
# Retry-decorated HTTP call — credentials as QUERY PARAMS
# ---------------------------------------------------------------------------


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _api_post(endpoint: str, params: dict[str, Any]) -> Any:
    """POST to a Sonitus API endpoint with credentials as query params.

    Args:
        endpoint: Path under BASE_URL, e.g. ``/api/monitors``.
        params: Query params dict — must already include ``username`` and
            ``password``.

    Returns:
        Parsed JSON response (list or dict).

    Raises:
        requests.RequestException: After all retry attempts are exhausted.
    """
    url = f"{BASE_URL}{endpoint}"
    logger.debug("POST %s  params_keys=%s", url, list(params.keys()))
    resp = requests.post(url, params=params, verify=False, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Fetch monitors
# ---------------------------------------------------------------------------


def fetch_monitors() -> list[dict[str, Any]]:
    """Fetch all monitoring stations from ``POST /api/monitors``.

    Credentials are passed as query params per the Sonitus OpenAPI spec.

    Returns:
        List of station dicts.  Each has at least ``serial_number``, ``label``,
        ``latitude``, ``longitude``.  Empty list on failure.
    """
    creds = _build_creds()
    try:
        data = _api_post("/api/monitors", creds)
    except requests.RequestException as exc:
        logger.error("Failed to fetch monitors: %s", exc)
        return []

    if not isinstance(data, list):
        logger.error(
            "Unexpected /api/monitors response type %s — raw: %.500s",
            type(data).__name__,
            str(data),
        )
        return []

    if not data:
        logger.warning("/api/monitors returned an empty list.")
        return []

    first = data[0]
    logger.info(
        "Monitor sample keys: %s  (total stations: %d)",
        list(first.keys()) if isinstance(first, dict) else type(first).__name__,
        len(data),
    )
    return data


# ---------------------------------------------------------------------------
# Fetch one station / one month — with disk cache
# ---------------------------------------------------------------------------


_MAX_WINDOW_SECONDS = 7 * 24 * 3600  # API rejects windows > 7 days


def _fetch_window(serial: str, start_ts: int, end_ts: int) -> list[Any]:
    """Fetch one ≤7-day window from /api/data.

    Args:
        serial: Station serial_number.
        start_ts: Window start as Unix timestamp (inclusive).
        end_ts: Window end as Unix timestamp (exclusive).

    Returns:
        List of raw row dicts, or empty list on failure.
    """
    creds = _build_creds()
    params: dict[str, Any] = {
        **creds,
        "monitor": serial,
        "start": start_ts,
        "end": end_ts,
    }
    try:
        data = _api_post("/api/data", params)
        time.sleep(SLEEP_BETWEEN_CALLS)
    except requests.RequestException as exc:
        logger.warning(
            "Fetch failed for station=%s start=%d end=%d: %s — skipping.",
            serial,
            start_ts,
            end_ts,
            exc,
        )
        return []

    # API returns {"error": "..."} dict on logical errors even with HTTP 200.
    if isinstance(data, dict) and "error" in data:
        logger.warning("API error for station=%s: %s", serial, data["error"])
        return []

    return data if isinstance(data, list) else []


def fetch_station_month(
    serial: str,
    year: int,
    month: int,
    cache_dir: Path,
) -> list[Any]:
    """Fetch one month of readings for one station, using a disk cache.

    The Sonitus API rejects windows longer than 7 days, so this function
    splits the month into ≤7-day chunks and combines the results.  The
    combined month data is cached as a single JSON file for idempotency.

    Args:
        serial: Station serial_number from the monitors list.
        year: Four-digit year.
        month: Calendar month (1–12).
        cache_dir: Directory where cached JSON files are stored.

    Returns:
        List of raw row dicts (wide format — one row per timestamp, pollutants
        as columns).  May be empty if station had no data for this period.
    """
    safe_serial = str(serial).replace("/", "_").replace("\\", "_")
    cache_file = cache_dir / f"{safe_serial}_{year}_{month:02d}.json"

    if cache_file.exists():
        logger.debug("Cache hit: %s", cache_file.name)
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache %s (%s) — re-fetching.", cache_file, exc)

    utc = timezone.utc
    month_start = int(datetime(year, month, 1, tzinfo=utc).timestamp())
    if month == 12:
        month_end = int(datetime(year + 1, 1, 1, tzinfo=utc).timestamp())
    else:
        month_end = int(datetime(year, month + 1, 1, tzinfo=utc).timestamp())

    # Chunk month into ≤7-day windows and collect all rows.
    all_rows: list[Any] = []
    cursor = month_start
    while cursor < month_end:
        window_end = min(cursor + _MAX_WINDOW_SECONDS, month_end)
        rows = _fetch_window(serial, cursor, window_end)
        all_rows.extend(rows)
        cursor = window_end

    # Cache combined result (even if empty) to avoid re-fetching on re-runs.
    try:
        cache_file.write_text(json.dumps(all_rows, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write cache %s: %s", cache_file, exc)

    return all_rows


# ---------------------------------------------------------------------------
# Parse wide-format rows → long-format documents
# ---------------------------------------------------------------------------


def parse_readings(
    raw: list[Any],
    monitor: dict[str, Any],
    year: int,
    month: int,
) -> list[dict[str, Any]]:
    """Convert wide-format API rows into normalised long-format documents.

    Each API row has one timestamp and multiple pollutant columns.  This
    function melts those into one document per (timestamp, pollutant) pair so
    the MongoDB schema is uniform and easily queryable.

    Args:
        raw: Raw rows from the API or cache (wide format).
        monitor: Station dict from ``/api/monitors`` for this station.
        year: Calendar year (metadata / fallback).
        month: Calendar month (metadata / fallback).

    Returns:
        List of long-format dicts ready for upsert into ``raw_air``.
    """
    if not raw:
        return []

    # Extract station-level fields — Sonitus monitors endpoint returns
    # serial_number, label, latitude, longitude fields.
    serial: str = str(
        monitor.get("serial_number")
        or monitor.get("monitor_id")
        or monitor.get("monitorId")
        or monitor.get("id")
        or "unknown"
    )
    station_name: str = str(
        monitor.get("label")
        or monitor.get("name")
        or monitor.get("station_name")
        or serial
    )
    lat: float | None = _try_float(
        monitor.get("latitude") or monitor.get("lat")
    )
    lon: float | None = _try_float(
        monitor.get("longitude") or monitor.get("lon") or monitor.get("lng")
    )

    # Log sample row structure once per call.
    if raw:
        first = raw[0]
        logger.debug(
            "Reading sample keys for station=%s: %s",
            serial,
            list(first.keys()) if isinstance(first, dict) else type(first).__name__,
        )

    # Determine which pollutant columns are present in this batch.
    all_pollutant_cols = _AQ_POLLUTANTS + _NOISE_POLLUTANTS
    present_pollutants: list[str] = []
    if raw and isinstance(raw[0], dict):
        first_row = raw[0]
        present_pollutants = [
            col for col in all_pollutant_cols if col in first_row
        ]
    if not present_pollutants:
        logger.debug("No recognised pollutant columns in station=%s response.", serial)

    documents: list[dict[str, Any]] = []

    for row in raw:
        if not isinstance(row, dict):
            continue

        # Parse timestamp — Sonitus /api/data typically returns "datetime" field.
        ts_raw = row.get("datetime") or row.get("timestamp") or row.get("date")
        timestamp = _coerce_timestamp(ts_raw)
        if timestamp is None:
            timestamp = f"{year}-{month:02d}-01T00:00:00+00:00"

        # Melt: one document per pollutant column.
        for col in present_pollutants:
            value = _try_float(row.get(col))
            if value is None:
                continue
            if value < 0:
                # Negative sensor readings are instrument errors.
                continue

            documents.append(
                {
                    "monitor_id": serial,
                    "station_name": station_name,
                    "lat": lat,
                    "lon": lon,
                    "pollutant": col,
                    "value": value,
                    "timestamp": timestamp,
                    "year": year,
                    "month": month,
                    "source": "sonitus_api",
                }
            )

    return documents


# ---------------------------------------------------------------------------
# MongoDB bulk upsert
# ---------------------------------------------------------------------------


def _bulk_upsert(collection: Collection, documents: list[dict[str, Any]]) -> int:
    """Bulk-upsert documents using monitor_id + timestamp + pollutant as key.

    Args:
        collection: Target PyMongo collection.
        documents: Documents to upsert.

    Returns:
        Count of inserted + modified documents.
    """
    if not documents:
        return 0

    ops = [
        UpdateOne(
            {
                "monitor_id": d["monitor_id"],
                "timestamp": d["timestamp"],
                "pollutant": d["pollutant"],
            },
            {"$set": d},
            upsert=True,
        )
        for d in documents
    ]
    result = collection.bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def _insert_synthetic_fallback(collection: Collection) -> None:
    """Insert clearly-labelled synthetic documents so downstream ETL can run.

    Called only when the live API is unreachable.  Documents have
    ``source: synthetic_fallback`` so they can be excluded from analysis.

    Args:
        collection: The ``raw_air`` MongoDB collection.
    """
    logger.warning(
        "API unavailable — inserting synthetic fallback documents so the "
        "downstream pipeline remains testable."
    )
    stations = [
        ("SYN_001", "Winetavern St (synthetic)", 53.3454, -6.2752),
        ("SYN_002", "Rathmines (synthetic)", 53.3239, -6.2630),
        ("SYN_003", "Finglas (synthetic)", 53.3876, -6.3028),
        ("SYN_004", "Dun Laoghaire (synthetic)", 53.2943, -6.1358),
        ("SYN_005", "Tallaght (synthetic)", 53.2884, -6.3744),
    ]
    pollutants = ["no2", "pm25", "pm10", "laeq"]
    docs: list[dict[str, Any]] = []
    base_values = {"no2": 25.0, "pm25": 12.0, "pm10": 20.0, "laeq": 55.0}

    for year in range(2015, 2022):
        for mid, name, lat, lon in stations:
            for pol in pollutants:
                docs.append(
                    {
                        "monitor_id": mid,
                        "station_name": name,
                        "lat": lat,
                        "lon": lon,
                        "pollutant": pol,
                        "value": base_values[pol],
                        "timestamp": f"{year}-06-01T00:00:00+00:00",
                        "year": year,
                        "month": 6,
                        "source": "synthetic_fallback",
                    }
                )

    _bulk_upsert(collection, docs)
    logger.info("Synthetic fallback documents inserted: %d", len(docs))


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _try_float(value: Any) -> float | None:
    """Coerce value to float; return None on failure.

    Args:
        value: Arbitrary input.

    Returns:
        Float or None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_timestamp(value: Any) -> str | None:
    """Normalise a timestamp value to an ISO-8601 UTC string.

    Args:
        value: Raw timestamp (string, int, float, or None).

    Returns:
        ISO-8601 string or None.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ):
            try:
                dt = datetime.strptime(stripped, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue
        logger.debug("Could not parse timestamp %r — storing raw.", stripped)
        return stripped

    return None


def _months(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> list[tuple[int, int]]:
    """Return ordered list of (year, month) tuples in range.

    Args:
        start_year: First year (inclusive).
        start_month: First month (1–12, inclusive).
        end_year: Last year (inclusive).
        end_month: Last month (1–12, inclusive).

    Returns:
        List of (year, month) tuples.
    """
    result: list[tuple[int, int]] = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        result.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run() -> None:
    """Orchestrate full air-quality ingest from Sonitus API into MongoDB.

    Steps:
        1. Fetch station list from ``/api/monitors``.
        2. For each station × month, fetch readings (disk-cached).
        3. Parse wide-format rows into long-format documents.
        4. Bulk-upsert into ``raw_air``.
        5. Log final count; fall back to synthetic data if API returned nothing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    db = get_mongo_db()
    collection: Collection = db[COLLECTION_NAME]

    collection.create_index(
        [("monitor_id", 1), ("timestamp", 1), ("pollutant", 1)],
        unique=True,
        background=True,
    )

    logger.info("Starting Sonitus air-quality ingest  base_url=%s", BASE_URL)
    logger.info(
        "Date range: %d-%02d → %d-%02d",
        INGEST_START_YEAR,
        INGEST_START_MONTH,
        INGEST_END_YEAR,
        INGEST_END_MONTH,
    )

    # Step 1: station list
    monitors = fetch_monitors()
    if not monitors:
        logger.error("No stations returned — falling back to synthetic data.")
        _insert_synthetic_fallback(collection)
        logger.info("raw_air total: %d", collection.count_documents({}))
        return

    logger.info("Stations to process: %d", len(monitors))

    # Step 2: fetch readings
    month_range = _months(
        INGEST_START_YEAR, INGEST_START_MONTH, INGEST_END_YEAR, INGEST_END_MONTH
    )
    total_upserted = 0
    total_parsed = 0

    for station_idx, monitor in enumerate(monitors, start=1):
        serial = str(
            monitor.get("serial_number")
            or monitor.get("monitor_id")
            or monitor.get("id")
            or f"station_{station_idx}"
        )
        station_name = str(
            monitor.get("label") or monitor.get("name") or serial
        )
        logger.info(
            "Station %d/%d  serial=%s  name=%s",
            station_idx,
            len(monitors),
            serial,
            station_name,
        )

        for month_idx, (year, month) in enumerate(month_range, start=1):
            raw_readings = fetch_station_month(serial, year, month, cache_dir)
            if not raw_readings:
                continue

            docs = parse_readings(raw_readings, monitor, year, month)
            total_parsed += len(docs)

            if docs:
                upserted = _bulk_upsert(collection, docs)
                total_upserted += upserted

            if month_idx % 12 == 0:
                logger.info(
                    "  %s — %d/%d months done  total parsed=%d",
                    serial,
                    month_idx,
                    len(month_range),
                    total_parsed,
                )

    # Step 3: fallback if nothing parsed
    if total_parsed == 0:
        logger.warning("Zero readings parsed — API may be down or schema changed.")
        _insert_synthetic_fallback(collection)

    # Step 4: final count
    total_in_db = collection.count_documents({})
    logger.info(
        "Ingest complete.  Parsed: %d  Upserted/modified: %d  raw_air total: %d",
        total_parsed,
        total_upserted,
        total_in_db,
    )

    if total_in_db < 1000:
        logger.warning(
            "raw_air has only %d docs (target: >=1,000). "
            "Expand date range or add more stations.",
            total_in_db,
        )
    else:
        logger.info("raw_air meets minimum requirement (>=1,000).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
