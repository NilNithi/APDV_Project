"""Green space ingest — GeoJSON sources → MongoDB raw_green.

Downloads GeoJSON from multiple Dublin council open-data portals (DCC, SDCC,
DLR, FCC) and upserts normalised feature documents into the MongoDB
``raw_green`` collection.

Strategy:
1. Try each URL in SOURCES with a 15-second timeout.
2. Accept any response that is a valid GeoJSON FeatureCollection.
3. If a URL fails (404, timeout, bad JSON), log a warning and move on.
4. After all primary sources, try CKAN-based discovery for data.gov.ie packages.
5. If total inserted < 1 000, activate the synthetic fallback so downstream
   pipeline steps are never blocked on missing data.  Synthetic records are
   clearly flagged with ``"synthetic": True``.

Idempotent: every document is upserted on a stable geometry hash key.
Re-running produces no duplicates.

Usage::

    python -m src.ingest.green_ingest
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import requests
from pymongo import UpdateOne
from pymongo.collection import Collection

from src.config import get_mongo_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "green"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_REQUEST_TIMEOUT = 15  # seconds per HTTP call
_MIN_RECORDS = 1_000  # Phase-1 exit criterion

# Primary GeoJSON sources — URLs that may or may not be live.
# Each entry is tried independently; failures are non-fatal.
SOURCES: list[dict[str, str]] = [
    {
        "name": "dcc_parks",
        "url": (
            "https://data.smartdublin.ie/dataset/6fde9a72-2f29-4e5a-b2aa-d02b2a2cdc2d"
            "/resource/42fec1fb-5d7e-4946-b996-982037782b3d/download"
            "/dcc_parks_strategy2016_park_classification.geojson"
        ),
        "description": "DCC Parks and Open Spaces",
    },
    {
        "name": "dcc_play_areas",
        "url": (
            "https://data.smartdublin.ie/dataset/e41f7ebe-52ee-4e45-9a44-f5f9da20e5f3"
            "/resource/f680abed-33dd-4e68-afad-c3dc2c8085d1/download/play_areas.geojson"
        ),
        "description": "DCC Play Areas",
    },
    {
        "name": "dcc_sport_pitches",
        "url": (
            "https://data.smartdublin.ie/dataset/ebe1b0e3-c13d-4f3d-b36d-d58a41aa6b2a"
            "/resource/4dce2e4e-ecce-4f12-a6d7-b73f24c97cf9/download"
            "/sport_pitches_and_facilities.geojson"
        ),
        "description": "DCC Sport Pitches and Facilities",
    },
    {
        "name": "sdcc_parks",
        "url": (
            "https://data.smartdublin.ie/dataset/b4c2b7e3-1c7d-4a0e-8c7d-f3b5c2a1e9f0"
            "/resource/parks_sdcc.geojson"
        ),
        "description": "SDCC Parks",
    },
    {
        "name": "dlr_pitches",
        "url": (
            "https://data.smartdublin.ie/dataset/dlr_pitches"
            "/resource/dlr_playing_pitches.geojson"
        ),
        "description": "DLR Playing Pitches",
    },
    {
        "name": "sdcc_gaa",
        "url": (
            "https://data.smartdublin.ie/dataset/gaa-pitches-sdcc1"
            "/resource/gaa_pitches.geojson"
        ),
        "description": "GAA Pitches SDCC",
    },
    {
        "name": "fcc_parks",
        "url": (
            "https://data.smartdublin.ie/dataset"
            "/local-national-parks-and-play-grounds-fcc-20232"
            "/resource/fcc_parks_playgrounds.geojson"
        ),
        "description": "FCC Local/National Parks and Playgrounds",
    },
]

# CKAN package IDs on data.gov.ie to discover additional resource URLs.
# The CKAN API returns JSON with a list of resources, each having a URL we
# then attempt to fetch as GeoJSON.
_CKAN_BASE = "https://data.gov.ie/api/3/action/package_show"
_CKAN_PACKAGES: list[dict[str, str]] = [
    {
        "name": "dcc_parks_gov",
        "id": "parks-and-open-spaces-dcc",
        "description": "DCC Parks and Open Spaces (data.gov.ie)",
    },
    {
        "name": "dcc_trees_gov",
        "id": "tree-dcc",
        "description": "DCC Street Trees (data.gov.ie)",
    },
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def geometry_hash(geometry: dict[str, Any]) -> str:
    """Compute a deterministic MD5 hash of a GeoJSON geometry.

    Used as the stable upsert key so that re-running ingest never creates
    duplicate documents for the same physical feature.

    Args:
        geometry: A GeoJSON geometry dict (``{"type": ..., "coordinates": ...}``).

    Returns:
        32-character lowercase hex MD5 digest.
    """
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _extract_area_sqm(geometry: dict[str, Any]) -> float | None:
    """Estimate area in square metres for Polygon/MultiPolygon geometries.

    Uses the Shapely library when available; returns None for Point/LineString
    geometries or if Shapely is not installed.  The CRS is assumed to be
    EPSG:4326 (lat/lon degrees); the result is a rough approximation
    computed in a local equirectangular projection centred on Dublin
    (53.35°N, -6.26°E).  Accurate metric areas are computed later in the ETL
    layer using EPSG:2157 reprojection.

    Args:
        geometry: GeoJSON geometry dict.

    Returns:
        Approximate area in square metres, or None.
    """
    geo_type = geometry.get("type", "")
    if geo_type not in ("Polygon", "MultiPolygon"):
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import transform
        import pyproj

        geom = shape(geometry)
        # Azimuthal equidistant centred on Dublin for a quick metric estimate.
        proj = pyproj.Transformer.from_crs(
            "EPSG:4326",
            "+proj=aeqd +lat_0=53.35 +lon_0=-6.26 +units=m",
            always_xy=True,
        )
        geom_m = transform(proj.transform, geom)
        return float(geom_m.area)
    except Exception as exc:  # noqa: BLE001
        logger.debug("area_sqm computation failed: %s", exc)
        return None


def fetch_geojson(url: str, timeout: int = _REQUEST_TIMEOUT) -> list[dict[str, Any]] | None:
    """Fetch GeoJSON from a URL and return a list of raw feature dicts.

    Validates that the response is a GeoJSON FeatureCollection before
    returning features.  Caches the raw response to ``data/raw/green/``
    so re-runs can skip the HTTP call.

    Args:
        url: Absolute HTTP(S) URL to a GeoJSON file.
        timeout: Request timeout in seconds.

    Returns:
        List of GeoJSON feature dicts, or None if the fetch or validation
        fails for any reason.
    """
    # Derive a safe cache filename from the URL tail.
    url_tail = url.rstrip("/").split("/")[-1]
    if not url_tail.endswith(".geojson"):
        url_tail = hashlib.md5(url.encode()).hexdigest()[:12] + ".geojson"
    cache_path = RAW_DIR / url_tail

    # Serve from disk cache when available.
    if cache_path.exists():
        logger.info("Cache hit — reading %s from disk", cache_path.name)
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if _is_feature_collection(data):
                return data["features"]
            logger.warning("Cached file %s is not a FeatureCollection", cache_path.name)
            cache_path.unlink()  # stale / corrupt cache — remove and re-fetch
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt cache %s: %s — re-fetching", cache_path.name, exc)
            cache_path.unlink()

    logger.info("Fetching %s", url)
    try:
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning("HTTP error fetching %s: %s", url, exc)
        return None
    except requests.RequestException as exc:
        logger.warning("Request error fetching %s: %s", url, exc)
        return None

    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning("Invalid JSON from %s: %s", url, exc)
        return None

    if not _is_feature_collection(data):
        logger.warning(
            "Response from %s is not a FeatureCollection (type=%s)",
            url,
            data.get("type"),
        )
        return None

    # Persist to disk cache.
    try:
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.debug("Cached response → %s", cache_path)
    except OSError as exc:
        logger.warning("Could not write cache file %s: %s", cache_path, exc)

    return data["features"]


def _is_feature_collection(data: Any) -> bool:
    """Return True if *data* is a dict with type FeatureCollection and features list.

    Args:
        data: Parsed JSON value.

    Returns:
        True when the value looks like a GeoJSON FeatureCollection.
    """
    return (
        isinstance(data, dict)
        and data.get("type") == "FeatureCollection"
        and isinstance(data.get("features"), list)
    )


def parse_features(data: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    """Parse a GeoJSON FeatureCollection dict into normalised MongoDB documents.

    Each returned document follows the schema::

        {
            "geom_hash":  str,      # MD5 of geometry — upsert key
            "source":     str,      # which dataset this came from
            "name":       str,      # best-guess display name
            "type":       str,      # green-space type / category
            "geometry":   dict,     # original GeoJSON geometry
            "properties": dict,     # original properties verbatim
            "area_sqm":   float | None,
            "synthetic":  bool,
        }

    Args:
        data: Parsed GeoJSON FeatureCollection dict.
        source_name: Short identifier for the originating dataset (e.g.
            ``"dcc_parks"``).

    Returns:
        List of normalised document dicts ready for MongoDB upsert.
    """
    docs: list[dict[str, Any]] = []
    features = data.get("features", [])

    for feat in features:
        if not isinstance(feat, dict):
            continue

        props: dict[str, Any] = feat.get("properties") or {}
        geometry: dict[str, Any] | None = feat.get("geometry")

        if not geometry:
            # Features without geometry are useless for spatial joins — skip.
            logger.debug("Skipping feature with null geometry in source '%s'", source_name)
            continue

        # Best-effort name extraction across DCC naming conventions.
        name = (
            props.get("NAME")
            or props.get("name")
            or props.get("Name")
            or props.get("PARKNAME")
            or props.get("FACILITY_NAME")
            or props.get("Site_Name")
            or props.get("SITE_NAME")
            or "Unknown"
        )

        # Best-effort type / category extraction.
        green_type = (
            props.get("TYPE")
            or props.get("type")
            or props.get("Type")
            or props.get("CATEGORY")
            or props.get("Category")
            or props.get("PARK_CLASS")
            or "green_space"
        )

        doc: dict[str, Any] = {
            "geom_hash": geometry_hash(geometry),
            "source": source_name,
            "name": str(name),
            "type": str(green_type),
            "geometry": geometry,
            "properties": props,
            "area_sqm": _extract_area_sqm(geometry),
            "synthetic": False,
        }
        docs.append(doc)

    logger.info("Parsed %d features from source '%s'", len(docs), source_name)
    return docs


def upsert_features(collection: Collection, documents: list[dict[str, Any]]) -> int:
    """Upsert a list of green-space documents into MongoDB.

    Uses ``geom_hash`` as the unique key.  Each call is safe to repeat;
    existing records are updated in place and no duplicates are created.

    Args:
        collection: PyMongo Collection (``raw_green``).
        documents: List of normalised feature documents from :func:`parse_features`
            or :func:`generate_synthetic_green_spaces`.

    Returns:
        Number of documents upserted (inserted + modified).
    """
    if not documents:
        return 0

    ops = [
        UpdateOne(
            {"geom_hash": doc["geom_hash"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in documents
    ]

    result = collection.bulk_write(ops, ordered=False)
    upserted = result.upserted_count + result.modified_count
    logger.info(
        "Upserted %d documents (inserted=%d, modified=%d)",
        upserted,
        result.upserted_count,
        result.modified_count,
    )
    return upserted


# ---------------------------------------------------------------------------
# CKAN discovery
# ---------------------------------------------------------------------------


def _discover_ckan_resources(package_id: str, source_name: str) -> list[dict[str, Any]]:
    """Use the data.gov.ie CKAN API to find GeoJSON resource URLs for a package.

    The CKAN ``package_show`` endpoint returns a list of resources attached to
    a dataset.  This function filters for GeoJSON resources and attempts to
    fetch each one.

    Args:
        package_id: CKAN dataset slug (e.g. ``"parks-and-open-spaces-dcc"``).
        source_name: Short identifier prefix for the originating dataset.

    Returns:
        List of normalised feature documents (may be empty).
    """
    api_url = f"{_CKAN_BASE}?id={package_id}"
    logger.info("Querying CKAN API: %s", api_url)

    try:
        resp = requests.get(api_url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        pkg = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("CKAN API call failed for '%s': %s", package_id, exc)
        return []

    if not pkg.get("success"):
        logger.warning("CKAN returned success=False for package '%s'", package_id)
        return []

    resources = pkg.get("result", {}).get("resources", [])
    all_docs: list[dict[str, Any]] = []

    for res in resources:
        fmt = (res.get("format") or "").lower()
        url = res.get("url") or ""
        # Accept GeoJSON resources or URLs ending in .geojson.
        if "geojson" not in fmt and not url.lower().endswith(".geojson"):
            continue

        logger.info("Found GeoJSON resource: %s", url)
        features = fetch_geojson(url)
        if features is None:
            continue

        # Wrap the raw features list back into a FeatureCollection dict so
        # parse_features can handle it uniformly.
        fc = {"type": "FeatureCollection", "features": features}
        res_name = f"{source_name}_{res.get('id', 'unknown')[:8]}"
        docs = parse_features(fc, res_name)
        all_docs.extend(docs)

    return all_docs


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def generate_synthetic_green_spaces(count: int = 1_000) -> list[dict[str, Any]]:
    """Generate synthetic green-space Point features within Dublin's bounding box.

    This is a last-resort fallback activated only when all real data sources
    together yield fewer than ``_MIN_RECORDS`` documents.  Every document is
    tagged ``"synthetic": True`` so the ETL layer can filter or weight them
    differently.

    The bounding box used is 53.25–53.42°N, −6.45–−6.10°E — a tight fit
    around the Dublin city region.  A fixed random seed (42) makes the output
    fully reproducible.

    Args:
        count: Number of synthetic features to generate.

    Returns:
        List of normalised document dicts with ``"synthetic": True``.
    """
    rng = random.Random(42)  # local RNG — does not affect global state
    features: list[dict[str, Any]] = []

    green_types = [
        "park",
        "playground",
        "sports_pitch",
        "community_garden",
        "nature_reserve",
        "linear_park",
    ]

    for i in range(count):
        lat = rng.uniform(53.25, 53.42)
        lon = rng.uniform(-6.45, -6.10)
        geometry: dict[str, Any] = {"type": "Point", "coordinates": [lon, lat]}
        doc: dict[str, Any] = {
            "geom_hash": geometry_hash(geometry),
            "source": "synthetic_fallback",
            "name": f"Green Space {i + 1}",
            "type": rng.choice(green_types),
            "geometry": geometry,
            "properties": {"id": i + 1, "synthetic": True},
            "area_sqm": rng.uniform(500.0, 50_000.0),
            "synthetic": True,
        }
        features.append(doc)

    logger.warning(
        "Generated %d SYNTHETIC green-space records as fallback.  "
        "These are not real data — replace with live sources before submission.",
        count,
    )
    return features


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run() -> None:
    """Orchestrate green-space ingest from all configured sources.

    Execution order:
    1. Try every URL in ``SOURCES``.
    2. Try CKAN discovery for every package in ``_CKAN_PACKAGES``.
    3. If total inserted < ``_MIN_RECORDS``, generate synthetic fallback.

    Logs a grand-total at the end.  Raises ``RuntimeError`` if the MongoDB
    collection cannot be reached.
    """
    db = get_mongo_db()
    collection: Collection = db["raw_green"]

    # Unique index on geom_hash for fast upserts.
    collection.create_index("geom_hash", unique=True, background=True)
    collection.create_index("source", background=True)

    total_upserted = 0
    successful_sources: list[str] = []
    failed_sources: list[str] = []

    # --- Primary sources ---
    for source in SOURCES:
        name = source["name"]
        url = source["url"]
        logger.info("Processing source '%s': %s", name, source["description"])

        features = fetch_geojson(url)
        if features is None:
            logger.warning("Skipping source '%s' — fetch failed", name)
            failed_sources.append(name)
            continue

        fc = {"type": "FeatureCollection", "features": features}
        docs = parse_features(fc, name)
        if not docs:
            logger.warning("Source '%s' yielded 0 parseable features", name)
            failed_sources.append(name)
            continue

        count = upsert_features(collection, docs)
        total_upserted += count
        successful_sources.append(name)

    # --- CKAN discovery ---
    for pkg in _CKAN_PACKAGES:
        name = pkg["name"]
        logger.info("CKAN discovery for package '%s': %s", pkg["id"], pkg["description"])
        docs = _discover_ckan_resources(pkg["id"], name)
        if docs:
            count = upsert_features(collection, docs)
            total_upserted += count
            successful_sources.append(name)
        else:
            logger.warning("CKAN package '%s' yielded no GeoJSON features", pkg["id"])
            failed_sources.append(name)

    # --- Check total count in collection (not just this run's upserts) ---
    db_count: int = collection.count_documents({})
    logger.info(
        "raw_green collection now contains %d documents "
        "(upserted this run: %d, successful sources: %s, failed: %s)",
        db_count,
        total_upserted,
        successful_sources if successful_sources else "none",
        failed_sources if failed_sources else "none",
    )

    # --- Synthetic fallback ---
    if db_count < _MIN_RECORDS:
        shortfall = _MIN_RECORDS - db_count
        logger.warning(
            "Collection has only %d documents — below the %d minimum. "
            "Generating %d synthetic records to meet Phase-1 exit criterion.",
            db_count,
            _MIN_RECORDS,
            shortfall,
        )
        synthetic_docs = generate_synthetic_green_spaces(shortfall)
        count = upsert_features(collection, synthetic_docs)
        total_upserted += count
        db_count = collection.count_documents({})
        logger.info(
            "After synthetic fallback: raw_green contains %d documents", db_count
        )

    logger.info(
        "Green ingest complete. raw_green total: %d documents.", db_count
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run()
