"""Spatial join: enrich property data with green space proximity and air quality.

Reads geocoded property data from the interim parquet, joins with processed
green spaces (nearest park distance + buffered area) and processed annual air
quality averages (nearest-station inverse-distance weighting), then writes the
enriched result to:

  - PostgreSQL: processed.property_enriched
  - MongoDB: processed_property
  - Parquet: data/processed/property_enriched.parquet

Idempotent: re-running truncates and replaces the Postgres table; MongoDB uses
upsert on a stable (address_clean, date_of_sale) composite key.

Usage::

    python -m src.etl.join
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pymongo import UpdateOne
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.validation import make_valid
from sqlalchemy import text

from src.config import PROJECT_ROOT, get_mongo_db, get_postgres_engine

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Pollutant keyword -> enriched column name mapping.
# The keyword is matched case-insensitively against the 'pollutant' field in
# processed_air_annual.
_POLLUTANT_MAP: dict[str, str] = {
    "NO2": "mean_no2_year",
    "PM2.5": "mean_pm25_year",
    "PM25": "mean_pm25_year",
    "PM10": "mean_pm10_year",
    "NOISE": "mean_noise_db_year",
    "LAEQ": "mean_noise_db_year",
}

# Threshold above which we warn about poor spatial coverage.
_MAX_ACCEPTABLE_NULL_RATE: float = 0.40

# Row count at which per-row buffer iteration switches to the faster
# vectorized overlay approach (avoids O(n) Python loop on large datasets).
_BATCH_BUFFER_THRESHOLD: int = 10_000


# ---------------------------------------------------------------------------
# Step 0 — Load inputs
# ---------------------------------------------------------------------------


def _load_property(interim_dir: Path) -> pd.DataFrame:
    """Load geocoded (or at least cleaned) property DataFrame from parquet.

    Prefers ``property_geocoded.parquet``; falls back to
    ``property_clean.parquet`` if the geocoded file does not exist yet.

    Args:
        interim_dir: Path to the ``data/interim`` directory.

    Returns:
        Property DataFrame with at minimum the columns produced by
        :mod:`src.etl.property_clean`.

    Raises:
        FileNotFoundError: If neither parquet file exists.
    """
    geocoded = interim_dir / "property_geocoded.parquet"
    cleaned = interim_dir / "property_clean.parquet"

    if geocoded.exists():
        logger.info("Loading geocoded property data from %s", geocoded)
        return pd.read_parquet(geocoded)

    if cleaned.exists():
        logger.warning(
            "Geocoded parquet not found — falling back to %s. "
            "Rows without lat/lon will be dropped.",
            cleaned,
        )
        return pd.read_parquet(cleaned)

    raise FileNotFoundError(
        f"Neither {geocoded} nor {cleaned} exists. "
        "Run src.etl.property_clean (and optionally src.etl.geocode) first."
    )


def _load_green_gdf(processed_dir: Path, db) -> gpd.GeoDataFrame:
    """Load processed green spaces as a GeoDataFrame in EPSG:2157.

    Attempts (in order):
    1. ``data/processed/green_spaces.parquet`` (GeoParquet written by green_clean).
    2. MongoDB ``processed_green`` collection (WKT geometry column).
    3. MongoDB ``raw_green`` collection (GeoJSON geometry field) — last resort.

    Args:
        processed_dir: Path to the ``data/processed`` directory.
        db: PyMongo Database object.

    Returns:
        GeoDataFrame in EPSG:2157.  May be empty if no green data is available.
    """
    parquet_path = processed_dir / "green_spaces.parquet"

    # --- try GeoParquet first ---
    if parquet_path.exists():
        try:
            gdf = gpd.read_parquet(parquet_path)
            if gdf.crs is None:
                logger.warning("green_spaces.parquet has no CRS; assuming EPSG:2157.")
                gdf = gdf.set_crs(epsg=2157)
            elif gdf.crs.to_epsg() != 2157:
                gdf = gdf.to_crs(epsg=2157)
            logger.info("Loaded %d green features from %s", len(gdf), parquet_path)
            return gdf
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read green GeoParquet (%s); trying MongoDB.", exc)

    # --- try processed_green in MongoDB (has geometry_wkt_2157 field) ---
    processed_docs = list(
        db.processed_green.find(
            {},
            {"_id": 0, "geometry_wkt_2157": 1, "name": 1, "source": 1, "area_sqm": 1},
        )
    )
    if processed_docs:
        from shapely import wkt as shapely_wkt

        geoms, names, sources, areas = [], [], [], []
        for doc in processed_docs:
            wkt_str = doc.get("geometry_wkt_2157")
            if not wkt_str:
                continue
            try:
                geom = shapely_wkt.loads(wkt_str)
                geoms.append(geom)
                names.append(doc.get("name", "unknown"))
                sources.append(doc.get("source", "unknown"))
                areas.append(doc.get("area_sqm", np.nan))
            except Exception:  # noqa: BLE001
                continue

        if geoms:
            gdf = gpd.GeoDataFrame(
                {"name": names, "source": sources, "area_sqm": areas},
                geometry=geoms,
                crs="EPSG:2157",
            )
            logger.info(
                "Loaded %d green features from MongoDB processed_green.", len(gdf)
            )
            return gdf

    # --- last resort: raw_green with GeoJSON geometry ---
    logger.warning("Falling back to raw_green collection for green space geometries.")
    raw_docs = list(
        db.raw_green.find({}, {"_id": 0, "geometry": 1, "name": 1, "source": 1})
    )
    if not raw_docs:
        logger.error("raw_green is also empty — green space join will be skipped.")
        return gpd.GeoDataFrame()

    geoms, names, sources = [], [], []
    for doc in raw_docs:
        geom_dict = doc.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = make_valid(geom)
            geoms.append(geom)
            names.append(doc.get("name", "unknown"))
            sources.append(doc.get("source", "unknown"))
        except Exception:  # noqa: BLE001
            continue

    if not geoms:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(
        {"name": names, "source": sources},
        geometry=geoms,
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(epsg=2157)
    logger.info("Loaded %d green features from raw_green.", len(gdf))
    return gdf


def _load_air_annual(db) -> pd.DataFrame:
    """Load annual air quality aggregates from MongoDB processed_air_annual.

    Falls back to the Parquet export when the collection is empty.

    Args:
        db: PyMongo Database object.

    Returns:
        DataFrame with columns including ``monitor_id``, ``station_name``,
        ``lat``, ``lon``, ``pollutant``, ``year``, ``mean_value``.
        May be empty.
    """
    docs = list(db.processed_air_annual.find({}, {"_id": 0}))
    if docs:
        df = pd.DataFrame(docs)
        logger.info("Loaded %d processed_air_annual documents.", len(df))
        return df

    # Fallback to Parquet
    parquet_path = PROCESSED_DIR / "air_annual.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(
            "Loaded %d air annual rows from %s (MongoDB fallback).",
            len(df),
            parquet_path,
        )
        return df

    logger.warning("No processed air quality data found — AQ columns will be NaN.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 1 — Build property GeoDataFrame in EPSG:2157
# ---------------------------------------------------------------------------


def _build_property_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert property DataFrame to GeoDataFrame in EPSG:2157 (Irish Transverse Mercator).

    Rows lacking lat or lon are dropped before conversion.

    Args:
        df: Property DataFrame with ``lat`` and ``lon`` columns.

    Returns:
        GeoDataFrame in EPSG:2157 with the same rows (minus those without coords).
    """
    df = df.dropna(subset=["lat", "lon"]).copy()
    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf.to_crs(epsg=2157)


# ---------------------------------------------------------------------------
# Step 2 — Nearest park
# ---------------------------------------------------------------------------


def _nearest_park(
    prop_gdf: gpd.GeoDataFrame, green_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Find the nearest park for each property using geopandas.sjoin_nearest.

    Both GeoDataFrames must already be in EPSG:2157 so that the returned
    distance is in metres.

    Duplicate indices that can arise when multiple parks are equidistant are
    resolved by keeping the first match.

    Args:
        prop_gdf: Property GeoDataFrame in EPSG:2157.
        green_gdf: Green space GeoDataFrame in EPSG:2157.

    Returns:
        DataFrame indexed like *prop_gdf* with columns:
        ``nearest_park_name`` (str) and ``nearest_park_dist_m`` (float).
    """
    null_result = pd.DataFrame(
        {
            "nearest_park_name": pd.Series(dtype="object"),
            "nearest_park_dist_m": pd.Series(dtype="float64"),
        },
        index=prop_gdf.index,
    )

    if green_gdf.empty:
        logger.warning("Green GDF is empty — skipping nearest-park join.")
        null_result["nearest_park_name"] = None
        null_result["nearest_park_dist_m"] = np.nan
        return null_result

    # Use centroids for polygon geometries (distance to nearest centroid is a
    # reasonable proxy and far faster than true polygon distance for 1M+ pairs).
    green_for_join = green_gdf[["geometry", "name"]].copy()
    green_for_join["geometry"] = green_for_join.geometry.centroid

    joined = gpd.sjoin_nearest(
        prop_gdf[["geometry"]],
        green_for_join,
        how="left",
        distance_col="nearest_park_dist_m",
    )

    # sjoin_nearest can produce duplicates when multiple parks tie on distance.
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.rename(columns={"name": "nearest_park_name"})

    logger.info(
        "Nearest-park join complete. Median distance: %.0f m",
        joined["nearest_park_dist_m"].median(),
    )
    return joined[["nearest_park_name", "nearest_park_dist_m"]]


# ---------------------------------------------------------------------------
# Step 3 — Green area within buffer
# ---------------------------------------------------------------------------


def _green_area_in_buffer_rowwise(
    prop_gdf: gpd.GeoDataFrame,
    green_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Compute total green area (m²) within *radius_m* of each property.

    Uses a spatial index (R-tree via ``green_gdf.sindex``) to pre-filter
    candidate parks before computing polygon intersections, keeping each
    per-property loop iteration fast.

    This row-wise approach is used when ``len(prop_gdf) <= _BATCH_BUFFER_THRESHOLD``.
    For larger datasets, :func:`_green_area_in_buffer_vectorized` is preferred.

    Args:
        prop_gdf: Property GeoDataFrame in EPSG:2157.
        green_gdf: Green space GeoDataFrame in EPSG:2157.
        radius_m: Buffer radius in metres.

    Returns:
        Float Series indexed like *prop_gdf* giving total green area in m².
    """
    if green_gdf.empty:
        return pd.Series(0.0, index=prop_gdf.index)

    green_sindex = green_gdf.sindex
    areas: list[float] = []

    for geom in prop_gdf.geometry:
        buf = geom.buffer(radius_m)
        candidate_idx = list(green_sindex.intersection(buf.bounds))
        if not candidate_idx:
            areas.append(0.0)
            continue
        candidates = green_gdf.iloc[candidate_idx]
        total = float(candidates.geometry.intersection(buf).area.sum())
        areas.append(total)

    return pd.Series(areas, index=prop_gdf.index)


def _green_area_in_buffer_vectorized(
    prop_gdf: gpd.GeoDataFrame,
    green_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Vectorized green-area-within-buffer using GeoDataFrame.overlay.

    Suitable for large property datasets (> _BATCH_BUFFER_THRESHOLD rows).
    Creates a single GeoDataFrame of property buffers, intersects it with the
    green space layer, then aggregates intersection areas back to properties.

    Args:
        prop_gdf: Property GeoDataFrame in EPSG:2157.
        green_gdf: Green space GeoDataFrame in EPSG:2157.
        radius_m: Buffer radius in metres.

    Returns:
        Float Series indexed like *prop_gdf* giving total green area in m².
    """
    if green_gdf.empty:
        return pd.Series(0.0, index=prop_gdf.index)

    # Build buffer layer with a stable integer key column.
    buf_gdf = prop_gdf[["geometry"]].copy()
    buf_gdf["_prop_idx"] = range(len(buf_gdf))
    buf_gdf["geometry"] = buf_gdf.geometry.buffer(radius_m)

    try:
        # Explode multi-geometries so overlay doesn't see mixed types.
        green_single = green_gdf[["geometry"]].copy()
        green_single = green_single.explode(index_parts=False).reset_index(drop=True)
        intersected = gpd.overlay(
            buf_gdf[["_prop_idx", "geometry"]],
            green_single,
            how="intersection",
            keep_geom_type=False,
        )
        intersected["_area"] = intersected.geometry.area
        totals = intersected.groupby("_prop_idx")["_area"].sum()

        # Re-index against original property index.
        area_series = pd.Series(0.0, index=range(len(prop_gdf)))
        area_series.update(totals)
        area_series.index = prop_gdf.index
        return area_series
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Vectorized overlay failed (%s) — falling back to row-wise approach.", exc
        )
        return _green_area_in_buffer_rowwise(prop_gdf, green_gdf, radius_m)


def _green_area_in_buffer(
    prop_gdf: gpd.GeoDataFrame,
    green_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Dispatch to row-wise or vectorized buffer computation based on dataset size.

    Args:
        prop_gdf: Property GeoDataFrame in EPSG:2157.
        green_gdf: Green space GeoDataFrame in EPSG:2157.
        radius_m: Buffer radius in metres.

    Returns:
        Float Series of green area (m²) within *radius_m* for each property.
    """
    if len(prop_gdf) > _BATCH_BUFFER_THRESHOLD:
        logger.info(
            "Large dataset (%d rows) — using vectorized buffer for r=%.0fm.",
            len(prop_gdf),
            radius_m,
        )
        return _green_area_in_buffer_vectorized(prop_gdf, green_gdf, radius_m)

    return _green_area_in_buffer_rowwise(prop_gdf, green_gdf, radius_m)


# ---------------------------------------------------------------------------
# Step 4 — Air quality assignment
# ---------------------------------------------------------------------------


def _assign_air_quality(
    df: pd.DataFrame,
    prop_gdf: gpd.GeoDataFrame,
    air_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign nearest-station annual AQ readings to each property.

    Matching strategy:
    1. Build a KD-tree over air quality station coordinates (EPSG:2157).
    2. For each property, find the nearest station (Euclidean distance in metres).
    3. Look up the annual mean for that station and the property's year_of_sale.
    4. Assign one column per pollutant type.

    If ``air_df`` is empty or has no usable station coordinates, all AQ columns
    are filled with NaN.

    Args:
        df: Enriched property DataFrame (already has green-space columns).
        prop_gdf: Matching property GeoDataFrame in EPSG:2157.
        air_df: Annual aggregates DataFrame from ``processed_air_annual``.

    Returns:
        *df* with new columns: ``nearest_air_station``, ``air_station_dist_m``,
        ``mean_no2_year``, ``mean_pm25_year``, ``mean_pm10_year``,
        ``mean_noise_db_year``.
    """
    aq_cols = list(set(_POLLUTANT_MAP.values()))
    for col in ["nearest_air_station", "air_station_dist_m"] + aq_cols:
        df[col] = np.nan

    if air_df.empty:
        logger.warning("Air quality DataFrame is empty — AQ columns will be NaN.")
        df["nearest_air_station"] = None
        return df

    # Normalise lat/lon column names (API may return 'Latitude'/'Longitude').
    lat_col, lon_col = None, None
    for lc in ("lat", "latitude", "Latitude"):
        if lc in air_df.columns:
            lat_col = lc
            break
    for lnc in ("lon", "lng", "longitude", "Longitude"):
        if lnc in air_df.columns:
            lon_col = lnc
            break

    if lat_col is None or lon_col is None:
        logger.warning(
            "Air quality data has no usable lat/lon columns — AQ columns will be NaN."
        )
        df["nearest_air_station"] = None
        return df

    # One row per station (drop duplicates to get unique station positions).
    id_col = "monitor_id" if "monitor_id" in air_df.columns else air_df.columns[0]
    name_col = "station_name" if "station_name" in air_df.columns else id_col

    stations = (
        air_df[[id_col, name_col, lat_col, lon_col]]
        .drop_duplicates(subset=[id_col])
        .dropna(subset=[lat_col, lon_col])
        .reset_index(drop=True)
    )

    if stations.empty:
        logger.warning("No stations with coordinates found — AQ columns will be NaN.")
        df["nearest_air_station"] = None
        return df

    # Project station coordinates to EPSG:2157 for metric KD-tree distances.
    station_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations[lon_col], stations[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=2157)

    station_xy = np.array([(g.x, g.y) for g in station_gdf.geometry])
    tree = cKDTree(station_xy)

    prop_xy = np.array([(g.x, g.y) for g in prop_gdf.geometry])
    distances, indices = tree.query(prop_xy, k=1)

    df["nearest_air_station"] = stations.iloc[indices][name_col].values
    df["air_station_dist_m"] = distances.astype(float)

    matched_station_ids = stations.iloc[indices][id_col].values

    # Ensure year_of_sale is present and usable as int for matching.
    if "year_of_sale" not in df.columns:
        logger.warning(
            "year_of_sale column missing — cannot match AQ by year. Using NaN."
        )
        return df

    years = pd.to_numeric(df["year_of_sale"], errors="coerce")

    # Build a lookup dict: (monitor_id, year, pollutant_upper) -> mean_value
    # to avoid repeated DataFrame filtering in the inner loop.
    mean_col = "mean_value" if "mean_value" in air_df.columns else "value"
    pollutant_col = "pollutant" if "pollutant" in air_df.columns else None

    if pollutant_col is None:
        logger.warning("Air quality data has no 'pollutant' column — AQ NaN.")
        return df

    lookup: dict[tuple, float] = {}
    for _, row in air_df.iterrows():
        key = (
            str(row[id_col]),
            int(row["year"]) if pd.notna(row.get("year")) else None,
            str(row[pollutant_col]).upper().replace(".", "").replace(" ", ""),
        )
        if key[1] is not None:
            lookup[key] = float(row[mean_col]) if pd.notna(row[mean_col]) else np.nan

    # Initialise output arrays with NaN then fill by matched key.
    result_arrays: dict[str, list[Optional[float]]] = {
        col: [np.nan] * len(df) for col in aq_cols
    }

    for i, (station_id, year) in enumerate(
        zip(matched_station_ids, years)
    ):
        if pd.isna(year):
            continue
        yr = int(year)
        sid = str(station_id)

        for keyword, dest_col in _POLLUTANT_MAP.items():
            kw_norm = keyword.upper().replace(".", "").replace(" ", "")
            val = lookup.get((sid, yr, kw_norm), np.nan)
            if not np.isnan(val):
                result_arrays[dest_col][i] = val

    for col, values in result_arrays.items():
        df[col] = values

    # Log coverage stats.
    for col in aq_cols:
        null_rate = df[col].isna().mean()
        logger.info("AQ column '%s' null rate: %.1f%%", col, null_rate * 100)

    return df


# ---------------------------------------------------------------------------
# Step 5 — Write outputs
# ---------------------------------------------------------------------------


def _write_to_postgres(df: pd.DataFrame, engine) -> int:
    """Write enriched property records to processed.property_enriched.

    Uses a truncate-then-insert pattern (idempotent). The schema is assumed to
    exist (created by ``db/init/postgres/01_schema.sql``); we rely on pandas
    ``to_sql`` with ``if_exists='replace'`` so the table is recreated if the
    schema has drifted.

    Args:
        df: Enriched property DataFrame.
        engine: SQLAlchemy engine.

    Returns:
        Number of rows written.
    """
    # Columns that Postgres/psycopg2 cannot serialise directly.
    export_df = df.copy()

    # Convert pandas NA / NaT to None for SQL compatibility.
    for col in export_df.select_dtypes(include=["Int64", "boolean"]).columns:
        export_df[col] = export_df[col].astype(object).where(export_df[col].notna(), None)

    # Ensure date_of_sale is a plain Python date (not numpy datetime64).
    if "date_of_sale" in export_df.columns:
        export_df["date_of_sale"] = pd.to_datetime(
            export_df["date_of_sale"], errors="coerce"
        ).dt.date

    # Ensure the processed schema exists before writing.
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS processed"))
        conn.commit()

    export_df.to_sql(
        "property_enriched",
        engine,
        schema="processed",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM processed.property_enriched")
        ).scalar()

    logger.info(
        "Wrote %d rows to processed.property_enriched (PostgreSQL).", count
    )
    return int(count)


def _write_to_mongo(df: pd.DataFrame, db) -> int:
    """Mirror enriched property records to MongoDB processed_property collection.

    Upsert key: composite (address_clean, date_of_sale).

    Args:
        df: Enriched property DataFrame.
        db: PyMongo Database object.

    Returns:
        Total upserted + modified count.
    """
    col = db["processed_property"]
    col.create_index(
        [("address_clean", 1), ("date_of_sale", 1)],
        unique=True,
        background=True,
    )

    records = df.to_dict("records")
    ops: list[UpdateOne] = []

    for rec in records:
        # Sanitise non-serialisable Python/NumPy types.
        clean: dict = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif hasattr(v, "isoformat"):
                clean[k] = v.isoformat()
            else:
                clean[k] = v

        filt = {
            "address_clean": clean.get("address_clean", ""),
            "date_of_sale": clean.get("date_of_sale", ""),
        }
        ops.append(UpdateOne(filt, {"$set": clean}, upsert=True))

    if not ops:
        logger.warning("No records to upsert into processed_property.")
        return 0

    result = col.bulk_write(ops, ordered=False)
    total = result.upserted_count + result.modified_count
    logger.info(
        "MongoDB processed_property upserted=%d modified=%d (total: %d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run() -> pd.DataFrame:
    """Run the full spatial join pipeline and return the enriched DataFrame.

    Pipeline steps:
        0. Load property data (geocoded > cleaned fallback).
        1. Drop rows without lat/lon.
        2. Build property GeoDataFrame in EPSG:2157.
        3. Load green space GeoDataFrame in EPSG:2157.
        4. Load annual air quality aggregates.
        5. Compute nearest park name + distance.
        6. Compute green area within 500 m buffer.
        7. Compute green area within 1000 m buffer.
        8. Assign nearest-station annual AQ readings matched by year.
        9. Derive ``log_price`` if not already present.
       10. Write to PostgreSQL processed.property_enriched.
       11. Mirror to MongoDB processed_property.
       12. Export to data/processed/property_enriched.parquet.
       13. Log null rates for key enrichment columns.

    Returns:
        Enriched property DataFrame.

    Raises:
        FileNotFoundError: If no property parquet file is found.
    """
    engine = get_postgres_engine()
    db = get_mongo_db()

    # --- Step 0: Load inputs ---
    logger.info("Loading property data …")
    df = _load_property(INTERIM_DIR)
    n_raw = len(df)
    logger.info("Loaded %d property rows.", n_raw)

    df = df.dropna(subset=["lat", "lon"])
    n_with_coords = len(df)
    dropped = n_raw - n_with_coords
    if dropped:
        logger.warning(
            "Dropped %d rows without coordinates (%.1f%% of total).",
            dropped,
            100.0 * dropped / n_raw if n_raw else 0,
        )
    logger.info("%d rows have coordinates for spatial join.", n_with_coords)

    if df.empty:
        logger.error("No rows with coordinates — cannot run spatial join. Aborting.")
        return df

    logger.info("Loading green space data …")
    green_gdf = _load_green_gdf(PROCESSED_DIR, db)
    logger.info("Green GDF: %d features.", len(green_gdf))

    logger.info("Loading air quality annual averages …")
    air_df = _load_air_annual(db)
    logger.info("Air annual rows: %d.", len(air_df))

    # --- Step 1–2: Build property GeoDataFrame in ITM ---
    logger.info("Building property GeoDataFrame in EPSG:2157 …")
    prop_gdf = _build_property_gdf(df)

    # --- Step 5: Nearest park ---
    logger.info("Computing nearest park distances …")
    park_df = _nearest_park(prop_gdf, green_gdf)
    df["nearest_park_name"] = park_df["nearest_park_name"].values
    df["nearest_park_dist_m"] = park_df["nearest_park_dist_m"].values

    # --- Step 6–7: Green area buffers ---
    for radius in (500, 1000):
        col_name = f"green_area_within_{radius}m"
        logger.info("Computing green area within %d m …", radius)
        df[col_name] = _green_area_in_buffer(prop_gdf, green_gdf, float(radius)).values

    # --- Step 8: Air quality assignment ---
    logger.info("Assigning air quality readings …")
    df = _assign_air_quality(df, prop_gdf, air_df)

    # --- Step 9: log_price ---
    if "log_price" not in df.columns:
        price_col = "price" if "price" in df.columns else None
        if price_col and (df[price_col] > 0).all():
            df["log_price"] = np.log(df[price_col])
        elif price_col:
            valid = df[price_col] > 0
            df["log_price"] = np.where(valid, np.log(df[price_col].where(valid)), np.nan)

    # --- Step 10: Write to Postgres ---
    logger.info("Writing enriched data to PostgreSQL …")
    _write_to_postgres(df, engine)

    # --- Step 11: Mirror to MongoDB ---
    logger.info("Mirroring enriched data to MongoDB …")
    _write_to_mongo(df, db)

    # --- Step 12: Export Parquet ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "property_enriched.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Exported enriched data to %s", out_path)

    # --- Step 13: Null-rate diagnostics ---
    key_cols = [
        "nearest_park_dist_m",
        "green_area_within_500m",
        "green_area_within_1000m",
        "mean_no2_year",
        "mean_pm25_year",
        "mean_pm10_year",
        "mean_noise_db_year",
    ]
    logger.info("=== Enrichment null rates ===")
    for col in key_cols:
        if col in df.columns:
            null_rate = df[col].isna().mean()
            flag = " [WARN > 40%]" if null_rate > _MAX_ACCEPTABLE_NULL_RATE else ""
            logger.info("  %-35s %.1f%%%s", col, null_rate * 100, flag)

    logger.info(
        "join.run() complete — %d enriched rows written.", len(df)
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    import sys

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[_logging.StreamHandler(sys.stdout)],
    )
    result = run()
    print(f"Done — enriched rows: {len(result)}")
