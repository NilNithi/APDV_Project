"""Clean and validate green space geometries from MongoDB raw_green collection.

Reads every document from ``raw_green``, validates and repairs Shapely
geometries, reprojects to EPSG:2157 (Irish Transverse Mercator) for accurate
metric area / distance calculations, upserts processed documents back into
``processed_green``, and exports a GeoParquet file.

Idempotent: re-running replaces existing processed documents via upsert on a
stable geometry-hash key (or ``{source, name}`` as fallback).

Usage::

    python -m src.etl.green_clean
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from pymongo import UpdateOne
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from src.config import get_mongo_db, PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# MongoDB collection names.
_COL_RAW = "raw_green"
_COL_PROCESSED = "processed_green"

# Source CRS of GeoJSON (WGS-84) and target CRS for spatial analysis.
_CRS_SOURCE = "EPSG:4326"
_CRS_TARGET = "EPSG:2157"  # Irish Transverse Mercator — metres


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------


def _parse_geometry(doc: dict) -> Optional[object]:
    """Parse and validate a Shapely geometry from a raw_green document.

    If the geometry is invalid (according to the OGC spec), ``make_valid``
    is applied.  Points are accepted as-is; their area will be recorded as 0.

    Args:
        doc: A single document dict from MongoDB ``raw_green``.

    Returns:
        A valid Shapely geometry object, or ``None`` if the geometry field is
        absent, null, or un-parseable.
    """
    geom_dict = doc.get("geometry")
    if not geom_dict:
        return None
    try:
        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = make_valid(geom)
        return geom
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
        logger.warning(
            "Invalid geometry for feature '%s': %s", doc.get("name", "<unnamed>"), exc
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(db) -> list[dict]:
    """Load all documents from raw_green.

    Args:
        db: PyMongo Database object.

    Returns:
        List of raw document dicts (``_id`` stripped).
    """
    logger.info("Loading documents from %s …", _COL_RAW)
    docs = list(db[_COL_RAW].find({}, {"_id": 0}))
    logger.info("Loaded %d raw_green documents.", len(docs))
    return docs


def _build_geodataframe(docs: list[dict]) -> gpd.GeoDataFrame:
    """Convert raw documents to a GeoDataFrame in WGS-84.

    Rows whose geometry cannot be parsed are dropped with a warning.

    Args:
        docs: Raw documents from MongoDB ``raw_green``.

    Returns:
        GeoDataFrame in EPSG:4326 with ``geometry`` column populated.
    """
    geometries = []
    props = []

    for doc in docs:
        geom = _parse_geometry(doc)
        if geom is None:
            logger.debug(
                "Dropping feature '%s' — no parseable geometry.",
                doc.get("name", "<unnamed>"),
            )
            continue
        geometries.append(geom)
        # Store all non-geometry fields as properties.
        row = {k: v for k, v in doc.items() if k != "geometry"}
        props.append(row)

    if not props:
        logger.warning("No features with valid geometry found in raw_green.")
        return gpd.GeoDataFrame(geometry=[], crs=_CRS_SOURCE)

    gdf = gpd.GeoDataFrame(props, geometry=geometries, crs=_CRS_SOURCE)
    logger.info(
        "Built GeoDataFrame: %d features (dropped %d with null geometry).",
        len(gdf),
        len(docs) - len(gdf),
    )
    return gdf


def _compute_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute ``area_sqm`` from the reprojected geometry.

    For Point geometries the area is 0 by definition; those rows are assigned
    the median area of all non-zero-area features so that downstream spatial
    analysis has a reasonable substitute.

    This function must be called **after** reprojection to EPSG:2157 so that
    ``.area`` returns values in square metres.

    Args:
        gdf: GeoDataFrame in EPSG:2157.

    Returns:
        The same GeoDataFrame with a new ``area_sqm`` column (float).
    """
    gdf = gdf.copy()
    gdf["area_sqm"] = gdf.geometry.area

    # Point / zero-area features: substitute median of real areas.
    zero_mask = gdf["area_sqm"] == 0
    if zero_mask.any():
        non_zero = gdf.loc[~zero_mask, "area_sqm"]
        fallback = float(non_zero.median()) if not non_zero.empty else 0.0
        gdf.loc[zero_mask, "area_sqm"] = fallback
        logger.info(
            "Assigned median area (%.1f m²) to %d zero-area (Point) features.",
            fallback,
            zero_mask.sum(),
        )
    return gdf


def _upsert_processed(db, gdf: gpd.GeoDataFrame) -> int:
    """Upsert processed green-space documents into ``processed_green``.

    Upsert key precedence (first available):
    1. ``geom_hash`` field (set by the ingest step when a stable hash exists).
    2. Composite ``{source, name}`` (may have collisions for unnamed features).
    3. Row index as last resort (not truly stable across re-runs — logged).

    Geometry is serialised back to GeoJSON for MongoDB storage.

    Args:
        db: PyMongo Database object.
        gdf: Processed GeoDataFrame in EPSG:2157.

    Returns:
        Total upserted + modified count.
    """
    if gdf.empty:
        logger.warning("Processed GeoDataFrame is empty — nothing to upsert.")
        return 0

    col = db[_COL_PROCESSED]
    # Ensure indexes for the two most-common upsert keys.
    col.create_index("geom_hash", sparse=True, background=True)
    col.create_index([("source", 1), ("name", 1)], sparse=True, background=True)

    ops: list[UpdateOne] = []
    for _, row in gdf.iterrows():
        row_dict = row.to_dict()

        # Convert Shapely geometry to GeoJSON dict for Mongo storage.
        geom_obj = row_dict.pop("geometry", None)
        if geom_obj is not None:
            # Store geometry in the source CRS (WGS-84) for GeoJSON compliance.
            # We keep area_sqm (EPSG:2157-derived) as a separate scalar field.
            try:
                geom_wgs84 = (
                    gpd.GeoSeries([geom_obj], crs=_CRS_TARGET)
                    .to_crs(_CRS_SOURCE)
                    .iloc[0]
                )
                row_dict["geometry"] = mapping(geom_wgs84)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not reproject geometry for storage: %s", exc)
                row_dict["geometry"] = mapping(geom_obj)

        # Also store the WKT of the EPSG:2157 geometry for fast loading in ETL.
        if geom_obj is not None:
            row_dict["geometry_wkt_2157"] = geom_obj.wkt

        # Build the filter (upsert key).
        if "geom_hash" in row_dict and row_dict["geom_hash"]:
            filt = {"geom_hash": row_dict["geom_hash"]}
        elif row_dict.get("source") and row_dict.get("name"):
            filt = {"source": row_dict["source"], "name": row_dict["name"]}
        else:
            # Last-resort key — log so the operator knows this is fragile.
            filt = {
                "geometry_wkt_2157": row_dict.get("geometry_wkt_2157", str(row.name))
            }
            logger.debug(
                "Feature has no stable key (no geom_hash, no source+name) — "
                "using WKT as upsert key (row %s).",
                row.name,
            )

        # Sanitise NaN floats so MongoDB doesn't reject them.
        clean_doc = {
            k: (None if isinstance(v, float) and pd.isna(v) else v)
            for k, v in row_dict.items()
        }

        ops.append(UpdateOne(filt, {"$set": clean_doc}, upsert=True))

    result = col.bulk_write(ops, ordered=False)
    total = result.upserted_count + result.modified_count
    logger.info(
        "processed_green upserted=%d modified=%d (total in collection: %d)",
        result.upserted_count,
        result.modified_count,
        col.count_documents({}),
    )
    return total


def _export_parquet(gdf: gpd.GeoDataFrame, filename: str) -> None:
    """Export the GeoDataFrame to GeoParquet (or plain Parquet as fallback).

    Geopandas >= 0.12 supports native GeoParquet via ``.to_parquet()``.  If
    that fails (e.g. older pyarrow without geoarrow support), the geometry
    column is converted to WKT strings and saved as a plain Parquet file.

    Args:
        gdf: GeoDataFrame in EPSG:2157 with ``area_sqm`` column.
        filename: Output filename (e.g. ``"green_spaces.parquet"``).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename

    try:
        gdf.to_parquet(out_path, index=False)
        logger.info("Exported GeoParquet (%d features) to %s", len(gdf), out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "GeoParquet export failed (%s) — falling back to WKT Parquet.", exc
        )
        fallback_df = gdf.copy()
        fallback_df["geometry"] = fallback_df.geometry.apply(
            lambda g: g.wkt if g is not None else None
        )
        # Drop the geometry dtype column; keep the WKT string column.
        plain_df = pd.DataFrame(fallback_df)
        plain_df.to_parquet(out_path, index=False)
        logger.info(
            "Exported plain Parquet with WKT geometry (%d features) to %s",
            len(plain_df),
            out_path,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run() -> gpd.GeoDataFrame:
    """Clean and validate green space data, writing processed collection.

    Orchestration steps:
        1. Load all documents from MongoDB ``raw_green``.
        2. Parse and validate geometries (``make_valid`` on broken polygons).
        3. Build a GeoDataFrame in EPSG:4326.
        4. Drop features with null geometry.
        5. Reproject to EPSG:2157 (Irish Transverse Mercator).
        6. Compute ``area_sqm`` from reprojected geometry.
        7. Add ``geometry_wkt`` column (WKT of EPSG:2157 geometry).
        8. Upsert processed documents to ``processed_green`` (idempotent).
        9. Export to ``data/processed/green_spaces.parquet``.

    Returns:
        The processed GeoDataFrame in EPSG:2157 with ``area_sqm`` populated.
        Returns an empty GeoDataFrame if ``raw_green`` contains no usable data.
    """
    db = get_mongo_db()

    # --- load ---
    docs = _load_raw(db)
    if not docs:
        logger.warning("raw_green is empty — skipping ETL. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame()

    # --- parse geometries & build GeoDataFrame ---
    gdf = _build_geodataframe(docs)
    if gdf.empty:
        logger.warning(
            "No features with valid geometry after parsing. "
            "Returning empty GeoDataFrame."
        )
        return gdf

    # --- reproject to EPSG:2157 ---
    logger.info("Reprojecting %d features from %s to %s …", len(gdf), _CRS_SOURCE, _CRS_TARGET)
    gdf = gdf.to_crs(epsg=2157)

    # --- compute area ---
    gdf = _compute_area(gdf)

    # --- add WKT column for downstream convenience ---
    gdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g is not None else None)

    # --- persist to MongoDB ---
    _upsert_processed(db, gdf)

    # --- export Parquet ---
    _export_parquet(gdf, "green_spaces.parquet")

    logger.info(
        "green_clean complete. Valid features: %d  CRS: %s",
        len(gdf),
        gdf.crs,
    )
    return gdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run()
    print(f"Done — features processed: {len(result)}")
