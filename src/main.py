"""Main orchestrator — runs the full Green Premium Dublin pipeline end-to-end.

Execution order:

  Phase 1 — Ingest
    1a. property_ingest  : PSRA CSV → PostgreSQL raw.property
    1b. air_ingest       : Sonitus API → MongoDB raw_air
    1c. green_ingest     : DCC GeoJSON → MongoDB raw_green

  Phase 2 — ETL
    2a. property_clean   : raw.property → data/interim/property_clean.parquet
    2b. geocode          : address strings → lat/lon (SQLite-cached Nominatim)
    2c. air_clean        : raw_air → processed_air_monthly / processed_air_annual
    2d. green_clean      : raw_green → processed_green + green_spaces.parquet
    2e. join             : all processed layers → processed.property_enriched

Usage::

    python -m src.main

Individual phases can also be run in isolation, e.g.::

    python -m src.ingest.property_ingest
    python -m src.etl.join
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Module-level logger — must come after basicConfig so it picks up the handler.
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full Green Premium Dublin data pipeline.

    Each phase is run sequentially. Failures in Phase 1 sub-steps are logged
    and surfaced as exceptions so the pipeline halts rather than silently
    producing partial results.

    Raises:
        Exception: Propagated from any sub-step that raises.
    """
    logger.info("=== GREEN PREMIUM DUBLIN PIPELINE START ===")

    # ------------------------------------------------------------------
    # Phase 1: Ingest
    # ------------------------------------------------------------------
    logger.info("--- Phase 1: Ingest ---")

    logger.info("[1a] Property ingest (PSRA CSV → PostgreSQL raw.property) …")
    from src.ingest.property_ingest import run as property_ingest

    property_ingest()

    logger.info("[1b] Air quality ingest (Sonitus API → MongoDB raw_air) …")
    from src.ingest.air_ingest import run as air_ingest

    air_ingest()

    logger.info("[1c] Green space ingest (DCC GeoJSON → MongoDB raw_green) …")
    from src.ingest.green_ingest import run as green_ingest

    green_ingest()

    # ------------------------------------------------------------------
    # Phase 2: ETL
    # ------------------------------------------------------------------
    logger.info("--- Phase 2: ETL ---")

    logger.info("[2a] Property clean (raw.property → property_clean.parquet) …")
    from src.etl.property_clean import run as property_clean

    df_clean = property_clean()
    logger.info("property_clean produced %d rows.", len(df_clean))

    logger.info("[2b] Geocode (address → lat/lon via Nominatim + cache) …")
    from src.etl.geocode import run as geocode

    df_geocoded = geocode(df_clean)
    logger.info(
        "geocode produced %d rows; null lat rate: %.1f%%.",
        len(df_geocoded),
        df_geocoded["lat"].isna().mean() * 100,
    )

    logger.info("[2c] Air clean (raw_air → processed_air_annual) …")
    from src.etl.air_clean import run as air_clean

    air_results = air_clean()
    logger.info(
        "air_clean produced %d monthly rows, %d annual rows.",
        len(air_results.get("monthly", [])),
        len(air_results.get("annual", [])),
    )

    logger.info("[2d] Green clean (raw_green → processed_green + GeoParquet) …")
    from src.etl.green_clean import run as green_clean

    gdf_green = green_clean()
    logger.info("green_clean produced %d features.", len(gdf_green))

    logger.info("[2e] Spatial join (all layers → property_enriched) …")
    from src.etl.join import run as join_data

    df_enriched = join_data()
    logger.info("join produced %d enriched property rows.", len(df_enriched))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info(
        "Final enriched dataset: %d rows with %d columns.",
        len(df_enriched),
        len(df_enriched.columns),
    )

    # Log presence of key enrichment columns as a quick sanity check.
    key_cols = [
        "nearest_park_dist_m",
        "green_area_within_500m",
        "mean_no2_year",
        "mean_pm25_year",
        "log_price",
    ]
    missing = [c for c in key_cols if c not in df_enriched.columns]
    if missing:
        logger.warning(
            "The following expected columns are absent from the enriched dataset: %s",
            missing,
        )
    else:
        logger.info("All key enrichment columns are present.")


if __name__ == "__main__":
    main()
