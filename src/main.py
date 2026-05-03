"""Main orchestrator — runs the full Green Premium Dublin pipeline end-to-end.

Supports two modes:

  python -m src.main          # Full pipeline (real data, ~1-2 hours)
  python -m src.main --demo   # Demo mode (synthetic data, ~30 seconds, no DB needed)

Execution order:

  Phase 1 — Ingest (parallel: property | air | green)
  Phase 2 — ETL
    2a. Clean (parallel: property_clean | air_clean | green_clean)
    2b. Geocode (depends on property_clean)
    2c. Spatial join (depends on all clean + geocode)
  Phase 3 — Analysis + Visualisation (parallel: descriptive | correlation | regression | temporal | static_plots | maps)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Parallel runner helper
# ---------------------------------------------------------------------------

def _run_parallel(tasks: dict[str, callable], label: str) -> dict[str, any]:
    """Run named callables in parallel threads, return {name: result}.

    Args:
        tasks: Dict of {task_name: callable}.
        label: Label for logging (e.g. "Ingest", "Clean").

    Returns:
        Dict of {task_name: return_value}.
    """
    results = {}
    logger.info("--- %s: launching %d tasks in parallel ---", label, len(tasks))
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                logger.info("[%s] %s completed.", label, name)
            except Exception:
                logger.exception("[%s] %s FAILED.", label, name)
                raise
    return results


# ---------------------------------------------------------------------------
# Demo mode: generate small synthetic dataset (no DB needed)
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Run a fast demo pipeline with synthetic data (~500 rows, ~30 seconds).

    No databases, no API calls, no network. Generates synthetic property,
    air quality, and green space data, runs ETL in-memory, produces figures,
    and writes the enriched parquet so the dashboard can load it.
    """
    logger.info("=== DEMO MODE — synthetic data, no DB required ===")
    t0 = time.perf_counter()

    rng = np.random.default_rng(42)
    n = 500

    # --- Synthetic property data ---
    logger.info("[Demo] Generating %d synthetic property records...", n)
    postcodes = [f"Dublin {d}" for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 22, 24]]
    years = [2015, 2016, 2017, 2018, 2019, 2020]

    # Dublin bbox: lat 53.25-53.42, lon -6.45 to -6.10
    lats = rng.uniform(53.25, 53.42, n)
    lons = rng.uniform(-6.45, -6.10, n)
    prices = rng.lognormal(mean=12.5, sigma=0.5, size=n).astype(float)

    df = pd.DataFrame({
        "id": range(1, n + 1),
        "date_of_sale": pd.to_datetime(
            [f"{rng.choice(years)}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" for _ in range(n)]
        ),
        "address": [f"{rng.integers(1, 200)} Sample Street {i}" for i in range(n)],
        "address_clean": [f"{rng.integers(1, 200)} sample street {i}" for i in range(n)],
        "postal_code": [rng.choice(postcodes) for _ in range(n)],
        "county": "Dublin",
        "price": prices,
        "not_full_market_price": False,
        "construction": rng.choice(
            ["New Dwelling house /Apartment", "Second-Hand Dwelling house /Apartment"],
            size=n,
        ),
        "vat_exclusive": False,
        "source_file": "demo_synthetic",
        "ingested_at": pd.Timestamp.now(),
        "year_of_sale": pd.array([int(rng.choice(years)) for _ in range(n)], dtype="Int64"),
        "log_price": np.log(prices),
        "floor_area_sqm": rng.uniform(40, 200, n),
        "floor_area_category": None,
        "lat": lats,
        "lon": lons,
        "geocode_source": "synthetic_demo",
    })

    # --- Synthetic green space data ---
    logger.info("[Demo] Generating synthetic green spaces...")
    n_parks = 50
    park_lats = rng.uniform(53.25, 53.42, n_parks)
    park_lons = rng.uniform(-6.45, -6.10, n_parks)
    park_names = [f"Demo Park {i+1}" for i in range(n_parks)]

    # Compute nearest park distance + green area
    from scipy.spatial import cKDTree

    park_coords = np.column_stack([park_lats, park_lons])
    prop_coords = np.column_stack([lats, lons])
    tree = cKDTree(park_coords)
    dist_deg, idx = tree.query(prop_coords)
    # Rough degrees to meters at Dublin latitude
    dist_m = dist_deg * 111_000 * np.cos(np.radians(53.35))

    df["nearest_park_name"] = [park_names[i] for i in idx]
    df["nearest_park_dist_m"] = dist_m
    df["green_area_within_500m"] = rng.uniform(0, 150_000, n)
    df["green_area_within_1000m"] = df["green_area_within_500m"] + rng.uniform(0, 100_000, n)

    # --- Synthetic air quality data ---
    logger.info("[Demo] Generating synthetic air quality readings...")
    station_names = ["Winetavern St", "Rathmines", "Finglas", "Dun Laoghaire", "Tallaght"]
    df["nearest_air_station"] = rng.choice(station_names, n)
    df["air_station_dist_m"] = rng.uniform(500, 8000, n)
    # NO2 higher in inner city (lower district numbers)
    district_num = df["postal_code"].str.extract(r"(\d+)").astype(float).values.flatten()
    df["mean_no2_year"] = 30 - district_num * 0.5 + rng.normal(0, 3, n)
    df["mean_pm25_year"] = rng.uniform(8, 18, n)
    df["mean_pm10_year"] = rng.uniform(15, 35, n)
    df["mean_noise_db_year"] = rng.uniform(45, 75, n)

    # --- Write outputs to data/demo/ (never overwrites real data) ---
    demo_dir = PROJECT_ROOT / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = demo_dir / "property_enriched.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("[Demo] Wrote %d rows to %s", len(df), parquet_path)

    # Symlink/copy so dashboard and analysis modules can find it
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    prod_parquet = processed_dir / "property_enriched.parquet"
    # Only copy if no real data exists (never overwrite real 92K data)
    if not prod_parquet.exists() or prod_parquet.stat().st_size < 1_000_000:
        import shutil
        shutil.copy2(parquet_path, prod_parquet)
        logger.info("[Demo] Copied demo parquet to %s (no real data found)", prod_parquet)
    else:
        logger.info("[Demo] Real data exists at %s (%d MB) -- NOT overwriting. "
                     "Dashboard will show real data. Demo parquet at %s",
                     prod_parquet, prod_parquet.stat().st_size // 1_000_000, parquet_path)

    # --- Run analysis ---
    logger.info("[Demo] Running analysis on synthetic data...")
    try:
        from src.analysis import descriptive, correlation, regression, temporal
        analysis_tasks = {
            "descriptive": descriptive.run,
            "correlation": correlation.run,
            "regression": regression.run,
            "temporal": temporal.run,
        }
        _run_parallel(analysis_tasks, "Analysis")
    except Exception:
        logger.warning("[Demo] Analysis step had errors (non-fatal for demo).", exc_info=True)

    # --- Run visualization ---
    logger.info("[Demo] Generating figures...")
    try:
        from src.viz import static_plots, maps
        viz_tasks = {
            "static_plots": static_plots.run,
            "maps": maps.run,
        }
        _run_parallel(viz_tasks, "Viz")
    except Exception:
        logger.warning("[Demo] Viz step had errors (non-fatal for demo).", exc_info=True)

    elapsed = time.perf_counter() - t0
    logger.info("=== DEMO PIPELINE COMPLETE in %.1f seconds ===", elapsed)
    logger.info("Dashboard: python -m src.dashboard.app  -->  http://localhost:8050")


# ---------------------------------------------------------------------------
# Full pipeline (real data, parallel where possible)
# ---------------------------------------------------------------------------

def _run_full() -> None:
    """Run the full pipeline with real data and parallel execution."""
    logger.info("=== GREEN PREMIUM DUBLIN PIPELINE START ===")
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: Ingest (parallel — 3 independent data sources)
    # ------------------------------------------------------------------
    from src.ingest.property_ingest import run as property_ingest
    from src.ingest.air_ingest import run as air_ingest
    from src.ingest.green_ingest import run as green_ingest

    ingest_tasks = {
        "property_ingest": property_ingest,
        "air_ingest": air_ingest,
        "green_ingest": green_ingest,
    }
    _run_parallel(ingest_tasks, "Phase 1 Ingest")

    # ------------------------------------------------------------------
    # Phase 2a: Clean (parallel — 3 independent clean steps)
    # ------------------------------------------------------------------
    from src.etl.property_clean import run as property_clean
    from src.etl.air_clean import run as air_clean
    from src.etl.green_clean import run as green_clean

    clean_tasks = {
        "property_clean": property_clean,
        "air_clean": air_clean,
        "green_clean": green_clean,
    }
    clean_results = _run_parallel(clean_tasks, "Phase 2a Clean")

    df_clean = clean_results["property_clean"]
    logger.info("property_clean produced %d rows.", len(df_clean))

    # ------------------------------------------------------------------
    # Phase 2b: Geocode (depends on property_clean)
    # ------------------------------------------------------------------
    logger.info("[2b] Geocode (address -> lat/lon via Nominatim + cache)...")
    from src.etl.geocode import run as geocode

    df_geocoded = geocode(df_clean)
    logger.info(
        "geocode produced %d rows; null lat rate: %.1f%%.",
        len(df_geocoded),
        df_geocoded["lat"].isna().mean() * 100,
    )

    # ------------------------------------------------------------------
    # Phase 2c: Spatial join (depends on all clean + geocode)
    # ------------------------------------------------------------------
    logger.info("[2c] Spatial join (all layers -> property_enriched)...")
    from src.etl.join import run as join_data

    df_enriched = join_data()
    logger.info("join produced %d enriched property rows.", len(df_enriched))

    # ------------------------------------------------------------------
    # Phase 3: Analysis + Viz (parallel — all independent)
    # ------------------------------------------------------------------
    from src.analysis import descriptive, correlation, regression, temporal
    from src.viz import static_plots, maps

    phase3_tasks = {
        "descriptive": descriptive.run,
        "correlation": correlation.run,
        "regression": regression.run,
        "temporal": temporal.run,
        "static_plots": static_plots.run,
        "maps": maps.run,
    }
    _run_parallel(phase3_tasks, "Phase 3 Analysis+Viz")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    logger.info("=== PIPELINE COMPLETE in %.1f seconds ===", elapsed)
    logger.info(
        "Final enriched dataset: %d rows with %d columns.",
        len(df_enriched),
        len(df_enriched.columns),
    )

    key_cols = [
        "nearest_park_dist_m",
        "green_area_within_500m",
        "mean_no2_year",
        "mean_pm25_year",
        "log_price",
    ]
    missing = [c for c in key_cols if c not in df_enriched.columns]
    if missing:
        logger.warning("Missing expected columns: %s", missing)
    else:
        logger.info("All key enrichment columns present.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and dispatch to full or demo pipeline."""
    parser = argparse.ArgumentParser(description="Green Premium Dublin Pipeline")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data (~500 rows, ~30s, no DB needed)",
    )
    args = parser.parse_args()

    if args.demo:
        _run_demo()
    else:
        _run_full()


if __name__ == "__main__":
    main()
