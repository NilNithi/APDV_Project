"""Generate F4 (NO₂ scatter-map / choropleth) and F9 (interactive Folium map).

F4 — Mean annual NO₂ levels across Dublin properties, rendered as a
     scatter-mapbox coloured by NO₂ concentration.

F9 — Interactive Folium map of Dublin: property sale prices (coloured by
     price quartile), green-space markers, and air-quality station markers.
     Saved as a self-contained HTML file for embedding in the dashboard.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go

from src.config import get_mongo_db, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants (mirror static_plots.py)
# ---------------------------------------------------------------------------
TEMPLATE: str = "plotly_white"
VIRIDIS: str = "Viridis"
import os as _os
FIGURE_DIR: Path = Path(_os.environ.get("GREEN_PREMIUM_FIGURE_DIR", PROJECT_ROOT / "report" / "figures"))
SCALE: int = int(_os.environ.get("GREEN_PREMIUM_FIGURE_SCALE", "3"))

# Dublin city centre approx centroid
_DUBLIN_LAT: float = 53.349805
_DUBLIN_LON: float = -6.260310


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_property_df() -> pd.DataFrame:
    """Load enriched or cleaned property data from parquet.

    Returns:
        Property DataFrame, potentially empty if no parquet files exist yet.
    """
    import os
    processed = Path(os.environ.get("GREEN_PREMIUM_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
    interim = PROJECT_ROOT / "data" / "interim"

    for candidate in [
        processed / "property_enriched.parquet",
        interim / "property_clean.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            logger.info("maps: loaded %d property rows from %s", len(df), candidate)
            return df

    logger.warning("maps: no property parquet found — returning empty DataFrame")
    return pd.DataFrame()


def _synthetic_geo_df(n: int = 3000) -> pd.DataFrame:
    """Generate synthetic geo-located property data for demos.

    Spreads n points in a bounding box around Dublin with realistic price and
    NO₂ values so every map renders even before the pipeline has run.

    Args:
        n: Number of synthetic records to generate.

    Returns:
        DataFrame with columns: lat, lon, price, mean_no2_year, postal_code,
        address_clean, year_of_sale.
    """
    rng = np.random.default_rng(99)
    lat = rng.uniform(53.27, 53.43, n)
    lon = rng.uniform(-6.43, -6.07, n)
    # NO₂ higher near city centre (decreases with distance from centroid)
    dist_km = np.sqrt((lat - _DUBLIN_LAT) ** 2 + (lon - _DUBLIN_LON) ** 2) * 111
    no2 = np.clip(55 - 1.8 * dist_km + rng.normal(0, 5, n), 5, 80)
    # Price negatively correlated with NO₂, positively with distance from centre
    # (suburbs can be expensive too — add noise)
    price = np.exp(12.5 - 0.007 * no2 + rng.normal(0, 0.45, n))

    postcodes = ["Dublin " + str(i) for i in rng.integers(1, 24, n)]
    years = rng.integers(2015, 2025, n)

    return pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "price": price,
            "mean_no2_year": no2,
            "postal_code": postcodes,
            "address_clean": [f"Sample Address {i}" for i in range(n)],
            "year_of_sale": years,
        }
    )


# ---------------------------------------------------------------------------
# F4 — NO₂ map
# ---------------------------------------------------------------------------


def plot_f4_no2_map(df: pd.DataFrame) -> go.Figure:
    """F4: Mean NO₂ by location across Dublin, shown as a scatter-mapbox.

    If the DataFrame lacks geocoordinates or NO₂ data the function falls back
    to synthetic data so the figure is never blank.

    Args:
        df: Enriched property DataFrame.  Expected columns: ``lat``, ``lon``,
            ``mean_no2_year``.

    Returns:
        A Plotly Figure (scatter-mapbox, Viridis_r colour scale).
    """
    no2_col = "mean_no2_year"
    needed = {"lat", "lon", no2_col}

    if df.empty or not needed.issubset(df.columns) or df[no2_col].isna().all():
        df = _synthetic_geo_df()
        logger.warning("F4: using synthetic demo data")

    plot_df = (
        df.dropna(subset=["lat", "lon", no2_col])
        .sample(min(5_000, len(df)), random_state=42)
    )

    fig = px.scatter_mapbox(
        plot_df,
        lat="lat",
        lon="lon",
        color=no2_col,
        color_continuous_scale="Viridis_r",
        title="F4: Mean Annual NO₂ Levels Across Dublin Properties",
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": _DUBLIN_LAT, "lon": _DUBLIN_LON},
        opacity=0.65,
        size_max=8,
        labels={no2_col: "Mean NO₂ (μg/m³)"},
        template=TEMPLATE,
        hover_data={
            "lat": False,
            "lon": False,
            no2_col: ":.1f",
            "postal_code": True,
        } if "postal_code" in plot_df.columns else None,
    )
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    return fig


# ---------------------------------------------------------------------------
# F9 — Interactive Folium map
# ---------------------------------------------------------------------------


def _price_color(price: float, q25: float, q50: float, q75: float) -> str:
    """Map a price to a Viridis-inspired quartile colour.

    Args:
        price: Individual sale price.
        q25: 25th-percentile price.
        q50: 50th-percentile price.
        q75: 75th-percentile price.

    Returns:
        Hex colour string.
    """
    if price < q25:
        return "#440154"  # dark purple — cheapest
    elif price < q50:
        return "#31688e"  # steel blue
    elif price < q75:
        return "#35b779"  # green
    return "#fde725"      # yellow — most expensive


def create_f9_folium_map(
    df: pd.DataFrame,
    db=None,
) -> folium.Map:
    """F9: Interactive Dublin map with properties, parks, and AQ stations.

    Property circles are coloured by price quartile (Viridis palette).
    Park centroids (from MongoDB ``raw_green``) are shown in green.
    Air-quality stations (from MongoDB ``raw_air``) are shown in orange.

    Args:
        df: Enriched property DataFrame.  Must contain ``lat``, ``lon``,
            ``price``.  Falls back to synthetic data if absent.
        db: PyMongo Database object.  If ``None``, MongoDB overlays are
            silently skipped.

    Returns:
        A :class:`folium.Map` instance ready to be saved as HTML.
    """
    needed = {"lat", "lon", "price"}
    if df.empty or not needed.issubset(df.columns):
        df = _synthetic_geo_df()
        logger.warning("F9: using synthetic property data")

    sample = (
        df.dropna(subset=["lat", "lon", "price"])
        .sample(min(2_000, len(df)), random_state=42)
    )

    q25 = float(sample["price"].quantile(0.25))
    q50 = float(sample["price"].quantile(0.50))
    q75 = float(sample["price"].quantile(0.75))

    # Base map
    m = folium.Map(
        location=[_DUBLIN_LAT, _DUBLIN_LON],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # -- Property circles -------------------------------------------------
    prop_group = folium.FeatureGroup(name="Properties", show=True)
    for row in sample.itertuples(index=False):
        colour = _price_color(row.price, q25, q50, q75)
        addr = getattr(row, "address_clean", "") or ""
        popup_text = (
            f"<b>€{row.price:,.0f}</b><br>"
            f"{str(addr)[:50]}<br>"
            f"Year: {getattr(row, 'year_of_sale', 'N/A')}"
        )
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=4,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            weight=0,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(prop_group)
    prop_group.add_to(m)

    # -- Park markers from MongoDB ----------------------------------------
    if db is not None:
        park_group = folium.FeatureGroup(name="Green Spaces", show=True)
        try:
            parks = list(
                db.raw_green.find(
                    {},
                    {"_id": 0, "name": 1, "geometry": 1, "properties": 1},
                ).limit(300)
            )
            added = 0
            for park in parks:
                geom = park.get("geometry") or {}
                coords = None

                if geom.get("type") == "Point":
                    coords = geom["coordinates"]  # [lon, lat]
                elif geom.get("type") in {"Polygon", "MultiPolygon"}:
                    # Use first ring centroid as a representative point
                    try:
                        ring = (
                            geom["coordinates"][0]
                            if geom["type"] == "Polygon"
                            else geom["coordinates"][0][0]
                        )
                        lons = [c[0] for c in ring]
                        lats = [c[1] for c in ring]
                        coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
                    except (IndexError, KeyError, TypeError):
                        pass

                if coords is None:
                    continue

                lon_p, lat_p = coords[0], coords[1]
                name = (
                    park.get("name")
                    or (park.get("properties") or {}).get("NAME", "Park")
                )
                folium.CircleMarker(
                    location=[lat_p, lon_p],
                    radius=6,
                    color="#1a9641",
                    fill=True,
                    fill_color="#1a9641",
                    fill_opacity=0.55,
                    weight=1,
                    popup=folium.Popup(str(name), max_width=150),
                ).add_to(park_group)
                added += 1

            logger.info("F9: added %d park markers", added)
        except Exception as exc:
            logger.warning("F9: could not add park markers — %s", exc)
        park_group.add_to(m)

    # -- Air quality station markers from MongoDB -------------------------
    if db is not None:
        aq_group = folium.FeatureGroup(name="AQ Stations", show=True)
        try:
            stations = list(
                db.raw_air.aggregate(
                    [
                        {"$group": {
                            "_id": "$monitor_id",
                            "name": {"$first": "$station_name"},
                            "lat": {"$first": "$lat"},
                            "lon": {"$first": "$lon"},
                        }},
                        {"$limit": 60},
                    ]
                )
            )
            for st in stations:
                lat_s = st.get("lat")
                lon_s = st.get("lon")
                if lat_s is None or lon_s is None:
                    continue
                try:
                    lat_s, lon_s = float(lat_s), float(lon_s)
                except (TypeError, ValueError):
                    continue
                folium.Marker(
                    location=[lat_s, lon_s],
                    icon=folium.Icon(color="orange", icon="cloud", prefix="fa"),
                    popup=folium.Popup(
                        f"<b>AQ Station</b><br>{st.get('name', st['_id'])}",
                        max_width=180,
                    ),
                ).add_to(aq_group)
            logger.info("F9: added %d AQ station markers", len(stations))
        except Exception as exc:
            logger.warning("F9: could not add AQ station markers — %s", exc)
        aq_group.add_to(m)

    # -- Layer control & legend -------------------------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px; left: 50px;
        background: white;
        padding: 12px 16px;
        border: 1px solid #aaa;
        border-radius: 6px;
        z-index: 9999;
        font-family: sans-serif;
        font-size: 13px;
        line-height: 1.6;
        box-shadow: 2px 2px 6px rgba(0,0,0,.15);
    ">
    <b>Property Price Quartile</b><br>
    <span style="color:#440154;font-size:16px">&#9679;</span> Q1 — lowest (&lt; 25th pct)<br>
    <span style="color:#31688e;font-size:16px">&#9679;</span> Q2 — 25–50th pct<br>
    <span style="color:#35b779;font-size:16px">&#9679;</span> Q3 — 50–75th pct<br>
    <span style="color:#fde725;font-size:16px">&#9679;</span> Q4 — highest (&gt; 75th pct)<br>
    <hr style="margin:6px 0">
    <span style="color:#1a9641;font-size:16px">&#9679;</span> Green Space<br>
    <span style="color:orange;font-size:16px">&#9632;</span> AQ Station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run() -> None:
    """Generate F4 (Plotly) and F9 (Folium) and save to report/figures/.

    F4 is saved as PNG (with HTML fallback if kaleido is absent).
    F9 is saved as a self-contained HTML file.
    """
    # Re-read env vars at call time (not import time) so --demo overrides work
    fig_dir = Path(_os.environ.get("GREEN_PREMIUM_FIGURE_DIR", PROJECT_ROOT / "report" / "figures"))
    scale = int(_os.environ.get("GREEN_PREMIUM_FIGURE_SCALE", "3"))

    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_property_df()

    # F4 — NO₂ map
    f4 = plot_f4_no2_map(df)
    f4_png = fig_dir / "F4_no2_map.png"
    try:
        f4.write_image(str(f4_png), scale=scale, width=1200, height=700)
        logger.info("Saved F4 -> %s", f4_png)
    except Exception as exc:
        logger.warning("Cannot write F4 PNG (%s) -- saving HTML fallback", exc)
        f4_html = f4_png.with_suffix(".html")
        f4.write_html(str(f4_html))
        logger.info("Saved F4 fallback -> %s", f4_html)

    # F9 — Folium interactive map
    db = None
    if not _os.environ.get("GREEN_PREMIUM_DATA_DIR"):  # skip MongoDB in demo
        try:
            db = get_mongo_db()
        except Exception as exc:
            logger.warning("F9: MongoDB unavailable (%s) — map will omit overlays", exc)

    has_geo = not df.empty and {"lat", "lon", "price"}.issubset(df.columns)
    map_df = df if has_geo else _synthetic_geo_df()

    f9_map = create_f9_folium_map(map_df, db)
    f9_path = fig_dir / "F9_dublin_map.html"
    f9_map.save(str(f9_path))
    logger.info("Saved F9 -> %s", f9_path)

    logger.info("maps.run() complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
