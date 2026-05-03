"""Dash application entry point for the Green Premium Dublin dashboard.

Usage::

    python -m src.dashboard.app

The dashboard will be available at http://localhost:8050.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import dash
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data load (once at startup)
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
_ENRICHED_PATH = _DATA_PATH / "property_enriched.parquet"


def load_data() -> pd.DataFrame:
    if not _ENRICHED_PATH.exists():
        logger.warning("Enriched parquet not found at %s", _ENRICHED_PATH)
        return pd.DataFrame()
    df = pd.read_parquet(_ENRICHED_PATH)
    if "date_of_sale" in df.columns:
        df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")
    logger.info("Loaded %d enriched property rows from %s", len(df), _ENRICHED_PATH)
    return df


DF: pd.DataFrame = load_data()

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Green Premium Dublin",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    serve_locally=True,
)

server = app.server

# Import layout and register callbacks *after* app is created.
from src.dashboard.layout import build_layout  # noqa: E402
from src.dashboard.callbacks import register_callbacks  # noqa: E402

app.layout = build_layout(DF)
register_callbacks(app, DF)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )
    app.run(debug=False, port=8050)
