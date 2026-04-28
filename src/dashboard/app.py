"""Dash application entry point for the Green Premium Dublin dashboard.

Initialises the Dash app, loads the enriched property dataset once at startup
(no DB round-trips per callback), and wires in the layout.

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
# Data load (once at startup — all callbacks filter this in-memory)
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed"
_ENRICHED_PATH = _DATA_PATH / "property_enriched.parquet"


def load_data() -> pd.DataFrame:
    """Load the enriched property dataset from parquet.

    Returns:
        DataFrame with all enriched columns, or an empty DataFrame if the
        file does not yet exist (pipeline not yet run).
    """
    if not _ENRICHED_PATH.exists():
        logger.warning(
            "Enriched parquet not found at %s — dashboard will show empty data. "
            "Run `python -m src.main` to build the pipeline first.",
            _ENRICHED_PATH,
        )
        return pd.DataFrame()

    df = pd.read_parquet(_ENRICHED_PATH)

    # Ensure date_of_sale is datetime for range filtering.
    if "date_of_sale" in df.columns:
        df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")

    logger.info("Loaded %d enriched property rows from %s", len(df), _ENRICHED_PATH)
    return df


# Load data globally so all callbacks share the same object.
DF: pd.DataFrame = load_data()

# ---------------------------------------------------------------------------
# Dash app initialisation
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Green Premium Dublin",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Expose the underlying Flask server for WSGI deployment if needed.
server = app.server

# Import layout and callbacks *after* app is created (avoids circular imports).
from src.dashboard.layout import build_layout  # noqa: E402
from src.dashboard import callbacks  # noqa: E402, F401

app.layout = build_layout(DF)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    app.run(debug=True, port=8050)
