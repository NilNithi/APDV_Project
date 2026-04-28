"""Dashboard layout — 6-tab structure for the Green Premium Dublin app.

Each tab maps to one section of the research:
    1. Overview        — KPIs + price distribution + scatter
    2. Geographic      — Interactive map + year/pollutant filters
    3. Green Premium   — Boxplot by quintile, temporal lines, correlation heatmap
    4. Air Quality     — NO₂ choropleth / scatter, PM2.5 scatter
    5. Statistical     — OLS forest plot + coefficient table
    6. Methodology     — Pipeline description

All figures are loaded from pre-rendered PNG files in ``report/figures/``.
Interactive Plotly figures are generated on-the-fly from the enriched DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import dcc, html

_FIGURES_DIR = Path(__file__).resolve().parents[2] / "report" / "figures"
_TEMPLATE = "plotly_white"
_VIRIDIS = px.colors.sequential.Viridis


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _fig_img(filename: str, style: dict | None = None) -> html.Img:
    """Return an <img> tag pointing to a pre-rendered figure PNG.

    Args:
        filename: Filename (not path) of the PNG in ``report/figures/``.
        style: Optional CSS style dict.

    Returns:
        Dash html.Img component.
    """
    return html.Img(
        src=f"/assets/figures/{filename}",
        style=style or {"width": "100%", "height": "auto"},
    )


def _kpi_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    """Build a KPI metric card.

    Args:
        title: Card header label.
        value: Formatted value string.
        color: Bootstrap colour name.

    Returns:
        dbc.Card component.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(title, className="card-title text-muted small"),
                html.H4(value, className=f"text-{color} fw-bold"),
            ]
        ),
        className="shadow-sm h-100",
    )


def _price_histogram(df: pd.DataFrame) -> go.Figure:
    """Build F1 — property price histogram with log x-axis.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty or "price" not in df.columns:
        return go.Figure()

    prices = df["price"].dropna()
    prices = prices[prices > 0]

    fig = px.histogram(
        prices,
        x=prices,
        nbins=80,
        log_x=True,
        template=_TEMPLATE,
        title="F1 — Distribution of Dublin Property Prices",
        labels={"x": "Sale Price (€, log scale)", "y": "Count"},
        color_discrete_sequence=[_VIRIDIS[3]],
    )
    median_val = float(prices.median())
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median €{median_val:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, b=40))
    return fig


def _park_scatter(df: pd.DataFrame) -> go.Figure:
    """Build F2 — price vs distance to nearest park.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty or "nearest_park_dist_m" not in df.columns:
        return go.Figure()

    sub = df[["nearest_park_dist_m", "price"]].dropna().sample(
        min(5000, len(df)), random_state=42
    )
    fig = px.scatter(
        sub,
        x="nearest_park_dist_m",
        y="price",
        log_y=True,
        trendline="ols",
        template=_TEMPLATE,
        title="F2 — Price vs Distance to Nearest Park",
        labels={
            "nearest_park_dist_m": "Distance to Nearest Park (m)",
            "price": "Sale Price (€, log scale)",
        },
        color_discrete_sequence=[_VIRIDIS[5]],
        opacity=0.4,
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


def _green_quintile_box(df: pd.DataFrame) -> go.Figure:
    """Build F3 — price boxplot by green-area-within-500m quintile.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty or "green_area_within_500m" not in df.columns:
        return go.Figure()

    sub = df[["green_area_within_500m", "price"]].dropna()
    sub = sub[sub["price"] > 0]
    try:
        sub["quintile"] = pd.qcut(
            sub["green_area_within_500m"], q=5,
            labels=["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"],
        )
    except ValueError:
        # Duplicate bin edges when many values are equal — fall back without labels.
        sub["quintile"] = pd.qcut(sub["green_area_within_500m"], q=5, duplicates="drop")
    fig = px.box(
        sub,
        x="quintile",
        y="price",
        log_y=True,
        template=_TEMPLATE,
        title="F3 — Price by Green Area Within 500 m (Quintiles)",
        labels={
            "quintile": "Green Area Quintile",
            "price": "Sale Price (€, log scale)",
        },
        color="quintile",
        color_discrete_sequence=_VIRIDIS[::2][:5],
        notched=True,
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, b=40))
    return fig


def _no2_scatter(df: pd.DataFrame) -> go.Figure:
    """Build F5 — price vs mean annual NO₂.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty or "mean_no2_year" not in df.columns:
        return go.Figure()

    sub = df[["mean_no2_year", "price"]].dropna().sample(
        min(5000, len(df)), random_state=42
    )
    fig = px.scatter(
        sub,
        x="mean_no2_year",
        y="price",
        log_y=True,
        trendline="ols",
        template=_TEMPLATE,
        title="F5 — Price vs Mean Annual NO₂",
        labels={
            "mean_no2_year": "Mean Annual NO₂ (µg/m³)",
            "price": "Sale Price (€, log scale)",
        },
        color_discrete_sequence=[px.colors.sequential.Reds[5]],
        opacity=0.4,
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


def _temporal_lines(df: pd.DataFrame) -> go.Figure:
    """Build F6 — year-over-year median price by green-proximity quintile.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty or "nearest_park_dist_m" not in df.columns:
        return go.Figure()

    sub = df[["year_of_sale", "nearest_park_dist_m", "price"]].dropna()
    sub = sub[sub["price"] > 0]
    try:
        sub["quintile"] = pd.qcut(
            sub["nearest_park_dist_m"], q=5,
            labels=["Q1 (closest)", "Q2", "Q3", "Q4", "Q5 (furthest)"],
        )
    except ValueError:
        sub["quintile"] = pd.qcut(sub["nearest_park_dist_m"], q=5, duplicates="drop")
    grouped = (
        sub.groupby(["year_of_sale", "quintile"], observed=True)["price"]
        .median()
        .reset_index()
    )
    fig = px.line(
        grouped,
        x="year_of_sale",
        y="price",
        color="quintile",
        template=_TEMPLATE,
        title="F6 — Median Price by Green-Proximity Quintile Over Time",
        labels={
            "year_of_sale": "Year",
            "price": "Median Sale Price (€)",
            "quintile": "Park Distance Quintile",
        },
        color_discrete_sequence=_VIRIDIS[::2][:5],
        markers=True,
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


def _correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Build F7 — correlation heatmap of numeric features.

    Args:
        df: Enriched property DataFrame.

    Returns:
        Plotly Figure.
    """
    if df.empty:
        return go.Figure()

    cols = [
        "log_price",
        "nearest_park_dist_m",
        "green_area_within_500m",
        "green_area_within_1000m",
        "mean_no2_year",
        "mean_pm10_year",
        "mean_noise_db_year",
    ]
    available = [c for c in cols if c in df.columns]
    sub = df[available].dropna()

    if len(sub) < 10:
        return go.Figure()

    corr = sub.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=_TEMPLATE,
        title="F7 — Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=50, b=40))
    return fig


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _tab_overview(df: pd.DataFrame) -> dbc.Tab:
    """Build the Overview tab.

    Args:
        df: Enriched property DataFrame.

    Returns:
        dbc.Tab component.
    """
    n_sales = f"{len(df):,}" if not df.empty else "—"
    median_price = (
        f"€{df['price'].median():,.0f}" if not df.empty and "price" in df.columns else "—"
    )
    mean_no2 = (
        f"{df['mean_no2_year'].mean():.1f} µg/m³"
        if not df.empty and "mean_no2_year" in df.columns and df["mean_no2_year"].notna().any()
        else "—"
    )
    n_stations = (
        str(df["nearest_air_station"].nunique())
        if not df.empty and "nearest_air_station" in df.columns
        else "—"
    )

    return dbc.Tab(
        label="Overview",
        tab_id="tab-overview",
        children=dbc.Container(
            [
                html.H4("Dublin Property Market Overview", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(_kpi_card("Total Sales", n_sales, "primary"), md=3),
                        dbc.Col(_kpi_card("Median Price", median_price, "success"), md=3),
                        dbc.Col(_kpi_card("Mean Annual NO₂", mean_no2, "warning"), md=3),
                        dbc.Col(_kpi_card("Air Stations", n_stations, "info"), md=3),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="fig-histogram", figure=_price_histogram(df)), md=6),
                        dbc.Col(dcc.Graph(id="fig-park-scatter", figure=_park_scatter(df)), md=6),
                    ]
                ),
            ],
            fluid=True,
        ),
    )


def _tab_geographic(df: pd.DataFrame) -> dbc.Tab:
    """Build the Geographic Explorer tab.

    Args:
        df: Enriched property DataFrame.

    Returns:
        dbc.Tab component.
    """
    years = sorted(df["year_of_sale"].dropna().unique().tolist()) if not df.empty else []
    year_min = int(min(years)) if years else 2015
    year_max = int(max(years)) if years else 2021

    return dbc.Tab(
        label="Geographic",
        tab_id="tab-geographic",
        children=dbc.Container(
            [
                html.H4("Geographic Explorer", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Year Range:"),
                                dcc.RangeSlider(
                                    id="year-slider",
                                    min=year_min,
                                    max=year_max,
                                    step=1,
                                    value=[year_min, year_max],
                                    marks={y: str(y) for y in range(year_min, year_max + 1)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Label("Pollutant:"),
                                dcc.Dropdown(
                                    id="pollutant-select",
                                    options=[
                                        {"label": "NO₂", "value": "mean_no2_year"},
                                        {"label": "PM10", "value": "mean_pm10_year"},
                                        {"label": "PM2.5", "value": "mean_pm25_year"},
                                        {"label": "Noise (LAeq)", "value": "mean_noise_db_year"},
                                    ],
                                    value="mean_no2_year",
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="fig-map",
                                style={"height": "600px"},
                            ),
                            md=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        ),
    )


def _tab_green(df: pd.DataFrame) -> dbc.Tab:
    """Build the Green Premium Analysis tab.

    Args:
        df: Enriched property DataFrame.

    Returns:
        dbc.Tab component.
    """
    return dbc.Tab(
        label="Green Premium",
        tab_id="tab-green",
        children=dbc.Container(
            [
                html.H4("Green Space Premium Analysis", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="fig-green-box", figure=_green_quintile_box(df)), md=6),
                        dbc.Col(dcc.Graph(id="fig-temporal", figure=_temporal_lines(df)), md=6),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="fig-heatmap", figure=_correlation_heatmap(df)),
                            md=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        ),
    )


def _tab_air(df: pd.DataFrame) -> dbc.Tab:
    """Build the Air Quality Analysis tab.

    Args:
        df: Enriched property DataFrame.

    Returns:
        dbc.Tab component.
    """
    return dbc.Tab(
        label="Air Quality",
        tab_id="tab-air",
        children=dbc.Container(
            [
                html.H4("Air Quality & Property Price", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="fig-no2-scatter", figure=_no2_scatter(df)),
                            md=12,
                        )
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="fig-no2-map"),
                            md=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        ),
    )


def _tab_model() -> dbc.Tab:
    """Build the Statistical Model tab.

    Returns:
        dbc.Tab component.
    """
    ols_summary_path = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "ols_summary.txt"
    )
    ols_text = ""
    if ols_summary_path.exists():
        ols_text = ols_summary_path.read_text(encoding="utf-8")[:3000]

    return dbc.Tab(
        label="Statistical Model",
        tab_id="tab-model",
        children=dbc.Container(
            [
                html.H4("OLS Regression: Log(Price) ~ Green + Air Quality", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id="fig-forest"),
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("OLS Summary", className="mb-2"),
                                html.Pre(
                                    ols_text or "Run src.analysis.regression to generate summary.",
                                    style={
                                        "fontSize": "0.75rem",
                                        "maxHeight": "500px",
                                        "overflowY": "auto",
                                        "background": "#f8f9fa",
                                        "padding": "12px",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
    )


def _tab_methodology() -> dbc.Tab:
    """Build the Methodology tab.

    Returns:
        dbc.Tab component.
    """
    return dbc.Tab(
        label="Methodology",
        tab_id="tab-methodology",
        children=dbc.Container(
            [
                html.H4("Data Pipeline & Methodology", className="mt-3 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Architecture"),
                                html.P(
                                    "Three datasets are ingested: (1) PSRA property price "
                                    "CSVs from data.smartdublin.ie into PostgreSQL, (2) "
                                    "Sonitus air quality API readings into MongoDB, and (3) "
                                    "DCC green space GeoJSON into MongoDB. Each passes through "
                                    "an ETL stage before being spatially joined into a single "
                                    "enriched table."
                                ),
                                html.H5("Geocoding Strategy"),
                                html.P(
                                    "Property addresses are geocoded via a three-tier strategy: "
                                    "(1) SQLite cache for seen addresses, (2) Nominatim OSM "
                                    "geocoder at 1 req/s, (3) Dublin postcode centroid fallback. "
                                    "This achieves <30% null rate."
                                ),
                                html.H5("Spatial Join"),
                                html.P(
                                    "All spatial operations use EPSG:2157 (Irish Transverse "
                                    "Mercator) for accurate metric distances. Nearest-park "
                                    "distance uses geopandas.sjoin_nearest with an R-tree index. "
                                    "Green area within 500 m and 1000 m buffers uses polygon "
                                    "intersection. Air quality assignment uses a KD-tree "
                                    "(scipy.spatial.cKDTree) to find the nearest monitoring "
                                    "station, matched by year of sale."
                                ),
                                html.H5("Statistical Modelling"),
                                html.P(
                                    "OLS regression via statsmodels with HC3 robust standard "
                                    "errors. Dependent variable: log(price). Key predictors: "
                                    "log(nearest_park_dist_m), green_area_within_500m, "
                                    "mean_no2_year, year_of_sale fixed effects, construction type."
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.H5("Databases"),
                                dbc.Table(
                                    [
                                        html.Thead(
                                            html.Tr(
                                                [
                                                    html.Th("Store"),
                                                    html.Th("Raw"),
                                                    html.Th("Processed"),
                                                ]
                                            )
                                        ),
                                        html.Tbody(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Td("PostgreSQL"),
                                                        html.Td("raw.property"),
                                                        html.Td("processed.property_enriched"),
                                                    ]
                                                ),
                                                html.Tr(
                                                    [
                                                        html.Td("MongoDB"),
                                                        html.Td("raw_air / raw_green"),
                                                        html.Td("processed_air / processed_property"),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    bordered=True,
                                    size="sm",
                                    className="mt-2",
                                ),
                            ],
                            md=4,
                        ),
                    ]
                ),
            ],
            fluid=True,
        ),
    )


# ---------------------------------------------------------------------------
# Filters sidebar / controls (shared across tabs via callbacks)
# ---------------------------------------------------------------------------


def _control_row(df: pd.DataFrame) -> dbc.Row:
    """Build the global filter controls rendered above the tabs.

    Args:
        df: Enriched property DataFrame (used to derive dropdown options).

    Returns:
        dbc.Row with postcode dropdown and CSV download button.
    """
    postcodes = (
        sorted(df["postal_code"].dropna().unique().tolist())
        if not df.empty and "postal_code" in df.columns
        else []
    )
    options = [{"label": p, "value": p} for p in postcodes]

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Filter by Postcode:", className="small fw-bold"),
                    dcc.Dropdown(
                        id="postcode-select",
                        options=options,
                        multi=True,
                        placeholder="All postcodes…",
                        style={"fontSize": "0.85rem"},
                    ),
                ],
                md=8,
            ),
            dbc.Col(
                [
                    html.Br(),
                    dbc.Button(
                        "⬇ Download CSV",
                        id="download-btn",
                        color="secondary",
                        size="sm",
                        className="mt-1",
                    ),
                    dcc.Download(id="download-csv"),
                ],
                md=4,
                className="d-flex align-items-end",
            ),
        ],
        className="mb-3 p-2 bg-light rounded",
    )


# ---------------------------------------------------------------------------
# Top-level layout builder
# ---------------------------------------------------------------------------


def build_layout(df: pd.DataFrame) -> dbc.Container:
    """Assemble the full dashboard layout.

    Args:
        df: Enriched property DataFrame loaded at startup.

    Returns:
        Root dbc.Container component assigned to ``app.layout``.
    """
    return dbc.Container(
        [
            # Header
            dbc.Row(
                dbc.Col(
                    html.H2(
                        "🌿 The Green Premium — Dublin Property & Environment",
                        className="text-center py-3 text-success",
                    )
                )
            ),
            html.Hr(),
            # Global filters
            _control_row(df),
            # Tabs
            dbc.Tabs(
                [
                    _tab_overview(df),
                    _tab_geographic(df),
                    _tab_green(df),
                    _tab_air(df),
                    _tab_model(),
                    _tab_methodology(),
                ],
                id="main-tabs",
                active_tab="tab-overview",
            ),
        ],
        fluid=True,
        className="px-4",
    )
