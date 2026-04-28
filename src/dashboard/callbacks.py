"""Dashboard callbacks — interactivity for the Green Premium Dublin app.

All callbacks filter the globally-loaded ``DF`` DataFrame in-memory.
No database calls occur per user interaction.

Registered controls:
    - year-slider      : RangeSlider filtering year_of_sale
    - postcode-select  : Multi-select dropdown filtering postal_code
    - pollutant-select : Single dropdown selecting which AQ column to colour
    - download-btn     : Triggers CSV download of the filtered dataset
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, callback, dcc

from src.dashboard.app import DF
from src.dashboard.layout import (
    _correlation_heatmap,
    _green_quintile_box,
    _no2_scatter,
    _park_scatter,
    _price_histogram,
    _temporal_lines,
    _TEMPLATE,
    _VIRIDIS,
)

# ---------------------------------------------------------------------------
# Filtered-data helper
# ---------------------------------------------------------------------------


def _filter(
    year_range: list[int] | None,
    postcodes: list[str] | None,
) -> pd.DataFrame:
    """Apply year-range and postcode filters to the global DataFrame.

    Args:
        year_range: [min_year, max_year] from the year slider.
        postcodes: List of selected postal_code strings, or empty/None for all.

    Returns:
        Filtered copy of DF.
    """
    df = DF.copy()
    if df.empty:
        return df

    if year_range and "year_of_sale" in df.columns:
        df = df[
            (df["year_of_sale"] >= year_range[0])
            & (df["year_of_sale"] <= year_range[1])
        ]

    if postcodes and "postal_code" in df.columns:
        df = df[df["postal_code"].isin(postcodes)]

    return df


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("fig-histogram", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def update_histogram(year_range: list[int], postcodes: list[str]) -> go.Figure:
    """Update price histogram on filter change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.

    Returns:
        Updated Plotly Figure.
    """
    return _price_histogram(_filter(year_range, postcodes))


@callback(
    Output("fig-park-scatter", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def update_park_scatter(year_range: list[int], postcodes: list[str]) -> go.Figure:
    """Update price vs park distance scatter on filter change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.

    Returns:
        Updated Plotly Figure.
    """
    return _park_scatter(_filter(year_range, postcodes))


@callback(
    Output("fig-green-box", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def update_green_box(year_range: list[int], postcodes: list[str]) -> go.Figure:
    """Update green-quintile boxplot on filter change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.

    Returns:
        Updated Plotly Figure.
    """
    return _green_quintile_box(_filter(year_range, postcodes))


@callback(
    Output("fig-temporal", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def update_temporal(year_range: list[int], postcodes: list[str]) -> go.Figure:
    """Update temporal lines on filter change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.

    Returns:
        Updated Plotly Figure.
    """
    return _temporal_lines(_filter(year_range, postcodes))


@callback(
    Output("fig-heatmap", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def update_heatmap(year_range: list[int], postcodes: list[str]) -> go.Figure:
    """Update correlation heatmap on filter change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.

    Returns:
        Updated Plotly Figure.
    """
    return _correlation_heatmap(_filter(year_range, postcodes))


@callback(
    Output("fig-no2-scatter", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    Input("pollutant-select", "value"),
    prevent_initial_call=True,
)
def update_no2_scatter(
    year_range: list[int], postcodes: list[str], pollutant_col: str
) -> go.Figure:
    """Update AQ scatter on filter or pollutant change.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.
        pollutant_col: Column name for selected pollutant.

    Returns:
        Updated Plotly Figure.
    """
    df = _filter(year_range, postcodes)
    if df.empty or pollutant_col not in df.columns:
        return go.Figure()

    label_map = {
        "mean_no2_year": "Mean Annual NO₂ (µg/m³)",
        "mean_pm10_year": "Mean Annual PM10 (µg/m³)",
        "mean_pm25_year": "Mean Annual PM2.5 (µg/m³)",
        "mean_noise_db_year": "Mean Annual Noise (dB LAeq)",
    }
    x_label = label_map.get(pollutant_col, pollutant_col)

    sub = df[[pollutant_col, "price"]].dropna().sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sub,
        x=pollutant_col,
        y="price",
        log_y=True,
        trendline="ols",
        template=_TEMPLATE,
        title=f"Price vs {x_label}",
        labels={pollutant_col: x_label, "price": "Sale Price (€, log scale)"},
        color_discrete_sequence=[px.colors.sequential.Reds[5]],
        opacity=0.4,
    )
    return fig


@callback(
    Output("fig-no2-map", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    Input("pollutant-select", "value"),
    prevent_initial_call=True,
)
def update_aq_map(
    year_range: list[int], postcodes: list[str], pollutant_col: str
) -> go.Figure:
    """Build scatter mapbox coloured by selected pollutant.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.
        pollutant_col: Column name for selected pollutant.

    Returns:
        Updated Plotly Figure.
    """
    df = _filter(year_range, postcodes)
    if df.empty or "lat" not in df.columns or pollutant_col not in df.columns:
        return go.Figure()

    sub = df[["lat", "lon", "price", pollutant_col]].dropna().sample(
        min(3000, len(df)), random_state=42
    )
    label_map = {
        "mean_no2_year": "NO₂",
        "mean_pm10_year": "PM10",
        "mean_pm25_year": "PM2.5",
        "mean_noise_db_year": "Noise",
    }
    label = label_map.get(pollutant_col, pollutant_col)

    fig = px.scatter_mapbox(
        sub,
        lat="lat",
        lon="lon",
        color=pollutant_col,
        size="price",
        size_max=10,
        color_continuous_scale="Viridis",
        zoom=10,
        center={"lat": 53.35, "lon": -6.26},
        template=_TEMPLATE,
        title=f"F4 — {label} by Property Location",
        labels={pollutant_col: label, "price": "Sale Price (€)"},
        hover_data={"lat": False, "lon": False, "price": ":,.0f"},
        mapbox_style="open-street-map",
        opacity=0.6,
    )
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=500)
    return fig


@callback(
    Output("fig-map", "figure"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    Input("pollutant-select", "value"),
    prevent_initial_call=False,
)
def update_main_map(
    year_range: list[int], postcodes: list[str], pollutant_col: str
) -> go.Figure:
    """Build main geographic map (F9 equivalent) coloured by price.

    Args:
        year_range: Selected [min, max] years.
        postcodes: Selected postcode list.
        pollutant_col: Column name for AQ overlay (not used in this map).

    Returns:
        Updated Plotly Figure.
    """
    df = _filter(year_range, postcodes)
    if df.empty or "lat" not in df.columns:
        return go.Figure()

    sub = df[["lat", "lon", "price", "address", "postal_code", "year_of_sale"]].dropna(
        subset=["lat", "lon", "price"]
    ).sample(min(4000, len(df)), random_state=42)

    fig = px.scatter_mapbox(
        sub,
        lat="lat",
        lon="lon",
        color="price",
        color_continuous_scale="Viridis",
        zoom=10,
        center={"lat": 53.35, "lon": -6.26},
        template=_TEMPLATE,
        title="F9 — Dublin Property Prices",
        labels={"price": "Sale Price (€)"},
        hover_data={
            "lat": False,
            "lon": False,
            "price": ":,.0f",
            "postal_code": True,
            "year_of_sale": True,
        },
        mapbox_style="open-street-map",
        opacity=0.6,
        size_max=8,
    )
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=560)
    return fig


@callback(
    Output("fig-forest", "figure"),
    Input("year-slider", "value"),
    prevent_initial_call=False,
)
def update_forest_plot(year_range: list[int]) -> go.Figure:
    """Build OLS coefficient forest plot (F8) from saved results parquet.

    Args:
        year_range: Not used for this plot (OLS is computed on full data).

    Returns:
        Plotly Figure.
    """
    from pathlib import Path

    results_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "regression_results.parquet"
    if not results_path.exists():
        return go.Figure(layout={"title": "Run src.analysis.regression to generate model results."})

    coef_df = pd.read_parquet(results_path)

    # Filter to main model non-intercept, non-year, non-construction coefficients.
    mask = (
        ~coef_df["term"].str.startswith("C(", na=False)
        & ~coef_df["term"].str.startswith("Intercept", na=False)
        & coef_df["term"].notna()
    )
    plot_df = coef_df[mask].copy()

    if plot_df.empty:
        plot_df = coef_df.head(20)

    plot_df = plot_df.sort_values("coef")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["coef"],
            y=plot_df["term"],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(plot_df["ci_upper"] - plot_df["coef"]).tolist(),
                arrayminus=(plot_df["coef"] - plot_df["ci_lower"]).tolist(),
            ),
            marker=dict(size=8, color=_VIRIDIS[5]),
            hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title="F8 — OLS Regression Coefficients (95% CI)",
        xaxis_title="Coefficient (log-price units)",
        yaxis_title="",
        template=_TEMPLATE,
        height=500,
        margin=dict(t=50, b=40, l=200),
    )
    return fig


@callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    Input("year-slider", "value"),
    Input("postcode-select", "value"),
    prevent_initial_call=True,
)
def download_filtered_csv(
    n_clicks: int | None,
    year_range: list[int],
    postcodes: list[str],
) -> dict | None:
    """Export the filtered dataset as a CSV download.

    Args:
        n_clicks: Button click count (trigger).
        year_range: Active year range filter.
        postcodes: Active postcode filter.

    Returns:
        Dash dcc.send_data_frame dict, or None if button not clicked.
    """
    if not n_clicks:
        return None

    df = _filter(year_range, postcodes)
    return dcc.send_data_frame(df.to_csv, "green_premium_filtered.csv", index=False)
