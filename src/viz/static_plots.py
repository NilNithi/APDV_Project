"""Generate static publication figures F1–F8 as PNG files.

Each function returns a :class:`plotly.graph_objects.Figure` and is
intentionally side-effect-free (no file I/O). The :func:`run` orchestrator
calls all of them and persists the results to ``report/figures/``.

Figures produced
----------------
F1 – Distribution of Dublin property prices (histogram, log-x)
F2 – Price vs distance to nearest park (scatter + OLS trendline)
F3 – Price by green-area-within-500m quintile (boxplot)
F5 – Price vs mean annual NO₂ (scatter + OLS trendline)
F6 – Year-over-year median price by green-proximity quintile (line)
F7 – Feature correlation heatmap (Pearson r)
F8 – OLS regression coefficients with 95 % CI (forest plot)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants (shared with maps.py)
# ---------------------------------------------------------------------------
TEMPLATE: str = "plotly_white"
VIRIDIS: str = "Viridis"
CATEGORICAL: str = "Plotly"  # px.colors.qualitative.Plotly
FIGURE_DIR: Path = PROJECT_ROOT / "report" / "figures"
SCALE: int = 3  # ×3 -> ~300 DPI equivalent at 1200×700 px canvas


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load enriched property data and pre-computed analysis results.

    Falls back gracefully through the available data files so the visualisation
    layer still functions even if the full ETL has not been run yet.

    Returns:
        A 2-tuple of:
        - ``df``: enriched property DataFrame (may be empty).
        - ``results``: mapping of analysis-result name -> DataFrame.
    """
    processed = PROJECT_ROOT / "data" / "processed"
    interim = PROJECT_ROOT / "data" / "interim"

    # Property data — prefer fully enriched, fall back to cleaned
    for candidate in [
        processed / "property_enriched.parquet",
        interim / "property_clean.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            logger.info("Loaded property data from %s (%d rows)", candidate, len(df))
            break
    else:
        logger.warning("No property parquet found — using empty DataFrame")
        df = pd.DataFrame()

    # Analysis result artefacts
    result_names = [
        "correlation_pearson",
        "regression_results",
        "temporal_median_price",
        "descriptive_quintiles",
    ]
    results: dict[str, pd.DataFrame] = {}
    for name in result_names:
        p = processed / f"{name}.parquet"
        if p.exists():
            results[name] = pd.read_parquet(p)
            logger.info("Loaded analysis result: %s", name)

    return df, results


# ---------------------------------------------------------------------------
# Figure functions — F1 through F8 (F4 lives in maps.py as a choropleth/map)
# ---------------------------------------------------------------------------


def plot_f1_price_distribution(df: pd.DataFrame) -> go.Figure:
    """F1: Histogram of property prices with a log x-axis.

    Justifies the log-transform applied before OLS regression by showing
    the strong right-skew of the raw price distribution.

    Args:
        df: Enriched property DataFrame. Must contain a ``price`` column.

    Returns:
        A Plotly Figure (histogram, log-x scale).
    """
    plot_df = df[df["price"] > 0].copy() if not df.empty and "price" in df.columns else pd.DataFrame({"price": []})

    fig = px.histogram(
        plot_df,
        x="price",
        nbins=80,
        log_x=True,
        title="F1: Distribution of Dublin Residential Property Prices (2015–2024)",
        labels={"price": "Sale Price (€, log scale)", "count": "Number of Properties"},
        template=TEMPLATE,
        color_discrete_sequence=["#2E86AB"],
    )

    if not plot_df.empty:
        median_price = float(plot_df["price"].median())
        fig.add_vline(
            x=median_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: €{median_price:,.0f}",
            annotation_position="top right",
        )

    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


def plot_f2_price_vs_park_dist(df: pd.DataFrame) -> go.Figure:
    """F2: Scatter of property price vs distance to nearest park.

    A negative trendline slope would provide direct evidence for a green
    premium — i.e. prices fall as distance to green space increases.

    Args:
        df: Enriched property DataFrame. Must contain ``nearest_park_dist_m``
            and ``price`` columns.

    Returns:
        A Plotly Figure (scatter + OLS trendline, log y-axis).
    """
    dist_col = "nearest_park_dist_m"
    needed = {"price", dist_col}

    if df.empty or not needed.issubset(df.columns):
        # Synthetic demo data so the dashboard tab is never blank
        rng = np.random.default_rng(42)
        n = 2000
        dist = rng.uniform(50, 2000, n)
        price = np.exp(12.5 - 0.0002 * dist + rng.normal(0, 0.4, n))
        df = pd.DataFrame({"nearest_park_dist_m": dist, "price": price})
        logger.warning("F2: using synthetic demo data")

    sample = (
        df[df[dist_col].notna() & df["price"].notna()]
        .sample(min(5_000, len(df)), random_state=42)
    )

    fig = px.scatter(
        sample,
        x=dist_col,
        y="price",
        log_y=True,
        trendline="ols",
        title="F2: Property Price vs Distance to Nearest Park",
        labels={
            dist_col: "Distance to Nearest Park (m)",
            "price": "Sale Price (€, log scale)",
        },
        template=TEMPLATE,
        opacity=0.4,
        color_discrete_sequence=["#2E86AB"],
    )
    return fig


def plot_f3_price_by_green_quintile(df: pd.DataFrame) -> go.Figure:
    """F3: Boxplot of property price by green-area-within-500m quintile.

    Uses ``green_area_within_500m`` if available, otherwise falls back to
    inverse distance so the quintile ordering still reflects green access.

    Args:
        df: Enriched property DataFrame.

    Returns:
        A Plotly Figure (boxplot, log y-axis).
    """
    if df.empty or "price" not in df.columns:
        rng = np.random.default_rng(1)
        n = 3000
        green = rng.uniform(0, 50000, n)
        price = np.exp(12 + 2e-6 * green + rng.normal(0, 0.45, n))
        df = pd.DataFrame({"green_area_within_500m": green, "price": price})
        logger.warning("F3: using synthetic demo data")

    col = "green_area_within_500m" if "green_area_within_500m" in df.columns else "nearest_park_dist_m"
    df2 = df.copy()

    quintile_labels = ["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"]
    # For distance, invert so Q5 = most green (closest park)
    if col == "nearest_park_dist_m":
        df2["_sort_col"] = -df2[col]
    else:
        df2["_sort_col"] = df2[col]

    try:
        df2["green_quintile"] = pd.qcut(df2["_sort_col"], 5, labels=quintile_labels)
    except ValueError:
        # Duplicate edges: fall back to cut
        df2["green_quintile"] = pd.cut(df2["_sort_col"], 5, labels=quintile_labels)

    fig = px.box(
        df2[df2["price"] > 0],
        x="green_quintile",
        y="price",
        log_y=True,
        title="F3: Property Price by Green Area Within 500 m (Quintiles)",
        labels={
            "green_quintile": "Green Access Quintile",
            "price": "Sale Price (€, log scale)",
        },
        template=TEMPLATE,
        color="green_quintile",
        color_discrete_sequence=px.colors.sequential.Viridis_r,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_f5_price_vs_no2(df: pd.DataFrame) -> go.Figure:
    """F5: Scatter of property price vs mean annual NO₂ level.

    A negative trendline suggests that areas with worse air quality command
    lower prices — consistent with the hedonic-pricing literature.

    Args:
        df: Enriched property DataFrame. Should contain ``mean_no2_year`` and
            ``price`` columns. Falls back to synthetic data if absent.

    Returns:
        A Plotly Figure (scatter + OLS trendline, log y-axis).
    """
    no2_col = "mean_no2_year"

    if df.empty or no2_col not in df.columns or df[no2_col].isna().all():
        rng = np.random.default_rng(5)
        n = 3000
        no2 = rng.uniform(10, 60, n)
        price = np.exp(13.2 - 0.008 * no2 + rng.normal(0, 0.4, n))
        df = pd.DataFrame({no2_col: no2, "price": price})
        logger.warning("F5: using synthetic demo data (no2 column absent or all-NA)")
    else:
        df = df.copy()

    sample = (
        df[df[no2_col].notna() & df["price"].notna()]
        .sample(min(3_000, len(df)), random_state=42)
    )

    fig = px.scatter(
        sample,
        x=no2_col,
        y="price",
        log_y=True,
        trendline="ols",
        title="F5: Property Price vs Mean Annual NO₂ Level",
        labels={
            no2_col: "Mean Annual NO₂ (μg/m³)",
            "price": "Sale Price (€, log scale)",
        },
        template=TEMPLATE,
        opacity=0.4,
        color_discrete_sequence=["#E76F51"],
    )
    return fig


def plot_f6_temporal_by_quintile(results: dict[str, pd.DataFrame]) -> go.Figure:
    """F6: Year-over-year median property price by green-proximity quintile.

    Reveals whether the green premium has widened, narrowed, or remained
    stable over the study period (2015–2024).

    Args:
        results: Mapping returned by :func:`_load_data`. May contain
            ``temporal_median_price`` — a wide DataFrame indexed by year with
            one column per quintile.

    Returns:
        A Plotly Figure (multi-line chart).
    """
    df = results.get("temporal_median_price")

    if df is None or df.empty:
        # Synthetic: slight positive trend, Q5 (most green) commands premium
        years = list(range(2015, 2025))
        base = [200_000 + i * 12_000 for i in range(len(years))]
        quintile_offsets = [0, 10_000, 20_000, 30_000, 45_000]
        df = pd.DataFrame(
            {
                f"Q{q+1}{'_closest' if q == 4 else ''}": [b + quintile_offsets[q] for b in base]
                for q in range(5)
            },
            index=years,
        )
        df.index.name = "year_of_sale"
        logger.warning("F6: using synthetic demo data")

    df_melt = df.reset_index().melt(
        id_vars=["year_of_sale"],
        var_name="Green Quintile",
        value_name="Median Price",
    )

    fig = px.line(
        df_melt,
        x="year_of_sale",
        y="Median Price",
        color="Green Quintile",
        markers=True,
        title="F6: Year-over-Year Median Property Price by Green-Proximity Quintile",
        labels={
            "year_of_sale": "Year",
            "Median Price": "Median Sale Price (€)",
        },
        template=TEMPLATE,
    )
    fig.update_layout(hovermode="x unified")
    return fig


def plot_f7_correlation_heatmap(results: dict[str, pd.DataFrame]) -> go.Figure:
    """F7: Pearson correlation heatmap across key features.

    Provides a compact multivariate overview to motivate the regression
    design choices.

    Args:
        results: Mapping returned by :func:`_load_data`. May contain
            ``correlation_pearson`` — a square correlation DataFrame.

    Returns:
        A Plotly Figure (annotated heatmap, RdBu diverging scale).
    """
    corr = results.get("correlation_pearson")

    if corr is None or corr.empty:
        cols = ["log_price", "park_dist_m", "green_500m", "NO₂", "PM2.5", "PM10", "noise_dB"]
        rng = np.random.default_rng(7)
        raw = rng.uniform(-0.8, 0.8, (len(cols), len(cols)))
        # Ensure symmetry and unit diagonal
        raw = (raw + raw.T) / 2
        np.fill_diagonal(raw, 1.0)
        corr = pd.DataFrame(raw, index=cols, columns=cols)
        logger.warning("F7: using synthetic demo data")

    labels = [
        c.replace("_", " ").replace("mean ", "").replace(" year", "")
        for c in corr.columns
    ]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Pearson r"),
        )
    )
    fig.update_layout(
        title="F7: Feature Correlation Matrix (Pearson r)",
        template=TEMPLATE,
        xaxis=dict(tickangle=-30),
    )
    return fig


def plot_f8_regression_forest(results: dict[str, pd.DataFrame]) -> go.Figure:
    """F8: OLS regression coefficients with 95 % CI (forest plot).

    The forest plot makes effect sizes and their uncertainty immediately
    legible for the report's results section.

    Args:
        results: Mapping returned by :func:`_load_data`. May contain
            ``regression_results`` — a DataFrame with columns ``coef``,
            ``ci_lower``, ``ci_upper`` and an index of variable names.

    Returns:
        A Plotly Figure (horizontal scatter with error bars).
    """
    coef_df = results.get("regression_results")

    if coef_df is None or coef_df.empty:
        coef_df = pd.DataFrame(
            {
                "coef":     [-0.12,  0.08, -0.15, -0.05,  0.03],
                "ci_lower": [-0.18,  0.02, -0.22, -0.09, -0.01],
                "ci_upper": [-0.06,  0.14, -0.08, -0.01,  0.07],
            },
            index=[
                "log(park_dist_m)",
                "green_area_500m",
                "mean_NO₂",
                "mean_PM2.5",
                "mean_noise_dB",
            ],
        )
        logger.warning("F8: using synthetic demo data")

    # Strip year dummies and intercept — their coefficients are not the story
    mask = ~(
        coef_df.index.str.startswith("C(year")
        | coef_df.index.str.startswith("Intercept")
        | coef_df.index.str.startswith("year_")
    )
    df_plot = coef_df[mask].copy()

    # Sort by coefficient for readability
    df_plot = df_plot.sort_values("coef")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["coef"],
            y=df_plot.index.tolist(),
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(df_plot["ci_upper"] - df_plot["coef"]).tolist(),
                arrayminus=(df_plot["coef"] - df_plot["ci_lower"]).tolist(),
                color="#2E86AB",
                thickness=2,
                width=6,
            ),
            marker=dict(size=10, color="#2E86AB", symbol="circle"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coefficient: %{x:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
    fig.update_layout(
        title="F8: OLS Regression Coefficients (95 % CI, HC3 Robust SE)",
        xaxis_title="Coefficient (log-price scale)",
        yaxis_title="Predictor Variable",
        template=TEMPLATE,
        margin=dict(l=180),
    )
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run() -> dict[str, go.Figure]:
    """Generate all static figures (F1–F3, F5–F8) and save to report/figures/.

    Figures are written as PNG at 300 DPI equivalent.  If ``kaleido`` is not
    installed, falls back to writing HTML so the run never hard-fails.

    Returns:
        Mapping of figure label (e.g. ``"F1"``) to its :class:`go.Figure`.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df, results = _load_data()

    figures: dict[str, go.Figure] = {
        "F1": plot_f1_price_distribution(df),
        "F2": plot_f2_price_vs_park_dist(df),
        "F3": plot_f3_price_by_green_quintile(df),
        "F5": plot_f5_price_vs_no2(df),
        "F6": plot_f6_temporal_by_quintile(results),
        "F7": plot_f7_correlation_heatmap(results),
        "F8": plot_f8_regression_forest(results),
    }

    saved: list[str] = []
    for label, fig in figures.items():
        png_path = FIGURE_DIR / f"{label}_plot.png"
        try:
            fig.write_image(str(png_path), scale=SCALE, width=1200, height=700)
            saved.append(str(png_path))
            logger.info("Saved %s -> %s", label, png_path)
        except Exception as exc:  # kaleido absent or other render error
            logger.warning(
                "Cannot write PNG for %s (%s) — saving HTML fallback", label, exc
            )
            html_path = png_path.with_suffix(".html")
            fig.write_html(str(html_path))
            saved.append(str(html_path))

    logger.info("Saved %d figures to %s", len(saved), FIGURE_DIR)
    return figures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
