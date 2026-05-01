# The Green Premium: Correlating Urban Real Estate Prices, Air Quality, and Public Green Spaces in Dublin, Ireland

**Team P — National College of Ireland, MSc Data Analytics, 2025/26**

Members: Sabeeha Shaik (x24336807) · Subhash Lora Jat (x25171496) · Nila Ilangovan (x25117726)

---

## Key Results

| Finding | Metric | Value |
|---------|--------|-------|
| Green area within 500 m correlates with higher prices | Pearson r | **0.095** (p < 0.001) |
| Higher NO2 correlates with lower prices | Pearson r | **-0.110** (p < 0.001) |
| Green quintile premium (top 20% vs bottom 20%) | Median price gap | **~15%** |
| NO2 effect in OLS (per 1 ug/m3 increase) | Coefficient | **-0.0102** (p < 0.001) |
| OLS model fit (with year + construction controls) | Adj. R-squared | **0.019** |
| Park distance coefficient (spatial confounding) | OLS coef | **+0.075** (p < 0.001) |
| Green-AQ interaction effect | OLS interaction | Not significant (p = 0.138) |

**Interpretation:** A measurable green premium exists — properties near larger green areas command higher prices. Higher NO2 is associated with lower prices. However, the park distance coefficient is positive (counterintuitive), reflecting spatial confounding: affluent suburbs (D4, D6) have large parks at moderate distance, while inner-city areas (D1, D7) have small parks nearby but lower prices. The interaction between green proximity and air quality is not statistically significant.

---

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+

### 1. Start databases

```bash
docker-compose up -d
```

PostgreSQL 16 + PostGIS on port 5432 (default) and MongoDB 7 on port 27017.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if your Docker ports differ (e.g. POSTGRES_PORT=5433)
```

### 4. Run the full pipeline

```bash
python -m src.main
```

This runs: ingest (property CSV + Sonitus API + GeoJSON) -> ETL (clean, geocode, spatial join) -> analysis (descriptive, correlation, OLS regression, temporal) -> visualisation (static plots + maps).

### 5. Launch the dashboard

```bash
python -m src.dashboard.app
# Opens at http://localhost:8050
```

Six interactive tabs: Overview, Geographic Explorer, Green Premium Analysis, Air Quality Analysis, Statistical Model, Methodology.

### 6. Run tests

```bash
pytest
```

### 7. Lint check

```bash
ruff check . && black --check .
```

---

## Data Sources

| Dataset | Records | Source | Format | Database |
|---------|---------|--------|--------|----------|
| Property prices (2015-2020) | 92,254 (Dublin) | PSRA via data.smartdublin.ie | CSV | PostgreSQL |
| Air quality readings | 21M+ raw readings | Sonitus API (data.smartdublin.ie/sonitus-api) | JSON via REST API | MongoDB |
| Green spaces | 1,000 features | DCC/SDCC/DLR/FCC open data portals | GeoJSON | MongoDB |

- Property and green space files are downloaded automatically by the ingest scripts.
- Air quality data is fetched live from the Sonitus API (POST requests, 7-day max window per call, credentials as query params).
- All three datasets exceed the 1,000-record minimum requirement.

---

## Pipeline Overview

```
PSRA CSV ─────> PostgreSQL (raw.property)  ─┐
Sonitus API ──> MongoDB (raw_air)           ├─> ETL (clean, geocode, spatial join)
GeoJSON ──────> MongoDB (raw_green)         ─┘          │
                                                        v
                                            PostgreSQL (processed.property_enriched)
                                            MongoDB (processed_property)
                                                        │
                                            ┌───────────┴────────────┐
                                            v                        v
                                     Analysis Layer           Dash Dashboard
                                  (stats, OLS, corr)        (localhost:8050)
                                            │
                                            v
                                    report/figures/ (F1-F9)
```

---

## Project Structure

```
src/
  config.py       # loads .env, DB connections
  main.py         # orchestrator — runs full pipeline
  ingest/         # raw data ingestion (property CSV, Sonitus API, GeoJSON)
  etl/            # cleaning, geocoding, spatial joins, feature engineering
  analysis/       # descriptive stats, correlation, OLS regression, temporal trends
  viz/            # static plots (report figures F1-F8) and maps (F4, F9)
  dashboard/      # interactive Dash app (6 tabs, year slider, postcode filter)
data/
  raw/            # downloaded source files (gitignored if large)
  interim/        # geocode SQLite cache, cleaned parquets
  processed/      # enriched parquet exports, analysis outputs
db/init/          # PostgreSQL schema SQL (raw + processed schemas)
report/           # LaTeX report (IEEE format) and figures
docs/             # architecture diagrams and video script
tests/            # pytest suite (ingest, ETL, analysis)
```

## Architecture

See `docs/architecture.png` for the full data pipeline diagram.

- **PostgreSQL 16 + PostGIS**: stores property data (raw and enriched), spatial queries
- **MongoDB 7**: stores air quality (21M+ readings) and green space data (raw and processed)
- **GeoPandas + EPSG:2157**: Irish Transverse Mercator for metric distance calculations
- **Geocoding**: SQLite cache -> Nominatim (1 req/s) -> postcode centroid fallback
- **Dash + Plotly + Folium**: interactive dashboard with 6 tabs

---

## Figures

| # | Description | File |
|---|-------------|------|
| F1 | Distribution of property prices (histogram, log-x) | `report/figures/F1_plot.png` |
| F2 | Price vs distance to nearest park (scatter + trendline) | `report/figures/F2_plot.png` |
| F3 | Price by green-area-within-500m quintile (boxplot) | `report/figures/F3_plot.png` |
| F4 | Mean NO2 by Dublin area (choropleth map) | `report/figures/F4_no2_map.png` |
| F5 | Price vs mean annual NO2 (scatter + trendline) | `report/figures/F5_plot.png` |
| F6 | Year-over-year median price by green quintile (line) | `report/figures/F6_plot.png` |
| F7 | Correlation heatmap of all features | `report/figures/F7_plot.png` |
| F8 | OLS regression coefficients with CIs (forest plot) | `report/figures/F8_plot.png` |
| F9 | Interactive Dublin map (properties + parks + AQ stations) | `report/figures/F9_dublin_map.html` |
