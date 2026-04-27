# The Green Premium: Correlating Urban Real Estate Prices, Air Quality, and Public Green Spaces in Dublin, Ireland

**Team P — National College of Ireland, MSc Data Analytics, 2025/26**

Members: Sabeeha Shaik (x24336807) · Subhash Lora Jat (x25171496) · Nila Ilangovan (x25117726)

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+

### 1. Start databases

```bash
docker-compose up -d
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if your Docker ports differ
```

### 4. Run the full pipeline

```bash
python -m src.main
```

### 5. Launch the dashboard

```bash
python -m src.dashboard.app
# Opens at http://localhost:8050
```

### 6. Run tests

```bash
pytest
```

### 7. Lint check

```bash
ruff check . && black --check .
```

## Data Sources

| Dataset | Source | Format |
|---------|--------|--------|
| Property prices (2015–2024) | PSRA via data.smartdublin.ie | CSV (year-by-year) |
| Air quality readings | Sonitus API (data.smartdublin.ie/sonitus-api) | JSON via REST API |
| Green spaces | DCC/SDCC/DLR/FCC open data portals | GeoJSON |

Property and green space files are downloaded automatically by `python -m src.ingest.property_ingest` and `python -m src.ingest.green_ingest`. Air quality data is fetched live from the Sonitus API by `python -m src.ingest.air_ingest`.

## Project Structure

```
src/
  ingest/     # raw data ingestion scripts
  etl/        # cleaning, geocoding, spatial joins
  analysis/   # descriptive stats, correlation, OLS regression
  viz/        # static plots (report figures) and maps
  dashboard/  # interactive Dash app
data/
  raw/        # downloaded source files (gitignored)
  interim/    # geocode cache
  processed/  # enriched parquet exports
db/init/      # PostgreSQL schema SQL
report/       # LaTeX report and figures
docs/         # architecture diagrams and video script
```

## Architecture

See `docs/architecture.png` for the full data pipeline diagram.

PostgreSQL 16 + PostGIS stores property data (raw and enriched).
MongoDB 7 stores air quality and green space data (raw and processed).
