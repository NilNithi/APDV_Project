# CLAUDE.md — Green Premium Dublin Project

This file is the single source of truth for Claude Code working on this project. Read this fully before doing anything. Re-read sections as relevant during work.

---

## 0. PROJECT IDENTITY

**Project Title:** *The Green Premium: Correlating Urban Real Estate Prices, Air Quality, and Public Green Spaces in Dublin, Ireland*

**Submitted to:** National College of Ireland — MSc in Data Analytics
**Module:** Analytics Programming & Data Visualisation (Semester 2, 2025/26)
**Weight:** 70% of module
**Deadline:** 3rd May 2026
**Group:** Team P (3 members)

- Member 1: Sabeeha Shaik (x24336807)
- Member 2: Subhash Lora Jat (x25171496)
- Member 3: Nila Ilangovan (x25117726)

**Submission file naming:**
- Report: `TeamP.pdf`
- Code archive: `TeamP.zip` or `TeamP.gz`
- Video: `TeamP.mp4`
- Individual work-breakdown: `x24336807.pdf`, `x25171496.pdf`, `x25117726.pdf`

---

## 1. CORE OBJECTIVES (do not deviate)

The project must answer this central question:

> **Do air quality and proximity to green spaces correlate with — and potentially explain — variation in residential property prices in Dublin?**

Sub-questions to answer in the report:

1. Is there a measurable price premium for properties closer to public green spaces, after controlling for year of sale?
2. Does air quality (NO₂, PM2.5, PM10, noise) correlate with property prices at the postcode/electoral-division level?
3. Do the effects of greenery and air quality interact — i.e., are properties with both high greenery and clean air priced disproportionately higher?
4. How have these relationships changed over time (year-over-year trends from ~2010 onward)?
5. (Bonus) Can we predict property price from air quality + green proximity using a simple regression model?

Every visualisation, table, and section of the report must serve answering these questions. No filler.

---

## 2. THE THREE DATASETS

### Dataset 1 — Real Estate (STRUCTURED, CSV)
- **Source:** Dublin Residential Property Price Register (PSRA)
- **URL:** https://data.gov.ie/dataset/dublin-residential-property-price-register
- **Owner in team:** Member 1 (Sabeeha)
- **Initial DB:** PostgreSQL
- **Expected size:** ≥ 1,000 records (PSRA has hundreds of thousands — filter to Dublin only)
- **Key fields:** date_of_sale, address, postcode/eircode, county, price, description (new/second-hand), property_size_description
- **Notes:** Addresses are messy free-text. Geocoding will be needed to get lat/lon.

### Dataset 2 — Air Quality (SEMI-STRUCTURED, JSON via API)
- **Source:** Dublin City Council Sonitus Air Quality Monitoring API
- **URL:** https://data.gov.ie/dataset/sonitus
- **Base API:** https://data.smartdublin.ie/sonitus-api
- **Owner in team:** Member 2 (Subhash)
- **Initial DB:** MongoDB
- **Expected size:** ≥ 1,000 records (multi-station, multi-pollutant time series — easy to exceed)
- **Key fields:** monitor_id, station_name, lat, lon, pollutant_type (NO2, PM2.5, PM10, noise dB), value, timestamp
- **Notes:** Use the API endpoint to pull historical readings. Aggregate to monthly or yearly averages per station for joining.

### Dataset 3 — Green Spaces (SEMI-STRUCTURED, GeoJSON)
- **Source:** Parks, Gardens and Public Spaces — Dublin City Council
- **URL:** https://data.gov.ie/dataset/parks-and-open-spaces-dcc
- **Owner in team:** Member 3 (Nila)
- **Initial DB:** MongoDB
- **Expected size:** ≥ 1,000 features (if base dataset has fewer, supplement with DCC playgrounds/sports facilities GeoJSON to cross 1,000)
- **Key fields:** park_id, name, geometry (Polygon/MultiPolygon), area_sqm, type, ward
- **Notes:** Geometries must be preserved. Use GeoPandas for spatial joins.

**HARD RULE from the brief:** Each dataset must contain **at least 1,000 records**. If any falls short after filtering, supplement with a related open-data source from the same provider — but document this clearly in the report.

---

## 3. ARCHITECTURE OVERVIEW

```
┌────────────────────────────────────────────────────────────────┐
│                       RAW DATA SOURCES                         │
│   PSRA CSV download   Sonitus REST API   DCC GeoJSON download  │
└──────────┬──────────────────────┬─────────────────┬────────────┘
           │                      │                 │
           ▼                      ▼                 ▼
   ┌──────────────┐      ┌─────────────────────────────┐
   │  PostgreSQL  │      │          MongoDB            │
   │  raw_property│      │  raw_air  │  raw_green      │
   └──────┬───────┘      └──────┬──────────┬───────────┘
          │                     │          │
          └─────────┬───────────┴──────────┘
                    ▼
           ┌─────────────────┐
           │   ETL PIPELINE  │  (Python — pandas, geopandas)
           │  clean ▸ geocode│
           │  spatial-join   │
           │  feature-engg   │
           └────────┬────────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  ┌──────────────┐        ┌──────────────┐
  │ PostgreSQL   │        │   MongoDB    │
  │ processed_*  │        │ processed_*  │
  │ (star schema)│        │ (enriched    │
  │              │        │  documents)  │
  └──────┬───────┘        └──────┬───────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  ANALYSIS LAYER │  (stats, regression, correlation)
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │  DASH DASHBOARD │  (interactive map + charts)
            └─────────────────┘
```

---

## 4. TECH STACK (PINNED)

**Language:** Python 3.11+

**Databases (Docker, see §5):**
- PostgreSQL 16 (with PostGIS extension for spatial queries)
- MongoDB 7

**Python libraries:**
- Data: `pandas`, `numpy`, `geopandas`, `shapely`, `pyproj`
- DB: `psycopg2-binary`, `sqlalchemy`, `pymongo`
- HTTP / data fetch: `requests`, `tenacity` (for retries)
- Geocoding: `geopy` (Nominatim — free, rate-limited; cache results aggressively)
- Stats / ML: `scipy`, `statsmodels`, `scikit-learn`
- Viz: `plotly`, `folium`
- Dashboard: `dash`, `dash-bootstrap-components`, `dash-leaflet`
- Orchestration: plain Python with a `Makefile` (default — simpler to demo on video; `prefect` only if time permits)
- Testing: `pytest`
- Lint/format: `ruff`, `black`

**IDE:** VS Code (mention this in the report; Jupyter for exploratory notebooks only)

**Version control:** Git, GitHub. Commit messages must look human (not robotic). Use Conventional Commits style: `feat:`, `fix:`, `docs:`, `refactor:`.

---

## 5. DEPLOYMENT — LOCAL DOCKER

Everything runs locally via Docker Compose. The grader / video viewer must be able to spin up the project with **one command**.

### `docker-compose.yml` (must exist at repo root)

```yaml
version: "3.9"

services:
  postgres:
    image: postgis/postgis:16-3.4
    container_name: green_premium_postgres
    environment:
      POSTGRES_USER: green_user
      POSTGRES_PASSWORD: green_pass
      POSTGRES_DB: green_premium
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init/postgres:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U green_user -d green_premium"]
      interval: 5s
      timeout: 5s
      retries: 10

  mongo:
    image: mongo:7
    container_name: green_premium_mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: green_user
      MONGO_INITDB_ROOT_PASSWORD: green_pass
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
  mongo_data:
```

### `.env` template (commit `.env.example`, NOT `.env`)

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=green_user
POSTGRES_PASSWORD=green_pass
POSTGRES_DB=green_premium

MONGO_URI=mongodb://green_user:green_pass@localhost:27017/?authSource=admin
MONGO_DB=green_premium

NOMINATIM_USER_AGENT=green_premium_nci_2026
```

---

## 6. PROJECT DIRECTORY STRUCTURE

```
TeamP/
├── README.md                   # quick-start: docker-compose up, then python src/main.py
├── CLAUDE.md                   # this file
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml              # ruff/black config
├── Makefile                    # make up / make etl / make dashboard / make test
│
├── data/
│   ├── raw/                    # downloaded source files (gitignored if large)
│   │   ├── property/
│   │   ├── air/
│   │   └── green/
│   ├── interim/                # geocoded addresses cache, etc.
│   └── processed/              # final CSVs/parquet exports for the report
│
├── db/
│   └── init/
│       └── postgres/
│           └── 01_schema.sql   # creates schemas, extensions, tables
│
├── src/
│   ├── __init__.py
│   ├── config.py               # loads .env, single source of truth for credentials
│   ├── main.py                 # orchestrator: runs full pipeline end-to-end
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── property_ingest.py  # PSRA CSV → PostgreSQL.raw_property
│   │   ├── air_ingest.py       # Sonitus API → MongoDB.raw_air
│   │   └── green_ingest.py     # GeoJSON → MongoDB.raw_green
│   │
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── property_clean.py   # parse addresses, prices, dates
│   │   ├── air_clean.py        # parse timestamps, coerce numerics, aggregate
│   │   ├── green_clean.py      # validate geometries, project to EPSG:2157 (Irish grid)
│   │   ├── geocode.py          # geopy + on-disk cache (joblib or sqlite)
│   │   └── join.py             # spatial joins → unified processed table
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── descriptive.py      # summary stats per question
│   │   ├── correlation.py      # Pearson/Spearman between price ↔ AQ ↔ green proximity
│   │   ├── regression.py       # OLS: log_price ~ green_proximity + AQ + year + size
│   │   └── temporal.py         # year-over-year trends
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── static_plots.py     # plotly figures used in the IEEE report (saved as PNG)
│   │   └── maps.py             # folium / plotly choropleth and bubble maps
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py              # Dash entry point
│       ├── layout.py           # tabs/cards/filters
│       ├── callbacks.py        # interactivity
│       └── assets/
│           └── style.css
│
├── notebooks/                  # exploratory only — NOT part of the pipeline
│   ├── 01_property_eda.ipynb
│   ├── 02_air_eda.ipynb
│   ├── 03_green_eda.ipynb
│   └── 04_joined_eda.ipynb
│
├── tests/
│   ├── test_ingest.py
│   ├── test_etl.py
│   └── test_analysis.py
│
├── report/
│   ├── TeamP.tex               # IEEE conference template (LaTeX)
│   ├── references.bib
│   ├── figures/                # all plots saved here for the report
│   └── TeamP.pdf               # final compiled report
│
└── docs/
    ├── architecture.png        # the diagram from §3
    ├── pipeline_flow.png
    ├── er_diagram.png
    └── video_script.md         # presentation script for the user to record
```

---

## 7. EXECUTION ORDER (what Claude Code should build, in this order)

Build in **phases**. Do not skip ahead. Each phase ends with a working, testable artifact.

### Phase 0 — Bootstrap (Day 1, morning)
1. Initialize git repo, create `.gitignore` (Python + data/raw/ + .env).
2. Write `docker-compose.yml`, `.env.example`, `requirements.txt`, `pyproject.toml`, `Makefile`.
3. Write `db/init/postgres/01_schema.sql` (raw + processed schemas).
4. Run `docker-compose up -d` and verify both DBs are reachable from Python.
5. Write `src/config.py` and a `tests/test_config.py` smoke test.

**Phase 0 exit criteria:** `make up && python -c "from src.config import test_connections; test_connections()"` returns OK for both databases.

### Phase 1 — Ingest (Day 1, afternoon)
1. `src/ingest/property_ingest.py` — download PSRA CSV (full year files, filter Dublin), load to `raw.property` in Postgres.
2. `src/ingest/air_ingest.py` — paginate Sonitus API, store raw JSON documents in MongoDB collection `raw_air`. Implement `tenacity` retries with exponential backoff. Cache to `data/raw/air/` as well.
3. `src/ingest/green_ingest.py` — download GeoJSON, store features as documents in MongoDB collection `raw_green`.

Each ingest script must be **idempotent** — running it twice should not duplicate rows. Use `ON CONFLICT DO NOTHING` (Postgres) and unique-keyed upserts (Mongo).

**Phase 1 exit criteria:** Each raw collection/table has ≥ 1,000 records. Print counts at the end of each ingest.

### Phase 2 — ETL (Day 2)
1. `src/etl/property_clean.py`
   - Parse `date_of_sale` to date type
   - Parse `price` (strip €, commas, cast to numeric)
   - Standardize address (uppercase, strip)
   - Extract postcode/eircode where present
2. `src/etl/geocode.py`
   - Use Nominatim via geopy with a 1-second rate limit
   - Cache results in a SQLite/joblib cache (geocoding the same address twice = waste)
   - For un-geocodable addresses, fall back to postcode centroid
3. `src/etl/air_clean.py`
   - Coerce numerics, drop nulls, drop sensor errors (negative readings, impossible spikes)
   - Aggregate to monthly mean per station per pollutant
4. `src/etl/green_clean.py`
   - Validate geometries (`is_valid` check, buffer(0) fix)
   - Reproject to **EPSG:2157** (Irish Transverse Mercator) for accurate metric distance calculations
5. `src/etl/join.py`
   - For each property: compute distance to nearest park (in metres), park area within 500m and 1000m buffers
   - For each property: assign air-quality reading from nearest station, weighted by inverse distance, matched to year of sale
   - Output: `processed.property_enriched` table in Postgres
   - Mirror enriched documents to `processed_property` in Mongo

**Phase 2 exit criteria:** `processed.property_enriched` has rows with non-null `nearest_park_dist_m`, `green_area_within_500m`, `mean_no2_year`, `mean_pm25_year`.

### Phase 3 — Analysis (Day 3, morning)
1. `src/analysis/descriptive.py` — table of means/medians/quartiles per year, per Dublin postcode area.
2. `src/analysis/correlation.py` — Pearson + Spearman correlations among: log_price, nearest_park_dist_m, green_area_within_500m, mean_no2, mean_pm25, mean_pm10, noise_db.
3. `src/analysis/regression.py` — OLS via statsmodels:
   `log(price) ~ log(nearest_park_dist_m) + green_area_500m + mean_no2 + mean_pm25 + year + property_size_desc`
   Report coefficients, p-values, R². Discuss whether coefficients confirm the "green premium" hypothesis.
4. `src/analysis/temporal.py` — year-over-year median price by green-proximity quintile.

### Phase 4 — Visualisation (Day 3, afternoon)
1. `src/viz/static_plots.py` — generate every figure used in the report as PNG saved to `report/figures/`. Title every figure. Use a consistent colour palette (`plotly.colors.sequential.Viridis` for sequential, `Plotly` for categorical).
2. `src/viz/maps.py` — choropleth of median price per electoral division, bubble map of price vs nearest park, air-quality station overlay.

Required plots (minimum, all must appear in the report AND the dashboard):

| # | Plot | Type | Insight |
|---|------|------|---------|
| F1 | Distribution of property prices in Dublin | Histogram + log-x | Skewness justifies log-transform |
| F2 | Price vs distance to nearest park | Scatter w/ trendline | Green premium exists / doesn't |
| F3 | Price by green-area-within-500m quintile | Boxplot | Categorical green effect |
| F4 | Mean NO₂ by Dublin area | Choropleth | Geographic AQ disparity |
| F5 | Price vs mean annual NO₂ | Scatter w/ trendline | AQ ↔ price relation |
| F6 | Year-over-year median price by green quintile | Line chart | Temporal evolution |
| F7 | Correlation heatmap of all features | Heatmap | Multivariate overview |
| F8 | OLS regression coefficients (with CIs) | Forest plot | Effect-size summary |
| F9 | Interactive Dublin map: properties coloured by price, parks overlaid | Map | Dashboard centerpiece |

### Phase 5 — Dashboard (Day 4)
Build `src/dashboard/app.py` using Dash + dash-bootstrap-components + dash-leaflet.

**Tabs / sections (top-level):**
1. **Overview** — KPIs (total sales, median price, mean NO₂, # parks), F1, F2 side-by-side.
2. **Geographic Explorer** — F9 large interactive map; year filter slider; pollutant dropdown.
3. **Green Premium Analysis** — F3, F6, F7.
4. **Air Quality Analysis** — F4, F5.
5. **Statistical Model** — F8 with explanatory text and OLS table.
6. **Methodology** — short page describing the pipeline (one-paragraph version of §3).

**Interactivity requirements (the rubric rewards this):**
- Year range slider (affects all charts)
- Postcode/area multi-select dropdown
- Pollutant selector (NO₂ / PM2.5 / PM10 / noise)
- Hover tooltips on every chart
- Download-as-CSV button for the filtered dataset

The dashboard must run with `python -m src.dashboard.app` and open at `http://localhost:8050`.

### Phase 6 — Report (Day 4–5)
Write `report/TeamP.tex` in **IEEE conference format** (download template from https://www.ieee.org/conferences_events/conferences/publishing/templates.html).

**Word count target:** ~3,000 words (excluding references and figure captions).

**Section-by-section guide:**

1. **Abstract (~150 words)** — one-sentence motivation, one-sentence method, one-sentence dataset summary, two-sentence findings, one-sentence implication.
2. **Introduction (~400 words)** — Why does urban green/air-quality affect property markets? Why Dublin? State the research questions verbatim. End with paper roadmap.
3. **Related Work (~500 words)** — Cite at least 6 academic works. Topics to cover:
   - Hedonic pricing models (Rosen 1974 is the canonical citation)
   - Green premium literature (e.g., Conway et al., Jim & Chen)
   - Air quality and house prices (Chay & Greenstone)
   - ETL / data orchestration tooling (cite Stonebraker, Ackerman or similar)
   - GeoPandas / spatial data processing
   - Dash / interactive visualization theory (Munzner)

   **Critical:** Don't just summarize — say what each paper did, what it missed, and how this project addresses that gap.
4. **Data Processing Methodology (~700 words)** — describe each dataset (size, source, structure), justify DB choices (Postgres for tabular relational joins, Mongo for schemaless geo/JSON), describe the geocoding strategy and rate-limiting, describe the spatial-join algorithm (KD-tree for nearest-park, polygon intersection for buffer-area). Include the architecture diagram (§3).
5. **Data Visualisation Methodology (~400 words)** — for each major chart, justify (a) the chart type (cite Munzner / Tufte), (b) colour choices (sequential vs diverging vs categorical, accessibility), (c) interactivity choices in the dashboard. Justify the dashboard layout using a tab-based information-architecture argument.
6. **Results and Evaluation (~600 words)** — present each finding with the corresponding figure. Three or more **non-arbitrary** findings (rubric requirement for H1). Discuss effect sizes, not just p-values.
7. **Conclusions and Future Work (~250 words)** — restate findings, discuss limitations (geocoding errors, sensor coverage gaps, no causal identification), suggest follow-ups (causal estimation via difference-in-differences around new park openings; richer microdata).
8. **Bibliography** — IEEE citation style.

**Length limit is real — don't blow past 3,000 words.** Cut filler ruthlessly.

### Phase 7 — Video & WBS (Day 5)
- The user will record a max-10-minute MP4. Generate a presentation script as `docs/video_script.md` covering: project intro (1 min), architecture walkthrough (2 min), code highlights (2 min), live dashboard demo (3 min), findings discussion (1.5 min), conclusion (0.5 min).
- The user has said they will handle the work-breakdown report separately.

---

## 8. CODING STANDARDS (non-negotiable)

- **Type hints everywhere.** Even in scripts.
- **Docstrings** on every public function, in Google style.
- **Logging** with the `logging` module, not `print`. Module-level loggers: `logger = logging.getLogger(__name__)`.
- **Configuration** via `src/config.py` and `.env`. No hardcoded credentials. Ever.
- **Idempotency.** Every ingest and ETL step must be safe to re-run.
- **Tests** for the ETL transformations (at least: address-cleaning, geocode-cache hit, distance-calc correctness on a known geometry, regression coefficient sanity-check on synthetic data).
- **Errors over silent failures.** Don't `except: pass`. Catch specific exceptions, log, raise or skip with a counter.
- **`requirements.txt` is authoritative.** Pin major versions.
- **Black + Ruff** must pass clean before any commit.
- **Comments where the *why* is non-obvious**, not the *what*.

---

## 9. WHAT THE BRIEF EXPLICITLY REWARDS (rubric mapping)

The grading rubric is in the brief. Here's how to hit each top-tier criterion:

| Rubric Criterion | Weight | How to hit Solid H1 (≥80%) |
|---|---|---|
| Project Objectives | 10% | Five clearly-stated research questions answered with quantified findings. |
| Literature Review | 10% | 8+ critically-evaluated references, each with stated limitation and how this project addresses it. |
| Data Complexity & Handling | 15% | Three datasets ≥1,000 rows; one programmatically retrieved via API (Sonitus); both Postgres AND Mongo used pre- and post-processing; one dataset is geospatial (high complexity). |
| Data Processing | 20% | Multiple techniques: SQL transforms, GeoPandas spatial joins, KD-tree nearest-neighbour, geocoding with caching, OLS regression, temporal aggregation. Document why each was chosen. |
| Data Visualisation | 15% | 9+ figures, justified using Munzner/Tufte theory; consistent palette; interactive dashboard with multi-control filtering. |
| Results & Conclusions | 20% | At least 3 non-arbitrary findings with effect sizes and confidence intervals; discussed in context of hedonic-pricing literature. |
| Quality of Writing | 10% | Strict IEEE template adherence, ≤3,000 words, captioned figures, complete IEEE-style references, zero typos. |

---

## 10. THINGS THAT WILL TANK THE GRADE — DO NOT DO

- **Do not use a Kaggle dataset.** The brief explicitly forbids datasets with existing online implementations.
- **Do not store credentials in code.**
- **Do not commit `data/raw/` if files exceed ~50MB total.** Use `.gitignore`. Document download steps in README.
- **Do not skip the IEEE template.** Word/LaTeX template only.
- **Do not exceed 3,000 words** in the report.
- **Do not use only one database.** Both Postgres AND Mongo must be used, both pre- and post-processing.
- **Do not produce static visualisations only.** Dashboard must be interactive (sliders, dropdowns, hover, multi-control).
- **Do not write a "summary of summaries" related-work section.** Critical evaluation only.
- **Do not invent findings.** If a coefficient is not significant, say so. The rubric rewards honest interpretation.

---

## 11. REPRODUCIBILITY CHECKLIST (must all be true at submission)

- [ ] `git clone <repo> && docker-compose up -d && pip install -r requirements.txt && python -m src.main` runs the full pipeline end-to-end on a clean machine.
- [ ] `python -m src.dashboard.app` launches the dashboard at localhost:8050.
- [ ] All figures used in the report exist in `report/figures/` and are regenerated by `src.viz`.
- [ ] `pytest` passes.
- [ ] `ruff check .` and `black --check .` pass.
- [ ] README has a "How to run" section with commands.
- [ ] `.env.example` exists; `.env` is gitignored.
- [ ] All three datasets' raw files are either committed (if small) or have download URLs in README + automated `python -m src.ingest.<x>` to pull them.
- [ ] Architecture diagram (§3 of this file, rendered as PNG) is in `docs/` AND embedded in the report.
- [ ] Report is in `report/TeamP.pdf`, max 3,000 words, IEEE format, with all student names/numbers on page 1.

---

## 12. OPEN DECISIONS LEFT TO CLAUDE CODE'S JUDGMENT

- Specific year window for property data (suggested: 2015–2024 — wide enough for trends, narrow enough to keep geocoding cost manageable)
- Whether to use parquet or CSV for `data/processed/` (suggest parquet — smaller, typed)
- Choice of map tile provider in dash-leaflet (OpenStreetMap is fine and free)
- Whether to add a Random Forest as a secondary model alongside OLS (only if time permits — OLS is mandatory because it's interpretable; RF would just be a robustness check)

---

## 13. WHEN STUCK

- The Sonitus API may have changed structure or rate limits. Inspect the API response with `curl` first, then write the parser.
- If geocoding rates are too slow with Nominatim, fall back to postcode-centroid lookup using a one-time Eircode-to-coords reference table.
- If a spatial join is slow, reproject to EPSG:2157 (metres) and use `geopandas.sjoin_nearest` rather than custom loops.
- If the report runs over 3,000 words, cut the Related Work first (compress, don't drop citations).

---

## 14. DELIVERABLES SUMMARY (final checklist before submission)

1. `TeamP.pdf` — IEEE-formatted report, ≤3,000 words, all team member names and student numbers on page 1.
2. `TeamP.zip` — entire repo zipped (excluding `data/raw/` if huge, excluding `.env`, excluding `__pycache__`).
3. `TeamP.mp4` — ≤10-minute presentation video (user records using `docs/video_script.md` as guide).
4. Three individual work-breakdown PDFs (`x24336807.pdf`, `x25171496.pdf`, `x25117726.pdf`) — user handles these manually.

---

**End of CLAUDE.md.** When in doubt, re-read §1 (objectives) and §9 (rubric mapping). Every line of code or prose should serve one or both.
