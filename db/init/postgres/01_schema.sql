-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- RAW SCHEMA — stores data exactly as ingested
-- ============================================================
CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.property (
    id               SERIAL PRIMARY KEY,
    date_of_sale     TEXT,
    address          TEXT,
    postal_code      TEXT,
    county           TEXT,
    price            TEXT,
    not_full_market_price TEXT,
    construction     TEXT,
    floor_area       TEXT,
    vat_exclusive    TEXT,
    source_file      TEXT,
    ingested_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date_of_sale, address, price)
);

-- ============================================================
-- PROCESSED SCHEMA — enriched, typed, spatially-joined output
-- ============================================================
CREATE SCHEMA IF NOT EXISTS processed;

CREATE TABLE IF NOT EXISTS processed.property_enriched (
    id                       SERIAL PRIMARY KEY,
    date_of_sale             DATE,
    year_of_sale             INTEGER,
    address                  TEXT,
    address_clean            TEXT,
    postal_code              TEXT,
    county                   TEXT,
    price                    NUMERIC(12, 2),
    log_price                NUMERIC(8, 4),
    not_full_market_price    BOOLEAN,
    construction             TEXT,
    floor_area               NUMERIC(8, 2),
    lat                      DOUBLE PRECISION,
    lon                      DOUBLE PRECISION,
    geocode_source           TEXT,
    geom                     GEOMETRY(Point, 4326),
    geom_itm                 GEOMETRY(Point, 2157),
    nearest_park_name        TEXT,
    nearest_park_dist_m      NUMERIC(10, 2),
    green_area_within_500m   NUMERIC(14, 2),
    green_area_within_1000m  NUMERIC(14, 2),
    nearest_air_station      TEXT,
    air_station_dist_m       NUMERIC(10, 2),
    mean_no2_year            NUMERIC(8, 3),
    mean_pm25_year           NUMERIC(8, 3),
    mean_pm10_year           NUMERIC(8, 3),
    mean_noise_db_year       NUMERIC(8, 3),
    source_file              TEXT,
    enriched_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index on WGS84 geometry (for map queries)
CREATE INDEX IF NOT EXISTS idx_enriched_geom
    ON processed.property_enriched USING GIST (geom);

-- Spatial index on ITM geometry (for distance queries)
CREATE INDEX IF NOT EXISTS idx_enriched_geom_itm
    ON processed.property_enriched USING GIST (geom_itm);

-- B-tree indexes for common filter columns
CREATE INDEX IF NOT EXISTS idx_enriched_year
    ON processed.property_enriched (year_of_sale);

CREATE INDEX IF NOT EXISTS idx_enriched_postal
    ON processed.property_enriched (postal_code);
