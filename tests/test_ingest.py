"""Unit tests for ingest modules.

Tests are deliberately offline — no real database or HTTP connections are made.
MongoDB and PostgreSQL interactions are mocked so the suite runs on a clean
machine without Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Property ingest tests
# ---------------------------------------------------------------------------


class TestPropertyIngest:
    """Tests for src.ingest.property_ingest."""

    def test_dublin_filter(self) -> None:
        """Only Dublin rows should survive the county filter."""
        from src.ingest.property_ingest import COLUMN_MAP  # noqa: F401 — checks import

        df = pd.DataFrame(
            {
                "Date of Sale (dd/mm/yyyy)": ["01/01/2020", "01/01/2020"],
                "Address": ["1 Main St", "2 Country Rd"],
                "Postal Code": ["D01", ""],
                "County": ["Co. Dublin", "Co. Cork"],
                "Price (\u20ac)": ["\u20ac300,000.00", "\u20ac200,000.00"],
                "Not Full Market Price": ["No", "No"],
                "VAT Exclusive": ["No", "No"],
                "Description of Property": [
                    "Second-Hand Dwelling house /Apartment",
                    "Second-Hand Dwelling house /Apartment",
                ],
                "Property Size Description": ["", ""],
            }
        )
        dublin_mask = df["County"].str.contains("Dublin", case=False, na=False)
        filtered = df[dublin_mask]

        assert len(filtered) == 1
        assert filtered.iloc[0]["County"] == "Co. Dublin"

    def test_column_map_completeness(self) -> None:
        """COLUMN_MAP must include every expected source column."""
        from src.ingest.property_ingest import COLUMN_MAP

        expected_keys = [
            "Date of Sale (dd/mm/yyyy)",
            "Address",
            "Postal Code",
            "County",
            "Price (\u20ac)",
            "Not Full Market Price",
            "VAT Exclusive",
            "Description of Property",
            "Property Size Description",
        ]
        for key in expected_keys:
            assert key in COLUMN_MAP, f"COLUMN_MAP is missing key: {key!r}"

    def test_column_map_values_are_strings(self) -> None:
        """All COLUMN_MAP values (DB column names) must be non-empty strings."""
        from src.ingest.property_ingest import COLUMN_MAP

        for src_col, db_col in COLUMN_MAP.items():
            assert isinstance(db_col, str) and db_col, (
                f"COLUMN_MAP[{src_col!r}] is not a non-empty string: {db_col!r}"
            )

    def test_download_skips_cached(self, tmp_path: Path) -> None:
        """download_year must skip HTTP when the cache file already exists."""
        from src.ingest import property_ingest

        cached = tmp_path / "PPR-2020.csv"
        cached.write_text("date,address,county\n01/01/2020,1 Main St,Co. Dublin")

        with patch.object(property_ingest, "RAW_DIR", tmp_path):
            with patch("requests.get") as mock_get:
                result = property_ingest.download_year(2020)
                mock_get.assert_not_called()

        assert result == cached

    def test_download_returns_none_on_all_failures(self, tmp_path: Path) -> None:
        """download_year must return None and not leave a partial file when all retries fail."""
        import requests as req
        from src.ingest import property_ingest

        with patch.object(property_ingest, "RAW_DIR", tmp_path):
            with patch.object(property_ingest, "_RETRY_ATTEMPTS", 2):
                with patch.object(property_ingest, "_RETRY_SLEEP", 0):
                    with patch("requests.get", side_effect=req.RequestException("timeout")):
                        result = property_ingest.download_year(2021)

        assert result is None
        # Partial file must be cleaned up.
        assert not (tmp_path / "PPR-2021.csv").exists()

    def test_load_year_filters_dublin_and_inserts(self, tmp_path: Path) -> None:
        """load_year must filter non-Dublin rows and call executemany for the rest."""
        from src.ingest import property_ingest

        # Write a tiny CSV (two rows: one Dublin, one Cork).
        csv_content = (
            "Date of Sale (dd/mm/yyyy),Address,Postal Code,County,"
            "Price (\u20ac),Not Full Market Price,VAT Exclusive,"
            "Description of Property,Property Size Description\n"
            "01/01/2020,1 Main St,D01,Co. Dublin,\u20ac300000.00,No,No,"
            "Second-Hand Dwelling house /Apartment,\n"
            "01/01/2020,2 Cork Rd,,Co. Cork,\u20ac200000.00,No,No,"
            "Second-Hand Dwelling house /Apartment,\n"
        )
        csv_path = tmp_path / "PPR-2020.csv"
        csv_path.write_bytes(csv_content.encode("cp1252"))

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        inserted = property_ingest.load_year(csv_path, 2020, mock_engine)
        assert inserted == 1
        assert mock_conn.execute.call_count == 1  # one batch for 1 Dublin row


# ---------------------------------------------------------------------------
# Air ingest tests
# ---------------------------------------------------------------------------


class TestAirIngest:
    """Tests for src.ingest.air_ingest."""

    def test_bulk_upsert_empty_list(self) -> None:
        """_bulk_upsert with an empty list must return 0 and not touch MongoDB."""
        from src.ingest.air_ingest import _bulk_upsert

        mock_collection = MagicMock()
        result = _bulk_upsert(mock_collection, [])

        assert result == 0
        mock_collection.bulk_write.assert_not_called()

    def test_bulk_upsert_calls_bulk_write(self) -> None:
        """_bulk_upsert with documents must call bulk_write exactly once."""
        from src.ingest.air_ingest import _bulk_upsert

        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.upserted_count = 2
        mock_result.modified_count = 0
        mock_collection.bulk_write.return_value = mock_result

        docs = [
            {"monitor_id": "A", "timestamp": "2023-01-01T00:00:00Z", "pollutant": "NO2", "value": 10.0},
            {"monitor_id": "B", "timestamp": "2023-01-01T00:00:00Z", "pollutant": "PM2.5", "value": 5.0},
        ]
        result = _bulk_upsert(mock_collection, docs)

        mock_collection.bulk_write.assert_called_once()
        assert result == 2  # upserted_count + modified_count


# ---------------------------------------------------------------------------
# Green ingest tests
# ---------------------------------------------------------------------------


class TestGreenIngest:
    """Tests for src.ingest.green_ingest."""

    # --- geometry_hash ---

    def test_geometry_hash_deterministic(self) -> None:
        """The same geometry must always produce the same hash."""
        from src.ingest.green_ingest import geometry_hash

        geom = {"type": "Point", "coordinates": [-6.26, 53.35]}
        assert geometry_hash(geom) == geometry_hash(geom)

    def test_geometry_hash_is_md5_hex(self) -> None:
        """geometry_hash must return a 32-character lowercase hex string."""
        from src.ingest.green_ingest import geometry_hash

        geom = {"type": "Point", "coordinates": [-6.26, 53.35]}
        h = geometry_hash(geom)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_geometry_hash_differs_for_different_geometries(self) -> None:
        """Different geometries must produce different hashes."""
        from src.ingest.green_ingest import geometry_hash

        g1 = {"type": "Point", "coordinates": [-6.26, 53.35]}
        g2 = {"type": "Point", "coordinates": [-6.27, 53.36]}
        assert geometry_hash(g1) != geometry_hash(g2)

    def test_geometry_hash_key_order_independent(self) -> None:
        """Key insertion order in the geometry dict must not affect the hash."""
        from src.ingest.green_ingest import geometry_hash

        g1 = {"coordinates": [-6.26, 53.35], "type": "Point"}
        g2 = {"type": "Point", "coordinates": [-6.26, 53.35]}
        assert geometry_hash(g1) == geometry_hash(g2)

    # --- parse_features ---

    def test_parse_features_adds_source(self) -> None:
        """parse_features must inject 'source' into every document."""
        from src.ingest.green_ingest import parse_features

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {"name": "Test Park", "type": "park"},
                }
            ],
        }
        docs = parse_features(geojson, "test_source")

        assert len(docs) == 1
        assert docs[0]["source"] == "test_source"

    def test_parse_features_skips_null_geometry(self) -> None:
        """Features without geometry must be silently skipped."""
        from src.ingest.green_ingest import parse_features

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {"name": "No Geometry"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {"name": "Has Geometry"},
                },
            ],
        }
        docs = parse_features(geojson, "src")
        assert len(docs) == 1
        assert docs[0]["name"] == "Has Geometry"

    def test_parse_features_synthetic_false(self) -> None:
        """Real parsed features must have synthetic=False."""
        from src.ingest.green_ingest import parse_features

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {},
                }
            ],
        }
        docs = parse_features(geojson, "real_source")
        assert docs[0]["synthetic"] is False

    def test_parse_features_name_fallback(self) -> None:
        """parse_features must fall back to 'Unknown' when no name property exists."""
        from src.ingest.green_ingest import parse_features

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {},
                }
            ],
        }
        docs = parse_features(geojson, "src")
        assert docs[0]["name"] == "Unknown"

    def test_parse_features_preserves_properties(self) -> None:
        """Original properties dict must be stored verbatim."""
        from src.ingest.green_ingest import parse_features

        props = {"NAME": "Phoenix Park", "AREA_HA": 707, "WARD": "Cabra"}
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.35, 53.36]},
                    "properties": props,
                }
            ],
        }
        docs = parse_features(geojson, "src")
        assert docs[0]["properties"] == props
        assert docs[0]["name"] == "Phoenix Park"

    def test_parse_features_geom_hash_present(self) -> None:
        """Every parsed document must carry a geom_hash field."""
        from src.ingest.green_ingest import parse_features

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {},
                }
            ],
        }
        docs = parse_features(geojson, "src")
        assert "geom_hash" in docs[0]
        assert len(docs[0]["geom_hash"]) == 32

    # --- generate_synthetic_green_spaces ---

    def test_synthetic_fallback_count(self) -> None:
        """generate_synthetic_green_spaces must return exactly the requested count."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        spaces = generate_synthetic_green_spaces(100)
        assert len(spaces) == 100

    def test_synthetic_fallback_all_synthetic_true(self) -> None:
        """Every synthetic document must have synthetic=True."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        spaces = generate_synthetic_green_spaces(50)
        assert all(s["synthetic"] is True for s in spaces)

    def test_synthetic_fallback_within_dublin_bbox(self) -> None:
        """All synthetic coordinates must lie within Dublin's bounding box."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        spaces = generate_synthetic_green_spaces(200)
        for s in spaces:
            lon, lat = s["geometry"]["coordinates"]
            assert 53.25 <= lat <= 53.42, f"lat {lat} out of range"
            assert -6.45 <= lon <= -6.10, f"lon {lon} out of range"

    def test_synthetic_fallback_reproducible(self) -> None:
        """Two calls with the same internal seed must produce identical output."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        a = generate_synthetic_green_spaces(10)
        b = generate_synthetic_green_spaces(10)
        assert [s["name"] for s in a] == [s["name"] for s in b]
        assert [s["geometry"] for s in a] == [s["geometry"] for s in b]

    def test_synthetic_fallback_default_count(self) -> None:
        """Default count argument must produce 1 000 documents."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        spaces = generate_synthetic_green_spaces()
        assert len(spaces) == 1_000

    def test_synthetic_fallback_has_geom_hash(self) -> None:
        """Synthetic documents must carry a valid geom_hash."""
        from src.ingest.green_ingest import generate_synthetic_green_spaces

        spaces = generate_synthetic_green_spaces(5)
        for s in spaces:
            assert "geom_hash" in s
            assert len(s["geom_hash"]) == 32

    # --- fetch_geojson ---

    def test_fetch_geojson_returns_none_on_http_error(self, tmp_path: Path) -> None:
        """fetch_geojson must return None when the server returns 4xx/5xx."""
        import requests as req
        from src.ingest import green_ingest

        with patch.object(green_ingest, "RAW_DIR", tmp_path):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.side_effect = req.HTTPError("404")
                mock_get.return_value = mock_resp

                result = green_ingest.fetch_geojson("https://example.com/parks.geojson")

        assert result is None

    def test_fetch_geojson_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """fetch_geojson must return None when the response body is not valid JSON."""
        from src.ingest import green_ingest

        with patch.object(green_ingest, "RAW_DIR", tmp_path):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.side_effect = ValueError("not JSON")
                mock_get.return_value = mock_resp

                result = green_ingest.fetch_geojson("https://example.com/bad.geojson")

        assert result is None

    def test_fetch_geojson_returns_features_on_success(self, tmp_path: Path) -> None:
        """fetch_geojson must return the features list from a valid FeatureCollection."""
        from src.ingest import green_ingest

        fake_fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {"name": "Park A"},
                }
            ],
        }
        with patch.object(green_ingest, "RAW_DIR", tmp_path):
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = fake_fc
                mock_get.return_value = mock_resp

                result = green_ingest.fetch_geojson("https://example.com/parks.geojson")

        assert result is not None
        assert len(result) == 1
        assert result[0]["properties"]["name"] == "Park A"

    def test_fetch_geojson_serves_from_cache(self, tmp_path: Path) -> None:
        """fetch_geojson must not make an HTTP call when a valid cache file exists."""
        from src.ingest import green_ingest

        cached_fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                    "properties": {},
                }
            ],
        }
        cache_file = tmp_path / "parks.geojson"
        cache_file.write_text(json.dumps(cached_fc), encoding="utf-8")

        with patch.object(green_ingest, "RAW_DIR", tmp_path):
            with patch("requests.get") as mock_get:
                result = green_ingest.fetch_geojson(
                    "https://example.com/parks.geojson"
                )
                mock_get.assert_not_called()

        assert result is not None
        assert len(result) == 1

    # --- upsert_features ---

    def test_upsert_features_empty_returns_zero(self) -> None:
        """upsert_features with an empty list must return 0 without touching MongoDB."""
        from src.ingest.green_ingest import upsert_features

        mock_col = MagicMock()
        result = upsert_features(mock_col, [])

        assert result == 0
        mock_col.bulk_write.assert_not_called()

    def test_upsert_features_calls_bulk_write(self) -> None:
        """upsert_features must call bulk_write once for a non-empty document list."""
        from src.ingest.green_ingest import upsert_features

        mock_col = MagicMock()
        mock_result = MagicMock()
        mock_result.upserted_count = 1
        mock_result.modified_count = 0
        mock_col.bulk_write.return_value = mock_result

        docs = [
            {
                "geom_hash": "abc123",
                "source": "test",
                "name": "Park",
                "type": "park",
                "geometry": {"type": "Point", "coordinates": [-6.26, 53.35]},
                "properties": {},
                "area_sqm": None,
                "synthetic": False,
            }
        ]
        result = upsert_features(mock_col, docs)

        mock_col.bulk_write.assert_called_once()
        assert result == 1
