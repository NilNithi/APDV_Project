"""Unit tests for ETL transformation functions.

All tests are offline — no real database, network, or filesystem connections
are made unless the test explicitly constructs them using tmp_path. External
calls (PostgreSQL, MongoDB, Nominatim) are mocked.

Run with::

    pytest tests/test_etl.py -v
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon


# ---------------------------------------------------------------------------
# Test 1 — price parsing
# ---------------------------------------------------------------------------


class TestPriceParsing:
    """Tests for src.etl.property_clean._parse_price."""

    def test_standard_price(self) -> None:
        """Euro symbol + comma-formatted value must parse to correct float."""
        from src.etl.property_clean import _parse_price

        series = pd.Series(["€250,000.00"])
        result = _parse_price(series)
        assert result.iloc[0] == pytest.approx(250_000.0)

    def test_large_price(self) -> None:
        """Prices over one million with multiple commas must parse correctly."""
        from src.etl.property_clean import _parse_price

        series = pd.Series(["€1,250,000"])
        result = _parse_price(series)
        assert result.iloc[0] == pytest.approx(1_250_000.0)

    def test_invalid_string_returns_nan(self) -> None:
        """Non-numeric strings such as 'N/A' must produce NaN."""
        from src.etl.property_clean import _parse_price

        series = pd.Series(["N/A"])
        result = _parse_price(series)
        assert pd.isna(result.iloc[0])

    def test_batch_mixed(self) -> None:
        """Mixed-validity series — valid entries parse correctly, invalid → NaN."""
        from src.etl.property_clean import _parse_price

        series = pd.Series(["€250,000.00", "€1,250,000", "N/A", ""])
        result = _parse_price(series)
        assert result.iloc[0] == pytest.approx(250_000.0)
        assert result.iloc[1] == pytest.approx(1_250_000.0)
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])


# ---------------------------------------------------------------------------
# Test 2 — date parsing
# ---------------------------------------------------------------------------


class TestDateParsing:
    """Tests for date_of_sale parsing inside clean_property_df."""

    def _make_minimal_df(self, dates: list[str]) -> pd.DataFrame:
        """Build a minimal raw property DataFrame with the given date strings."""
        n = len(dates)
        return pd.DataFrame(
            {
                "date_of_sale": dates,
                "price": ["€300,000.00"] * n,
                "address": ["1 Test St, Dublin"] * n,
                "postal_code": ["D01"] * n,
                "county": ["Co. Dublin"] * n,
                "not_full_market_price": ["No"] * n,
                "description": ["Second-Hand Dwelling house /Apartment"] * n,
                "property_size_description": [""] * n,
            }
        )

    def test_dd_mm_yyyy_parsed_correctly(self) -> None:
        """'01/06/2020' must parse to 2020-06-01 and year_of_sale 2020."""
        from src.etl.property_clean import clean_property_df

        df = self._make_minimal_df(["01/06/2020"])
        result = clean_property_df(df)

        assert len(result) == 1
        assert result["date_of_sale"].iloc[0] == pd.Timestamp("2020-06-01")
        assert result["year_of_sale"].iloc[0] == 2020

    def test_multiple_dates(self) -> None:
        """Multiple valid dates must all parse."""
        from src.etl.property_clean import clean_property_df

        df = self._make_minimal_df(["01/06/2020", "15/12/2018"])
        result = clean_property_df(df)

        assert len(result) == 2
        assert result["date_of_sale"].iloc[0] == pd.Timestamp("2020-06-01")
        assert result["date_of_sale"].iloc[1] == pd.Timestamp("2018-12-15")

    def test_invalid_date_row_dropped(self) -> None:
        """Rows with unparseable dates must be silently dropped."""
        from src.etl.property_clean import clean_property_df

        df = self._make_minimal_df(["01/06/2020", "not-a-date"])
        result = clean_property_df(df)

        # Only the valid row should survive.
        assert len(result) == 1
        assert result["date_of_sale"].iloc[0] == pd.Timestamp("2020-06-01")


# ---------------------------------------------------------------------------
# Test 3 — geocode cache hit
# ---------------------------------------------------------------------------


class TestGeocodeCacheHit:
    """Verify that a cached geocode result is returned without calling Nominatim."""

    def test_cache_hit_skips_nominatim(self, tmp_path: Path) -> None:
        """When an address is already in the SQLite cache, Nominatim must not be called."""
        from src.etl.geocode import _init_cache, _cache_store, geocode_address

        cache_db = tmp_path / "geocode_cache.db"
        conn = _init_cache(cache_db)

        address = "1 GRAFTON STREET, DUBLIN"
        _cache_store(conn, address, 53.3411, -6.2598, "nominatim")
        conn.commit()

        mock_geolocator = MagicMock()

        lat, lon, source = geocode_address(
            address, "D02", mock_geolocator, conn
        )

        mock_geolocator.geocode.assert_not_called()
        assert lat == pytest.approx(53.3411)
        assert lon == pytest.approx(-6.2598)
        assert source == "cache"

        conn.close()

    def test_cache_null_entry_skips_nominatim(self, tmp_path: Path) -> None:
        """A cached 'failed' entry (NULL coords) must also suppress Nominatim calls."""
        from src.etl.geocode import _init_cache, _cache_store, geocode_address

        cache_db = tmp_path / "geocode_cache.db"
        conn = _init_cache(cache_db)

        address = "NOWHERE LAND, DUBLIN"
        _cache_store(conn, address, None, None, "failed")
        conn.commit()

        mock_geolocator = MagicMock()
        lat, lon, source = geocode_address(address, "", mock_geolocator, conn)

        mock_geolocator.geocode.assert_not_called()
        assert lat is None
        assert lon is None

        conn.close()


# ---------------------------------------------------------------------------
# Test 4 — postcode centroid fallback
# ---------------------------------------------------------------------------


class TestPostcodeCentroidFallback:
    """Verify Dublin 2 centroid is used when Nominatim returns None."""

    def test_nominatim_none_uses_postcode_centroid(self, tmp_path: Path) -> None:
        """When Nominatim returns None, geocode_address must fall back to the
        Dublin area centroid defined in DUBLIN_CENTROIDS."""
        from src.etl.geocode import (
            _init_cache,
            geocode_address,
            DUBLIN_CENTROIDS,
        )

        cache_db = tmp_path / "geocode_cache.db"
        conn = _init_cache(cache_db)

        address = "99 MYSTERY LANE, DUBLIN 2"

        # Nominatim returns None (simulate failed lookup).
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None

        with patch("src.etl.geocode._RATE_LIMIT_SECS", 0):
            lat, lon, source = geocode_address(
                address, "DUBLIN 2", mock_geolocator, conn
            )

        expected_lat, expected_lon = DUBLIN_CENTROIDS["Dublin 2"]
        assert lat == pytest.approx(expected_lat)
        assert lon == pytest.approx(expected_lon)
        assert source == "postcode"

        conn.close()

    def test_eircode_prefix_fallback(self, tmp_path: Path) -> None:
        """An Eircode routing key (e.g. 'D02') must also resolve to a centroid."""
        from src.etl.geocode import _postcode_centroid, DUBLIN_CENTROIDS

        result = _postcode_centroid("D02 XY45")
        expected = DUBLIN_CENTROIDS["Dublin 2"]
        assert result == expected


# ---------------------------------------------------------------------------
# Test 5 — distance calculation
# ---------------------------------------------------------------------------


class TestDistanceCalculation:
    """Verify that EPSG:2157 distance between two known points is correct."""

    def test_known_itm_distance(self) -> None:
        """Distance between Dublin city centre and Dun Laoghaire in ITM should
        be approximately 11.5 km (within ±500 m tolerance)."""
        import geopandas as gpd
        from shapely.geometry import Point

        # WGS-84 coordinates for the two locations.
        dublin_centre_wgs = Point(-6.2603, 53.3498)  # lon, lat
        dun_laoghaire_wgs = Point(-6.1340, 53.2933)

        pts = gpd.GeoSeries(
            [dublin_centre_wgs, dun_laoghaire_wgs], crs="EPSG:4326"
        ).to_crs(epsg=2157)

        distance_m = pts.iloc[0].distance(pts.iloc[1])

        # Haversine reference: ~11.5 km
        assert 10_500 < distance_m < 12_500, (
            f"Expected ~11 500 m, got {distance_m:.0f} m"
        )

    def test_same_point_distance_is_zero(self) -> None:
        """Distance from a point to itself must be 0."""
        import geopandas as gpd
        from shapely.geometry import Point

        pt_wgs = Point(-6.2603, 53.3498)
        pts = gpd.GeoSeries([pt_wgs, pt_wgs], crs="EPSG:4326").to_crs(epsg=2157)
        assert pts.iloc[0].distance(pts.iloc[1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 6 — buffer area calculation
# ---------------------------------------------------------------------------


class TestBufferAreaCalculation:
    """Verify green-area-in-buffer logic using synthetic geometries in EPSG:2157."""

    def _make_square_park(
        self, centre_x: float, centre_y: float, half_side: float
    ) -> Polygon:
        """Create an axis-aligned square Polygon in EPSG:2157 coordinates.

        Args:
            centre_x: ITM easting of the square centre (metres).
            centre_y: ITM northing of the square centre (metres).
            half_side: Half the side length (metres).

        Returns:
            Shapely Polygon.
        """
        return Polygon(
            [
                (centre_x - half_side, centre_y - half_side),
                (centre_x + half_side, centre_y - half_side),
                (centre_x + half_side, centre_y + half_side),
                (centre_x - half_side, centre_y + half_side),
            ]
        )

    def test_park_within_500m_buffer_counted(self) -> None:
        """A 100 m × 100 m park centred 300 m from a property must appear in
        the 500 m buffer sum (intersection area ≈ 10 000 m²)."""
        import geopandas as gpd
        from src.etl.join import _green_area_in_buffer_rowwise

        # Property at an arbitrary ITM origin.
        prop_x, prop_y = 715_000.0, 734_000.0
        prop_pt = Point(prop_x, prop_y)

        # Park centre 300 m east of the property; 100 m side → 10 000 m².
        park = self._make_square_park(prop_x + 300, prop_y, half_side=50)

        prop_gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[prop_pt], crs="EPSG:2157"
        )
        green_gdf = gpd.GeoDataFrame(
            {"name": ["Test Park"]}, geometry=[park], crs="EPSG:2157"
        )

        areas = _green_area_in_buffer_rowwise(prop_gdf, green_gdf, radius_m=500)

        # The park is entirely within the 500 m buffer, so intersection ≈ 10 000 m².
        assert areas.iloc[0] == pytest.approx(10_000.0, rel=0.01)

    def test_park_outside_500m_not_counted(self) -> None:
        """A park centred 600 m away must not appear in the 500 m buffer sum."""
        import geopandas as gpd
        from src.etl.join import _green_area_in_buffer_rowwise

        prop_x, prop_y = 715_000.0, 734_000.0
        prop_pt = Point(prop_x, prop_y)

        # Park centre 600 m north — entirely outside the 500 m buffer.
        park = self._make_square_park(prop_x, prop_y + 600, half_side=50)

        prop_gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[prop_pt], crs="EPSG:2157"
        )
        green_gdf = gpd.GeoDataFrame(
            {"name": ["Far Park"]}, geometry=[park], crs="EPSG:2157"
        )

        areas = _green_area_in_buffer_rowwise(prop_gdf, green_gdf, radius_m=500)

        assert areas.iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_park_partially_in_buffer(self) -> None:
        """A large park that straddles the 500 m boundary must be partially counted."""
        import geopandas as gpd
        from src.etl.join import _green_area_in_buffer_rowwise

        prop_x, prop_y = 715_000.0, 734_000.0
        prop_pt = Point(prop_x, prop_y)

        # A very large park centred at 450 m — extends from 250 m to 650 m,
        # so about half is inside the 500 m buffer.
        park = self._make_square_park(prop_x + 450, prop_y, half_side=200)

        prop_gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[prop_pt], crs="EPSG:2157"
        )
        green_gdf = gpd.GeoDataFrame(
            {"name": ["Straddling Park"]}, geometry=[park], crs="EPSG:2157"
        )

        areas = _green_area_in_buffer_rowwise(prop_gdf, green_gdf, radius_m=500)

        # Intersection must be positive and less than the park's full area.
        park_area = park.area  # 400 m × 400 m = 160 000 m²
        assert 0 < areas.iloc[0] < park_area

    def test_empty_green_gdf_returns_zero(self) -> None:
        """An empty green GeoDataFrame must produce area = 0 for all properties."""
        import geopandas as gpd
        from src.etl.join import _green_area_in_buffer_rowwise

        prop_gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(715_000, 734_000), Point(716_000, 735_000)],
            crs="EPSG:2157",
        )
        empty_green = gpd.GeoDataFrame(geometry=[], crs="EPSG:2157")

        areas = _green_area_in_buffer_rowwise(prop_gdf, empty_green, radius_m=500)

        assert (areas == 0.0).all()
