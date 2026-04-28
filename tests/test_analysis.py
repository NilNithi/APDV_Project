"""Tests for analysis modules.

All tests are offline — no database connections or filesystem side effects
outside of tmp_path fixtures.  Synthetic DataFrames are used throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_enriched_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a minimal synthetic enriched-property DataFrame.

    Args:
        n: Number of rows.
        seed: NumPy random seed for reproducibility.

    Returns:
        DataFrame with columns expected by all analysis modules.
    """
    rng = np.random.default_rng(seed)
    park_dist = rng.uniform(50, 3000, size=n)
    no2 = rng.uniform(10, 60, size=n)
    price = np.exp(
        12.0
        - 0.3 * np.log1p(park_dist)
        - 0.02 * no2
        + rng.normal(0, 0.3, size=n)
    )
    years = rng.integers(2015, 2025, size=n)
    return pd.DataFrame(
        {
            "price": price,
            "log_price": np.log(price),
            "nearest_park_dist_m": park_dist,
            "green_area_within_500m": rng.uniform(0, 50_000, size=n),
            "green_area_within_1000m": rng.uniform(0, 200_000, size=n),
            "mean_no2_year": no2,
            "mean_pm25_year": rng.uniform(5, 25, size=n),
            "mean_pm10_year": rng.uniform(10, 40, size=n),
            "year_of_sale": years,
            "postal_code": rng.choice(["D01", "D02", "D04", "D06", "D08"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — correlation symmetry
# ---------------------------------------------------------------------------


class TestCorrelationSymmetry:
    """Verify that _pearson_matrix / _spearman_matrix produce symmetric outputs."""

    def test_pearson_matrix_symmetric(self) -> None:
        """Pearson r matrix must be symmetric (r[i,j] == r[j,i])."""
        from src.analysis.correlation import _pearson_matrix

        df = _make_enriched_df(20)[
            ["log_price", "nearest_park_dist_m", "mean_no2_year"]
        ]
        r_mat, _ = _pearson_matrix(df)

        cols = r_mat.columns.tolist()
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                assert r_mat.loc[c1, c2] == pytest.approx(r_mat.loc[c2, c1], abs=1e-12), (
                    f"Pearson matrix not symmetric at [{c1}, {c2}]"
                )

    def test_pearson_diagonal_is_one(self) -> None:
        """Diagonal entries of the Pearson r matrix must all be 1.0."""
        from src.analysis.correlation import _pearson_matrix

        df = _make_enriched_df(20)[["log_price", "nearest_park_dist_m"]]
        r_mat, _ = _pearson_matrix(df)

        for col in r_mat.columns:
            assert r_mat.loc[col, col] == pytest.approx(1.0, abs=1e-12)

    def test_pearson_pval_diagonal_is_zero(self) -> None:
        """Diagonal p-values must be exactly 0.0."""
        from src.analysis.correlation import _pearson_matrix

        df = _make_enriched_df(20)[["log_price", "nearest_park_dist_m"]]
        _, p_mat = _pearson_matrix(df)

        for col in p_mat.columns:
            assert p_mat.loc[col, col] == pytest.approx(0.0, abs=1e-12)

    def test_spearman_matrix_symmetric(self) -> None:
        """Spearman rho matrix must be symmetric."""
        from src.analysis.correlation import _spearman_matrix

        df = _make_enriched_df(20)[
            ["log_price", "nearest_park_dist_m", "mean_no2_year"]
        ]
        r_mat, _ = _spearman_matrix(df)

        cols = r_mat.columns.tolist()
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                assert r_mat.loc[c1, c2] == pytest.approx(r_mat.loc[c2, c1], abs=1e-12), (
                    f"Spearman matrix not symmetric at [{c1}, {c2}]"
                )

    def test_pearson_values_in_range(self) -> None:
        """All Pearson r values must lie in [-1, 1]."""
        from src.analysis.correlation import _pearson_matrix

        df = _make_enriched_df(50)[
            ["log_price", "nearest_park_dist_m", "mean_no2_year", "mean_pm25_year"]
        ]
        r_mat, _ = _pearson_matrix(df)

        values = r_mat.values.flatten()
        assert np.all(values >= -1.0 - 1e-9)
        assert np.all(values <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Test 2 — regression recovers known coefficient
# ---------------------------------------------------------------------------


class TestRegressionSynthetic:
    """Verify that OLS recovers a known slope on synthetic data."""

    def _make_green_only_df(self, n: int = 300, seed: int = 0) -> pd.DataFrame:
        """Return a DataFrame where log_price = 2.0 * green + noise.

        Args:
            n: Number of rows.
            seed: NumPy random seed.

        Returns:
            DataFrame with columns ``log_price`` and ``green``.
        """
        rng = np.random.default_rng(seed)
        green = rng.uniform(0, 10, size=n)
        log_price = 2.0 * green + rng.normal(0, 0.5, size=n)
        return pd.DataFrame({"log_price": log_price, "green": green})

    def test_ols_recovers_coefficient_near_two(self) -> None:
        """OLS on log_price ~ green should estimate coef close to 2.0."""
        df = self._make_green_only_df()
        model = smf.ols("log_price ~ green", data=df).fit()

        coef = model.params["green"]
        ci_lo, ci_hi = model.conf_int().loc["green"]

        # The true value (2.0) should lie within the 95% CI
        assert ci_lo <= 2.0 <= ci_hi, (
            f"True coef 2.0 not within 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]"
        )
        # Point estimate should be within 0.3 of 2.0
        assert abs(coef - 2.0) < 0.3, (
            f"OLS coef {coef:.3f} is too far from true value 2.0"
        )

    def test_ols_r2_reasonable(self) -> None:
        """R² for the synthetic known-relationship should be reasonably high."""
        df = self._make_green_only_df(n=500)
        model = smf.ols("log_price ~ green", data=df).fit()
        # With noise std=0.5 and signal range ~20, R² should be > 0.8
        assert model.rsquared > 0.8, (
            f"R²={model.rsquared:.3f} unexpectedly low for synthetic data"
        )

    def test_extract_coef_df_shape(self) -> None:
        """_extract_coef_df must return a DataFrame with the right columns."""
        from src.analysis.regression import _extract_coef_df

        df = self._make_green_only_df(n=100)
        model = smf.ols("log_price ~ green", data=df).fit()
        coef_df = _extract_coef_df(model)

        expected_cols = {"coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"}
        assert expected_cols.issubset(set(coef_df.columns)), (
            f"Missing columns: {expected_cols - set(coef_df.columns)}"
        )
        # Intercept + green = 2 rows
        assert len(coef_df) == 2

    def test_extract_coef_df_ci_ordering(self) -> None:
        """ci_lower must always be <= ci_upper for every parameter."""
        from src.analysis.regression import _extract_coef_df

        df = self._make_green_only_df(n=200)
        model = smf.ols("log_price ~ green", data=df).fit()
        coef_df = _extract_coef_df(model)

        assert (coef_df["ci_lower"] <= coef_df["ci_upper"]).all(), (
            "ci_lower > ci_upper for some parameters"
        )


# ---------------------------------------------------------------------------
# Test 3 — quintile count
# ---------------------------------------------------------------------------


class TestQuintileCount:
    """Verify that pd.qcut on park distance yields exactly 5 distinct groups."""

    def test_quintile_five_groups(self) -> None:
        """100-row DataFrame qcut into 5 bins must produce exactly 5 distinct labels."""
        df = _make_enriched_df(100)
        df["green_quintile"] = pd.qcut(
            df["nearest_park_dist_m"],
            q=5,
            labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"],
            duplicates="drop",
        )
        n_groups = df["green_quintile"].nunique()
        assert n_groups == 5, f"Expected 5 quintile groups, got {n_groups}"

    def test_quintile_labels_match(self) -> None:
        """Quintile labels must be exactly the five expected strings."""
        df = _make_enriched_df(100)
        expected_labels = {"Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"}
        df["green_quintile"] = pd.qcut(
            df["nearest_park_dist_m"],
            q=5,
            labels=list(expected_labels),
            duplicates="drop",
        )
        actual_labels = set(df["green_quintile"].dropna().unique().astype(str))
        assert actual_labels == expected_labels, (
            f"Unexpected labels: {actual_labels}"
        )

    def test_quintile_roughly_equal_bins(self) -> None:
        """Each quintile bin should contain roughly n/5 rows (within factor of 2)."""
        n = 200
        df = _make_enriched_df(n)
        df["green_quintile"] = pd.qcut(
            df["nearest_park_dist_m"],
            q=5,
            labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"],
            duplicates="drop",
        )
        counts = df["green_quintile"].value_counts()
        expected = n / 5
        for label, count in counts.items():
            assert count >= expected / 2, (
                f"Quintile {label} has only {count} rows (expected ~{expected:.0f})"
            )

    def test_quintile_no_nulls_on_clean_data(self) -> None:
        """qcut on data with no NaN park distances should produce no NaN quintiles."""
        df = _make_enriched_df(50)
        assert df["nearest_park_dist_m"].isna().sum() == 0, "Fixture has NaN park dist"

        df["green_quintile"] = pd.qcut(
            df["nearest_park_dist_m"],
            q=5,
            labels=["Q1_closest", "Q2", "Q3", "Q4", "Q5_farthest"],
            duplicates="drop",
        )
        assert df["green_quintile"].isna().sum() == 0, (
            "Unexpected NaN in quintile column"
        )
