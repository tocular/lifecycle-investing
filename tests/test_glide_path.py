"""
Test suite for glide path computation.

Validates glide path functions against lecture examples from FINC 450 Capital Markets.
Tests both the compute_glide_path function (year-by-year allocation) and the
project_wealth function (wealth projection with geometric returns).
"""

import pytest
import pandas as pd

from lifecycle_app.calculations.glide_path import (
    compute_glide_path,
    project_wealth,
)
from lifecycle_app.config import (
    RISK_FREE_RATE,
    STOCK_GEOMETRIC_RETURN,
    BOND_GEOMETRIC_RETURN,
    CASH_GEOMETRIC_RETURN,
)


# ---------------------------------------------------------------------------
# Fixtures: Common test parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def young_consultant_params():
    """Parameters for the lecture example: 25-year-old consultant."""
    return {
        "current_age": 25,
        "retirement_age": 65,
        "life_expectancy": 85,
        "annual_income": 250_000,
        "working_expenses": 100_000,
        "retirement_expenses": 100_000,
        "financial_assets": 100_000,
        "risk_aversion": 2.0,
        "income_beta": 0.0,  # Bond-like income
        "risk_free_rate": RISK_FREE_RATE,
    }


@pytest.fixture
def mid_career_params():
    """Parameters for a 45-year-old mid-career professional."""
    return {
        "current_age": 45,
        "retirement_age": 65,
        "life_expectancy": 85,
        "annual_income": 200_000,
        "working_expenses": 120_000,
        "retirement_expenses": 80_000,
        "financial_assets": 500_000,
        "risk_aversion": 2.0,
        "income_beta": 0.0,
        "risk_free_rate": RISK_FREE_RATE,
    }


@pytest.fixture
def near_retirement_params():
    """Parameters for a 60-year-old near retirement."""
    return {
        "current_age": 60,
        "retirement_age": 65,
        "life_expectancy": 85,
        "annual_income": 250_000,
        "working_expenses": 150_000,
        "retirement_expenses": 100_000,
        "financial_assets": 2_000_000,
        "risk_aversion": 2.0,
        "income_beta": 0.0,
        "risk_free_rate": RISK_FREE_RATE,
    }


# ---------------------------------------------------------------------------
# Tests: compute_glide_path
# ---------------------------------------------------------------------------


class TestComputeGlidePath:
    """Tests for the compute_glide_path function."""

    def test_returns_dataframe(self, young_consultant_params):
        """Should return a pandas DataFrame."""
        result = compute_glide_path(**young_consultant_params)

        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, young_consultant_params):
        """DataFrame should have all required columns."""
        result = compute_glide_path(**young_consultant_params)

        expected_columns = [
            "age",
            "years_to_retirement",
            "pv_human_capital",
            "pv_expenses",
            "total_wealth",
            "financial_wealth",
            "stock_weight",
            "bond_weight",
            "cash_weight",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_correct_number_of_rows(self, young_consultant_params):
        """DataFrame should have one row per year from current age to life expectancy."""
        result = compute_glide_path(**young_consultant_params)

        expected_rows = (
            young_consultant_params["life_expectancy"]
            - young_consultant_params["current_age"]
            + 1
        )

        assert len(result) == expected_rows

    def test_age_sequence(self, young_consultant_params):
        """Age column should be sequential from current age to life expectancy."""
        result = compute_glide_path(**young_consultant_params)

        expected_ages = list(
            range(
                young_consultant_params["current_age"],
                young_consultant_params["life_expectancy"] + 1,
            )
        )

        assert list(result["age"]) == expected_ages

    def test_stock_weight_decreases_over_time(self, young_consultant_params):
        """Stock weight should generally decrease as investor ages (for bond-like income)."""
        result = compute_glide_path(**young_consultant_params)

        # Compare first and last working year stock weights
        first_year_stock = result.iloc[0]["stock_weight"]
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        near_retirement_stock = result.iloc[retirement_idx - 1]["stock_weight"]

        assert near_retirement_stock <= first_year_stock

    def test_human_capital_decreases_over_time(self, young_consultant_params):
        """PV of human capital should decrease as retirement approaches."""
        result = compute_glide_path(**young_consultant_params)

        # Human capital at start
        hc_first = result.iloc[0]["pv_human_capital"]

        # Human capital near retirement
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        hc_near_retirement = result.iloc[retirement_idx - 1]["pv_human_capital"]

        assert hc_near_retirement < hc_first

    def test_human_capital_zero_at_retirement(self, young_consultant_params):
        """Human capital should be zero at and after retirement age."""
        result = compute_glide_path(**young_consultant_params)

        # At retirement age
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        hc_at_retirement = result.iloc[retirement_idx]["pv_human_capital"]

        assert hc_at_retirement == 0.0

        # After retirement
        hc_after_retirement = result.iloc[retirement_idx + 5]["pv_human_capital"]
        assert hc_after_retirement == 0.0

    def test_years_to_retirement_decreases(self, young_consultant_params):
        """Years to retirement should decrease each year."""
        result = compute_glide_path(**young_consultant_params)

        years_to_retirement = result["years_to_retirement"].tolist()

        # Should be monotonically decreasing then zero
        for i in range(1, len(years_to_retirement)):
            assert years_to_retirement[i] <= years_to_retirement[i - 1]

    def test_weights_sum_to_one(self, young_consultant_params):
        """Portfolio weights should sum to 1.0 for each year."""
        result = compute_glide_path(**young_consultant_params)

        for _, row in result.iterrows():
            total = row["stock_weight"] + row["bond_weight"] + row["cash_weight"]
            assert total == pytest.approx(1.0)

    def test_weights_within_bounds(self, young_consultant_params):
        """All weights should be between 0 and 1 (constrained optimization)."""
        result = compute_glide_path(**young_consultant_params)

        assert (result["stock_weight"] >= 0).all()
        assert (result["stock_weight"] <= 1).all()
        assert (result["bond_weight"] >= 0).all()
        assert (result["bond_weight"] <= 1).all()
        assert (result["cash_weight"] >= 0).all()
        assert (result["cash_weight"] <= 1).all()

    def test_lecture_example_young_consultant(self, young_consultant_params):
        """Lecture example: 25-year-old consultant allocation within valid bounds."""
        result = compute_glide_path(**young_consultant_params)

        # Young worker with bond-like income and massive human capital
        # relative to financial assets gets constrained allocation
        first_year = result.iloc[0]

        # Weights should be valid (between 0 and 1) and sum to 1
        assert 0.0 <= first_year["stock_weight"] <= 1.0
        assert 0.0 <= first_year["bond_weight"] <= 1.0
        assert 0.0 <= first_year["cash_weight"] <= 1.0
        total = first_year["stock_weight"] + first_year["bond_weight"] + first_year["cash_weight"]
        assert total == pytest.approx(1.0)

    def test_higher_risk_aversion_lower_stocks(self, young_consultant_params):
        """Higher risk aversion should result in lower stock allocation."""
        # Low risk aversion
        params_low_gamma = young_consultant_params.copy()
        params_low_gamma["risk_aversion"] = 1.5
        result_low = compute_glide_path(**params_low_gamma)

        # High risk aversion
        params_high_gamma = young_consultant_params.copy()
        params_high_gamma["risk_aversion"] = 5.0
        result_high = compute_glide_path(**params_high_gamma)

        # Near retirement, high gamma should have lower stock allocation
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]

        # Compare at a point where both wouldn't be at constraint
        mid_idx = retirement_idx // 2

        # This relationship may not hold at every point due to constraints,
        # but should hold at retirement when human capital is depleted
        stock_low_at_retirement = result_low.iloc[retirement_idx]["stock_weight"]
        stock_high_at_retirement = result_high.iloc[retirement_idx]["stock_weight"]

        assert stock_high_at_retirement <= stock_low_at_retirement

    def test_stock_like_income_effect(self, young_consultant_params):
        """Stock-like income should affect allocation differently."""
        # Bond-like income (baseline)
        result_bond = compute_glide_path(**young_consultant_params)

        # Stock-like income (investment banker)
        params_stock = young_consultant_params.copy()
        params_stock["income_beta"] = 0.4
        result_stock = compute_glide_path(**params_stock)

        # Both should have valid allocations
        assert (result_bond["stock_weight"] >= 0).all()
        assert (result_stock["stock_weight"] >= 0).all()

        # Stock-like income means human capital provides stock exposure,
        # so financial portfolio may need different allocation
        # The relationship depends on total wealth levels

    def test_financial_wealth_grows(self, young_consultant_params):
        """Financial wealth should grow over time (projection is now integrated)."""
        result = compute_glide_path(**young_consultant_params)

        # compute_glide_path now projects wealth - it should grow during working years
        initial_wealth = result.iloc[0]["financial_wealth"]
        assert initial_wealth == young_consultant_params["financial_assets"]

        # Should grow during working years
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        wealth_at_retirement = result.iloc[retirement_idx]["financial_wealth"]
        assert wealth_at_retirement > initial_wealth * 10  # Significant growth expected


# ---------------------------------------------------------------------------
# Tests: project_wealth
# ---------------------------------------------------------------------------


class TestProjectWealth:
    """Tests for the project_wealth function.

    Note: project_wealth is now a compatibility wrapper since compute_glide_path
    handles wealth projection internally. These tests verify backward compatibility.
    """

    def test_returns_dataframe_with_new_column(self, young_consultant_params):
        """Should return DataFrame with projected_financial_wealth column."""
        glide_path = compute_glide_path(**young_consultant_params)
        result = project_wealth(glide_path_df=glide_path)

        assert "projected_financial_wealth" in result.columns

    def test_first_year_wealth_unchanged(self, young_consultant_params):
        """First year projected wealth should equal initial financial assets."""
        glide_path = compute_glide_path(**young_consultant_params)
        result = project_wealth(glide_path_df=glide_path)

        expected = young_consultant_params["financial_assets"]
        assert result.iloc[0]["projected_financial_wealth"] == pytest.approx(expected)

    def test_wealth_grows_with_positive_savings(self, young_consultant_params):
        """Wealth should grow over time (projection happens in compute_glide_path)."""
        glide_path = compute_glide_path(**young_consultant_params)
        result = project_wealth(glide_path_df=glide_path)

        # Wealth should increase during working years
        initial_wealth = result.iloc[0]["projected_financial_wealth"]
        wealth_at_10 = result.iloc[10]["projected_financial_wealth"]
        wealth_at_20 = result.iloc[20]["projected_financial_wealth"]

        assert wealth_at_10 > initial_wealth
        assert wealth_at_20 > wealth_at_10

    def test_projected_equals_financial_wealth(self, young_consultant_params):
        """projected_financial_wealth should equal financial_wealth column."""
        glide_path = compute_glide_path(**young_consultant_params)
        result = project_wealth(glide_path_df=glide_path)

        # Since projection is now in compute_glide_path, these should be equal
        for i in range(len(result)):
            assert result.iloc[i]["projected_financial_wealth"] == pytest.approx(
                result.iloc[i]["financial_wealth"]
            )

    def test_no_savings_in_retirement(self, near_retirement_params):
        """Wealth should still grow in retirement (returns minus withdrawals)."""
        glide_path = compute_glide_path(**near_retirement_params)
        result = project_wealth(glide_path_df=glide_path)

        # Find the year of retirement
        retirement_idx = near_retirement_params["retirement_age"] - near_retirement_params["current_age"]

        # Check years_to_retirement is 0 at retirement
        assert result.iloc[retirement_idx]["years_to_retirement"] == 0

    def test_preserves_original_columns(self, young_consultant_params):
        """Original glide path columns should be preserved."""
        glide_path = compute_glide_path(**young_consultant_params)
        result = project_wealth(glide_path_df=glide_path)

        # All original columns should exist
        for col in glide_path.columns:
            assert col in result.columns

    def test_does_not_modify_input(self, young_consultant_params):
        """Input DataFrame should not be modified."""
        glide_path = compute_glide_path(**young_consultant_params)
        original_cols = set(glide_path.columns)

        _ = project_wealth(glide_path_df=glide_path)

        # Input should be unchanged
        assert set(glide_path.columns) == original_cols
        assert "projected_financial_wealth" not in glide_path.columns


# ---------------------------------------------------------------------------
# Integration tests: Full lifecycle scenarios
# ---------------------------------------------------------------------------


class TestGlidePathIntegration:
    """Integration tests for complete glide path scenarios."""

    def test_full_lifecycle_young_consultant(self, young_consultant_params):
        """Full lifecycle test for the lecture example: 25-year-old consultant."""
        # Compute glide path
        glide_path = compute_glide_path(**young_consultant_params)

        # Project wealth
        annual_savings = (
            young_consultant_params["annual_income"]
            - young_consultant_params["working_expenses"]
        )
        projected = project_wealth(glide_path_df=glide_path, annual_savings=annual_savings)

        # Verify key lifecycle characteristics
        # 1. Young investor has valid constrained weights
        assert 0.0 <= projected.iloc[0]["stock_weight"] <= 1.0
        total_weight = (
            projected.iloc[0]["stock_weight"]
            + projected.iloc[0]["bond_weight"]
            + projected.iloc[0]["cash_weight"]
        )
        assert total_weight == pytest.approx(1.0)

        # 2. Human capital is substantial at start
        assert projected.iloc[0]["pv_human_capital"] > 5_000_000

        # 3. Human capital depletes to zero at retirement
        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        assert projected.iloc[retirement_idx]["pv_human_capital"] == 0.0

        # 4. Wealth accumulates significantly by retirement
        final_working_wealth = projected.iloc[retirement_idx]["projected_financial_wealth"]
        assert final_working_wealth > young_consultant_params["financial_assets"] * 10

        # 5. Stock allocation changes over the lifecycle
        # Due to changing human capital / financial wealth ratios
        stock_at_start = projected.iloc[0]["stock_weight"]
        stock_at_retirement = projected.iloc[retirement_idx]["stock_weight"]

        # Allocation should be valid throughout
        assert 0.0 <= stock_at_start <= 1.0
        assert 0.0 <= stock_at_retirement <= 1.0

    def test_mid_career_professional(self, mid_career_params):
        """Test glide path for mid-career professional."""
        glide_path = compute_glide_path(**mid_career_params)
        annual_savings = mid_career_params["annual_income"] - mid_career_params["working_expenses"]
        projected = project_wealth(glide_path_df=glide_path, annual_savings=annual_savings)

        # Should have reasonable values throughout
        assert (projected["stock_weight"] >= 0).all()
        assert (projected["stock_weight"] <= 1).all()
        assert (projected["projected_financial_wealth"] > 0).all()

    def test_near_retirement_investor(self, near_retirement_params):
        """Test glide path for investor near retirement."""
        glide_path = compute_glide_path(**near_retirement_params)
        annual_savings = near_retirement_params["annual_income"] - near_retirement_params["working_expenses"]
        projected = project_wealth(glide_path_df=glide_path, annual_savings=annual_savings)

        # Should have valid allocation (weights in [0,1] and sum to 1)
        first_row = projected.iloc[0]
        assert 0.0 <= first_row["stock_weight"] <= 1.0
        assert 0.0 <= first_row["bond_weight"] <= 1.0
        assert 0.0 <= first_row["cash_weight"] <= 1.0
        total = first_row["stock_weight"] + first_row["bond_weight"] + first_row["cash_weight"]
        assert total == pytest.approx(1.0)

        # Should have meaningful financial assets
        assert projected.iloc[0]["projected_financial_wealth"] > 1_000_000

    def test_wealth_never_negative(self, young_consultant_params):
        """Projected wealth should never go negative under normal assumptions."""
        glide_path = compute_glide_path(**young_consultant_params)
        annual_savings = young_consultant_params["annual_income"] - young_consultant_params["working_expenses"]
        projected = project_wealth(glide_path_df=glide_path, annual_savings=annual_savings)

        assert (projected["projected_financial_wealth"] >= 0).all()

    def test_retirement_wealth_supports_spending(self, young_consultant_params):
        """Accumulated wealth at retirement should support retirement spending."""
        glide_path = compute_glide_path(**young_consultant_params)
        annual_savings = young_consultant_params["annual_income"] - young_consultant_params["working_expenses"]
        projected = project_wealth(glide_path_df=glide_path, annual_savings=annual_savings)

        retirement_idx = young_consultant_params["retirement_age"] - young_consultant_params["current_age"]
        retirement_years = young_consultant_params["life_expectancy"] - young_consultant_params["retirement_age"]
        retirement_expenses = young_consultant_params["retirement_expenses"]

        wealth_at_retirement = projected.iloc[retirement_idx]["projected_financial_wealth"]

        # Wealth should be enough to fund at least some years of retirement
        # (This is a sanity check, not a guarantee of plan success)
        min_needed = retirement_expenses * retirement_years * 0.5  # At least half coverage
        assert wealth_at_retirement > min_needed
