"""
Test suite for present value calculations.

Validates PV functions against lecture examples from FINC 450 Capital Markets.
Uses pytest fixtures for common test parameters and pytest.approx for
floating-point tolerance in assertions.
"""

import pytest

from lifecycle_app.calculations.present_value import (
    pv_annuity,
    pv_human_capital,
    pv_expenses,
    total_wealth,
)


# ---------------------------------------------------------------------------
# Fixtures: Common test parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def lecture_rate():
    """Standard discount rate from lecture examples (2% real)."""
    return 0.02


@pytest.fixture
def standard_annuity_params():
    """Parameters for standard annuity test (lecture example)."""
    return {
        "payment": 100_000,
        "rate": 0.02,
        "n_periods": 50,
    }


@pytest.fixture
def human_capital_params():
    """Parameters for human capital PV test (lecture example)."""
    return {
        "annual_income": 250_000,
        "years_working": 35,
        "risk_free_rate": 0.02,
    }


@pytest.fixture
def expense_params():
    """Parameters for expense PV calculations."""
    return {
        "working_expenses": 100_000,
        "retirement_expenses": 100_000,
        "years_working": 35,
        "years_retirement": 15,
        "risk_free_rate": 0.02,
    }


# ---------------------------------------------------------------------------
# Tests: pv_annuity
# ---------------------------------------------------------------------------


class TestPvAnnuity:
    """Tests for the pv_annuity function."""

    def test_lecture_example(self, standard_annuity_params):
        """Lecture example: $100k/year for 50 years at 2% should equal ~$3.14M."""
        result = pv_annuity(**standard_annuity_params)

        # Lecture states approximately $3.14M
        # Using standard ordinary annuity formula: PV = PMT * [(1 - (1+r)^-n) / r]
        # PV = 100,000 * [(1 - (1.02)^-50) / 0.02] = 3,142,360.59
        assert result == pytest.approx(3_142_360.59, rel=0.001)

    def test_zero_rate(self):
        """When rate=0, PV should be simple sum of payments."""
        payment = 100_000
        n_periods = 10
        result = pv_annuity(payment, rate=0, n_periods=n_periods)

        assert result == pytest.approx(payment * n_periods)

    def test_zero_periods(self):
        """When n_periods=0, PV should be 0."""
        result = pv_annuity(payment=100_000, rate=0.02, n_periods=0)

        assert result == 0.0

    def test_negative_periods(self):
        """Negative periods should return 0."""
        result = pv_annuity(payment=100_000, rate=0.02, n_periods=-5)

        assert result == 0.0

    def test_single_period(self):
        """Single period annuity is simply discounted payment."""
        payment = 100_000
        rate = 0.02
        result = pv_annuity(payment, rate, n_periods=1)

        # PV of single payment = payment / (1 + rate)
        expected = payment / (1 + rate)
        assert result == pytest.approx(expected)

    def test_high_rate(self):
        """Higher discount rate should result in lower PV."""
        payment = 100_000
        n_periods = 20

        pv_low_rate = pv_annuity(payment, rate=0.02, n_periods=n_periods)
        pv_high_rate = pv_annuity(payment, rate=0.10, n_periods=n_periods)

        assert pv_high_rate < pv_low_rate

    def test_more_periods_increases_pv(self):
        """More periods should increase PV (more payments received)."""
        payment = 100_000
        rate = 0.02

        pv_short = pv_annuity(payment, rate, n_periods=10)
        pv_long = pv_annuity(payment, rate, n_periods=30)

        assert pv_long > pv_short


# ---------------------------------------------------------------------------
# Tests: pv_human_capital
# ---------------------------------------------------------------------------


class TestPvHumanCapital:
    """Tests for the pv_human_capital function."""

    def test_lecture_example(self, human_capital_params):
        """Lecture example: $250k/year for 35 years at 2% should equal ~$6.25M."""
        result = pv_human_capital(**human_capital_params)

        # Using annuity formula: 250,000 * [(1 - (1.02)^-35) / 0.02]
        # = 250,000 * 24.99862 = 6,249,654.78
        assert result == pytest.approx(6_249_654.78, rel=0.001)

    def test_stock_like_income_beta(self):
        """Stock-like income (beta=0.4) should have higher discount rate, lower PV."""
        annual_income = 250_000
        years_working = 35
        risk_free_rate = 0.02
        equity_premium = 0.04

        # Bond-like income (beta=0)
        pv_bond_like = pv_human_capital(
            annual_income=annual_income,
            years_working=years_working,
            risk_free_rate=risk_free_rate,
            income_beta=0.0,
            equity_premium=equity_premium,
        )

        # Stock-like income (beta=0.4)
        # Discount rate = 0.02 + 0.4 * 0.04 = 0.036 (3.6%)
        pv_stock_like = pv_human_capital(
            annual_income=annual_income,
            years_working=years_working,
            risk_free_rate=risk_free_rate,
            income_beta=0.4,
            equity_premium=equity_premium,
        )

        # Stock-like income is riskier, so has lower PV
        assert pv_stock_like < pv_bond_like

        # Verify approximate value with 3.6% discount rate
        # 250,000 * [(1 - (1.036)^-35) / 0.036] = 4,930,506.57
        assert pv_stock_like == pytest.approx(4_930_506.57, rel=0.01)

    def test_zero_years_working(self):
        """Zero working years should return 0."""
        result = pv_human_capital(
            annual_income=250_000,
            years_working=0,
            risk_free_rate=0.02,
        )

        assert result == 0.0

    def test_default_rate_used(self):
        """Should use RISK_FREE_RATE when not specified."""
        from lifecycle_app.config import RISK_FREE_RATE

        result_explicit = pv_human_capital(
            annual_income=100_000,
            years_working=20,
            risk_free_rate=RISK_FREE_RATE,
        )

        result_default = pv_human_capital(
            annual_income=100_000,
            years_working=20,
        )

        assert result_default == pytest.approx(result_explicit)


# ---------------------------------------------------------------------------
# Tests: pv_expenses
# ---------------------------------------------------------------------------


class TestPvExpenses:
    """Tests for the pv_expenses function."""

    def test_working_and_retirement_combined(self, expense_params):
        """Test combined working and retirement expenses properly discounted."""
        result = pv_expenses(**expense_params)

        # Working expenses: $100k/year for 35 years at 2%
        # = 100,000 * [(1 - (1.02)^-35) / 0.02] = 2,499,861.91
        pv_working_expected = 2_499_861.91

        # Retirement expenses: $100k/year for 15 years at 2%, discounted 35 years
        # = 100,000 * [(1 - (1.02)^-15) / 0.02] * (1.02)^-35
        # = 1,284,936.37 * 0.5000 = 642,468.18 (approximately)
        pv_retirement_expected = 642_468.18

        expected_total = pv_working_expected + pv_retirement_expected

        assert result == pytest.approx(expected_total, rel=0.01)

    def test_only_working_expenses(self, lecture_rate):
        """Test with zero retirement expenses."""
        result = pv_expenses(
            working_expenses=100_000,
            retirement_expenses=0,
            years_working=35,
            years_retirement=15,
            risk_free_rate=lecture_rate,
        )

        # Should equal just the PV of working expenses
        expected = pv_annuity(100_000, lecture_rate, 35)
        assert result == pytest.approx(expected)

    def test_only_retirement_expenses(self, lecture_rate):
        """Test with zero working expenses."""
        result = pv_expenses(
            working_expenses=0,
            retirement_expenses=100_000,
            years_working=35,
            years_retirement=15,
            risk_free_rate=lecture_rate,
        )

        # Should be PV of retirement annuity, discounted back 35 years
        pv_at_retirement = pv_annuity(100_000, lecture_rate, 15)
        discount_factor = (1 + lecture_rate) ** (-35)
        expected = pv_at_retirement * discount_factor

        assert result == pytest.approx(expected)

    def test_zero_rate(self):
        """When rate=0, PV should be simple sum of all expenses."""
        result = pv_expenses(
            working_expenses=100_000,
            retirement_expenses=80_000,
            years_working=35,
            years_retirement=15,
            risk_free_rate=0,
        )

        expected = (100_000 * 35) + (80_000 * 15)
        assert result == pytest.approx(expected)

    def test_no_working_years_remaining(self, lecture_rate):
        """Test when already at retirement."""
        result = pv_expenses(
            working_expenses=100_000,
            retirement_expenses=80_000,
            years_working=0,
            years_retirement=20,
            risk_free_rate=lecture_rate,
        )

        # Only retirement expenses matter
        expected = pv_annuity(80_000, lecture_rate, 20)
        assert result == pytest.approx(expected)

    def test_higher_retirement_expenses(self, lecture_rate):
        """Verify calculation handles different working vs retirement expenses."""
        result_higher_retirement = pv_expenses(
            working_expenses=50_000,
            retirement_expenses=100_000,
            years_working=35,
            years_retirement=20,
            risk_free_rate=lecture_rate,
        )

        result_lower_retirement = pv_expenses(
            working_expenses=50_000,
            retirement_expenses=50_000,
            years_working=35,
            years_retirement=20,
            risk_free_rate=lecture_rate,
        )

        assert result_higher_retirement > result_lower_retirement


# ---------------------------------------------------------------------------
# Tests: total_wealth
# ---------------------------------------------------------------------------


class TestTotalWealth:
    """Tests for the total_wealth function."""

    def test_basic_calculation(self):
        """Verify total wealth formula: financial + human_capital - expenses."""
        financial_assets = 50_000
        hc_pv = 6_200_000
        exp_pv = 3_100_000

        result = total_wealth(financial_assets, hc_pv, exp_pv)

        expected = financial_assets + hc_pv - exp_pv  # 3,150,000
        assert result == pytest.approx(expected)

    def test_lecture_style_example(self):
        """Lecture example: $50k assets, $6.2M human capital, $3.1M expenses."""
        result = total_wealth(
            financial_assets=50_000,
            pv_human_capital=6_200_000,
            pv_expenses=3_100_000,
        )

        assert result == pytest.approx(3_150_000)

    def test_negative_total_wealth(self):
        """Total wealth can be negative if liabilities exceed assets."""
        result = total_wealth(
            financial_assets=10_000,
            pv_human_capital=1_000_000,
            pv_expenses=2_000_000,
        )

        # 10,000 + 1,000,000 - 2,000,000 = -990,000
        assert result == pytest.approx(-990_000)

    def test_zero_human_capital(self):
        """Test with zero human capital (retiree)."""
        result = total_wealth(
            financial_assets=2_000_000,
            pv_human_capital=0,
            pv_expenses=1_500_000,
        )

        assert result == pytest.approx(500_000)

    def test_zero_expenses(self):
        """Test with zero expenses."""
        result = total_wealth(
            financial_assets=100_000,
            pv_human_capital=5_000_000,
            pv_expenses=0,
        )

        assert result == pytest.approx(5_100_000)

    def test_all_zeros(self):
        """Test with all zero inputs."""
        result = total_wealth(
            financial_assets=0,
            pv_human_capital=0,
            pv_expenses=0,
        )

        assert result == pytest.approx(0)


# ---------------------------------------------------------------------------
# Integration tests: End-to-end scenarios
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple PV functions."""

    def test_full_lifecycle_calculation(self, lecture_rate):
        """Test complete lifecycle wealth calculation matching lecture scenario."""
        # 25-year-old consultant (lecture example)
        annual_income = 250_000
        working_expenses = 100_000
        retirement_expenses = 100_000
        years_working = 40  # retire at 65
        years_retirement = 20  # live to 85
        financial_assets = 100_000

        # Calculate human capital PV
        hc_pv = pv_human_capital(
            annual_income=annual_income,
            years_working=years_working,
            risk_free_rate=lecture_rate,
        )

        # Calculate expenses PV
        exp_pv = pv_expenses(
            working_expenses=working_expenses,
            retirement_expenses=retirement_expenses,
            years_working=years_working,
            years_retirement=years_retirement,
            risk_free_rate=lecture_rate,
        )

        # Calculate total wealth
        tw = total_wealth(financial_assets, hc_pv, exp_pv)

        # Human capital should be substantial for young worker
        assert hc_pv > 6_000_000

        # Expenses should be less than income present value
        assert exp_pv < hc_pv

        # Total wealth should be positive
        assert tw > 0

    def test_near_retirement_scenario(self, lecture_rate):
        """Test scenario for worker near retirement."""
        # 60-year-old with 5 years to retirement
        hc_pv = pv_human_capital(
            annual_income=250_000,
            years_working=5,
            risk_free_rate=lecture_rate,
        )

        exp_pv = pv_expenses(
            working_expenses=100_000,
            retirement_expenses=100_000,
            years_working=5,
            years_retirement=20,
            risk_free_rate=lecture_rate,
        )

        tw = total_wealth(
            financial_assets=2_000_000,
            pv_human_capital=hc_pv,
            pv_expenses=exp_pv,
        )

        # Human capital much smaller near retirement
        assert hc_pv < 1_500_000

        # Financial assets now dominate
        assert tw > 0
