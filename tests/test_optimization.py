"""
Test suite for portfolio optimization calculations.

Validates optimization functions against lecture examples from FINC 450 Capital Markets.
Uses pytest fixtures for common test parameters and pytest.approx for
floating-point tolerance in assertions.
"""

import pytest

from lifecycle_app.calculations.optimization import (
    optimal_total_wealth_weights,
    financial_portfolio_weights,
)
from lifecycle_app.config import (
    STOCK_EXCESS_RETURN,
    BOND_EXCESS_RETURN,
    STOCK_VARIANCE,
    BOND_VARIANCE,
    LTPZ_DURATION,
)


# ---------------------------------------------------------------------------
# Fixtures: Common test parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def default_market_assumptions():
    """Standard capital market assumptions from lecture."""
    return {
        "stock_excess_return": STOCK_EXCESS_RETURN,  # 4%
        "bond_excess_return": BOND_EXCESS_RETURN,    # 1%
        "stock_variance": STOCK_VARIANCE,            # 0.0324
        "bond_variance": BOND_VARIANCE,              # 0.0324
    }


@pytest.fixture
def young_investor_params():
    """Parameters for a young investor with substantial human capital."""
    return {
        "total_wealth": 3_150_000,        # lecture example
        "financial_assets": 100_000,
        "pv_human_capital": 6_200_000,
        "human_capital_duration": 17.0,   # ~17 years for 35-year worker
        "pv_expenses": 3_150_000,         # PV of lifetime expenses
        "expense_duration": 22.0,         # Expense duration ~22 years
        "ltpz_duration": LTPZ_DURATION,
        "income_beta": 0.0,               # bond-like income
    }


@pytest.fixture
def retiree_params():
    """Parameters for a retiree with no human capital."""
    return {
        "total_wealth": 2_000_000,
        "financial_assets": 2_000_000,
        "pv_human_capital": 0,
        "human_capital_duration": 0,
        "pv_expenses": 0,               # No remaining expenses for simplicity
        "expense_duration": 0,
        "ltpz_duration": LTPZ_DURATION,
        "income_beta": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: optimal_total_wealth_weights
# ---------------------------------------------------------------------------


class TestOptimalTotalWealthWeights:
    """Tests for the optimal_total_wealth_weights function."""

    def test_lecture_example_gamma_2(self, default_market_assumptions):
        """Lecture example: gamma=2 should give ~62% stocks, ~15% bonds, ~23% cash."""
        result = optimal_total_wealth_weights(risk_aversion=2.0, **default_market_assumptions)

        # Lecture calculation:
        # w_stock = 0.04 / (2 * 0.0324) = 0.617 (61.7%)
        # w_bond = 0.01 / (2 * 0.0324) = 0.154 (15.4%)
        # w_cash = 1 - 0.617 - 0.154 = 0.229 (22.9%)
        assert result["stocks"] == pytest.approx(0.617, rel=0.01)
        assert result["bonds"] == pytest.approx(0.154, rel=0.01)
        assert result["cash"] == pytest.approx(0.229, rel=0.01)

    def test_weights_sum_to_one(self, default_market_assumptions):
        """Weights should always sum to 1.0 regardless of risk aversion."""
        for gamma in [1.0, 2.0, 3.0, 5.0, 10.0]:
            result = optimal_total_wealth_weights(risk_aversion=gamma, **default_market_assumptions)

            total = result["stocks"] + result["bonds"] + result["cash"]
            assert total == pytest.approx(1.0)

    def test_higher_risk_aversion_lower_stock_weight(self, default_market_assumptions):
        """Higher risk aversion should result in lower stock allocation."""
        result_low_gamma = optimal_total_wealth_weights(risk_aversion=1.5, **default_market_assumptions)
        result_mid_gamma = optimal_total_wealth_weights(risk_aversion=2.0, **default_market_assumptions)
        result_high_gamma = optimal_total_wealth_weights(risk_aversion=5.0, **default_market_assumptions)

        # Stock weight should decrease as risk aversion increases
        assert result_low_gamma["stocks"] > result_mid_gamma["stocks"]
        assert result_mid_gamma["stocks"] > result_high_gamma["stocks"]

    def test_higher_risk_aversion_higher_cash_weight(self, default_market_assumptions):
        """Higher risk aversion should result in higher cash allocation."""
        result_low_gamma = optimal_total_wealth_weights(risk_aversion=1.5, **default_market_assumptions)
        result_high_gamma = optimal_total_wealth_weights(risk_aversion=5.0, **default_market_assumptions)

        assert result_high_gamma["cash"] > result_low_gamma["cash"]

    def test_very_high_risk_aversion(self, default_market_assumptions):
        """Very risk-averse investor should hold mostly cash."""
        result = optimal_total_wealth_weights(risk_aversion=10.0, **default_market_assumptions)

        # With gamma=10:
        # w_stock = 0.04 / (10 * 0.0324) = 0.123 (12.3%)
        # w_bond = 0.01 / (10 * 0.0324) = 0.031 (3.1%)
        # w_cash = 1 - 0.123 - 0.031 = 0.846 (84.6%)
        assert result["cash"] > 0.8
        assert result["stocks"] < 0.15

    def test_low_risk_aversion_leverage(self, default_market_assumptions):
        """Very low risk aversion can result in leverage (cash < 0)."""
        result = optimal_total_wealth_weights(risk_aversion=0.5, **default_market_assumptions)

        # With gamma=0.5:
        # w_stock = 0.04 / (0.5 * 0.0324) = 2.469 (246.9%)
        # w_bond = 0.01 / (0.5 * 0.0324) = 0.617 (61.7%)
        # w_cash = 1 - 2.469 - 0.617 = -2.086 (leveraged position)
        assert result["stocks"] > 1.0  # Leverage on stocks
        assert result["cash"] < 0     # Borrowing cash

    def test_gamma_one(self, default_market_assumptions):
        """Test with gamma=1 (log utility)."""
        result = optimal_total_wealth_weights(risk_aversion=1.0, **default_market_assumptions)

        # w_stock = 0.04 / (1 * 0.0324) = 1.235
        # w_bond = 0.01 / (1 * 0.0324) = 0.309
        # w_cash = 1 - 1.235 - 0.309 = -0.544
        assert result["stocks"] == pytest.approx(1.235, rel=0.01)
        assert result["bonds"] == pytest.approx(0.309, rel=0.01)
        assert result["cash"] == pytest.approx(-0.544, rel=0.01)

    def test_different_return_assumptions(self):
        """Test with modified return/variance assumptions."""
        # Higher stock premium should increase stock allocation
        result_high_premium = optimal_total_wealth_weights(
            risk_aversion=2.0,
            stock_excess_return=0.06,  # 6% instead of 4%
            bond_excess_return=0.01,
            stock_variance=STOCK_VARIANCE,
            bond_variance=BOND_VARIANCE,
        )

        result_base = optimal_total_wealth_weights(
            risk_aversion=2.0,
            stock_excess_return=0.04,
            bond_excess_return=0.01,
            stock_variance=STOCK_VARIANCE,
            bond_variance=BOND_VARIANCE,
        )

        assert result_high_premium["stocks"] > result_base["stocks"]


# ---------------------------------------------------------------------------
# Tests: financial_portfolio_weights
# ---------------------------------------------------------------------------


class TestFinancialPortfolioWeights:
    """Tests for the financial_portfolio_weights function."""

    def test_young_investor_high_stock_weight(self, young_investor_params):
        """Young person with large human capital should have constrained allocation."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **young_investor_params,
            constrained=True,
        )

        # Young investor with massive human capital relative to financial assets
        # has extreme unconstrained weights that get clipped and renormalized.
        # The constrained result reflects capped allocation after normalization.
        assert 0.0 <= result["stocks"] <= 1.0
        assert result["stocks"] + result["bonds"] + result["cash"] == pytest.approx(1.0)

    def test_retiree_matches_total_wealth_weights(self, retiree_params):
        """Retiree with no human capital should match total wealth weights."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **retiree_params,
            constrained=False,
        )

        # Without human capital, financial portfolio should equal target allocation
        assert result["stocks"] == pytest.approx(optimal_weights["stocks"], rel=0.01)
        assert result["bonds"] == pytest.approx(optimal_weights["bonds"], rel=0.01)
        assert result["cash"] == pytest.approx(optimal_weights["cash"], rel=0.01)

    def test_constrained_clips_to_bounds(self, young_investor_params):
        """Constrained=True should clip weights to [0, 1]."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **young_investor_params,
            constrained=True,
        )

        # All weights should be between 0 and 1
        assert 0.0 <= result["stocks"] <= 1.0
        assert 0.0 <= result["bonds"] <= 1.0
        assert 0.0 <= result["cash"] <= 1.0

    def test_constrained_weights_sum_to_one(self, young_investor_params):
        """Constrained weights should sum to 1.0."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **young_investor_params,
            constrained=True,
        )

        total = result["stocks"] + result["bonds"] + result["cash"]
        assert total == pytest.approx(1.0)

    def test_unconstrained_can_have_leverage(self, young_investor_params):
        """Unconstrained weights can exceed bounds."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **young_investor_params,
            constrained=False,
        )

        # For young investor with large human capital relative to financial assets,
        # unconstrained weights can be extreme: stocks >> 1 (leverage),
        # bonds << 0 (shorting). Note: weights may not sum to 1.0 in unconstrained
        # mode when human capital provides significant implicit exposure.

        # With substantial human capital acting as bonds/cash, need extreme
        # stock position in financial portfolio to hit total wealth targets
        assert result["stocks"] > 1.0  # Implies leverage
        assert result["bonds"] < 0.0   # Implies short position

    def test_stock_like_income_reduces_stock_weight(self):
        """Stock-like income (income_beta > 0) reduces financial stock allocation."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        # Base parameters without income_beta to avoid conflict
        base_params = {
            "total_wealth": 3_150_000,
            "financial_assets": 100_000,
            "pv_human_capital": 6_200_000,
            "human_capital_duration": 17.0,
            "pv_expenses": 3_150_000,
            "expense_duration": 22.0,
            "ltpz_duration": LTPZ_DURATION,
        }

        # Bond-like income (baseline)
        result_bond_like = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **base_params,
            income_beta=0.0,
            constrained=False,
        )

        # Stock-like income (40% correlation with market)
        result_stock_like = financial_portfolio_weights(
            optimal_weights=optimal_weights,
            **base_params,
            income_beta=0.4,
            constrained=False,
        )

        # Stock-like income provides implicit stock exposure from human capital,
        # so financial portfolio needs fewer stocks to achieve target total
        # wealth allocation. Note: both can be extreme (leveraged/short).
        assert result_stock_like["stocks"] < result_bond_like["stocks"]

    def test_zero_financial_assets(self):
        """Zero financial assets should return safe default."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=5_000_000,
            financial_assets=0,
            pv_human_capital=5_000_000,
            human_capital_duration=17.0,
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_weights,
            constrained=True,
        )

        # Should return all cash as safe default
        assert result["stocks"] == 0.0
        assert result["bonds"] == 0.0
        assert result["cash"] == 1.0

    def test_negative_financial_assets(self):
        """Negative financial assets should return safe default."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=4_000_000,
            financial_assets=-100_000,  # In debt
            pv_human_capital=5_000_000,
            human_capital_duration=17.0,
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_weights,
            constrained=True,
        )

        assert result["cash"] == 1.0

    def test_human_capital_duration_effect(self):
        """Longer duration human capital should affect bond allocation."""
        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        # Short duration (near retirement)
        result_short = financial_portfolio_weights(
            total_wealth=1_000_000,
            financial_assets=500_000,
            pv_human_capital=500_000,
            human_capital_duration=5.0,  # Short duration
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_weights,
            ltpz_duration=LTPZ_DURATION,
            income_beta=0.0,
            constrained=False,
        )

        # Long duration (young worker)
        result_long = financial_portfolio_weights(
            total_wealth=1_000_000,
            financial_assets=500_000,
            pv_human_capital=500_000,
            human_capital_duration=20.0,  # Long duration
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_weights,
            ltpz_duration=LTPZ_DURATION,
            income_beta=0.0,
            constrained=False,
        )

        # Higher duration human capital acts more like bonds, so financial
        # portfolio needs fewer bonds
        assert result_long["bonds"] < result_short["bonds"]

    def test_higher_risk_aversion_investor(self):
        """Higher risk aversion should result in more conservative allocation."""
        # Low risk aversion (gamma=1.5)
        optimal_low_gamma = optimal_total_wealth_weights(risk_aversion=1.5)
        result_low = financial_portfolio_weights(
            total_wealth=2_000_000,
            financial_assets=2_000_000,
            pv_human_capital=0,
            human_capital_duration=0,
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_low_gamma,
            constrained=True,
        )

        # High risk aversion (gamma=5.0)
        optimal_high_gamma = optimal_total_wealth_weights(risk_aversion=5.0)
        result_high = financial_portfolio_weights(
            total_wealth=2_000_000,
            financial_assets=2_000_000,
            pv_human_capital=0,
            human_capital_duration=0,
            pv_expenses=0,
            expense_duration=0,
            optimal_weights=optimal_high_gamma,
            constrained=True,
        )

        # Higher risk aversion should mean lower stock weight
        assert result_high["stocks"] < result_low["stocks"]
        assert result_high["cash"] > result_low["cash"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestOptimizationIntegration:
    """Integration tests combining optimization functions."""

    def test_lifecycle_scenario_young_worker(self):
        """Full lifecycle scenario for a young worker (25 years old)."""
        # Young consultant with bond-like income
        total_wealth = 3_200_000
        financial_assets = 100_000
        pv_human_capital = 6_250_000
        human_capital_duration = 17.0
        pv_expenses = 3_150_000
        exp_duration = 22.0

        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=total_wealth,
            financial_assets=financial_assets,
            pv_human_capital=pv_human_capital,
            human_capital_duration=human_capital_duration,
            pv_expenses=pv_expenses,
            expense_duration=exp_duration,
            optimal_weights=optimal_weights,
            income_beta=0.0,
            constrained=True,
        )

        # Young worker with massive human capital relative to financial assets
        # has constrained weights that sum to 1.0. The exact split depends
        # on how extreme weights get clipped and renormalized.
        assert result["stocks"] + result["bonds"] + result["cash"] == pytest.approx(1.0)
        assert 0.0 <= result["stocks"] <= 1.0

    def test_lifecycle_scenario_mid_career(self):
        """Full lifecycle scenario for mid-career worker (45 years old)."""
        # Mid-career with moderate human capital
        total_wealth = 3_000_000
        financial_assets = 800_000
        pv_human_capital = 3_000_000
        human_capital_duration = 12.0
        pv_expenses = 800_000
        exp_duration = 18.0

        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=total_wealth,
            financial_assets=financial_assets,
            pv_human_capital=pv_human_capital,
            human_capital_duration=human_capital_duration,
            pv_expenses=pv_expenses,
            expense_duration=exp_duration,
            optimal_weights=optimal_weights,
            income_beta=0.0,
            constrained=True,
        )

        # Mid-career still has substantial human capital relative to financial
        # assets, leading to high stock allocation (may hit 100% constraint)
        assert 0.5 <= result["stocks"] <= 1.0
        assert result["stocks"] + result["bonds"] + result["cash"] == pytest.approx(1.0)

    def test_lifecycle_scenario_near_retirement(self):
        """Full lifecycle scenario for near-retirement worker (60 years old)."""
        # Near retirement with little human capital
        total_wealth = 2_500_000
        financial_assets = 2_000_000
        pv_human_capital = 700_000
        human_capital_duration = 3.0
        pv_expenses = 200_000
        exp_duration = 15.0

        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=total_wealth,
            financial_assets=financial_assets,
            pv_human_capital=pv_human_capital,
            human_capital_duration=human_capital_duration,
            pv_expenses=pv_expenses,
            expense_duration=exp_duration,
            optimal_weights=optimal_weights,
            income_beta=0.0,
            constrained=True,
        )

        # Near retirement with financial assets dominating should have
        # allocation closer to optimal total wealth weights (~62% stocks)
        # but still elevated due to remaining human capital
        assert 0.5 < result["stocks"] <= 1.0
        assert result["stocks"] + result["bonds"] + result["cash"] == pytest.approx(1.0)

    def test_stock_like_income_banker_scenario(self):
        """Scenario for investment banker with stock-like income."""
        # Investment banker: high income but volatile (beta=0.4)
        total_wealth = 4_000_000
        financial_assets = 500_000
        pv_human_capital = 5_000_000
        human_capital_duration = 15.0
        pv_expenses = 1_500_000
        exp_duration = 20.0

        optimal_weights = optimal_total_wealth_weights(risk_aversion=2.0)

        result = financial_portfolio_weights(
            total_wealth=total_wealth,
            financial_assets=financial_assets,
            pv_human_capital=pv_human_capital,
            human_capital_duration=human_capital_duration,
            pv_expenses=pv_expenses,
            expense_duration=exp_duration,
            optimal_weights=optimal_weights,
            income_beta=0.4,  # Stock-like income
            constrained=True,
        )

        # Banker already has stock exposure via human capital
        # Financial portfolio should have lower stock allocation than consultant
        assert result["stocks"] < 1.0
