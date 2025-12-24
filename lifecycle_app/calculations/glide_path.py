"""
Glide path computation for lifecycle investing.
Based on FINC 450 Capital Markets lecture notes.
"""

from typing import Optional

import pandas as pd

from ..config import (
    RISK_FREE_RATE,
    STOCK_EXCESS_RETURN,
    BOND_EXCESS_RETURN,
    STOCK_VARIANCE,
    BOND_VARIANCE,
    STOCK_VOLATILITY,
    BOND_VOLATILITY,
    LTPZ_DURATION,
    STOCK_GEOMETRIC_RETURN,
    BOND_GEOMETRIC_RETURN,
    CASH_GEOMETRIC_RETURN,
)
from .present_value import pv_human_capital, pv_expenses
from .duration import annuity_duration, expense_duration
from .optimization import (
    optimal_total_wealth_weights,
    financial_portfolio_weights,
)


def compute_glide_path(
    current_age: int,
    retirement_age: int,
    life_expectancy: int,
    annual_income: float,
    working_expenses: float,
    retirement_expenses: float,
    financial_assets: float,
    risk_aversion: float,
    income_beta: float = 0.0,
    risk_free_rate: float = RISK_FREE_RATE,
    stock_excess_return: float = None,
    bond_excess_return: float = None,
    stock_volatility: float = None,
    bond_volatility: float = None,
    ltpz_duration: float = None,
    stock_geo_return: float = STOCK_GEOMETRIC_RETURN,
    bond_geo_return: float = BOND_GEOMETRIC_RETURN,
    cash_return: float = CASH_GEOMETRIC_RETURN,
) -> pd.DataFrame:
    """
    Compute year-by-year optimal financial portfolio weights over the lifecycle.

    This function generates a glide path showing how portfolio allocation should
    evolve from current age through life expectancy. It simultaneously projects
    financial wealth growth, using the projected wealth at each age to compute
    the optimal allocation for that year.

    For each year, the function:
        1. Uses projected financial wealth from previous year
        2. Recalculates remaining working years and retirement years
        3. Recomputes present value of human capital (decreasing over time)
        4. Recomputes present value of expenses (decreasing as horizon shortens)
        5. Recalculates total economic wealth using projected financial wealth
        6. Derives optimal total wealth weights based on risk aversion
        7. Backs out required financial portfolio weights
        8. Projects next year's financial wealth based on returns + savings

    Args:
        current_age: Investor's current age in years.
        retirement_age: Target retirement age.
        life_expectancy: Expected age at death for planning purposes.
        annual_income: Annual labor income (assumed constant in real terms).
        working_expenses: Annual expenses during working years.
        retirement_expenses: Annual expenses during retirement.
        financial_assets: Current value of investable financial assets.
        risk_aversion: Coefficient of relative risk aversion (gamma).
        income_beta: Fraction of human capital that correlates with stocks.
        risk_free_rate: Real risk-free discount rate for PV calculations.
        stock_geo_return: Geometric return for stocks.
        bond_geo_return: Geometric return for bonds.
        cash_return: Return for cash (risk-free rate).

    Returns:
        DataFrame with columns:
            - age: Year of life
            - years_to_retirement: Years remaining until retirement
            - pv_human_capital: Present value of remaining labor income
            - pv_expenses: Present value of remaining lifetime expenses
            - total_wealth: Total economic wealth
            - financial_wealth: Projected financial assets at this age
            - stock_weight: Optimal stock allocation in financial portfolio
            - bond_weight: Optimal bond allocation in financial portfolio
            - cash_weight: Optimal cash allocation in financial portfolio
    """
    # Apply defaults from config if not provided
    if stock_excess_return is None:
        stock_excess_return = STOCK_EXCESS_RETURN
    if bond_excess_return is None:
        bond_excess_return = BOND_EXCESS_RETURN
    if stock_volatility is None:
        stock_volatility = STOCK_VOLATILITY
    if bond_volatility is None:
        bond_volatility = BOND_VOLATILITY
    if ltpz_duration is None:
        ltpz_duration = LTPZ_DURATION

    stock_variance = stock_volatility ** 2
    bond_variance = bond_volatility ** 2

    # Compute optimal total wealth weights (constant across the lifecycle)
    optimal_weights = optimal_total_wealth_weights(
        risk_aversion=risk_aversion,
        stock_excess_return=stock_excess_return,
        bond_excess_return=bond_excess_return,
        stock_variance=stock_variance,
        bond_variance=bond_variance,
    )

    # Storage for yearly calculations
    records = []

    # Track projected financial wealth (starts with initial assets)
    current_financial_wealth = financial_assets

    for age in range(current_age, life_expectancy + 1):
        # Calculate time horizons for this year
        years_to_retirement = max(0, retirement_age - age)
        years_working = years_to_retirement
        years_retired = max(0, life_expectancy - max(age, retirement_age))

        # Present value of human capital (future labor income)
        if years_working > 0:
            hc_pv = pv_human_capital(
                annual_income=annual_income,
                years_working=years_working,
                risk_free_rate=risk_free_rate,
            )
            hc_duration = annuity_duration(
                rate=risk_free_rate,
                n_periods=years_working,
            )
        else:
            hc_pv = 0.0
            hc_duration = 0.0

        # Present value of lifetime expenses
        exp_pv = pv_expenses(
            working_expenses=working_expenses,
            retirement_expenses=retirement_expenses,
            years_working=years_working,
            years_retirement=years_retired,
            risk_free_rate=risk_free_rate,
        )

        # Duration of expenses
        exp_dur = expense_duration(
            years_working=years_working,
            years_retirement=years_retired,
            working_expenses=working_expenses,
            retirement_expenses=retirement_expenses,
            risk_free_rate=risk_free_rate,
        )

        # Total economic wealth using PROJECTED financial wealth
        total_w = current_financial_wealth + hc_pv - exp_pv

        # Financial portfolio weights (constrained to [0,1])
        fin_weights = financial_portfolio_weights(
            total_wealth=total_w,
            financial_assets=current_financial_wealth,
            pv_human_capital=hc_pv,
            human_capital_duration=hc_duration,
            pv_expenses=exp_pv,
            expense_duration=exp_dur,
            optimal_weights=optimal_weights,
            ltpz_duration=ltpz_duration,
            income_beta=income_beta,
            constrained=True,
        )

        records.append({
            "age": age,
            "years_to_retirement": years_to_retirement,
            "pv_human_capital": hc_pv,
            "pv_expenses": exp_pv,
            "total_wealth": total_w,
            "financial_wealth": current_financial_wealth,
            "stock_weight": fin_weights["stocks"],
            "bond_weight": fin_weights["bonds"],
            "cash_weight": fin_weights["cash"],
        })

        # Project next year's financial wealth
        portfolio_return = (
            fin_weights["stocks"] * stock_geo_return
            + fin_weights["bonds"] * bond_geo_return
            + fin_weights["cash"] * cash_return
        )

        # Savings during working years, withdrawals during retirement
        if years_working > 0:
            annual_cashflow = annual_income - working_expenses
        else:
            annual_cashflow = -retirement_expenses

        current_financial_wealth = (
            current_financial_wealth * (1 + portfolio_return) + annual_cashflow
        )
        # Floor at zero (can't have negative financial wealth in practice)
        current_financial_wealth = max(0, current_financial_wealth)

    return pd.DataFrame(records)


def project_wealth(
    glide_path_df: pd.DataFrame,
    annual_savings: float = 0.0,
    stock_geo_return: float = STOCK_GEOMETRIC_RETURN,
    bond_geo_return: float = BOND_GEOMETRIC_RETURN,
    cash_return: float = CASH_GEOMETRIC_RETURN,
) -> pd.DataFrame:
    """
    Return glide path with projected wealth column for backward compatibility.

    Note: compute_glide_path now projects financial wealth internally.
    This function adds a 'projected_financial_wealth' column that matches
    the 'financial_wealth' column for compatibility with existing code.

    Args:
        glide_path_df: DataFrame from compute_glide_path().
        annual_savings: (Deprecated) No longer used - savings handled in glide path.
        stock_geo_return: (Deprecated) No longer used - returns handled in glide path.
        bond_geo_return: (Deprecated) No longer used.
        cash_return: (Deprecated) No longer used.

    Returns:
        DataFrame with 'projected_financial_wealth' column added.
    """
    df = glide_path_df.copy()
    # Wealth is already projected in compute_glide_path
    df["projected_financial_wealth"] = df["financial_wealth"]
    return df
