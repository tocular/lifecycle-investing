"""
Main dashboard display for lifecycle investing application.
Renders portfolio analysis results, charts, and key metrics.
"""

import streamlit as st
import pandas as pd

from ..calculations.present_value import (
    pv_human_capital,
    pv_expenses,
    total_wealth,
)
from ..calculations.duration import (
    human_capital_duration,
    expense_duration as calc_expense_duration,
)
from ..calculations.optimization import (
    optimal_total_wealth_weights,
    financial_portfolio_weights,
)
from ..calculations.glide_path import compute_glide_path, project_wealth
from .charts import plot_glide_path, plot_wealth_projection
from .. import config


def _format_currency(value: float) -> str:
    """Format a numeric value as USD currency with commas (accounting style)."""
    if value < 0:
        return f"(${abs(value):,.0f})"
    return f"${value:,.0f}"


def _format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    return f"{value * 100:.1f}%"


def render_dashboard(inputs: dict) -> None:
    """
    Display the lifecycle investing analysis dashboard.

    Renders a comprehensive view of the investor's optimal portfolio strategy,
    including present value calculations, recommended allocations, glide path
    visualization, and wealth projections based on geometric returns.

    Args:
        inputs: Dictionary containing user inputs with keys:
            - current_age: int, investor's current age
            - retirement_age: int, target retirement age
            - life_expectancy: int, expected lifespan for planning
            - annual_income: float, annual labor income
            - working_expenses: float, annual expenses during working years
            - retirement_expenses: float, annual expenses in retirement
            - financial_assets: float, current investable assets
            - risk_aversion: float, coefficient of relative risk aversion
            - income_beta: float, income correlation with market (0=bond-like)
    """
    # Extract inputs for calculations
    current_age = inputs["current_age"]
    retirement_age = inputs["retirement_age"]
    life_expectancy = inputs["life_expectancy"]
    annual_income = inputs["annual_income"]
    working_expenses = inputs["working_expenses"]
    retirement_expenses = inputs["retirement_expenses"]
    financial_assets = inputs["financial_assets"]
    risk_aversion = inputs["risk_aversion"]
    income_beta = inputs["income_beta"]

    # Editable Capital Market Assumptions
    with st.expander("Capital Market Assumptions", expanded=False):
        st.caption("Adjust these assumptions to see how they affect your portfolio allocation.")

        assumptions_col1, assumptions_col2 = st.columns(2)

        with assumptions_col1:
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=config.RISK_FREE_RATE * 100,
                step=0.1,
                format="%.1f",
            ) / 100

            stock_excess_return = st.number_input(
                "Stock Excess Return (%)",
                min_value=0.0,
                max_value=20.0,
                value=config.STOCK_EXCESS_RETURN * 100,
                step=0.1,
                format="%.1f",
            ) / 100

            stock_volatility = st.number_input(
                "Stock Volatility (%)",
                min_value=1.0,
                max_value=50.0,
                value=config.STOCK_VOLATILITY * 100,
                step=1.0,
                format="%.0f",
            ) / 100

        with assumptions_col2:
            bond_excess_return = st.number_input(
                "Bond Excess Return (%)",
                min_value=0.0,
                max_value=20.0,
                value=config.BOND_EXCESS_RETURN * 100,
                step=0.1,
                format="%.1f",
            ) / 100

            bond_volatility = st.number_input(
                "Bond Volatility (%)",
                min_value=1.0,
                max_value=50.0,
                value=config.BOND_VOLATILITY * 100,
                step=1.0,
                format="%.0f",
            ) / 100

            ltpz_duration = st.number_input(
                "LTPZ Duration (years)",
                min_value=1.0,
                max_value=30.0,
                value=config.LTPZ_DURATION,
                step=0.5,
                format="%.1f",
                help="LTPZ is a PIMCO exchange traded fund (ETF) that tracks a market-value-weighted index of long-term US Treasury Inflation Protected Securities",
            )

    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    # Calculate derived values
    years_working = max(0, retirement_age - current_age)
    years_retirement = max(0, life_expectancy - retirement_age)
    annual_savings = annual_income - working_expenses

    # Calculate variances from volatilities
    stock_variance = stock_volatility ** 2
    bond_variance = bond_volatility ** 2

    # Calculate geometric returns
    stock_geo_return = risk_free_rate + stock_excess_return - 0.5 * stock_variance
    bond_geo_return = risk_free_rate + bond_excess_return - 0.5 * bond_variance

    # Present value calculations
    pv_hc = pv_human_capital(
        annual_income=annual_income,
        years_working=years_working,
        risk_free_rate=risk_free_rate,
        income_beta=income_beta,
        equity_premium=stock_excess_return,
    )
    pv_exp = pv_expenses(
        working_expenses=working_expenses,
        retirement_expenses=retirement_expenses,
        years_working=years_working,
        years_retirement=years_retirement,
        risk_free_rate=risk_free_rate,
    )
    total_w = total_wealth(
        financial_assets=financial_assets,
        pv_human_capital=pv_hc,
        pv_expenses=pv_exp,
    )

    # Duration calculations
    hc_duration = human_capital_duration(years_working)
    exp_duration = calc_expense_duration(
        years_working=years_working,
        years_retirement=years_retirement,
        working_expenses=working_expenses,
        retirement_expenses=retirement_expenses,
    )

    # Portfolio optimization (Lecture 13's 4-step recipe)
    # Step 3: Compute target holdings in total wealth portfolio
    optimal_weights = optimal_total_wealth_weights(
        risk_aversion=risk_aversion,
        stock_excess_return=stock_excess_return,
        bond_excess_return=bond_excess_return,
        stock_variance=stock_variance,
        bond_variance=bond_variance,
    )
    # Step 2 & 4: Convert income/expenses to tradable portfolios, back out financial weights
    fin_weights = financial_portfolio_weights(
        total_wealth=total_w,
        financial_assets=financial_assets,
        pv_human_capital=pv_hc,
        human_capital_duration=hc_duration,
        pv_expenses=pv_exp,
        expense_duration=exp_duration,
        optimal_weights=optimal_weights,
        ltpz_duration=ltpz_duration,
        income_beta=income_beta,
        constrained=True,
    )

    # Calculate tradable portfolio equivalents (Lecture 13, Step 2)
    # Human Capital -> LONG bonds + cash based on duration
    hc_bond_equiv = (hc_duration / ltpz_duration) * pv_hc * (1.0 - income_beta)
    hc_cash_equiv = pv_hc * (1.0 - income_beta) - hc_bond_equiv
    hc_stock_equiv = pv_hc * income_beta

    # Expenses -> SHORT bonds + cash based on duration
    exp_bond_equiv = (exp_duration / ltpz_duration) * pv_exp
    exp_cash_equiv = pv_exp - exp_bond_equiv

    # NET "Human" position = Human Capital tradable - Expenses tradable
    net_human_stock = hc_stock_equiv
    net_human_bond = hc_bond_equiv - exp_bond_equiv
    net_human_cash = hc_cash_equiv - exp_cash_equiv
    net_human_total = pv_hc - pv_exp

    # Target dollar amounts for total wealth
    target_stock = optimal_weights["stocks"] * total_w
    target_bond = optimal_weights["bonds"] * total_w
    target_cash = optimal_weights["cash"] * total_w

    # Financial portfolio needed = Target - Net Human
    fin_stock = target_stock - net_human_stock
    fin_bond = target_bond - net_human_bond
    fin_cash = target_cash - net_human_cash

    # Section 1: Allocation Breakdown (Lecture-style table)
    st.subheader("Allocation Breakdown")

    show_normalized = st.toggle("Show as percentages", value=False)

    if show_normalized:
        allocation_data = {
            "Stocks": [
                _format_percent(net_human_stock / net_human_total) if net_human_total != 0 else "—",
                _format_percent(fin_stock / financial_assets) if financial_assets != 0 else "—",
                _format_percent(target_stock / total_w) if total_w != 0 else "—",
            ],
            "Bonds": [
                _format_percent(net_human_bond / net_human_total) if net_human_total != 0 else "—",
                _format_percent(fin_bond / financial_assets) if financial_assets != 0 else "—",
                _format_percent(target_bond / total_w) if total_w != 0 else "—",
            ],
            "Cash": [
                _format_percent(net_human_cash / net_human_total) if net_human_total != 0 else "—",
                _format_percent(fin_cash / financial_assets) if financial_assets != 0 else "—",
                _format_percent(target_cash / total_w) if total_w != 0 else "—",
            ],
            "Total": [
                "100.0%",
                "100.0%",
                "100.0%",
            ],
        }
    else:
        allocation_data = {
            "Stocks": [
                _format_currency(net_human_stock),
                _format_currency(fin_stock),
                _format_currency(target_stock),
            ],
            "Bonds": [
                _format_currency(net_human_bond),
                _format_currency(fin_bond),
                _format_currency(target_bond),
            ],
            "Cash": [
                _format_currency(net_human_cash),
                _format_currency(fin_cash),
                _format_currency(target_cash),
            ],
            "Total": [
                _format_currency(net_human_total),
                _format_currency(financial_assets),
                _format_currency(total_w),
            ],
        }

    allocation_df = pd.DataFrame(
        allocation_data,
        index=["Human (Net)", "Financial", "Target"]
    )

    # Style table headers to be bold
    st.markdown("""
        <style>
        table thead th, table tbody th {
            font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.table(allocation_df)

    st.caption(
        "**Human (Net)**: Implicit portfolio from human capital minus expenses liability.  \n"
        "**Financial**: What you should hold in your investment portfolio.  \n"
        "**Note**: Total Wealth = Financial Assets + PV(Human Capital) - PV(Expenses)."
    )

    st.divider()

    # Section 3: Glide Path Chart
    st.subheader("Portfolio Glide Path")

    glide_df = compute_glide_path(
        current_age=current_age,
        retirement_age=retirement_age,
        life_expectancy=life_expectancy,
        annual_income=annual_income,
        working_expenses=working_expenses,
        retirement_expenses=retirement_expenses,
        financial_assets=financial_assets,
        risk_aversion=risk_aversion,
        income_beta=income_beta,
        risk_free_rate=risk_free_rate,
        stock_excess_return=stock_excess_return,
        bond_excess_return=bond_excess_return,
        stock_volatility=stock_volatility,
        bond_volatility=bond_volatility,
        ltpz_duration=ltpz_duration,
        stock_geo_return=stock_geo_return,
        bond_geo_return=bond_geo_return,
        cash_return=risk_free_rate,
    )

    glide_fig = plot_glide_path(glide_df, retirement_age=retirement_age)
    st.plotly_chart(glide_fig, width="stretch")

    st.divider()

    # Section 4: Wealth Projection Chart
    st.subheader("Wealth Projection")

    projection_df = project_wealth(
        glide_path_df=glide_df,
        annual_savings=annual_savings,
    )

    wealth_fig = plot_wealth_projection(projection_df, retirement_age=retirement_age)
    st.plotly_chart(wealth_fig, width="stretch")
