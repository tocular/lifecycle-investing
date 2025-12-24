"""
Sidebar component for lifecycle investing application.
Collects user inputs for financial planning calculations.
"""

import streamlit as st
from ..config import (
    DEFAULT_AGE, DEFAULT_RETIREMENT_AGE, DEFAULT_LIFE_EXPECTANCY,
    DEFAULT_ANNUAL_INCOME, DEFAULT_EXPENSES_WORKING, DEFAULT_EXPENSES_RETIREMENT,
    DEFAULT_FINANCIAL_ASSETS, DEFAULT_RISK_AVERSION,
    MIN_AGE, MAX_AGE, MIN_RETIREMENT_AGE, MAX_RETIREMENT_AGE,
    MIN_LIFE_EXPECTANCY, MAX_LIFE_EXPECTANCY,
    MIN_INCOME, MAX_INCOME, MIN_EXPENSES, MAX_EXPENSES,
    MIN_ASSETS, MAX_ASSETS, MIN_RISK_AVERSION, MAX_RISK_AVERSION,
    BOND_LIKE_INCOME_BETA, STOCK_LIKE_INCOME_BETA,
)


def format_currency(value: int) -> str:
    """Format an integer value as currency string (e.g., '$150,000')."""
    return f"${value:,}"


def render_sidebar() -> dict:
    """
    Render the sidebar with all user input widgets.

    Returns:
        dict: User inputs with keys:
            - current_age: int
            - retirement_age: int
            - life_expectancy: int
            - annual_income: int
            - expenses_working: int
            - expenses_retirement: int
            - financial_assets: int
            - risk_aversion: float
            - income_beta: float
    """
    with st.sidebar:
        st.title("Parameters")

        # Age Settings Section
        st.header("Age Settings")

        current_age = st.slider(
            "Current Age",
            min_value=MIN_AGE,
            max_value=MAX_AGE,
            value=DEFAULT_AGE,
            step=1,
            help="Your current age in years"
        )

        retirement_age = st.slider(
            "Retirement Age",
            min_value=MIN_RETIREMENT_AGE,
            max_value=MAX_RETIREMENT_AGE,
            value=DEFAULT_RETIREMENT_AGE,
            step=1,
            help="Age at which you plan to retire"
        )

        life_expectancy = st.slider(
            "Life Expectancy",
            min_value=MIN_LIFE_EXPECTANCY,
            max_value=MAX_LIFE_EXPECTANCY,
            value=DEFAULT_LIFE_EXPECTANCY,
            step=1,
            help="Expected lifespan for planning purposes"
        )

        # Validation: retirement age must be greater than current age
        if retirement_age <= current_age:
            st.error("Retirement age must be greater than current age.")

        # Validation: life expectancy must be greater than retirement age
        if life_expectancy <= retirement_age:
            st.error("Life expectancy must be greater than retirement age.")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Income & Expenses Section
        st.header("Income & Expenses")

        annual_income = st.number_input(
            "Annual Income (post-tax)",
            min_value=MIN_INCOME,
            max_value=MAX_INCOME,
            value=DEFAULT_ANNUAL_INCOME,
            step=10_000,
            format="%d",
            help=f"Your annual post-tax income ({format_currency(DEFAULT_ANNUAL_INCOME)} default)"
        )

        expenses_working = st.number_input(
            "Expenses While Working",
            min_value=MIN_EXPENSES,
            max_value=MAX_EXPENSES,
            value=DEFAULT_EXPENSES_WORKING,
            step=5_000,
            format="%d",
            help="Annual expenses during your working years"
        )

        expenses_retirement = st.number_input(
            "Expenses in Retirement",
            min_value=MIN_EXPENSES,
            max_value=MAX_EXPENSES,
            value=DEFAULT_EXPENSES_RETIREMENT,
            step=5_000,
            format="%d",
            help="Expected annual expenses during retirement"
        )

        # Warning if expenses exceed income
        if expenses_working > annual_income:
            st.warning(
                "Your working expenses exceed your income. "
                "Consider adjusting your budget to ensure positive savings."
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Financial Assets Section
        st.header("Financial Assets")

        financial_assets = st.number_input(
            "Current Financial Assets",
            min_value=MIN_ASSETS,
            max_value=MAX_ASSETS,
            value=DEFAULT_FINANCIAL_ASSETS,
            step=10_000,
            format="%d",
            help="Total value of current investment portfolio"
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Investment Preferences Section
        st.header("Investment Preferences")

        risk_aversion = st.slider(
            "Risk Aversion (γ)",
            min_value=MIN_RISK_AVERSION,
            max_value=MAX_RISK_AVERSION,
            value=DEFAULT_RISK_AVERSION,
            step=0.5,
            help=(
                "Lower values (1-3) indicate aggressive investing with higher risk tolerance. "
                "Higher values (7-10) indicate conservative investing with lower risk tolerance. "
                "Moderate values (4-6) balance growth and stability."
            )
        )

        # Map income type selection to beta values
        income_type_options = {
            "Bond-like (stable)": BOND_LIKE_INCOME_BETA,
            "Stock-like (volatile)": STOCK_LIKE_INCOME_BETA,
        }

        income_type = st.selectbox(
            "Income Type",
            options=list(income_type_options.keys()),
            index=0,
            help=(
                "Bond-like: Stable income such as teacher, consultant, or government employee. "
                "Stock-like: Volatile income such as investment banker or trader."
            )
        )

        income_beta = income_type_options[income_type]

    # Return all collected inputs as a dictionary
    return {
        "current_age": current_age,
        "retirement_age": retirement_age,
        "life_expectancy": life_expectancy,
        "annual_income": annual_income,
        "expenses_working": expenses_working,
        "expenses_retirement": expenses_retirement,
        "financial_assets": financial_assets,
        "risk_aversion": risk_aversion,
        "income_beta": income_beta,
    }
