"""
Present value calculations for lifecycle investing.
Based on FINC 450 Capital Markets lecture formulas.
"""

from ..config import (
    RISK_FREE_RATE,
    STOCK_EXCESS_RETURN,
)


def pv_annuity(payment: float, rate: float, n_periods: int) -> float:
    """
    Calculate the present value of an ordinary annuity.

    Uses the standard annuity formula:
        PV = payment × [(1 - (1 + rate)^(-n_periods)) / rate]

    Args:
        payment: Fixed payment amount per period
        rate: Discount rate per period (as decimal, e.g., 0.02 for 2%)
        n_periods: Number of periods

    Returns:
        Present value of the annuity stream

    Example:
        >>> pv_annuity(100_000, 0.02, 50)  # $100k/year for 50 years at 2%
        3127667.68...  # approximately $3.1M
    """
    if n_periods <= 0:
        return 0.0

    # Handle edge case where rate is zero (simple sum of payments)
    if rate == 0:
        return payment * n_periods

    # Standard present value of annuity formula
    annuity_factor = (1 - (1 + rate) ** (-n_periods)) / rate
    return payment * annuity_factor


def pv_human_capital(
    annual_income: float,
    years_working: int,
    risk_free_rate: float = RISK_FREE_RATE,
    income_beta: float = 0.0,
    equity_premium: float = STOCK_EXCESS_RETURN,
) -> float:
    """
    Calculate the present value of human capital (future labor income).

    Human capital is the discounted value of expected future earnings.
    The discount rate depends on the riskiness of the income stream:
    - Bond-like income (beta=0): Stable jobs like teachers, consultants
    - Stock-like income (beta>0): Volatile jobs like traders, bankers

    The discount rate follows CAPM:
        discount_rate = risk_free_rate + income_beta × equity_premium

    Args:
        annual_income: Expected annual income during working years
        years_working: Number of years until retirement
        risk_free_rate: Real risk-free rate (default from config)
        income_beta: Sensitivity of income to market risk (0 = bond-like, >0 = stock-like)
        equity_premium: Expected excess return of equities over risk-free rate

    Returns:
        Present value of future labor income

    Example:
        >>> pv_human_capital(250_000, 35, 0.02, income_beta=0.0)
        6197767.86...  # approximately $6.2M (lecture example)
    """
    # CAPM-based discount rate reflecting income risk
    discount_rate = risk_free_rate + income_beta * equity_premium

    return pv_annuity(annual_income, discount_rate, years_working)


def pv_expenses(
    working_expenses: float,
    retirement_expenses: float,
    years_working: int,
    years_retirement: int,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate the present value of lifetime expenses.

    Expenses are discounted at the risk-free rate since they represent
    fixed obligations (consumption needs are relatively stable).

    The calculation has two components:
    1. Working period expenses: PV of annuity for years_working periods
    2. Retirement expenses: PV of annuity, discounted back to today

    Args:
        working_expenses: Annual expenses during working years
        retirement_expenses: Annual expenses during retirement
        years_working: Number of working years remaining
        years_retirement: Expected years in retirement
        risk_free_rate: Real risk-free rate (default from config)

    Returns:
        Total present value of all future expenses

    Example:
        >>> pv_expenses(100_000, 100_000, 35, 15, 0.02)
        # $100k/year for 50 total years ≈ $3.1M
    """
    # PV of expenses during working years (discounted to today)
    pv_working = pv_annuity(working_expenses, risk_free_rate, years_working)

    # PV of retirement expenses at the start of retirement
    pv_retirement_at_retirement = pv_annuity(
        retirement_expenses, risk_free_rate, years_retirement
    )

    # Discount retirement expenses back to today
    # This is the value at retirement × discount factor for years_working periods
    if risk_free_rate == 0:
        discount_factor = 1.0
    else:
        discount_factor = (1 + risk_free_rate) ** (-years_working)

    pv_retirement_today = pv_retirement_at_retirement * discount_factor

    return pv_working + pv_retirement_today


def total_wealth(
    financial_assets: float,
    pv_human_capital: float,
    pv_expenses: float,
) -> float:
    """
    Calculate total economic wealth available for investment.

    Total wealth represents the resources available for meeting lifetime
    consumption goals after accounting for human capital and obligations:

        Total Wealth = Financial Assets + Human Capital - PV(Expenses)

    This is the "pot" that must be allocated across risky and safe assets
    to optimize lifetime utility.

    Args:
        financial_assets: Current portfolio value (savings, investments)
        pv_human_capital: Present value of future labor income
        pv_expenses: Present value of future expenses/liabilities

    Returns:
        Total economic wealth (can be negative if obligations exceed assets)

    Example:
        >>> total_wealth(50_000, 6_200_000, 3_100_000)
        3150000.0  # $3.15M total wealth
    """
    return financial_assets + pv_human_capital - pv_expenses
