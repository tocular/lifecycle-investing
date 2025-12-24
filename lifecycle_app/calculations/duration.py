"""
Duration calculations for lifecycle investing.
Based on FINC 450 Capital Markets lecture formulas.

Duration measures the weighted-average time to receive cash flows,
which determines sensitivity to interest rate changes.
"""

from ..config import (
    RISK_FREE_RATE,
    STOCK_EXCESS_RETURN,
)

from .present_value import pv_annuity


def annuity_duration(rate: float, n_periods: int) -> float:
    """
    Calculate the Macaulay duration of an n-period annuity.

    Duration formula for an annuity:
        D = (1 + r) / r - n / ((1 + r)^n - 1)

    Duration represents the weighted-average time to receive cash flows,
    where weights are the present values of each payment.

    Args:
        rate: Discount rate per period (as decimal)
        n_periods: Number of periods in the annuity

    Returns:
        Macaulay duration in periods

    Example:
        >>> annuity_duration(0.02, 35)
        16.79...  # Duration of a 35-year annuity at 2%
    """
    if n_periods <= 0:
        return 0.0

    # Handle edge case where rate is zero
    # When r → 0, duration approaches (n + 1) / 2 (simple average of payment times)
    if rate == 0:
        return (n_periods + 1) / 2

    # Standard Macaulay duration formula for annuity
    term1 = (1 + rate) / rate
    term2 = n_periods / ((1 + rate) ** n_periods - 1)

    return term1 - term2


def human_capital_duration(
    years_working: int,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate the duration of human capital.

    Human capital is treated as an annuity of future income payments.
    Its duration determines how it behaves as an asset in the portfolio:
    - Short duration: acts like short-term bonds
    - Long duration: acts like long-term bonds

    Args:
        years_working: Number of years until retirement
        risk_free_rate: Discount rate for human capital

    Returns:
        Macaulay duration of human capital stream

    Example:
        >>> human_capital_duration(35, 0.02)
        16.79...  # Young worker has ~17 year duration
    """
    return annuity_duration(risk_free_rate, years_working)


def expense_duration(
    years_working: int,
    years_retirement: int,
    working_expenses: float,
    retirement_expenses: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calculate the weighted-average duration of lifetime expenses.

    Expenses consist of two annuities:
    1. Working period expenses: annuity starting now
    2. Retirement expenses: deferred annuity starting after working years

    The overall duration is the PV-weighted average of both streams.

    Args:
        years_working: Number of working years remaining
        years_retirement: Expected years in retirement
        working_expenses: Annual expenses during working years
        retirement_expenses: Annual expenses during retirement
        risk_free_rate: Discount rate for expenses

    Returns:
        Weighted-average duration of expense stream

    Example:
        >>> expense_duration(35, 20, 80_000, 60_000, 0.02)
        # Returns duration weighted by PV of each expense stream
    """
    if years_working <= 0 and years_retirement <= 0:
        return 0.0

    # Calculate PV of each expense stream (for weighting)
    pv_working = pv_annuity(working_expenses, risk_free_rate, years_working)

    # PV of retirement expenses at retirement start
    pv_retirement_at_retirement = pv_annuity(
        retirement_expenses, risk_free_rate, years_retirement
    )

    # Discount retirement PV back to today
    if risk_free_rate == 0:
        discount_factor = 1.0
    else:
        discount_factor = (1 + risk_free_rate) ** (-years_working)
    pv_retirement_today = pv_retirement_at_retirement * discount_factor

    total_pv = pv_working + pv_retirement_today

    # Handle edge case where total PV is zero
    if total_pv == 0:
        return 0.0

    # Duration of working expenses (standard annuity)
    duration_working = annuity_duration(risk_free_rate, years_working)

    # Duration of retirement expenses (deferred annuity)
    # The duration is the time to start of retirement + duration of retirement annuity
    # For a deferred annuity starting at time T with duration D_annuity:
    # Total duration = T + D_annuity (where T is the deferral period)
    duration_retirement_annuity = annuity_duration(risk_free_rate, years_retirement)
    duration_retirement = years_working + duration_retirement_annuity

    # Weighted average duration by present value
    weighted_duration = (
        (pv_working * duration_working + pv_retirement_today * duration_retirement)
        / total_pv
    )

    return weighted_duration
