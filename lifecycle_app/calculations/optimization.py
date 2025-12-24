"""
Portfolio optimization functions for lifecycle investing.
Based on FINC 450 Capital Markets lecture notes (p.31-35).
"""

from typing import Dict

from ..config import (
    STOCK_EXCESS_RETURN,
    BOND_EXCESS_RETURN,
    STOCK_VARIANCE,
    BOND_VARIANCE,
    LTPZ_DURATION,
)


def optimal_total_wealth_weights(
    risk_aversion: float,
    stock_excess_return: float = STOCK_EXCESS_RETURN,
    bond_excess_return: float = BOND_EXCESS_RETURN,
    stock_variance: float = STOCK_VARIANCE,
    bond_variance: float = BOND_VARIANCE,
) -> Dict[str, float]:
    """
    Compute optimal total wealth portfolio weights using mean-variance optimization.

    Assumes zero correlation between stocks and bonds, which simplifies the
    optimization to independent allocation decisions for each risky asset.

    Formula (with zero correlation):
        w_stock = stock_excess_return / (risk_aversion * stock_variance)
        w_bond = bond_excess_return / (risk_aversion * bond_variance)
        w_cash = 1 - w_stock - w_bond

    Lecture example with gamma=2:
        w_stock = 0.04 / (2 * 0.0324) = 0.617 (62%)
        w_bond = 0.01 / (2 * 0.0324) = 0.154 (15%)
        w_cash = 1 - 0.617 - 0.154 = 0.229 (23%)

    Args:
        risk_aversion: Coefficient of relative risk aversion (gamma).
                       Higher values indicate more risk-averse investors.
        stock_excess_return: Expected arithmetic excess return on stocks over cash.
        bond_excess_return: Expected arithmetic excess return on bonds over cash.
        stock_variance: Variance of stock returns (volatility squared).
        bond_variance: Variance of bond returns (volatility squared).

    Returns:
        Dictionary with keys 'stocks', 'bonds', 'cash' containing optimal weights.
        Weights sum to 1.0 but individual weights may be negative (short positions)
        or greater than 1.0 (leveraged positions) for extreme risk preferences.
    """
    # Optimal weight for each risky asset under mean-variance optimization
    # with uncorrelated assets: w = excess_return / (gamma * variance)
    w_stock = stock_excess_return / (risk_aversion * stock_variance)
    w_bond = bond_excess_return / (risk_aversion * bond_variance)

    # Residual allocation goes to risk-free cash
    w_cash = 1.0 - w_stock - w_bond

    return {
        "stocks": w_stock,
        "bonds": w_bond,
        "cash": w_cash,
    }


def financial_portfolio_weights(
    total_wealth: float,
    financial_assets: float,
    pv_human_capital: float,
    human_capital_duration: float,
    pv_expenses: float,
    expense_duration: float,
    optimal_weights: Dict[str, float],
    ltpz_duration: float = LTPZ_DURATION,
    income_beta: float = 0.0,
    constrained: bool = True,
) -> Dict[str, float]:
    """
    Convert total wealth target weights to financial portfolio weights.

    Implements Step 2 of Lecture 13's 4-step recipe:
    Convert future income AND future expenses into tradable portfolios with
    the same risk properties.

    Human capital (LONG position):
        - Bond-like income: acts like LONG bonds + cash based on duration
        - Stock-like income: portion correlates with stocks

    Expenses (SHORT position - it's a liability):
        - Expenses act like SHORT bonds + cash based on expense duration

    Net "Human" position = Human Capital tradable - Expenses tradable

    Financial portfolio weights are backed out by:
        1. Computing target $ amounts = optimal_weights * total_wealth
        2. Computing implicit exposure from human capital (LONG bonds/cash)
        3. Computing implicit exposure from expenses (SHORT bonds/cash)
        4. Net implicit = Human capital exposure - Expense exposure
        5. Financial $ needed = Target $ - Net implicit exposure
        6. Financial weights = Financial $ needed / financial_assets

    Args:
        total_wealth: Total economic wealth (financial assets + PV human capital
                      - PV expenses).
        financial_assets: Current value of investable financial assets.
        pv_human_capital: Present value of future labor income.
        human_capital_duration: Duration of human capital (years).
        pv_expenses: Present value of future expenses (working + retirement).
        expense_duration: Weighted-average duration of expense stream (years).
        optimal_weights: Target weights from optimal_total_wealth_weights().
        ltpz_duration: Duration of long-term TIPS used as bond benchmark.
        income_beta: Fraction of human capital that behaves like stocks.
                     0.0 = bond-like (teacher, government), 0.4 = stock-like (banker).
        constrained: If True, clip weights to [0, 1] to prevent short selling
                     and leverage. If False, allow unconstrained weights.

    Returns:
        Dictionary with keys 'stocks', 'bonds', 'cash' containing financial
        portfolio weights that sum to 1.0 (when constrained and feasible).
    """
    # Step 2a: Convert Human Capital to tradable portfolio (LONG position)
    # The bond-like portion of human capital (1 - income_beta)
    bond_like_hc = pv_human_capital * (1.0 - income_beta)

    # Bond-equivalent exposure: fraction based on duration relative to LTPZ
    hc_bond_equiv = (human_capital_duration / ltpz_duration) * bond_like_hc

    # Cash-equivalent exposure: remainder of bond-like human capital
    hc_cash_equiv = bond_like_hc - hc_bond_equiv

    # Stock-equivalent exposure: the stock-like portion of human capital
    hc_stock_equiv = pv_human_capital * income_beta

    # Step 2b: Convert Expenses to tradable portfolio (SHORT position)
    # Expenses are liabilities, so they represent a SHORT position in bonds/cash
    exp_bond_equiv = (expense_duration / ltpz_duration) * pv_expenses
    exp_cash_equiv = pv_expenses - exp_bond_equiv

    # Step 2c: Compute NET "Human" position (Human Capital - Expenses)
    # This is the implicit portfolio from non-financial wealth
    net_stock_equiv = hc_stock_equiv  # Expenses don't have stock component
    net_bond_equiv = hc_bond_equiv - exp_bond_equiv
    net_cash_equiv = hc_cash_equiv - exp_cash_equiv

    # Step 3: Target dollar amounts in each asset class based on total wealth
    target_stocks = optimal_weights["stocks"] * total_wealth
    target_bonds = optimal_weights["bonds"] * total_wealth
    target_cash = optimal_weights["cash"] * total_wealth

    # Step 4: Financial portfolio makes up the difference between targets
    # and net implicit exposure from human capital minus expenses
    financial_stocks_needed = target_stocks - net_stock_equiv
    financial_bonds_needed = target_bonds - net_bond_equiv
    financial_cash_needed = target_cash - net_cash_equiv

    # Handle edge case where financial assets are zero or negative
    if financial_assets <= 0:
        return {"stocks": 0.0, "bonds": 0.0, "cash": 1.0}

    # Convert dollar amounts to portfolio weights
    w_stock = financial_stocks_needed / financial_assets
    w_bond = financial_bonds_needed / financial_assets
    w_cash = financial_cash_needed / financial_assets

    # Apply constraints if requested (no short selling or leverage)
    if constrained:
        w_stock = max(0.0, min(1.0, w_stock))
        w_bond = max(0.0, min(1.0, w_bond))
        w_cash = max(0.0, min(1.0, w_cash))

        # Renormalize weights to sum to 1.0 after clipping
        total_weight = w_stock + w_bond + w_cash
        if total_weight > 0:
            w_stock /= total_weight
            w_bond /= total_weight
            w_cash /= total_weight
        else:
            # Fallback to all cash if all weights clipped to zero
            w_stock, w_bond, w_cash = 0.0, 0.0, 1.0

    return {
        "stocks": w_stock,
        "bonds": w_bond,
        "cash": w_cash,
    }
