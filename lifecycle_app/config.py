"""
Configuration constants for lifecycle investing calculations.
Based on FINC 450 Capital Markets lecture notes.
"""

# Capital Market Assumptions (from lecture p.31)
RISK_FREE_RATE = 0.02  # 2% real (inflation-adjusted)

# Stock assumptions
STOCK_EXCESS_RETURN = 0.04  # 4% arithmetic excess return
STOCK_VOLATILITY = 0.18  # 18% annual volatility
STOCK_VARIANCE = STOCK_VOLATILITY ** 2  # 0.0324

# Bond assumptions (long-term TIPS)
BOND_EXCESS_RETURN = 0.01  # 1% arithmetic excess return
BOND_VOLATILITY = 0.18  # 18% annual volatility
BOND_VARIANCE = BOND_VOLATILITY ** 2  # 0.0324

# Correlation between stocks and bonds
STOCK_BOND_CORRELATION = 0.0

# Duration of reference ETF for duration matching (lecture p.27)
LTPZ_DURATION = 18.5  # Long-term TIPS ETF duration (used for bond-equivalent calc)

# Geometric return adjustments (lecture p.55-56)
# Geometric return ≈ E[R] - (1/2)σ²
STOCK_GEOMETRIC_EXCESS = STOCK_EXCESS_RETURN - 0.5 * STOCK_VARIANCE  # ≈ 2.38%
BOND_GEOMETRIC_EXCESS = BOND_EXCESS_RETURN - 0.5 * BOND_VARIANCE    # ≈ -0.62%

# Total geometric returns (including risk-free rate)
STOCK_GEOMETRIC_RETURN = RISK_FREE_RATE + STOCK_GEOMETRIC_EXCESS  # ≈ 4.38%
BOND_GEOMETRIC_RETURN = RISK_FREE_RATE + BOND_GEOMETRIC_EXCESS    # ≈ 1.38%
CASH_GEOMETRIC_RETURN = RISK_FREE_RATE                             # 2%

# Income beta assumptions (lecture p.48-49)
BOND_LIKE_INCOME_BETA = 0.0   # Stable income (teacher, consultant, government)
STOCK_LIKE_INCOME_BETA = 0.4  # Volatile income (investment banker, trader)

# Default user inputs
DEFAULT_AGE = 25
DEFAULT_RETIREMENT_AGE = 65
DEFAULT_LIFE_EXPECTANCY = 85
DEFAULT_ANNUAL_INCOME = 150_000
DEFAULT_EXPENSES_WORKING = 80_000
DEFAULT_EXPENSES_RETIREMENT = 60_000
DEFAULT_FINANCIAL_ASSETS = 50_000
DEFAULT_RISK_AVERSION = 2.0
DEFAULT_INCOME_BETA = BOND_LIKE_INCOME_BETA

# Input ranges for UI
MIN_AGE = 18
MAX_AGE = 80
MIN_RETIREMENT_AGE = 30
MAX_RETIREMENT_AGE = 80
MIN_LIFE_EXPECTANCY = 60
MAX_LIFE_EXPECTANCY = 100
MIN_INCOME = 0
MAX_INCOME = 2_000_000
MIN_EXPENSES = 0
MAX_EXPENSES = 1_000_000
MIN_ASSETS = 0
MAX_ASSETS = 50_000_000
MIN_RISK_AVERSION = 1.0
MAX_RISK_AVERSION = 10.0
