# Lifecycle Investing Calculator

A web application for calculating optimal portfolio allocation using the total wealth framework.

## Overview

This tool helps you figure out how to allocate your investment portfolio. It uses the total wealth framework: your current assets plus the present value of future earnings, minus expected expenses.

## Features

- **Present Value Calculations**: Compute the present value of future income and expenses
- **Portfolio Optimization**: Mean-variance optimization for stocks, bonds, and cash allocation
- **Glide Path Visualization**: See how your allocation should evolve over time
- **Wealth Projection**: Project financial wealth using geometric returns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run run.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
lifecycle-investing/
├── lifecycle_app/
│   ├── main.py              # Streamlit entry point
│   ├── config.py            # Market assumptions and defaults
│   ├── calculations/        # Core financial calculations
│   └── ui/                  # Interface components
├── tests/                   # Test suite
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/
```

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice.
