"""
Plotly chart functions for lifecycle investing dashboard.
Provides visualizations for portfolio allocation glide paths and wealth projections.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_glide_path(df: pd.DataFrame, retirement_age: int = None) -> go.Figure:
    """
    Create a line chart showing optimal portfolio allocation over time.

    Displays the evolution of stock, bond, and cash weights in the financial
    portfolio as the investor ages. Younger investors typically have higher
    stock allocations due to substantial human capital, with a gradual shift
    toward bonds and cash as retirement approaches.

    Args:
        df: DataFrame from compute_glide_path() with columns:
            - age: Investor age
            - stock_weight: Stock allocation (0-1)
            - bond_weight: Bond allocation (0-1)
            - cash_weight: Cash allocation (0-1)
        retirement_age: Optional age at which to draw a vertical line
                       marking retirement transition.

    Returns:
        Plotly Figure object with three lines showing allocation percentages.
    """
    fig = go.Figure()

    # Add stock weight line (blue)
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=df["stock_weight"] * 100,
        name="Stocks",
        mode="lines",
        line=dict(color="#2563eb", width=2.5),
        hovertemplate="Age: %{x}<br>Stocks: %{y:.1f}%<extra></extra>",
    ))

    # Add bond weight line (orange)
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=df["bond_weight"] * 100,
        name="Bonds",
        mode="lines",
        line=dict(color="#ea580c", width=2.5),
        hovertemplate="Age: %{x}<br>Bonds: %{y:.1f}%<extra></extra>",
    ))

    # Add cash weight line (green)
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=df["cash_weight"] * 100,
        name="Cash",
        mode="lines",
        line=dict(color="#16a34a", width=2.5),
        hovertemplate="Age: %{x}<br>Cash: %{y:.1f}%<extra></extra>",
    ))

    # Add vertical line at retirement age if provided
    if retirement_age is not None:
        fig.add_vline(
            x=retirement_age,
            line_dash="dash",
            line_color="gray",
            line_width=1.5,
            annotation_text="Retirement",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="gray",
        )

    # Configure layout for professional appearance
    fig.update_layout(
        xaxis=dict(
            title="Age",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.5)",
            tickmode="linear",
            dtick=5,
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
        ),
        yaxis=dict(
            title="Portfolio Weight (%)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.5)",
            range=[0, 100],
            ticksuffix="%",
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="black"),
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=100),
    )

    return fig


def plot_wealth_projection(df: pd.DataFrame, retirement_age: int = None) -> go.Figure:
    """
    Create a line chart showing wealth projection over the lifecycle.

    Displays three wealth components:
    - Total wealth: financial assets + human capital PV - expenses PV
    - Financial wealth: projected investable assets
    - Human capital PV: present value of remaining labor income

    Uses geometric returns for projection, which account for volatility drag.

    Args:
        df: DataFrame from project_wealth() with columns:
            - age: Investor age
            - projected_financial_wealth: Projected portfolio value
            - pv_human_capital: Present value of remaining labor income
            - total_wealth: Total economic wealth
        retirement_age: Optional age at which to draw a vertical line.

    Returns:
        Plotly Figure object with wealth projection lines.
    """
    fig = go.Figure()

    # Calculate total projected wealth for display
    # Total wealth = financial wealth + human capital PV
    total_projected = df["projected_financial_wealth"] + df["pv_human_capital"]

    # Add total wealth line
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=total_projected,
        name="Total Wealth",
        mode="lines",
        line=dict(color="#7c3aed", width=2.5),
        hovertemplate="Age: %{x}<br>Total: $%{y:,.0f}<extra></extra>",
    ))

    # Add financial wealth line
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=df["projected_financial_wealth"],
        name="Financial Wealth",
        mode="lines",
        line=dict(color="#2563eb", width=2.5),
        hovertemplate="Age: %{x}<br>Financial: $%{y:,.0f}<extra></extra>",
    ))

    # Add human capital PV line
    fig.add_trace(go.Scatter(
        x=df["age"],
        y=df["pv_human_capital"],
        name="Human Capital PV",
        mode="lines",
        line=dict(color="#16a34a", width=2.5),
        hovertemplate="Age: %{x}<br>Human Capital: $%{y:,.0f}<extra></extra>",
    ))

    # Add vertical line at retirement age if provided
    if retirement_age is not None:
        fig.add_vline(
            x=retirement_age,
            line_dash="dash",
            line_color="gray",
            line_width=1.5,
            annotation_text="Retirement",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="gray",
        )

    # Configure layout with currency formatting
    fig.update_layout(
        xaxis=dict(
            title="Age",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.5)",
            tickmode="linear",
            dtick=5,
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
        ),
        yaxis=dict(
            title="Value ($)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.5)",
            tickformat="$,.0f",
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="black"),
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=100),
    )

    return fig
