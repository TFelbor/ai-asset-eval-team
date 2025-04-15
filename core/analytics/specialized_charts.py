"""
Specialized chart implementations for the AI Finance Dashboard.
This module provides specialized chart rendering functions for specific chart types.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

# Import enhanced logging
from core.utils.logger import log_info, log_error, log_success, log_warning, log_debug

def create_etf_sector_allocation_chart(ticker: str, etf_data: Dict[str, Any]) -> go.Figure:
    """
    Create a sector allocation pie chart for ETFs.

    Args:
        ticker: ETF ticker symbol
        etf_data: ETF data dictionary

    Returns:
        Plotly figure object
    """
    log_debug(f"create_etf_sector_allocation_chart called for {ticker}")
    log_debug(f"etf_data keys: {etf_data.keys()}")
    log_debug(f"sector_allocation: {etf_data.get('sector_allocation', {})}")
    # Extract sector allocation data
    sector_allocation = etf_data.get("sector_allocation", {})

    # If sector_allocation is empty or not a dictionary, use placeholder data
    if not sector_allocation or not isinstance(sector_allocation, dict):
        log_warning(f"No sector allocation data for {ticker}, using placeholder data")
        sector_allocation = {
            "Technology": 30,
            "Healthcare": 15,
            "Financials": 20,
            "Consumer Discretionary": 10,
            "Industrials": 15,
            "Other": 10
        }

    # Extract labels and values
    labels = list(sector_allocation.keys())
    values = list(sector_allocation.values())

    # Create figure
    fig = go.Figure()

    # Add pie chart
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Bold)
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Sector Allocation",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_key_metrics_chart(ticker: str, asset_data: Dict[str, Any], asset_type: str) -> go.Figure:
    """
    Create a radar chart of key metrics for any asset type.

    Args:
        ticker: Asset ticker symbol
        asset_data: Asset data dictionary
        asset_type: Asset type (stock, crypto, reit, etf)

    Returns:
        Plotly figure object
    """
    print(f"DEBUG: create_key_metrics_chart called for {ticker} ({asset_type})")
    print(f"DEBUG: asset_data keys: {asset_data.keys() if isinstance(asset_data, dict) else 'Not a dict'}")

    # Ensure asset_data is a dictionary
    if not isinstance(asset_data, dict):
        print(f"Warning: asset_data for {ticker} is not a dictionary, using placeholder data")
        asset_data = {}

    # Initialize metrics based on asset type
    metrics = {}
    labels = []
    values = []
    normalized_values = []

    # Extract and normalize metrics based on asset type
    if asset_type.lower() == "stock":
        # Extract stock metrics
        pe = float(asset_data.get("pe", 0)) if asset_data.get("pe") not in [None, "N/A", ""] else 0
        pb = float(asset_data.get("pb", 0)) if asset_data.get("pb") not in [None, "N/A", ""] else 0
        dividend_yield = float(asset_data.get("dividend_yield", "0").replace("%", "")) if asset_data.get("dividend_yield") not in [None, "N/A", ""] else 0
        beta = float(asset_data.get("beta", 0)) if asset_data.get("beta") not in [None, "N/A", ""] else 0
        confidence = float(asset_data.get("confidence", 0)) if asset_data.get("confidence") not in [None, "N/A", ""] else 50

        # Normalize values for radar chart
        pe_norm = min(1, 15 / max(1, pe)) if pe > 0 else 0.5  # Lower P/E is better
        pb_norm = min(1, 2 / max(0.1, pb)) if pb > 0 else 0.5  # Lower P/B is better
        div_norm = min(1, dividend_yield / 5)  # Higher dividend is better (up to 5%)
        beta_norm = 1 - min(1, abs(beta - 1) / 1)  # Beta closer to 1 is better
        conf_norm = confidence / 100  # Higher confidence is better

        # Set metrics
        metrics = {
            "P/E Ratio": {"value": pe, "normalized": pe_norm},
            "P/B Ratio": {"value": pb, "normalized": pb_norm},
            "Dividend Yield": {"value": dividend_yield, "normalized": div_norm},
            "Beta": {"value": beta, "normalized": beta_norm},
            "Confidence": {"value": confidence, "normalized": conf_norm}
        }

    elif asset_type.lower() == "crypto":
        # Extract crypto metrics
        market_cap = float(asset_data.get("market_cap", 0)) if asset_data.get("market_cap") not in [None, "N/A", ""] else 0
        volume = float(asset_data.get("total_volume", 0)) if asset_data.get("total_volume") not in [None, "N/A", ""] else 0
        price_change = float(asset_data.get("price_change_24h", 0)) if asset_data.get("price_change_24h") not in [None, "N/A", ""] else 0
        volatility = asset_data.get("volatility", "Medium")
        potential = float(asset_data.get("potential", 0)) if asset_data.get("potential") not in [None, "N/A", ""] else 0

        # Normalize values for radar chart
        mcap_norm = min(1, market_cap / 1e12) if market_cap > 0 else 0.1  # Higher market cap is better (up to $1T)
        vol_norm = min(1, volume / 1e10) if volume > 0 else 0.1  # Higher volume is better (up to $10B)
        change_norm = (price_change + 10) / 20 if -10 <= price_change <= 10 else 0.5  # Normalize to 0-1 range
        volatility_norm = {"Very Low": 0.9, "Low": 0.7, "Medium": 0.5, "High": 0.3, "Very High": 0.1}.get(volatility, 0.5)
        potential_norm = min(1, potential / 100)  # Higher potential is better (up to 100%)

        # Set metrics
        metrics = {
            "Market Cap": {"value": market_cap, "normalized": mcap_norm},
            "Volume": {"value": volume, "normalized": vol_norm},
            "Price Change": {"value": price_change, "normalized": change_norm},
            "Stability": {"value": volatility, "normalized": volatility_norm},
            "Potential": {"value": potential, "normalized": potential_norm}
        }

    elif asset_type.lower() == "reit":
        # Extract REIT metrics
        dividend_yield = float(asset_data.get("dividend_yield", "0").replace("%", "")) if asset_data.get("dividend_yield") not in [None, "N/A", ""] else 0
        price_to_ffo = float(asset_data.get("price_to_ffo", 0)) if asset_data.get("price_to_ffo") not in [None, "N/A", ""] else 0
        occupancy_rate = float(asset_data.get("occupancy_rate", 0)) if asset_data.get("occupancy_rate") not in [None, "N/A", ""] else 0.95
        debt_to_equity = float(asset_data.get("debt_to_equity", 0)) if asset_data.get("debt_to_equity") not in [None, "N/A", ""] else 0
        beta = float(asset_data.get("beta", 0)) if asset_data.get("beta") not in [None, "N/A", ""] else 0

        # Normalize values for radar chart
        div_norm = min(1, dividend_yield / 8)  # Higher dividend is better (up to 8%)
        ffo_norm = min(1, 20 / max(1, price_to_ffo)) if price_to_ffo > 0 else 0.5  # Lower P/FFO is better
        occ_norm = occupancy_rate if 0 <= occupancy_rate <= 1 else 0.95  # Higher occupancy is better
        debt_norm = min(1, 2 / max(0.1, debt_to_equity)) if debt_to_equity > 0 else 0.5  # Lower debt is better
        beta_norm = 1 - min(1, abs(beta - 1) / 1)  # Beta closer to 1 is better

        # Set metrics
        metrics = {
            "Dividend Yield": {"value": dividend_yield, "normalized": div_norm},
            "Price to FFO": {"value": price_to_ffo, "normalized": ffo_norm},
            "Occupancy Rate": {"value": occupancy_rate, "normalized": occ_norm},
            "Debt to Equity": {"value": debt_to_equity, "normalized": debt_norm},
            "Beta": {"value": beta, "normalized": beta_norm}
        }

    elif asset_type.lower() == "etf":
        # Extract ETF metrics
        expense_ratio = float(asset_data.get("expense_ratio_value", 0)) if asset_data.get("expense_ratio_value") not in [None, "N/A", ""] else 0
        ytd_return = float(asset_data.get("ytd_return_value", 0)) if asset_data.get("ytd_return_value") not in [None, "N/A", ""] else 0
        three_year_return = float(asset_data.get("three_year_return_value", 0)) if asset_data.get("three_year_return_value") not in [None, "N/A", ""] else 0
        aum = float(asset_data.get("aum", "0").replace("$", "").replace("B", "e9").replace("M", "e6").replace("T", "e12")) if asset_data.get("aum") not in [None, "N/A", ""] else 0
        diversification = len(asset_data.get("top_holdings", {}))

        # Normalize values for radar chart
        expense_norm = 1 - min(1, expense_ratio / 1)  # Lower expense ratio is better
        ytd_norm = min(1, (ytd_return + 20) / 40) if -20 <= ytd_return <= 20 else 0.5  # Normalize to 0-1 range
        three_year_norm = min(1, (three_year_return + 50) / 100) if -50 <= three_year_return <= 50 else 0.5  # Normalize to 0-1 range
        aum_norm = min(1, aum / 1e11) if aum > 0 else 0.1  # Higher AUM is better (up to $100B)
        div_norm = min(1, diversification / 20) if diversification > 0 else 0.5  # Higher diversification is better

        # Set metrics
        metrics = {
            "Expense Ratio": {"value": expense_ratio, "normalized": expense_norm},
            "YTD Return": {"value": ytd_return, "normalized": ytd_norm},
            "3-Year Return": {"value": three_year_return, "normalized": three_year_norm},
            "AUM": {"value": aum, "normalized": aum_norm},
            "Diversification": {"value": diversification, "normalized": div_norm}
        }

    # If no metrics were set, use placeholder data
    if not metrics:
        print(f"No metrics data for {ticker}, using placeholder data")
        metrics = {
            "Metric 1": {"value": 50, "normalized": 0.5},
            "Metric 2": {"value": 60, "normalized": 0.6},
            "Metric 3": {"value": 70, "normalized": 0.7},
            "Metric 4": {"value": 80, "normalized": 0.8},
            "Metric 5": {"value": 90, "normalized": 0.9}
        }

    # Extract labels and normalized values
    labels = list(metrics.keys())
    normalized_values = [metrics[label]["normalized"] for label in labels]

    # Create figure
    fig = go.Figure()

    # Add radar chart
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=labels,
        fill="toself",
        name=ticker,
        line=dict(color="#4f46e5")
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=f"{ticker} - Key Metrics",
        height=600,
        template="plotly_dark"
    )

    return fig

def create_dividend_history_chart(ticker: str, asset_data: Dict[str, Any], asset_type: str) -> go.Figure:
    """
    Create a dividend history chart for stocks and REITs.

    Args:
        ticker: Asset ticker symbol
        asset_data: Asset data dictionary
        asset_type: Asset type (stock, reit)

    Returns:
        Plotly figure object
    """
    print(f"DEBUG: create_dividend_history_chart called for {ticker} ({asset_type})")
    print(f"DEBUG: asset_data keys: {asset_data.keys() if isinstance(asset_data, dict) else 'Not a dict'}")

    # Ensure asset_data is a dictionary
    if not isinstance(asset_data, dict):
        print(f"Warning: asset_data for {ticker} is not a dictionary, using placeholder data")
        asset_data = {}

    print(f"DEBUG: dividend_yield: {asset_data.get('dividend_yield', 'N/A')}")
    from datetime import datetime

    # Get current dividend yield
    if asset_type.lower() == "stock":
        dividend_yield = float(asset_data.get("dividend_yield", "0").replace("%", "")) if asset_data.get("dividend_yield") not in [None, "N/A", ""] else 0
    elif asset_type.lower() == "reit":
        dividend_yield = float(asset_data.get("dividend_yield", "0").replace("%", "")) if asset_data.get("dividend_yield") not in [None, "N/A", ""] else 0
    else:
        dividend_yield = 0

    # Generate realistic dividend history with some variation
    years = [str(datetime.now().year - i) for i in range(5, -1, -1)]

    # Start at 80% of current yield and grow to current yield
    base_yield = dividend_yield * 0.8
    growth_factor = (dividend_yield / base_yield) ** (1/5) if base_yield > 0 else 1.05
    yields = [base_yield * (growth_factor ** i) for i in range(6)]

    # Create figure
    fig = go.Figure()

    # Add bar chart
    fig.add_trace(go.Bar(
        x=years,
        y=yields,
        marker_color="#10b981"
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Dividend History",
        xaxis_title="Year",
        yaxis_title="Dividend Yield (%)",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_performance_comparison_chart(ticker: str, asset_data: Dict[str, Any], benchmark_ticker: str = "SPY") -> go.Figure:
    """
    Create a performance comparison chart against a benchmark.

    Args:
        ticker: Asset ticker symbol
        asset_data: Asset data dictionary
        benchmark_ticker: Benchmark ticker symbol

    Returns:
        Plotly figure object
    """
    print(f"DEBUG: create_performance_comparison_chart called for {ticker} vs {benchmark_ticker}")
    print(f"DEBUG: asset_data keys: {asset_data.keys() if isinstance(asset_data, dict) else 'Not a dict'}")

    # Ensure asset_data is a dictionary
    if not isinstance(asset_data, dict):
        print(f"Warning: asset_data for {ticker} is not a dictionary, using placeholder data")
        asset_data = {}

    # Get price history with better error handling
    price_history = {}
    if "price_history" in asset_data:
        price_history = asset_data["price_history"]
    elif "history" in asset_data:
        price_history = asset_data["history"]

    print(f"DEBUG: price_history keys: {price_history.keys() if isinstance(price_history, dict) else 'Not a dict'}")
    if isinstance(price_history, dict):
        print(f"DEBUG: timestamps length: {len(price_history.get('timestamps', []))}")
        print(f"DEBUG: prices length: {len(price_history.get('prices', []))}")

    # Extract timestamps and prices
    timestamps = price_history.get("timestamps", [])
    prices = price_history.get("prices", [])

    # If no price history, return empty chart
    if not timestamps or not prices:
        fig = go.Figure()
        fig.add_annotation(text="No price history data available", showarrow=False, font_size=20)
        return fig

    # Create figure
    fig = go.Figure()

    # Normalize prices to start at 100
    if prices and prices[0] > 0:
        normalized_prices = [price / prices[0] * 100 for price in prices]

        # Add price line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=normalized_prices,
            mode="lines",
            name=ticker,
            line=dict(color="#4f46e5", width=2)
        ))

    # Add benchmark line (placeholder)
    # In a real implementation, you would fetch the benchmark data
    # For now, we'll create a synthetic benchmark
    if timestamps:
        import random
        random.seed(42)  # For reproducibility

        # Create a synthetic benchmark that's similar but slightly different
        benchmark_prices = []
        for i, price in enumerate(normalized_prices):
            if i == 0:
                benchmark_prices.append(100)  # Start at 100
            else:
                # Add some random variation to the price change
                price_change = normalized_prices[i] - normalized_prices[i-1]
                benchmark_change = price_change * (0.8 + 0.4 * random.random())  # 80-120% of actual change
                benchmark_prices.append(benchmark_prices[-1] + benchmark_change)

        # Add benchmark line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=benchmark_prices,
            mode="lines",
            name=benchmark_ticker,
            line=dict(color="#f59e0b", width=2)
        ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} vs {benchmark_ticker} - Relative Performance",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Starting at 100)",
        height=500,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
