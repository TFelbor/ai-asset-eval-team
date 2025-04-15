"""
Simple and reliable chart generation for the AI Finance Dashboard.
This module provides standardized functions for generating charts with minimal complexity.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

def create_simple_price_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a simple price chart with minimal complexity.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])

    # Validate data
    if not timestamps or not prices or len(timestamps) != len(prices):
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="No price data available", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)) and ts > 1e10:
                dates.append(datetime.fromtimestamp(ts/1000))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError):
            # Handle invalid timestamps
            continue

    # Create figure
    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name=f'{ticker} Price',
        line=dict(color='#4f46e5', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_simple_volume_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a simple volume chart with minimal complexity.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data
    timestamps = price_data.get("timestamps", [])
    volumes = price_data.get("volumes", [])

    # Validate data
    if not timestamps or not volumes or len(timestamps) != len(volumes):
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="No volume data available", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)) and ts > 1e10:
                dates.append(datetime.fromtimestamp(ts/1000))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError):
            # Handle invalid timestamps
            continue

    # Create figure
    fig = go.Figure()

    # Add volume bars
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name="Volume",
        marker_color="rgba(128, 128, 128, 0.5)"
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_simple_performance_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a simple performance chart showing percentage change.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])

    # Validate data
    if not timestamps or not prices or len(timestamps) != len(prices) or len(prices) < 2:
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="Insufficient price data for performance chart", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)) and ts > 1e10:
                dates.append(datetime.fromtimestamp(ts/1000))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError):
            # Handle invalid timestamps
            continue

    # Calculate percentage change from first day
    try:
        base_price = float(prices[0])
        if base_price <= 0:
            raise ValueError("Base price must be positive")

        percent_changes = []
        for price in prices:
            try:
                price_float = float(price)
                percent_changes.append((price_float / base_price - 1) * 100)
            except (ValueError, TypeError, ZeroDivisionError):
                # Handle invalid prices
                percent_changes.append(None)
    except (ValueError, TypeError, ZeroDivisionError):
        # Return empty chart if calculation fails
        fig = go.Figure()
        fig.add_annotation(text="Could not calculate performance", showarrow=False, font_size=20)
        return fig

    # Create figure
    fig = go.Figure()

    # Add performance line
    fig.add_trace(go.Scatter(
        x=dates,
        y=percent_changes,
        mode='lines',
        name=f'{ticker} Performance',
        line=dict(color='#4CAF50', width=2)
    ))

    # Add zero line
    fig.add_shape(
        type="line",
        x0=dates[0],
        y0=0,
        x1=dates[-1],
        y1=0,
        line=dict(color="#888888", width=1, dash="dash")
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Performance",
        xaxis_title="Date",
        yaxis_title="% Change",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_simple_comparison_chart(tickers: List[str], price_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create a simple comparison chart for multiple assets.

    Args:
        tickers: List of asset ticker symbols
        price_data: Dictionary mapping tickers to price data dictionaries

    Returns:
        Plotly figure object
    """
    # Validate data
    if not tickers or not price_data:
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="No comparison data available", showarrow=False, font_size=20)
        return fig

    # Create figure
    fig = go.Figure()

    # Add line for each ticker
    for ticker in tickers:
        if ticker not in price_data:
            continue

        ticker_data = price_data[ticker]
        timestamps = ticker_data.get("timestamps", [])
        prices = ticker_data.get("prices", [])

        # Validate data for this ticker
        if not timestamps or not prices or len(timestamps) != len(prices):
            continue

        # Convert timestamps to datetime objects if they're in milliseconds
        dates = []
        for ts in timestamps:
            try:
                if isinstance(ts, (int, float)) and ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                else:
                    dates.append(ts)
            except (ValueError, TypeError, OverflowError):
                # Handle invalid timestamps
                continue

        # Normalize prices to percentage change from first day
        try:
            base_price = float(prices[0])
            if base_price <= 0:
                raise ValueError("Base price must be positive")

            percent_changes = []
            for price in prices:
                try:
                    price_float = float(price)
                    percent_changes.append((price_float / base_price - 1) * 100)
                except (ValueError, TypeError, ZeroDivisionError):
                    # Handle invalid prices
                    percent_changes.append(None)

            # Add line for this ticker
            fig.add_trace(go.Scatter(
                x=dates,
                y=percent_changes,
                mode='lines',
                name=ticker
            ))
        except (ValueError, TypeError, ZeroDivisionError):
            # Skip this ticker if calculation fails
            continue

    # Check if any traces were added
    if not fig.data:
        fig.add_annotation(text="Could not generate comparison chart", showarrow=False, font_size=20)
        return fig

    # Add zero line
    first_dates = fig.data[0].x
    if len(first_dates) > 0:
        fig.add_shape(
            type="line",
            x0=first_dates[0],
            y0=0,
            x1=first_dates[-1],
            y1=0,
            line=dict(color="#888888", width=1, dash="dash")
        )

    # Update layout
    fig.update_layout(
        title="Asset Comparison",
        xaxis_title="Date",
        yaxis_title="% Change",
        height=500,
        template="plotly_dark"
    )

    return fig

def create_candlestick_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a candlestick chart for stock data.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data with OHLC values

    Returns:
        Plotly figure object
    """
    # Extract data
    timestamps = price_data.get("timestamps", [])
    opens = price_data.get("open", [])
    highs = price_data.get("high", [])
    lows = price_data.get("low", [])
    closes = price_data.get("close", [])
    volumes = price_data.get("volumes", [])

    # Validate data - we need all OHLC data
    if not timestamps or not opens or not highs or not lows or not closes or \
       len(timestamps) != len(opens) or len(timestamps) != len(highs) or \
       len(timestamps) != len(lows) or len(timestamps) != len(closes):
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="Insufficient OHLC data for candlestick chart", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)) and ts > 1e10:
                dates.append(datetime.fromtimestamp(ts/1000))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError):
            # Handle invalid timestamps
            continue

    # Create figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name=ticker
    ))

    # Add volume as a bar chart at the bottom if available
    if volumes and len(volumes) == len(dates):
        # Create a secondary y-axis for volume
        fig.add_trace(go.Bar(
            x=dates,
            y=volumes,
            name="Volume",
            marker_color="rgba(128, 128, 128, 0.5)",
            yaxis="y2"
        ))

        # Update layout to include secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,  # Taller chart for better visibility
        template="plotly_dark",
        xaxis_rangeslider_visible=False  # Disable rangeslider for cleaner look
    )

    # Add buttons for time range selection
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.1,
                y=1.1,
                showactive=True,
                buttons=[
                    dict(label="1M", method="relayout", args=[{"xaxis.range": [dates[-30] if len(dates) > 30 else dates[0], dates[-1]]}]),
                    dict(label="3M", method="relayout", args=[{"xaxis.range": [dates[-90] if len(dates) > 90 else dates[0], dates[-1]]}]),
                    dict(label="6M", method="relayout", args=[{"xaxis.range": [dates[-180] if len(dates) > 180 else dates[0], dates[-1]]}]),
                    dict(label="YTD", method="relayout", args=[{"xaxis.range": [datetime(datetime.now().year, 1, 1), dates[-1]]}]),
                    dict(label="1Y", method="relayout", args=[{"xaxis.range": [dates[-365] if len(dates) > 365 else dates[0], dates[-1]]}]),
                    dict(label="All", method="relayout", args=[{"xaxis.range": [dates[0], dates[-1]]}]),
                ]
            )
        ]
    )

    return fig

def generate_simple_chart(ticker: str, asset_data: Dict[str, Any], chart_type: str) -> go.Figure:
    """
    Generate a simple chart based on asset data and chart type.

    Args:
        ticker: Asset ticker symbol
        asset_data: Dictionary containing asset data
        chart_type: Type of chart to generate

    Returns:
        Plotly figure object
    """
    # Get price history data
    price_history = asset_data.get("price_history", {})

    # Debug price history data
    print(f"Price history keys for {ticker}: {price_history.keys() if isinstance(price_history, dict) else 'Not a dict'}")
    if isinstance(price_history, dict) and "prices" in price_history:
        print(f"Price data length: {len(price_history['prices'])}")
    if isinstance(price_history, dict) and "open" in price_history:
        print(f"OHLC data available: open={len(price_history['open'])}, high={len(price_history.get('high', []))}, low={len(price_history.get('low', []))}, close={len(price_history.get('close', []))}")

    # Generate chart based on type
    if chart_type == "price":
        return create_simple_price_chart(ticker, price_history)
    elif chart_type == "volume":
        return create_simple_volume_chart(ticker, price_history)
    elif chart_type == "performance":
        return create_simple_performance_chart(ticker, price_history)
    elif chart_type == "candlestick":
        return create_candlestick_chart(ticker, price_history)
    else:
        # Default to a simple placeholder chart
        fig = go.Figure()
        fig.add_annotation(text=f"Chart type '{chart_type}' not available", showarrow=False, font_size=20)
        return fig
