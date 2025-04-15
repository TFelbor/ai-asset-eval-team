"""
Direct chart implementation for the AI Finance Dashboard.
This module provides a more reliable approach for rendering charts in Streamlit.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import streamlit as st

# Import enhanced logging
from core.utils.logger import (
    log_info, log_error, log_success, log_warning, log_debug,
    log_api_call, log_data_operation, log_exception, performance_timer
)

# Import specialized chart functions
from core.analytics.specialized_charts import (
    create_etf_sector_allocation_chart,
    create_key_metrics_chart,
    create_dividend_history_chart,
    create_performance_comparison_chart
)

def format_price_label(price: float, currency: str = "$") -> str:
    """Format price labels consistently"""
    if price >= 1e6:
        return f"{currency}{price/1e6:.2f}M"
    elif price >= 1e3:
        return f"{currency}{price/1e3:.2f}K"
    else:
        return f"{currency}{price:.2f}"

def format_percent_label(value: float) -> str:
    """Format percentage labels consistently"""
    return f"{value:+.2f}%"

def standardize_chart_layout(fig: go.Figure, title: str, x_title: str, y_title: str):
    """Standardize chart layout across all visualizations"""
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=500,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

def add_volume_bars(fig: go.Figure, dates, volumes):
    """Add standardized volume bars to chart"""
    if volumes and len(volumes) == len(dates):
        fig.add_trace(go.Bar(
            x=dates,
            y=volumes,
            name='Volume',
            marker_color='rgba(79, 70, 229, 0.3)',
            yaxis='y2'
        ))

        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

def create_fallback_chart(error_message: str = "Chart could not be generated") -> go.Figure:
    """Create standardized fallback chart"""
    fig = go.Figure()
    fig.add_annotation(
        text=error_message,
        showarrow=False,
        font_size=20
    )
    standardize_chart_layout(
        fig,
        title="Error",
        x_title="",
        y_title=""
    )
    return fig

def create_direct_price_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a simple price chart with direct rendering.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data with enhanced error handling
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])
    volumes = price_data.get("volumes", [])

    # Log debug info
    log_info(f"Creating price chart for {ticker}")
    log_info(f"Data points: timestamps={len(timestamps)}, prices={len(prices)}, volumes={len(volumes)}")
    if timestamps and prices:
        log_info(f"Sample data - timestamp: {timestamps[0]}, price: {prices[0]}")

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
            if isinstance(ts, (int, float)):
                # Handle millisecond timestamps (13 digits)
                if ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                # Handle second timestamps (10 digits)
                else:
                    dates.append(datetime.fromtimestamp(ts))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError) as e:
            # Handle invalid timestamps
            log_error(f"Error converting timestamp {ts}: {e}")
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

    # Add volume bars if available
    add_volume_bars(fig, dates, volumes)

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified"
    )

    return fig

def create_direct_candlestick_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a candlestick chart with direct rendering.

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

    # Debug info
    log_info(f"Candlestick data for {ticker}:")
    log_info(f"  Timestamps: {len(timestamps)} items")
    log_info(f"  Opens: {len(opens)} items")
    log_info(f"  Highs: {len(highs)} items")
    log_info(f"  Lows: {len(lows)} items")
    log_info(f"  Closes: {len(closes)} items")
    log_info(f"  Volumes: {len(volumes)} items")

    # Check if we have OHLC data
    if not all([timestamps, opens, highs, lows, closes]) or \
       len(timestamps) != len(opens) or \
       len(timestamps) != len(highs) or \
       len(timestamps) != len(lows) or \
       len(timestamps) != len(closes):

        # If we only have close prices, create synthetic OHLC data
        if timestamps and "prices" in price_data:
            log_info(f"Creating synthetic OHLC data for {ticker}")
            # Ensure all prices are floats
            try:
                closes = [float(p) for p in price_data["prices"]]
                # Create synthetic OHLC data
                opens = [closes[0]] + closes[:-1]
                highs = [max(float(o), float(c)) * (1 + np.random.uniform(0, 0.005)) for o, c in zip(opens, closes)]
                lows = [min(float(o), float(c)) * (1 - np.random.uniform(0, 0.005)) for o, c in zip(opens, closes)]
            except (ValueError, TypeError) as e:
                # Log the error
                print(f"Error converting price data to floats: {str(e)}")
                # Not enough valid data for a candlestick chart
                fig = go.Figure()
                fig.add_annotation(text="Error processing price data for candlestick chart", showarrow=False, font_size=20)
                return fig
        else:
            # Not enough data for a candlestick chart
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for candlestick chart", showarrow=False, font_size=20)
            return fig

    # Convert timestamps to datetime if they're not already
    dates = []
    try:
        for ts in timestamps:
            if isinstance(ts, (int, float)) and ts > 1e10:
                dates.append(datetime.fromtimestamp(ts/1000))
            elif isinstance(ts, (int, float)):
                dates.append(datetime.fromtimestamp(ts))
            elif isinstance(ts, str):
                try:
                    dates.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                except ValueError:
                    from dateutil import parser
                    dates.append(parser.parse(ts))
            else:
                dates.append(ts)
    except Exception as e:
        log_error(f"Error converting timestamps: {str(e)}")
        # Create a fallback figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error processing timestamps: {str(e)}", showarrow=False, font_size=20)
        return fig

    # Create figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name=ticker,
        increasing_line_color='#10b981',  # Green
        decreasing_line_color='#ef4444'   # Red
    ))

    # Add volume bars if available
    add_volume_bars(fig, dates, volumes)

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,  # Disable rangeslider for cleaner look
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig

def create_direct_volume_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a volume chart with direct rendering.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data with enhanced error handling
    timestamps = price_data.get("timestamps", [])
    volumes = price_data.get("volumes", [])

    # Log debug info
    log_info(f"Creating volume chart for {ticker}")
    log_info(f"Data points: timestamps={len(timestamps)}, volumes={len(volumes)}")
    if timestamps and volumes:
        log_info(f"Sample data - timestamp: {timestamps[0]}, volume: {volumes[0]}")

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
            if isinstance(ts, (int, float)):
                # Handle millisecond timestamps (13 digits)
                if ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                # Handle second timestamps (10 digits)
                else:
                    dates.append(datetime.fromtimestamp(ts))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError) as e:
            # Handle invalid timestamps
            log_error(f"Error converting timestamp {ts}: {e}")
            continue

    # Process volumes with error handling
    processed_volumes = []
    for vol in volumes:
        try:
            processed_volumes.append(float(vol))
        except (ValueError, TypeError) as e:
            log_error(f"Error converting volume {vol}: {e}")
            processed_volumes.append(0)  # Use 0 as fallback

    # Create figure
    fig = go.Figure()

    # Add volume bars
    fig.add_trace(go.Bar(
        x=dates,
        y=processed_volumes,
        name="Volume",
        marker_color="rgba(79, 70, 229, 0.7)"
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=500,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified"
    )

    return fig

def create_direct_history_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create an enhanced price history chart with direct rendering.
    This chart includes both price and volume data with annotations for key events.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data with enhanced error handling
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])
    volumes = price_data.get("volumes", [])

    # Log debug info
    log_info(f"Creating history chart for {ticker}")
    log_info(f"Data points: timestamps={len(timestamps)}, prices={len(prices)}, volumes={len(volumes)}")
    if timestamps and prices:
        log_info(f"Sample data - timestamp: {timestamps[0]}, price: {prices[0]}")

    # Validate data
    if not timestamps or not prices or len(timestamps) != len(prices):
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="No price history data available", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)):
                # Handle millisecond timestamps (13 digits)
                if ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                # Handle second timestamps (10 digits)
                else:
                    dates.append(datetime.fromtimestamp(ts))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError) as e:
            # Handle invalid timestamps
            print(f"Error converting timestamp {ts}: {e}")
            continue

    # Create figure with subplots: price on top, volume on bottom
    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name=f'{ticker} Price',
        line=dict(color='#4f46e5', width=2)
    ))

    # Add volume bars if available
    add_volume_bars(fig, dates, volumes)

    # Calculate some statistics
    if len(prices) > 1:
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        start_price = prices[0]
        end_price = prices[-1]
        price_change = end_price - start_price
        percent_change = (price_change / start_price) * 100 if start_price > 0 else 0

        # Add annotations for key statistics
        fig.add_annotation(
            x=dates[0],
            y=start_price,
            text=f"Start: ${start_price:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-40,
            ay=-40
        )

        fig.add_annotation(
            x=dates[-1],
            y=end_price,
            text=f"End: ${end_price:.2f} ({percent_change:+.2f}%)",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40
        )

        # Add a horizontal line for the average price
        fig.add_shape(
            type="line",
            x0=dates[0],
            y0=avg_price,
            x1=dates[-1],
            y1=avg_price,
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
        )

        # Add annotation for average price
        fig.add_annotation(
            x=dates[int(len(dates)/2)],
            y=avg_price,
            text=f"Avg: ${avg_price:.2f}",
            showarrow=False,
            yshift=10
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,  # Make it taller for better visibility
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_direct_performance_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a performance chart with direct rendering.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data with enhanced error handling
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])

    # Print debug info
    print(f"Creating performance chart for {ticker}")
    print(f"Data points: timestamps={len(timestamps)}, prices={len(prices)}")
    if timestamps and prices:
        print(f"Sample data - timestamp: {timestamps[0]}, price: {prices[0]}")

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
            if isinstance(ts, (int, float)):
                # Handle millisecond timestamps (13 digits)
                if ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                # Handle second timestamps (10 digits)
                else:
                    dates.append(datetime.fromtimestamp(ts))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError) as e:
            # Handle invalid timestamps
            print(f"Error converting timestamp {ts}: {e}")
            continue

    # Process prices with error handling
    processed_prices = []
    for price in prices:
        try:
            processed_prices.append(float(price))
        except (ValueError, TypeError) as e:
            print(f"Error converting price {price}: {e}")
            # Skip invalid prices
            continue

    # Skip performance calculation if we don't have enough processed prices
    if len(processed_prices) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient valid price data for performance chart", showarrow=False, font_size=20)
        return fig

    # Calculate percentage change from first day
    try:
        base_price = processed_prices[0]
        if base_price <= 0:
            raise ValueError("Base price must be positive")

        percent_changes = []
        for price in processed_prices:
            try:
                percent_changes.append((price / base_price - 1) * 100)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"Error calculating percent change: {e}")
                # Handle invalid calculations
                percent_changes.append(None)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        # Return empty chart if calculation fails
        print(f"Performance calculation failed: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Could not calculate performance: {str(e)}", showarrow=False, font_size=20)
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
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified"
    )

    return fig

def create_direct_technical_chart(ticker: str, price_data: Dict[str, Any]) -> go.Figure:
    """
    Create a technical analysis chart with direct rendering.
    This chart includes price data with technical indicators like moving averages.

    Args:
        ticker: Asset ticker symbol
        price_data: Dictionary containing price history data

    Returns:
        Plotly figure object
    """
    # Extract data with enhanced error handling
    timestamps = price_data.get("timestamps", [])
    prices = price_data.get("prices", [])

    # Print debug info
    print(f"Creating technical chart for {ticker}")
    print(f"Data points: timestamps={len(timestamps)}, prices={len(prices)}")
    if timestamps and prices:
        print(f"Sample data - timestamp: {timestamps[0]}, price: {prices[0]}")

    # Validate data
    if not timestamps or not prices or len(timestamps) != len(prices):
        # Return empty chart with message if data is invalid
        fig = go.Figure()
        fig.add_annotation(text="No price data available for technical analysis", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime objects if they're in milliseconds
    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)):
                # Handle millisecond timestamps (13 digits)
                if ts > 1e10:
                    dates.append(datetime.fromtimestamp(ts/1000))
                # Handle second timestamps (10 digits)
                else:
                    dates.append(datetime.fromtimestamp(ts))
            else:
                dates.append(ts)
        except (ValueError, TypeError, OverflowError) as e:
            # Handle invalid timestamps
            print(f"Error converting timestamp {ts}: {e}")
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

    # Calculate and add moving averages
    if len(prices) >= 20:
        # 20-day moving average
        ma20 = [sum(prices[max(0, i-19):i+1])/min(20, i+1) for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma20,
            mode='lines',
            name='20-day MA',
            line=dict(color='#10b981', width=1.5)
        ))

    if len(prices) >= 50:
        # 50-day moving average
        ma50 = [sum(prices[max(0, i-49):i+1])/min(50, i+1) for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma50,
            mode='lines',
            name='50-day MA',
            line=dict(color='#f59e0b', width=1.5)
        ))

    if len(prices) >= 200:
        # 200-day moving average
        ma200 = [sum(prices[max(0, i-199):i+1])/min(200, i+1) for i in range(len(prices))]
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma200,
            mode='lines',
            name='200-day MA',
            line=dict(color='#ef4444', width=1.5)
        ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Technical Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_direct_comparison_chart(tickers: List[str], price_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create a direct comparison chart for multiple assets.

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
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified"
    )

    return fig

def render_chart_directly(fig: go.Figure, use_container_width: bool = True):
    """
    Render a Plotly figure directly in Streamlit with error handling.

    Args:
        fig: Plotly figure object
        use_container_width: Whether to use the full container width
    """
    try:
        # Render the chart directly
        st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
        st.info("Displaying a simplified version instead.")

        # Create a fallback figure
        fallback_fig = go.Figure()
        fallback_fig.add_annotation(text="Chart could not be rendered", showarrow=False, font_size=20)

        # Try to render the fallback figure
        try:
            st.plotly_chart(fallback_fig, use_container_width=use_container_width)
        except:
            # If even the fallback fails, show a simple message
            st.warning("Unable to display any chart. Please try again later.")

@performance_timer(category="charts")
def generate_direct_chart(ticker: str, asset_data: Dict[str, Any], chart_type: str):
    """
    Generate and directly render a chart based on asset data and chart type.

    Args:
        ticker: Asset ticker symbol
        asset_data: Dictionary containing asset data
        chart_type: Type of chart to generate
    """
    # Enhanced debugging for asset data
    log_debug(f"Asset data keys for {ticker}: {asset_data.keys() if isinstance(asset_data, dict) else 'Not a dict'}")

    # Get price history data with better error handling
    price_history = {}
    if isinstance(asset_data, dict):
        # Try different possible locations for price history data
        if "price_history" in asset_data:
            price_history = asset_data["price_history"]
        elif "history" in asset_data:
            price_history = asset_data["history"]

    # Debug price history data
    log_debug(f"Price history keys for {ticker}: {price_history.keys() if isinstance(price_history, dict) else 'Not a dict'}")
    if isinstance(price_history, dict) and "prices" in price_history:
        log_debug(f"Price data length: {len(price_history['prices'])}")
        log_debug(f"First few prices: {price_history['prices'][:5]}")
    if isinstance(price_history, dict) and "timestamps" in price_history:
        log_debug(f"Timestamps length: {len(price_history['timestamps'])}")
        log_debug(f"First few timestamps: {price_history['timestamps'][:5]}")
    if isinstance(price_history, dict) and "open" in price_history:
        log_debug(f"OHLC data available: open={len(price_history['open'])}, high={len(price_history.get('high', []))}, low={len(price_history.get('low', []))}, close={len(price_history.get('close', []))}")

    # If price_history is empty or not a dict, try to create a minimal structure
    if not price_history or not isinstance(price_history, dict):
        st.warning(f"No price history data available for {ticker}. Attempting to fetch data...")
        try:
            # Try to import data service and fetch data directly
            from app.services.data_service import DataService
            data_service = DataService()

            # Determine asset type from context
            asset_type = "stock"  # Default
            if "symbol" in asset_data and asset_data["symbol"].upper() in ["BTC", "ETH", "XRP", "LTC", "BCH"]:
                asset_type = "crypto"
            elif "type" in asset_data:
                asset_type = asset_data["type"]

            # Fetch price history
            price_history = data_service.get_price_history(ticker, asset_type, "1mo")
            print(f"Fetched new price history for {ticker}: {price_history.keys() if isinstance(price_history, dict) else 'Not a dict'}")
        except Exception as fetch_error:
            print(f"Error fetching price history: {str(fetch_error)}")
            # Create a minimal structure
            price_history = {"timestamps": [], "prices": []}

    # Determine asset type for specialized charts with improved detection
    asset_type = "stock"  # Default
    if isinstance(asset_data, dict):
        # Check for explicit type information
        if "type" in asset_data:
            asset_type = asset_data["type"]
        elif "asset_type" in asset_data:
            asset_type = asset_data["asset_type"]
        # Check for property-specific fields to identify REITs
        elif any(key in asset_data for key in ["property_type", "price_to_ffo", "funds_from_operations"]):
            asset_type = "reit"
            log_info(f"Detected REIT based on property-specific fields for {ticker}")
        # Check for sector allocation to identify ETFs
        elif "sector_allocation" in asset_data or "holdings" in asset_data:
            asset_type = "etf"
            log_info(f"Detected ETF based on sector allocation or holdings for {ticker}")
        # Check for crypto-specific fields
        elif any(key in asset_data for key in ["market_cap_rank", "total_volume", "circulating_supply"]):
            asset_type = "crypto"
            log_info(f"Detected cryptocurrency based on crypto-specific fields for {ticker}")
        # Check ticker patterns
        elif ticker.upper() in ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "SOL", "DOGE"]:
            asset_type = "crypto"
        elif ticker.upper() in ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VNQ", "VGT", "XLF", "XLE"]:
            asset_type = "etf"
        elif ticker.upper() in ["O", "AMT", "PLD", "SPG", "WELL", "PSA", "DLR", "AVB", "EQR", "VTR"]:
            asset_type = "reit"
        # Check for REIT-specific naming patterns
        elif ticker.upper().endswith("REIT") or "PROPERTIES" in ticker.upper() or "REALTY" in ticker.upper():
            asset_type = "reit"
            log_info(f"Detected REIT based on naming pattern for {ticker}")

    log_info(f"Detected asset type for {ticker}: {asset_type}")

    # Generate chart based on type
    try:
        if chart_type == "price":
            fig = create_direct_price_chart(ticker, price_history)
        elif chart_type == "volume":
            fig = create_direct_volume_chart(ticker, price_history)
        elif chart_type == "performance":
            fig = create_direct_performance_chart(ticker, price_history)
        elif chart_type == "candlestick":
            fig = create_direct_candlestick_chart(ticker, price_history)
        elif chart_type == "history":
            # Use the enhanced history chart
            fig = create_direct_history_chart(ticker, price_history)
        elif chart_type == "technical":
            # Use the technical analysis chart
            fig = create_direct_technical_chart(ticker, price_history)
        # Specialized chart types
        elif chart_type == "allocation" and asset_type.lower() == "etf":
            log_info(f"Generating ETF sector allocation chart for {ticker}")
            fig = create_etf_sector_allocation_chart(ticker, asset_data)
        elif chart_type == "metrics":
            log_info(f"Generating key metrics chart for {ticker} ({asset_type})")
            fig = create_key_metrics_chart(ticker, asset_data, asset_type)
        elif chart_type == "dividend" and asset_type.lower() in ["stock", "reit"]:
            log_info(f"Generating dividend history chart for {ticker} ({asset_type})")
            fig = create_dividend_history_chart(ticker, asset_data, asset_type)
        elif chart_type == "comparison":
            log_info(f"Generating performance comparison chart for {ticker}")
            fig = create_performance_comparison_chart(ticker, asset_data)
        else:
            # Default to a simple placeholder chart
            fig = go.Figure()
            fig.add_annotation(text=f"Chart type '{chart_type}' not available for {asset_type}", showarrow=False, font_size=20)

        # Render the chart directly
        render_chart_directly(fig)

        # Log success message
        log_success(f"Displayed chart using direct rendering: {chart_type} for {ticker}")

    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
        st.info("Chart could not be generated.")
        # Log detailed error for debugging
        import traceback
        log_error(f"Detailed error generating chart for {ticker}")
        log_exception(
            e,
            context={
                "function": "generate_direct_chart",
                "ticker": ticker,
                "chart_type": chart_type
            }
        )

def generate_direct_comparison(tickers: List[str], price_data: Dict[str, Dict[str, Any]]):
    """
    Generate and directly render a comparison chart for multiple assets.

    Args:
        tickers: List of asset ticker symbols
        price_data: Dictionary mapping tickers to price data dictionaries
    """
    try:
        # Create the comparison chart
        fig = create_direct_comparison_chart(tickers, price_data)

        # Render the chart directly
        render_chart_directly(fig)

    except Exception as e:
        st.error(f"Error generating comparison chart: {str(e)}")
        st.info("Comparison chart could not be generated.")
