"""
Advanced analytics module for the AI Finance Dashboard.
This module provides advanced analytics and chart generation functions.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from utils.serialization import make_json_serializable

from core.analytics.advanced_charts import (
    create_candlestick_chart,
    create_technical_chart,
    create_price_volume_chart,
    create_correlation_matrix
)

class AdvancedAnalytics:
    """Advanced analytics class for financial data analysis."""

    @staticmethod
    def create_price_chart(ticker: str, price_data: Dict[str, Any], chart_type: str = "line") -> go.Figure:
        """
        Create a price chart for a financial asset.

        Args:
            ticker: Asset ticker symbol
            price_data: Dictionary with price data
            chart_type: Type of chart to create (line, candlestick)

        Returns:
            Plotly figure object
        """
        if chart_type == "line":
            # Create a line chart
            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(
                x=price_data.get("timestamps", []),
                y=price_data.get("prices", []),
                mode="lines",
                name=ticker,
                line=dict(color="#4f46e5", width=2)
            ))

            # Add volume bars if available
            if "volumes" in price_data and price_data["volumes"]:
                # Add volume bars on secondary y-axis
                fig.add_trace(go.Bar(
                    x=price_data.get("timestamps", []),
                    y=price_data.get("volumes", []),
                    name="Volume",
                    marker_color="rgba(79, 70, 229, 0.3)",
                    yaxis="y2"
                ))

                # Update layout for dual y-axis
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
                title=f"{ticker} - Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                template="plotly_dark",
                hovermode="x unified"
            )

            return fig

        elif chart_type == "candlestick":
            # Create a candlestick chart
            fig = go.Figure()

            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=price_data.get("timestamps", []),
                open=price_data.get("open", []),
                high=price_data.get("high", []),
                low=price_data.get("low", []),
                close=price_data.get("close", []),
                name=ticker,
                increasing_line_color="#10b981",  # Green
                decreasing_line_color="#ef4444"   # Red
            ))

            # Update layout
            fig.update_layout(
                title=f"{ticker} - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                template="plotly_dark",
                xaxis_rangeslider_visible=True,
                hovermode="x unified"
            )

            return fig

        else:
            # Default to a simple placeholder chart
            fig = go.Figure()
            fig.add_annotation(text=f"Chart type '{chart_type}' not supported", showarrow=False, font_size=20)
            return fig

    @staticmethod
    def create_technical_chart(ticker: str, price_data: Dict[str, Any], indicators: List[str] = None) -> go.Figure:
        """
        Create a technical analysis chart with indicators.

        Args:
            ticker: Asset ticker symbol
            price_data: Dictionary with price data
            indicators: List of indicators to include

        Returns:
            Plotly figure object
        """
        # Use the imported function from advanced_charts.py
        return create_technical_chart(ticker, price_data, indicators)

    @staticmethod
    def create_comparison_chart(tickers: List[str], price_data: Dict[str, Dict[str, Any]], mode: str = "absolute") -> go.Figure:
        """
        Create a comparison chart for multiple assets.

        Args:
            tickers: List of asset ticker symbols
            price_data: Dictionary with price data for each ticker
            mode: Comparison mode (absolute, normalized, correlation)

        Returns:
            Plotly figure object
        """
        if mode == "correlation":
            # Create a correlation matrix
            return create_correlation_matrix(tickers, price_data)

        # Create figure
        fig = go.Figure()

        # Process each ticker
        for ticker in tickers:
            if ticker in price_data:
                timestamps = price_data[ticker].get("timestamps", [])
                prices = price_data[ticker].get("prices", [])

                if not timestamps or not prices:
                    continue

                if mode == "normalized" and prices:
                    # Normalize prices to start at 100
                    base_price = prices[0]
                    if base_price > 0:
                        prices = [(price / base_price) * 100 for price in prices]

                # Add line for this ticker
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode="lines",
                    name=ticker,
                    hovertemplate="%{y:.2f}"
                ))

        # Update layout
        title = f"Price Comparison - {', '.join(tickers)}"
        y_axis_title = "Price" if mode == "absolute" else "Normalized Price (Base=100)"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            height=500,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig
