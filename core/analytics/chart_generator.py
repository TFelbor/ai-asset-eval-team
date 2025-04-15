"""
Unified chart generation module for the AI Finance Dashboard.
This module provides standardized functions for generating charts across different asset types.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class ChartGenerator:
    """Unified chart generator for all asset types."""

    @staticmethod
    def create_price_chart(
        ticker: str,
        price_data: Dict[str, List],
        chart_type: str = "line"
    ) -> go.Figure:
        """
        Create a price chart for any asset type.

        Args:
            ticker: Asset ticker or symbol
            price_data: Dictionary containing timestamps, prices, and optionally volumes
            chart_type: Chart type (line or candlestick)

        Returns:
            Plotly figure object
        """
        timestamps = price_data.get("timestamps", [])
        prices = price_data.get("prices", [])
        volumes = price_data.get("volumes", [])

        # Convert timestamps to datetime if they're in milliseconds
        if timestamps and isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
        else:
            dates = timestamps

        fig = go.Figure()

        if chart_type == "candlestick" and all(k in price_data for k in ["open", "high", "low", "close"]):
            # Create candlestick chart
            fig.add_trace(go.Candlestick(
                x=dates,
                open=price_data.get("open", []),
                high=price_data.get("high", []),
                low=price_data.get("low", []),
                close=price_data.get("close", []),
                name=ticker
            ))
        else:
            # Create line chart
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=f'{ticker} Price',
                line=dict(color='#4f46e5', width=2)
            ))

        # Add volume as a bar chart if available
        if volumes and len(volumes) > 0:
            # Create a secondary y-axis for volume
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker=dict(color='rgba(128, 128, 128, 0.3)'),
                yaxis='y2'
            ))

            # Update layout for dual y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )

        # Update layout
        fig.update_layout(
            title=f"{ticker} - Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_metrics_chart(
        ticker: str,
        metrics: Dict[str, float],
        normalization_rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> go.Figure:
        """
        Create a radar chart of key metrics for any asset type.

        Args:
            ticker: Asset ticker or symbol
            metrics: Dictionary of metrics and their values
            normalization_rules: Rules for normalizing each metric (optional)

        Returns:
            Plotly figure object
        """
        # Default normalization rules if none provided
        if normalization_rules is None:
            normalization_rules = {}

        # Normalize metrics
        normalized_metrics = {}
        for key, value in metrics.items():
            if key in normalization_rules:
                rule = normalization_rules[key]
                if rule.get("type") == "inverse":
                    # Lower is better (e.g., P/E ratio)
                    target = rule.get("target", 1)
                    max_val = rule.get("max", 10)
                    normalized_metrics[key] = min(1, target / max(0.1, value)) if value > 0 else 0.5
                elif rule.get("type") == "direct":
                    # Higher is better (e.g., dividend yield)
                    max_val = rule.get("max", 1)
                    normalized_metrics[key] = min(1, value / max_val)
                elif rule.get("type") == "target":
                    # Closer to target is better (e.g., beta)
                    target = rule.get("target", 1)
                    range_val = rule.get("range", 1)
                    normalized_metrics[key] = 1 - min(1, abs(value - target) / range_val)
                else:
                    # Default: direct normalization between 0 and 1
                    normalized_metrics[key] = min(1, max(0, value))
            else:
                # Default normalization: assume value is already between 0 and 1
                normalized_metrics[key] = min(1, max(0, value))

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=list(normalized_metrics.values()),
            theta=list(normalized_metrics.keys()),
            fill='toself',
            name=ticker,
            line=dict(color="#4f46e5")
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"{ticker} - Key Metrics",
            height=600,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_performance_chart(
        ticker: str,
        price_data: Dict[str, List],
        benchmark_data: Optional[Dict[str, List]] = None
    ) -> go.Figure:
        """
        Create a performance chart showing percentage change over time.

        Args:
            ticker: Asset ticker or symbol
            price_data: Dictionary containing timestamps and prices
            benchmark_data: Optional benchmark data for comparison

        Returns:
            Plotly figure object
        """
        timestamps = price_data.get("timestamps", [])
        prices = price_data.get("prices", [])

        if not timestamps or not prices or len(timestamps) != len(prices):
            # Create empty chart if data is invalid
            fig = go.Figure()
            fig.add_annotation(text="No valid price data available", showarrow=False, font_size=20)
            return fig

        # Convert timestamps to datetime if they're in milliseconds
        if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
        else:
            dates = timestamps

        # Calculate percentage change from first day
        base_price = prices[0]
        price_percent_change = [(price / base_price - 1) * 100 for price in prices]

        fig = go.Figure()

        # Add asset performance line
        fig.add_trace(go.Scatter(
            x=dates,
            y=price_percent_change,
            mode='lines',
            name=f'{ticker}',
            line=dict(color='#4CAF50', width=2),
            fill='tozeroy'
        ))

        # Add benchmark if provided
        if benchmark_data and "timestamps" in benchmark_data and "prices" in benchmark_data:
            benchmark_timestamps = benchmark_data.get("timestamps", [])
            benchmark_prices = benchmark_data.get("prices", [])

            if benchmark_timestamps and benchmark_prices and len(benchmark_timestamps) == len(benchmark_prices):
                # Convert benchmark timestamps to datetime if needed
                if isinstance(benchmark_timestamps[0], (int, float)) and benchmark_timestamps[0] > 1e10:
                    benchmark_dates = [datetime.fromtimestamp(ts/1000) for ts in benchmark_timestamps]
                else:
                    benchmark_dates = benchmark_timestamps

                # Calculate benchmark percentage change
                benchmark_base_price = benchmark_prices[0]
                benchmark_percent_change = [(price / benchmark_base_price - 1) * 100 for price in benchmark_prices]

                # Add benchmark line
                fig.add_trace(go.Scatter(
                    x=benchmark_dates,
                    y=benchmark_percent_change,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#FF9800', width=2, dash='dash')
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
            height=600,
            hovermode="x unified",
            template="plotly_dark",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_allocation_chart(
        ticker: str,
        allocation_data: Dict[str, float]
    ) -> go.Figure:
        """
        Create a pie chart showing allocation (e.g., sector allocation for ETFs).

        Args:
            ticker: Asset ticker or symbol
            allocation_data: Dictionary of categories and their allocation percentages

        Returns:
            Plotly figure object
        """
        if not allocation_data:
            # Create empty chart if data is invalid
            fig = go.Figure()
            fig.add_annotation(text="No allocation data available", showarrow=False, font_size=20)
            return fig

        # Extract labels and values
        labels = list(allocation_data.keys())
        values = list(allocation_data.values())

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Bold)
        ))

        fig.update_layout(
            title=f"{ticker} - Allocation",
            height=600,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_comparison_chart(
        tickers: List[str],
        price_data: Dict[str, Dict[str, List]],
        chart_type: str = "absolute"
    ) -> go.Figure:
        """
        Create a comparison chart for multiple assets.

        Args:
            tickers: List of asset tickers or symbols
            price_data: Dictionary mapping tickers to price data dictionaries
            chart_type: Chart type (absolute or normalized)

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        for ticker in tickers:
            if ticker not in price_data:
                continue

            ticker_data = price_data[ticker]
            timestamps = ticker_data.get("timestamps", [])
            prices = ticker_data.get("prices", [])

            if not timestamps or not prices or len(timestamps) != len(prices):
                continue

            # Convert timestamps to datetime if they're in milliseconds
            if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:
                dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
            else:
                dates = timestamps

            if chart_type == "normalized":
                # Normalize prices to percentage change from first day
                base_price = prices[0]
                y_values = [(price / base_price - 1) * 100 for price in prices]
                y_axis_title = "% Change"
            else:
                # Use absolute prices
                y_values = prices
                y_axis_title = "Price"

            # Add line for this ticker
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_values,
                mode='lines',
                name=ticker
            ))

        # Update layout
        fig.update_layout(
            title="Asset Comparison",
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            height=600,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @staticmethod
    def create_correlation_matrix(
        tickers: List[str],
        price_data: Dict[str, Dict[str, List]]
    ) -> go.Figure:
        """
        Create a correlation matrix for multiple assets.

        Args:
            tickers: List of asset tickers or symbols
            price_data: Dictionary mapping tickers to price data dictionaries

        Returns:
            Plotly figure object
        """
        # Extract price series for each ticker
        price_series = {}

        for ticker in tickers:
            if ticker not in price_data:
                continue

            ticker_data = price_data[ticker]
            timestamps = ticker_data.get("timestamps", [])
            prices = ticker_data.get("prices", [])

            if not timestamps or not prices or len(timestamps) != len(prices):
                continue

            # Convert timestamps to datetime if they're in milliseconds
            if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:
                dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
            else:
                dates = timestamps

            # Create a DataFrame for this ticker
            ticker_df = pd.DataFrame({
                'date': dates,
                ticker: prices
            })
            ticker_df.set_index('date', inplace=True)

            price_series[ticker] = ticker_df

        if not price_series:
            # Create empty chart if no valid data
            fig = go.Figure()
            fig.add_annotation(text="No valid price data available for correlation", showarrow=False, font_size=20)
            return fig

        # Combine all price series into a single DataFrame
        combined_df = pd.concat(price_series.values(), axis=1)

        # Calculate correlation matrix
        corr_matrix = combined_df.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title="Correlation Matrix",
            height=600,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=80, b=50)
        )

        return fig

    @staticmethod
    def get_chart_insights(
        asset_type: str,
        asset_data: Dict[str, Any],
        chart_type: str
    ) -> List[str]:
        """
        Generate insights for a specific chart type.

        Args:
            asset_type: Type of asset (stock, crypto, reit, etf)
            asset_data: Asset data dictionary
            chart_type: Type of chart

        Returns:
            List of insight strings
        """
        insights = []

        if asset_type == "stock":
            if chart_type == "price":
                current_price = asset_data.get("current_price", "$0")
                intrinsic_value = asset_data.get("intrinsic_value", "N/A")
                dcf_value = asset_data.get("dcf", "$0")
                target_price = asset_data.get("target_price", "N/A")
                upside = asset_data.get("upside_potential", "0%")

                insights = [
                    f"Current price: {current_price}",
                    f"Intrinsic value: {intrinsic_value}",
                    f"DCF value: {dcf_value}",
                    f"Analyst target: {target_price}",
                    f"Upside potential: {upside}",
                    f"The intrinsic value is calculated using DCF, earnings-based valuation, book value, and analyst targets."
                ]
            elif chart_type == "metrics":
                pe = asset_data.get("raw", {}).get("pe", 0)
                pb = asset_data.get("raw", {}).get("pb", 0)
                dividend_yield = asset_data.get("dividend_yield", "0%")
                beta = asset_data.get("raw", {}).get("beta", 0)
                confidence = asset_data.get("confidence", 0)

                insights = [
                    f"P/E ratio: {pe:.2f}",
                    f"P/B ratio: {pb:.2f}",
                    f"Dividend yield: {dividend_yield}",
                    f"Beta: {beta:.2f}",
                    f"Confidence score: {confidence}/100"
                ]

        elif asset_type == "cryptocurrency":
            if chart_type == "price" or chart_type == "price_volume":
                current_price = asset_data.get("current_price", "$0")
                price_change_24h = asset_data.get("price_change_24h", "0%")
                price_change_7d = asset_data.get("price_change_7d", "0%")
                all_time_high = asset_data.get("all_time_high", "$0")

                insights = [
                    f"Current price: {current_price}",
                    f"24h change: {price_change_24h}",
                    f"7d change: {price_change_7d}",
                    f"All-time high: {all_time_high}"
                ]
            elif chart_type == "performance":
                # Calculate overall performance from price history
                price_history = asset_data.get("price_history", {})
                prices = price_history.get("prices", [])

                if prices:
                    start_price = prices[0]
                    end_price = prices[-1]
                    overall_change = ((end_price / start_price) - 1) * 100

                    insights = [
                        f"30-day performance: {overall_change:.2f}%",
                        f"Starting price (30 days ago): ${start_price:,.2f}",
                        f"Current price: ${end_price:,.2f}",
                        f"Volatility: {asset_data.get('volatility', 'Unknown')}"
                    ]
                else:
                    insights = ["No performance data available"]
            elif chart_type == "volume":
                # Calculate volume statistics from price history
                price_history = asset_data.get("price_history", {})
                volumes = price_history.get("volumes", [])

                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
                    max_volume = max(volumes)
                    min_volume = min(volumes)
                    current_volume = volumes[-1]

                    insights = [
                        f"Current 24h volume: ${current_volume:,.0f}",
                        f"Average 30-day volume: ${avg_volume:,.0f}",
                        f"Highest volume: ${max_volume:,.0f}",
                        f"Lowest volume: ${min_volume:,.0f}"
                    ]
                else:
                    insights = ["No volume data available"]

        elif asset_type == "reit":
            if chart_type == "price":
                insights = [
                    f"Property type: {asset_data.get('property_type', 'Commercial')}",
                    f"REITs typically invest in different types of real estate properties, with varying risk and return profiles."
                ]
            elif chart_type == "metrics":
                insights = [
                    f"Dividend yield: {asset_data.get('dividend_yield', '0%')}",
                    f"Price to FFO: {asset_data.get('price_to_ffo', 0):.2f}",
                    f"Debt to equity: {asset_data.get('debt_to_equity', 0):.2f}",
                    f"Beta: {asset_data.get('beta', 0):.2f}"
                ]
            elif chart_type == "dividend":
                insights = [
                    f"Current dividend yield: {asset_data.get('dividend_yield', '0%')}",
                    f"REITs are required to distribute at least 90% of their taxable income to shareholders as dividends.",
                    f"Dividend growth can be an indicator of a REIT's financial health and management's confidence in future cash flows."
                ]

        elif asset_type == "etf":
            if chart_type == "price":
                insights = [
                    f"ETF category: {asset_data.get('category', 'Unknown')}",
                    f"Expense ratio: {asset_data.get('expense_ratio', '0%')}",
                    f"AUM: {asset_data.get('aum', '$0')}"
                ]
            elif chart_type == "allocation":
                sector_allocation = asset_data.get('sector_allocation', {})
                top_sectors = [f'{k}: {v}' for k, v in sector_allocation.items()][:3]

                insights = [
                    f"Top sectors: {', '.join(top_sectors)}",
                    f"Sector allocation shows how the ETF's assets are distributed across different industries.",
                    f"Diversification across sectors can help reduce risk in the portfolio."
                ]
            elif chart_type == "performance":
                insights = [
                    f"YTD return: {asset_data.get('ytd_return', '0%')}",
                    f"1-year return: {asset_data.get('one_year_return', '0%')}",
                    f"3-year return: {asset_data.get('three_year_return', '0%')}",
                    f"Comparing the ETF's performance to its benchmark helps evaluate its management effectiveness."
                ]

        # If no specific insights were generated, return a default message
        if not insights:
            insights = ["No specific insights available for this chart type."]

        return insights
