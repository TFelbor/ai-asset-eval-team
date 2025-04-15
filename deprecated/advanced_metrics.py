"""
Advanced financial metrics and analytics.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta

class AdvancedAnalytics:
    """Advanced financial analytics and metrics."""

    @staticmethod
    def calculate_volatility(prices: List[float]) -> float:
        """
        Calculate the annualized volatility of a price series.

        Args:
            prices: List of historical prices

        Returns:
            Annualized volatility as a percentage
        """
        if len(prices) < 2:
            return 0.0

        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]

        # Calculate annualized volatility (standard deviation * sqrt(252))
        volatility = np.std(returns) * np.sqrt(252) * 100

        return volatility

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.03) -> float:
        """
        Calculate the Sharpe ratio.

        Args:
            returns: List of historical returns
            risk_free_rate: Annual risk-free rate (default: 3%)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252

        # Calculate excess returns
        excess_returns = np.array(returns) - daily_rf

        # Calculate Sharpe ratio
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        return sharpe

    @staticmethod
    def calculate_beta(stock_returns: List[float], market_returns: List[float]) -> float:
        """
        Calculate the beta of a stock relative to the market.

        Args:
            stock_returns: List of stock returns
            market_returns: List of market returns

        Returns:
            Beta coefficient
        """
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return 1.0

        # Calculate covariance and market variance
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        # Calculate beta
        beta = covariance / market_variance if market_variance > 0 else 1.0

        return beta

    @staticmethod
    def calculate_alpha(stock_returns: List[float], market_returns: List[float],
                       risk_free_rate: float = 0.03, beta: Optional[float] = None) -> float:
        """
        Calculate Jensen's alpha.

        Args:
            stock_returns: List of stock returns
            market_returns: List of market returns
            risk_free_rate: Annual risk-free rate (default: 3%)
            beta: Beta coefficient (if already calculated)

        Returns:
            Alpha value (annualized)
        """
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
            return 0.0

        # Calculate beta if not provided
        if beta is None:
            beta = AdvancedAnalytics.calculate_beta(stock_returns, market_returns)

        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252

        # Calculate average returns
        avg_stock_return = np.mean(stock_returns)
        avg_market_return = np.mean(market_returns)

        # Calculate alpha (annualized)
        alpha = (avg_stock_return - daily_rf) - beta * (avg_market_return - daily_rf)
        alpha_annualized = alpha * 252 * 100  # Convert to percentage and annualize

        return alpha_annualized

    @staticmethod
    def calculate_drawdown(prices: List[float]) -> Tuple[float, float]:
        """
        Calculate the maximum drawdown and current drawdown.

        Args:
            prices: List of historical prices

        Returns:
            Tuple of (maximum drawdown, current drawdown) as percentages
        """
        if len(prices) < 2:
            return 0.0, 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdowns
        drawdowns = (prices - running_max) / running_max * 100

        # Get maximum drawdown and current drawdown
        max_drawdown = np.min(drawdowns)
        current_drawdown = drawdowns[-1]

        return max_drawdown, current_drawdown

    @staticmethod
    def calculate_rsi(prices: List[float], window: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            prices: List of historical prices
            window: RSI window period (default: 14)

        Returns:
            RSI value (0-100)
        """
        if len(prices) <= window:
            return 50.0

        # Calculate price changes
        deltas = np.diff(prices)

        # Calculate gains and losses
        gains = np.copy(deltas)
        losses = np.copy(deltas)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_moving_averages(prices: List[float]) -> Dict[str, float]:
        """
        Calculate various moving averages.

        Args:
            prices: List of historical prices

        Returns:
            Dictionary of moving averages
        """
        result = {}

        # Calculate moving averages if enough data points
        if len(prices) >= 200:
            result["ma50"] = np.mean(prices[-50:])
            result["ma100"] = np.mean(prices[-100:])
            result["ma200"] = np.mean(prices[-200:])
        elif len(prices) >= 100:
            result["ma50"] = np.mean(prices[-50:])
            result["ma100"] = np.mean(prices[-100:])
        elif len(prices) >= 50:
            result["ma50"] = np.mean(prices[-50:])

        return result

    @staticmethod
    def get_advanced_stock_metrics(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get advanced metrics for a stock.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period (default: 1y)

        Returns:
            Dictionary of advanced metrics
        """
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return {"error": "No historical data available"}

            # Get S&P 500 data for market comparison
            spy = yf.Ticker("SPY")
            market_hist = spy.history(period=period)

            # Extract prices and calculate returns
            prices = hist["Close"].values
            returns = hist["Close"].pct_change().dropna().values

            market_returns = market_hist["Close"].pct_change().dropna().values

            # Trim to same length if needed
            min_len = min(len(returns), len(market_returns))
            returns = returns[-min_len:]
            market_returns = market_returns[-min_len:]

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices)
            beta = AdvancedAnalytics.calculate_beta(returns, market_returns)
            alpha = AdvancedAnalytics.calculate_alpha(returns, market_returns, beta=beta)
            sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
            max_drawdown, current_drawdown = AdvancedAnalytics.calculate_drawdown(prices)
            rsi = AdvancedAnalytics.calculate_rsi(prices)
            moving_averages = AdvancedAnalytics.calculate_moving_averages(prices)

            # Get additional data from yfinance
            info = stock.info

            # Compile results
            results = {
                "ticker": ticker,
                "volatility": volatility,
                "beta": beta,
                "alpha": alpha,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
                "rsi": rsi,
                "moving_averages": moving_averages,
                "current_price": prices[-1] if len(prices) > 0 else None,
                "price_to_book": info.get("priceToBook", None),
                "forward_pe": info.get("forwardPE", None),
                "peg_ratio": info.get("pegRatio", None),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "market_cap": info.get("marketCap", None),
                "enterprise_value": info.get("enterpriseValue", None),
                "enterprise_to_revenue": info.get("enterpriseToRevenue", None),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda", None),
                "profit_margins": info.get("profitMargins", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "return_on_equity": info.get("returnOnEquity", None),
                "return_on_assets": info.get("returnOnAssets", None),
                "free_cash_flow": info.get("freeCashflow", None),
                "operating_cash_flow": info.get("operatingCashflow", None),
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "target_mean_price": info.get("targetMeanPrice", None),
                "analyst_rating": info.get("recommendationKey", "N/A")
            }

            return results
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_advanced_crypto_metrics(coin_id: str, vs_currency: str = "usd", days: int = 365) -> Dict[str, Any]:
        """
        Get advanced metrics for a cryptocurrency using real data from CoinGecko API.

        Args:
            coin_id: Cryptocurrency ID (e.g., 'bitcoin') or ticker (e.g., 'BTC')
            vs_currency: Quote currency (default: usd)
            days: Number of days of historical data (default: 365)

        Returns:
            Dictionary of advanced metrics
        """
        try:
            from api_integrations.coingecko import CoinGeckoAPI

            # Initialize CoinGecko API client with API key from config
            from config.settings import COINGECKO_API_KEY
            coingecko = CoinGeckoAPI(api_key=COINGECKO_API_KEY)

            # Convert ticker to coin_id if needed
            coin_id_lower = coin_id.lower()

            # Common ticker mappings for faster lookup
            common_tickers = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'ada': 'cardano',
                'dot': 'polkadot',
                'doge': 'dogecoin',
                'xrp': 'ripple',
                'ltc': 'litecoin',
                'link': 'chainlink',
                'uni': 'uniswap',
                'bnb': 'binancecoin',
                'usdt': 'tether',
                'usdc': 'usd-coin',
                'matic': 'polygon',
            }

            # Check if it's a common ticker first
            if coin_id_lower in common_tickers:
                coin_id = common_tickers[coin_id_lower]
            elif len(coin_id_lower) <= 5:  # Most tickers are short
                # If not in common tickers, do the full lookup
                coin_list = coingecko.get_coin_list()
                if "error" in coin_list and "Rate limit exceeded" in str(coin_list.get("error", "")):
                    return {"error": "Rate limit exceeded. Please try again later."}

                for coin in coin_list:
                    if coin.get('symbol', '').lower() == coin_id_lower:
                        coin_id = coin.get('id')
                        break

            # Get coin data
            coin_data = coingecko.get_coin_data(coin_id)
            if "error" in coin_data:
                return {"error": f"Could not find data for {coin_id}: {coin_data.get('error')}"}

            # Get price history for calculations
            price_history = coingecko.get_coin_price_history(coin_id, vs_currency=vs_currency, days=days)
            if "error" in price_history:
                error_msg = price_history.get('error', '')
                if "Rate limit exceeded" in error_msg:
                    return {"error": "Rate limit exceeded. Please try again later."}
                return {"error": f"Could not fetch price history for {coin_id}: {error_msg}"}

            # Extract price data
            prices_data = price_history.get("prices", [])
            volumes_data = price_history.get("total_volumes", [])

            # Extract just the prices (second element of each timestamp-price pair)
            prices = [price[1] for price in prices_data] if prices_data else []
            volumes = [volume[1] for volume in volumes_data] if volumes_data else []
            timestamps = [price[0] for price in prices_data] if prices_data else []

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices) if len(prices) > 1 else 0
            max_drawdown, current_drawdown = AdvancedAnalytics.calculate_drawdown(prices) if len(prices) > 1 else (0, 0)
            rsi = AdvancedAnalytics.calculate_rsi(prices) if len(prices) > 14 else 50
            moving_averages = AdvancedAnalytics.calculate_moving_averages(prices) if len(prices) > 50 else {}

            # Calculate returns
            daily_returns = []
            if len(prices) > 1:
                daily_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

            weekly_return = np.mean(daily_returns[-7:]) * 100 if len(daily_returns) >= 7 else 0
            monthly_return = np.mean(daily_returns[-30:]) * 100 if len(daily_returns) >= 30 else 0
            yearly_return = np.mean(daily_returns) * 252 * 100 if len(daily_returns) > 0 else 0

            # Get market data
            market_data = coin_data.get("market_data", {})

            # Compile results with real data
            results = {
                "coin_id": coin_id,
                "name": coin_data.get("name", ""),
                "symbol": coin_data.get("symbol", "").upper(),
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
                "rsi": rsi,
                "moving_averages": moving_averages,
                "current_price": market_data.get("current_price", {}).get(vs_currency, 0),
                "daily_return": daily_returns[-1] * 100 if daily_returns else 0,
                "weekly_return": weekly_return,
                "monthly_return": monthly_return,
                "yearly_return": yearly_return,
                "sharpe_ratio": AdvancedAnalytics.calculate_sharpe_ratio(daily_returns) if daily_returns else 0,
                "market_cap_rank": market_data.get("market_cap_rank", 0),
                "market_cap": market_data.get("market_cap", {}).get(vs_currency, 0),
                "fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get(vs_currency, 0),
                "total_volume": market_data.get("total_volume", {}).get(vs_currency, 0),
                "high_24h": market_data.get("high_24h", {}).get(vs_currency, 0),
                "low_24h": market_data.get("low_24h", {}).get(vs_currency, 0),
                "price_change_24h": market_data.get("price_change_24h", 0),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h", 0),
                "market_cap_change_24h": market_data.get("market_cap_change_24h", 0),
                "market_cap_change_percentage_24h": market_data.get("market_cap_change_percentage_24h", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "total_supply": market_data.get("total_supply", 0),
                "max_supply": market_data.get("max_supply", 0),
                "ath": market_data.get("ath", {}).get(vs_currency, 0),
                "ath_change_percentage": market_data.get("ath_change_percentage", {}).get(vs_currency, 0),
                "ath_date": market_data.get("ath_date", {}).get(vs_currency, ""),
                "atl": market_data.get("atl", {}).get(vs_currency, 0),
                "atl_change_percentage": market_data.get("atl_change_percentage", {}).get(vs_currency, 0),
                "atl_date": market_data.get("atl_date", {}).get(vs_currency, ""),
                # Add price and volume history for charts
                "price_history": {
                    "timestamps": timestamps,
                    "prices": prices,
                    "volumes": volumes
                }
            }

            return results
        except Exception as e:
            raise e

    @staticmethod
    def create_price_chart(ticker: str, price_data: Dict[str, Any], chart_type: str = "line") -> go.Figure:
        """
        Create a price chart for an asset.

        Args:
            ticker: Asset ticker symbol
            price_data: Dictionary containing price history data
            chart_type: Chart type (line, candlestick, ohlc)

        Returns:
            Plotly figure object
        """
        # Extract data
        timestamps = price_data.get("timestamps", [])
        prices = price_data.get("prices", [])
        volumes = price_data.get("volumes", [])

        if not timestamps or not prices:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="No price data available", showarrow=False, font_size=20)
            fig.update_layout(title=f"{ticker} - Price Chart")
            return fig

        # Convert timestamps to datetime objects if they are in milliseconds
        if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:  # Likely milliseconds
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
        else:
            dates = timestamps

        # Create figure based on chart type
        if chart_type == "candlestick" and "open" in price_data and "high" in price_data and "low" in price_data and "close" in price_data:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=dates,
                open=price_data["open"],
                high=price_data["high"],
                low=price_data["low"],
                close=price_data["close"],
                name=ticker
            )])
        elif chart_type == "ohlc" and "open" in price_data and "high" in price_data and "low" in price_data and "close" in price_data:
            # Create OHLC chart
            fig = go.Figure(data=[go.Ohlc(
                x=dates,
                open=price_data["open"],
                high=price_data["high"],
                low=price_data["low"],
                close=price_data["close"],
                name=ticker
            )])
        else:
            # Create line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode="lines",
                name=f"{ticker} Price",
                line=dict(color="#2196F3", width=2)
            ))

        # Add volume as bar chart on secondary y-axis if available
        if volumes and len(volumes) == len(dates):
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name="Volume",
                marker_color="rgba(255, 152, 0, 0.3)",
                yaxis="y2"
            ))

            # Update layout with secondary y-axis
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
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark"
        )

        return fig

    @staticmethod
    def create_technical_chart(ticker: str, price_data: Dict[str, Any], indicators: List[str] = ["rsi", "macd"]) -> go.Figure:
        """
        Create a technical analysis chart for an asset.

        Args:
            ticker: Asset ticker symbol
            price_data: Dictionary containing price history data
            indicators: List of technical indicators to include

        Returns:
            Plotly figure object
        """
        # Extract data
        timestamps = price_data.get("timestamps", [])
        prices = price_data.get("prices", [])

        if not timestamps or not prices:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="No price data available", showarrow=False, font_size=20)
            fig.update_layout(title=f"{ticker} - Technical Analysis")
            return fig

        # Convert timestamps to datetime objects if they are in milliseconds
        if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:  # Likely milliseconds
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
        else:
            dates = timestamps

        # Create subplots based on indicators
        if "rsi" in indicators and "macd" in indicators:
            # Create figure with 3 rows (price, RSI, MACD)
            fig = go.Figure()
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.6, 0.2, 0.2],
                              subplot_titles=(f"{ticker} Price", "RSI", "MACD"))
        elif "rsi" in indicators or "macd" in indicators:
            # Create figure with 2 rows (price and one indicator)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.7, 0.3],
                              subplot_titles=(f"{ticker} Price", "RSI" if "rsi" in indicators else "MACD"))
        else:
            # Create figure with just price
            fig = go.Figure()

        # Add price trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name=f"{ticker} Price",
            line=dict(color="#2196F3", width=2)
        ), row=1, col=1)

        # Add Bollinger Bands if requested
        if "bollinger" in indicators:
            # Calculate Bollinger Bands
            window = 20
            rolling_mean = pd.Series(prices).rolling(window=window).mean()
            rolling_std = pd.Series(prices).rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)

            # Add bands to chart
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper_band,
                mode="lines",
                name="Upper Band",
                line=dict(color="rgba(250, 128, 114, 0.7)", width=1, dash="dash")
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_mean,
                mode="lines",
                name="SMA (20)",
                line=dict(color="rgba(255, 255, 255, 0.7)", width=1)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=dates,
                y=lower_band,
                mode="lines",
                name="Lower Band",
                line=dict(color="rgba(173, 216, 230, 0.7)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.1)"
            ), row=1, col=1)

        # Add RSI if requested
        if "rsi" in indicators:
            # Calculate RSI
            delta = pd.Series(prices).diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Add RSI to chart
            row = 2 if "macd" in indicators else 2
            fig.add_trace(go.Scatter(
                x=dates,
                y=rsi,
                mode="lines",
                name="RSI",
                line=dict(color="#FF9800", width=1.5)
            ), row=row, col=1)

            # Add overbought/oversold lines
            fig.add_shape(type="line", x0=dates[0], y0=70, x1=dates[-1], y1=70,
                        line=dict(color="red", width=1, dash="dash"),
                        row=row, col=1)
            fig.add_shape(type="line", x0=dates[0], y0=30, x1=dates[-1], y1=30,
                        line=dict(color="green", width=1, dash="dash"),
                        row=row, col=1)

        # Add MACD if requested
        if "macd" in indicators:
            # Calculate MACD
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            # Add MACD to chart
            row = 3 if "rsi" in indicators else 2
            fig.add_trace(go.Scatter(
                x=dates,
                y=macd_line,
                mode="lines",
                name="MACD",
                line=dict(color="#2196F3", width=1.5)
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=dates,
                y=signal_line,
                mode="lines",
                name="Signal",
                line=dict(color="#FF9800", width=1.5)
            ), row=row, col=1)

            # Add histogram
            colors = ["red" if val < 0 else "green" for val in histogram]
            fig.add_trace(go.Bar(
                x=dates,
                y=histogram,
                name="Histogram",
                marker_color=colors
            ), row=row, col=1)

        # Update layout
        fig.update_layout(
            title=f"{ticker} - Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            height=800 if "rsi" in indicators and "macd" in indicators else 600
        )

        return fig

    @staticmethod
    def create_comparison_chart(tickers: List[str], price_data: Dict[str, Dict[str, Any]], chart_type: str = "normalized") -> go.Figure:
        """
        Create a comparison chart for multiple assets.

        Args:
            tickers: List of asset ticker symbols
            price_data: Dictionary containing price history data for each ticker
            chart_type: Chart type (normalized, absolute, correlation)

        Returns:
            Plotly figure object
        """
        if not tickers or not price_data:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="No comparison data available", showarrow=False, font_size=20)
            fig.update_layout(title="Asset Comparison")
            return fig

        # Create figure
        fig = go.Figure()

        # Process each ticker
        for ticker in tickers:
            if ticker not in price_data:
                continue

            # Extract data
            ticker_data = price_data[ticker]
            timestamps = ticker_data.get("timestamps", [])
            prices = ticker_data.get("prices", [])

            if not timestamps or not prices:
                continue

            # Convert timestamps to datetime objects if they are in milliseconds
            if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1e10:  # Likely milliseconds
                dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
            else:
                dates = timestamps

            if chart_type == "normalized":
                # Normalize prices to start at 100
                base_price = prices[0]
                normalized_prices = [price / base_price * 100 for price in prices]

                # Add trace
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=normalized_prices,
                    mode="lines",
                    name=ticker,
                    line=dict(width=2)
                ))
            else:  # absolute
                # Add trace with absolute prices
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode="lines",
                    name=ticker,
                    line=dict(width=2)
                ))

        # Update layout
        title = "Asset Comparison - Normalized (Base=100)" if chart_type == "normalized" else "Asset Comparison - Absolute Prices"
        y_title = "Normalized Price (Base=100)" if chart_type == "normalized" else "Price"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_title,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark"
        )

        return fig

    @staticmethod
    def create_correlation_matrix(tickers: List[str], price_data: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create a correlation matrix for multiple assets.

        Args:
            tickers: List of asset ticker symbols
            price_data: Dictionary containing price history data for each ticker

        Returns:
            Plotly figure object
        """
        if not tickers or not price_data or len(tickers) < 2:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for correlation matrix", showarrow=False, font_size=20)
            fig.update_layout(title="Correlation Matrix")
            return fig

        # Extract returns for each ticker
        returns_data = {}
        for ticker in tickers:
            if ticker not in price_data:
                continue

            # Extract prices
            prices = price_data[ticker].get("prices", [])

            if len(prices) < 2:
                continue

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            returns_data[ticker] = returns

        # Create correlation matrix
        if len(returns_data) < 2:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for correlation matrix", showarrow=False, font_size=20)
            fig.update_layout(title="Correlation Matrix")
            return fig

        # Create DataFrame with returns
        df = pd.DataFrame(returns_data)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Asset Correlation Matrix"
        )

        # Update layout
        fig.update_layout(
            template="plotly_dark"
        )

        return fig
