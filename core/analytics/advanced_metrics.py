"""
Advanced financial metrics and analytics.
This module provides advanced financial metrics and analytics for different asset types.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta

from core.utils.serialization import make_json_serializable

# Import API clients
from core.api.coingecko_client import CoinGeckoClient
from core.api.yahoo_finance_client import YahooFinanceClient

# Initialize API clients
coingecko_client = CoinGeckoClient()
yahoo_finance_client = YahooFinanceClient()

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

        # Calculate average return
        avg_return = np.mean(returns)

        # Calculate standard deviation of returns
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Calculate daily risk-free rate
        daily_risk_free = risk_free_rate / 252

        # Calculate Sharpe ratio
        sharpe = (avg_return - daily_risk_free) / std_return

        # Annualize Sharpe ratio
        sharpe_annualized = sharpe * np.sqrt(252)

        return sharpe_annualized

    @staticmethod
    def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
        """
        Calculate the beta of an asset.

        Args:
            asset_returns: List of asset returns
            market_returns: List of market returns

        Returns:
            Beta
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0

        # Calculate covariance
        covariance = np.cov(asset_returns, market_returns)[0, 1]

        # Calculate market variance
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 1.0

        # Calculate beta
        beta = covariance / market_variance

        return beta

    @staticmethod
    def calculate_alpha(
        asset_returns: List[float],
        market_returns: List[float],
        risk_free_rate: float = 0.03,
        beta: Optional[float] = None
    ) -> float:
        """
        Calculate the alpha of an asset.

        Args:
            asset_returns: List of asset returns
            market_returns: List of market returns
            risk_free_rate: Annual risk-free rate (default: 3%)
            beta: Beta of the asset (optional, will be calculated if not provided)

        Returns:
            Alpha
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0

        # Calculate beta if not provided
        if beta is None:
            beta = AdvancedAnalytics.calculate_beta(asset_returns, market_returns)

        # Calculate average returns
        avg_asset_return = np.mean(asset_returns)
        avg_market_return = np.mean(market_returns)

        # Calculate daily risk-free rate
        daily_risk_free = risk_free_rate / 252

        # Calculate alpha
        alpha = avg_asset_return - (daily_risk_free + beta * (avg_market_return - daily_risk_free))

        # Annualize alpha
        alpha_annualized = alpha * 252

        return alpha_annualized

    @staticmethod
    def calculate_drawdown(prices: List[float]) -> Tuple[float, int, int]:
        """
        Calculate the maximum drawdown of a price series.

        Args:
            prices: List of historical prices

        Returns:
            Tuple of (maximum drawdown percentage, start index, end index)
        """
        if len(prices) < 2:
            return 0.0, 0, 0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        max_drawdown_idx = np.argmin(drawdown)

        # Find the start of the drawdown period
        start_idx = np.argmax(prices[:max_drawdown_idx])

        return max_drawdown * 100, start_idx, max_drawdown_idx

    @staticmethod
    def get_advanced_stock_metrics(ticker: str) -> Dict[str, Any]:
        """
        Get advanced metrics for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of advanced metrics
        """
        try:
            # Get stock data
            stock_data = yahoo_finance_client.get_stock_info(ticker)

            if "error" in stock_data:
                return {"error": stock_data["error"]}

            # Get historical prices
            historical_data = yahoo_finance_client.get_historical_data(ticker, period="1y")

            if "error" in historical_data:
                return {"error": historical_data["error"]}

            # Extract prices
            prices = historical_data.get("close", [])

            if not prices:
                return {"error": "No historical price data available"}

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Get market data (S&P 500)
            market_data = yahoo_finance_client.get_historical_data("^GSPC", period="1y")
            market_prices = market_data.get("close", [])

            if len(market_prices) != len(prices):
                # Adjust market data to match stock data length
                market_prices = market_prices[-len(prices):]

            market_returns = np.diff(market_prices) / market_prices[:-1]

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices)
            sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
            beta = AdvancedAnalytics.calculate_beta(returns, market_returns)
            alpha = AdvancedAnalytics.calculate_alpha(returns, market_returns, beta=beta)
            max_drawdown, _, _ = AdvancedAnalytics.calculate_drawdown(prices)

            # Return metrics
            return {
                "ticker": ticker,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "beta": beta,
                "alpha": alpha,
                "max_drawdown": max_drawdown,
                "current_price": stock_data.get("current_price", 0),
                "market_cap": stock_data.get("market_cap", 0),
                "pe_ratio": stock_data.get("pe_ratio", 0),
                "dividend_yield": stock_data.get("dividend_yield", 0),
                "52w_high": stock_data.get("52w_high", 0),
                "52w_low": stock_data.get("52w_low", 0)
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {str(e)}"}

    @staticmethod
    def get_advanced_crypto_metrics(coin: str) -> Dict[str, Any]:
        """
        Get advanced metrics for a cryptocurrency.

        Args:
            coin: Cryptocurrency symbol or ID

        Returns:
            Dictionary of advanced metrics
        """
        try:
            # Check if the input is a symbol or ID
            if len(coin) <= 5:  # Likely a symbol
                # Convert symbol to ID
                coin_id = coingecko_client.get_coin_id_from_symbol(coin)
                if not coin_id:
                    return {"error": f"Cryptocurrency with symbol '{coin}' not found"}
            else:
                # Assume it's already an ID
                coin_id = coin

            # Get coin data
            coin_data = coingecko_client.get_coin_by_id(coin_id)

            if "error" in coin_data:
                return {"error": coin_data["error"]}

            # Get historical market data
            market_data = coingecko_client.get_coin_price_history(coin_id, days=365)

            if "error" in market_data:
                return {"error": market_data["error"]}

            # Extract prices
            price_data = market_data.get("prices", [])
            if not price_data:
                return {"error": "No historical price data available"}

            prices = [price[1] for price in price_data]

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Get market data (Bitcoin as market proxy)
            btc_market_data = coingecko_client.get_coin_price_history("bitcoin", days=365)
            btc_price_data = btc_market_data.get("prices", [])

            if len(btc_price_data) != len(price_data):
                # Adjust market data to match coin data length
                btc_price_data = btc_price_data[-len(price_data):]

            btc_prices = [price[1] for price in btc_price_data]
            btc_returns = np.diff(btc_prices) / btc_prices[:-1]

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices)
            sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
            beta = AdvancedAnalytics.calculate_beta(returns, btc_returns)
            alpha = AdvancedAnalytics.calculate_alpha(returns, btc_returns, beta=beta)
            max_drawdown, _, _ = AdvancedAnalytics.calculate_drawdown(prices)

            # Get current price and market data
            current_price = coin_data.get("market_data", {}).get("current_price", {}).get("usd", 0)
            market_cap = coin_data.get("market_data", {}).get("market_cap", {}).get("usd", 0)
            volume_24h = coin_data.get("market_data", {}).get("total_volume", {}).get("usd", 0)
            price_change_24h = coin_data.get("market_data", {}).get("price_change_percentage_24h", 0)

            # Return metrics
            return {
                "coin": coin,
                "coin_id": coin_id,
                "name": coin_data.get("name", ""),
                "symbol": coin_data.get("symbol", "").upper(),
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "beta": beta,
                "alpha": alpha,
                "max_drawdown": max_drawdown,
                "current_price": current_price,
                "market_cap": market_cap,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h,
                "ath": coin_data.get("market_data", {}).get("ath", {}).get("usd", 0),
                "ath_change_percentage": coin_data.get("market_data", {}).get("ath_change_percentage", {}).get("usd", 0)
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {str(e)}"}

    @staticmethod
    def get_advanced_etf_metrics(ticker: str) -> Dict[str, Any]:
        """
        Get advanced metrics for an ETF.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary of advanced metrics
        """
        try:
            # Get ETF data
            etf_data = yahoo_finance_client.get_etf_info(ticker)

            if "error" in etf_data:
                return {"error": etf_data["error"]}

            # Get historical prices
            historical_data = yahoo_finance_client.get_historical_data(ticker, period="1y")

            if "error" in historical_data:
                return {"error": historical_data["error"]}

            # Extract prices
            prices = historical_data.get("close", [])

            if not prices:
                return {"error": "No historical price data available"}

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Get market data (S&P 500)
            market_data = yahoo_finance_client.get_historical_data("^GSPC", period="1y")
            market_prices = market_data.get("close", [])

            if len(market_prices) != len(prices):
                # Adjust market data to match ETF data length
                market_prices = market_prices[-len(prices):]

            market_returns = np.diff(market_prices) / market_prices[:-1]

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices)
            sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
            beta = AdvancedAnalytics.calculate_beta(returns, market_returns)
            alpha = AdvancedAnalytics.calculate_alpha(returns, market_returns, beta=beta)
            max_drawdown, _, _ = AdvancedAnalytics.calculate_drawdown(prices)

            # Return metrics
            return {
                "ticker": ticker,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "beta": beta,
                "alpha": alpha,
                "max_drawdown": max_drawdown,
                "current_price": etf_data.get("current_price", 0),
                "net_assets": etf_data.get("net_assets", 0),
                "expense_ratio": etf_data.get("expense_ratio", 0),
                "dividend_yield": etf_data.get("dividend_yield", 0),
                "ytd_return": etf_data.get("ytd_return", 0),
                "three_year_return": etf_data.get("three_year_return", 0),
                "five_year_return": etf_data.get("five_year_return", 0)
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {str(e)}"}

    @staticmethod
    def get_advanced_reit_metrics(ticker: str) -> Dict[str, Any]:
        """
        Get advanced metrics for a REIT.

        Args:
            ticker: REIT ticker symbol

        Returns:
            Dictionary of advanced metrics
        """
        try:
            # Get REIT data (using stock info as base)
            reit_data = yahoo_finance_client.get_stock_info(ticker)

            if "error" in reit_data:
                return {"error": reit_data["error"]}

            # Get historical prices
            historical_data = yahoo_finance_client.get_historical_data(ticker, period="1y")

            if "error" in historical_data:
                return {"error": historical_data["error"]}

            # Extract prices
            prices = historical_data.get("close", [])

            if not prices:
                return {"error": "No historical price data available"}

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Get market data (FTSE NAREIT All REITs Index or S&P 500 as fallback)
            try:
                market_data = yahoo_finance_client.get_historical_data("^FNAR", period="1y")
            except:
                # Fallback to S&P 500
                market_data = yahoo_finance_client.get_historical_data("^GSPC", period="1y")

            market_prices = market_data.get("close", [])

            if len(market_prices) != len(prices):
                # Adjust market data to match REIT data length
                market_prices = market_prices[-len(prices):]

            market_returns = np.diff(market_prices) / market_prices[:-1]

            # Calculate metrics
            volatility = AdvancedAnalytics.calculate_volatility(prices)
            sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
            beta = AdvancedAnalytics.calculate_beta(returns, market_returns)
            alpha = AdvancedAnalytics.calculate_alpha(returns, market_returns, beta=beta)
            max_drawdown, _, _ = AdvancedAnalytics.calculate_drawdown(prices)

            # REIT-specific metrics (estimated)
            ffo_per_share = reit_data.get("eps", 0) * 1.2  # Estimated FFO
            affo_per_share = ffo_per_share * 0.9  # Estimated AFFO
            price_to_ffo = reit_data.get("current_price", 0) / ffo_per_share if ffo_per_share else 0

            # Return metrics
            return {
                "ticker": ticker,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "beta": beta,
                "alpha": alpha,
                "max_drawdown": max_drawdown,
                "current_price": reit_data.get("current_price", 0),
                "market_cap": reit_data.get("market_cap", 0),
                "dividend_yield": reit_data.get("dividend_yield", 0),
                "ffo_per_share": ffo_per_share,
                "affo_per_share": affo_per_share,
                "price_to_ffo": price_to_ffo,
                "debt_to_equity": reit_data.get("debt_to_equity", 0),
                "property_type": "Unknown"  # Would need additional data source
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {str(e)}"}
