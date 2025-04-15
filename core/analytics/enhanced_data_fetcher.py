"""
Enhanced data fetching implementation for the AI Finance Dashboard.
This module provides a robust approach for fetching price history data from multiple sources.
"""
import yfinance as yf
import pandas as pd
import requests
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

# Import enhanced logging
from core.utils.logger import (
    log_info, log_error, log_success, log_warning,
    log_api_call, log_data_operation, log_exception, performance_timer
)

# Constants for API URLs
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
COINGECKO_URL = "https://api.coingecko.com/api/v3"

# API Keys (will be loaded from environment or config)
ALPHA_VANTAGE_API_KEY = None  # Will be set from config

class EnhancedDataFetcher:
    """Enhanced data fetcher with multiple API sources and fallbacks."""

    def __init__(self, alpha_vantage_api_key: Optional[str] = None):
        """
        Initialize the enhanced data fetcher.

        Args:
            alpha_vantage_api_key: Alpha Vantage API key (optional)
        """
        global ALPHA_VANTAGE_API_KEY

        # Set API key if provided
        if alpha_vantage_api_key:
            ALPHA_VANTAGE_API_KEY = alpha_vantage_api_key

        # Try to load from environment if not provided
        if not ALPHA_VANTAGE_API_KEY:
            try:
                from config import settings
                ALPHA_VANTAGE_API_KEY = settings.ALPHA_VANTAGE_API_KEY
            except (ImportError, AttributeError):
                print("Warning: Alpha Vantage API key not found. Some features may be limited.")

    @performance_timer(category="data")
    def fetch_stock_price_history(self, ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Fetch stock price history with multiple sources and fallbacks.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Dictionary containing price history data
        """
        log_info(f"Fetching stock price history for {ticker} with period {period}")

        # Log API call with enhanced logging
        log_api_call(
            api_name="Yahoo Finance",
            endpoint=f"get_price_history/{ticker}",
            params={"period": period},
            success=True
        )

        # Try Yahoo Finance first (most reliable for recent data)
        yahoo_data = self._fetch_from_yahoo(ticker, period)
        if not self._has_error(yahoo_data) and self._has_sufficient_data(yahoo_data):
            log_success(f"Successfully fetched data from Yahoo Finance for {ticker}")

            # Log successful data operation
            log_data_operation(
                operation="fetch",
                data_type="stock",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "period": period,
                    "data_points": len(yahoo_data.get("timestamps", []))
                },
                success=True
            )
            return yahoo_data

        # If Yahoo Finance failed or has insufficient data, try Alpha Vantage
        if ALPHA_VANTAGE_API_KEY:
            log_info(f"Yahoo Finance data insufficient, trying Alpha Vantage for {ticker}")
            alpha_data = self._fetch_from_alpha_vantage(ticker, period)
            if not self._has_error(alpha_data) and self._has_sufficient_data(alpha_data):
                log_success(f"Successfully fetched data from Alpha Vantage for {ticker}")

                # Log successful data operation
                log_data_operation(
                    operation="fetch",
                    data_type="stock",
                    details={
                        "ticker": ticker,
                        "source": "Alpha Vantage",
                        "period": period,
                        "data_points": len(alpha_data.get("timestamps", []))
                    },
                    success=True
                )
                return alpha_data

        # If both failed, try a longer period from Yahoo Finance
        extended_period = self._get_extended_period(period)
        if extended_period != period:
            log_warning(f"Trying extended period {extended_period} for {ticker}")
            yahoo_extended_data = self._fetch_from_yahoo(ticker, extended_period)
            if not self._has_error(yahoo_extended_data) and self._has_sufficient_data(yahoo_extended_data):
                log_success(f"Successfully fetched extended data from Yahoo Finance for {ticker}")

                # Log successful data operation
                log_data_operation(
                    operation="fetch",
                    data_type="stock",
                    details={
                        "ticker": ticker,
                        "source": "Yahoo Finance (extended)",
                        "period": extended_period,
                        "data_points": len(yahoo_extended_data.get("timestamps", []))
                    },
                    success=True
                )
                return yahoo_extended_data

        # If all attempts failed, log the error and return the original Yahoo data
        if self._has_error(yahoo_data):
            error_msg = yahoo_data.get("error", "Unknown error")
            log_error(f"Failed to fetch price history for {ticker}: {error_msg}")

            # Log failed data operation
            log_data_operation(
                operation="fetch",
                data_type="stock",
                details={
                    "ticker": ticker,
                    "error": error_msg
                },
                success=False,
                error=error_msg
            )
        else:
            log_warning(f"Insufficient data for {ticker}, returning potentially incomplete data")

        return yahoo_data

    @performance_timer(category="data")
    def fetch_crypto_price_history(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Fetch cryptocurrency price history with multiple sources and fallbacks.

        Args:
            ticker: Cryptocurrency ticker symbol (e.g., BTC, ETH)
            days: Number of days of history to fetch

        Returns:
            Dictionary containing price history data
        """
        log_info(f"Fetching crypto price history for {ticker} with days {days}")

        # Log API call with enhanced logging
        log_api_call(
            api_name="CoinGecko",
            endpoint=f"get_price_history/{ticker}",
            params={"days": days},
            success=True
        )

        # Try CoinGecko first (best for crypto)
        coingecko_data = self._fetch_from_coingecko(ticker, days)
        if not self._has_error(coingecko_data) and self._has_sufficient_data(coingecko_data):
            log_success(f"Successfully fetched data from CoinGecko for {ticker}")

            # Log successful data operation
            log_data_operation(
                operation="fetch",
                data_type="crypto",
                details={
                    "ticker": ticker,
                    "source": "CoinGecko",
                    "days": days,
                    "data_points": len(coingecko_data.get("timestamps", []))
                },
                success=True
            )
            return coingecko_data

        # If CoinGecko failed, try Yahoo Finance
        yahoo_ticker = f"{ticker}-USD"
        if ticker.lower() == "btc":
            yahoo_ticker = "BTC-USD"
        elif ticker.lower() == "eth":
            yahoo_ticker = "ETH-USD"

        log_info(f"CoinGecko data insufficient, trying Yahoo Finance for {yahoo_ticker}")
        period = self._days_to_period(days)
        yahoo_data = self._fetch_from_yahoo(yahoo_ticker, period)
        if not self._has_error(yahoo_data) and self._has_sufficient_data(yahoo_data):
            log_success(f"Successfully fetched data from Yahoo Finance for {yahoo_ticker}")

            # Log successful data operation
            log_data_operation(
                operation="fetch",
                data_type="crypto",
                details={
                    "ticker": yahoo_ticker,
                    "source": "Yahoo Finance",
                    "period": period,
                    "data_points": len(yahoo_data.get("timestamps", []))
                },
                success=True
            )
            return yahoo_data

        # If both failed, try a longer period from CoinGecko
        extended_days = days * 2
        if extended_days <= 365:  # Don't go beyond a year
            log_warning(f"Trying extended days {extended_days} for {ticker}")
            coingecko_extended_data = self._fetch_from_coingecko(ticker, extended_days)
            if not self._has_error(coingecko_extended_data) and self._has_sufficient_data(coingecko_extended_data):
                log_success(f"Successfully fetched extended data from CoinGecko for {ticker}")

                # Log successful data operation
                log_data_operation(
                    operation="fetch",
                    data_type="crypto",
                    details={
                        "ticker": ticker,
                        "source": "CoinGecko (extended)",
                        "days": extended_days,
                        "data_points": len(coingecko_extended_data.get("timestamps", []))
                    },
                    success=True
                )
                return coingecko_extended_data

        # If all attempts failed, log the error and return the original CoinGecko data
        if self._has_error(coingecko_data):
            error_msg = coingecko_data.get("error", "Unknown error")
            log_error(f"Failed to fetch price history for {ticker}: {error_msg}")

            # Log failed data operation
            log_data_operation(
                operation="fetch",
                data_type="crypto",
                details={
                    "ticker": ticker,
                    "error": error_msg
                },
                success=False,
                error=error_msg
            )
        else:
            log_warning(f"Insufficient data for {ticker}, returning potentially incomplete data")

        return coingecko_data

    @performance_timer(category="data")
    def fetch_price_history(self, ticker: str, asset_type: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Fetch price history for any asset type with appropriate source selection.

        Args:
            ticker: Asset ticker symbol
            asset_type: Asset type (stock, crypto, reit, etf)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Dictionary containing price history data
        """
        try:
            # Log the operation
            log_info(f"Fetching price history for {ticker} ({asset_type}) with period {period}")

            # Normalize asset type
            asset_type = asset_type.lower()

            # For crypto, use the crypto-specific function
            if asset_type in ["crypto", "cryptocurrency"]:
                # Convert period to days
                days = self._period_to_days(period)
                return self.fetch_crypto_price_history(ticker, days)

            # For stocks, REITs, and ETFs, use the stock function
            elif asset_type in ["stock", "reit", "etf"]:
                return self.fetch_stock_price_history(ticker, period)

            # Unsupported asset type
            else:
                return {"error": f"Unsupported asset type: {asset_type}"}

        except Exception as e:
            print(f"Error in fetch_price_history for {ticker} ({asset_type}): {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to fetch price history: {str(e)}"}

    def _fetch_from_yahoo(self, ticker: str, period: str) -> Dict[str, Any]:
        """
        Fetch price history from Yahoo Finance.

        Args:
            ticker: Asset ticker symbol
            period: Time period

        Returns:
            Dictionary containing price history data
        """
        try:
            # Fetch data from Yahoo Finance
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)

            if hist.empty:
                return {"error": f"No historical data available for {ticker} from Yahoo Finance"}

            # Convert to price history format
            price_history = {
                "timestamps": hist.index.tolist(),
                "prices": hist["Close"].tolist(),
                "volumes": hist["Volume"].tolist(),
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist()
            }

            # Print debug info
            print(f"Yahoo Finance data for {ticker}: {len(price_history['timestamps'])} data points")

            return price_history

        except Exception as e:
            print(f"Error fetching from Yahoo Finance for {ticker}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to fetch from Yahoo Finance: {str(e)}"}

    def _fetch_from_alpha_vantage(self, ticker: str, period: str) -> Dict[str, Any]:
        """
        Fetch price history from Alpha Vantage.

        Args:
            ticker: Asset ticker symbol
            period: Time period

        Returns:
            Dictionary containing price history data
        """
        try:
            if not ALPHA_VANTAGE_API_KEY:
                return {"error": "Alpha Vantage API key not set"}

            # Convert period to Alpha Vantage parameters
            function, outputsize = self._period_to_alpha_vantage_params(period)

            # Prepare request parameters
            params = {
                "function": function,
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": ALPHA_VANTAGE_API_KEY
            }

            # Make the request
            response = requests.get(ALPHA_VANTAGE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                return {"error": data["Error Message"]}

            if "Information" in data and "call frequency" in data["Information"]:
                return {"error": data["Information"]}

            # Parse the Alpha Vantage time series data
            price_history = self._parse_alpha_vantage_data(data, function)

            # Log successful data operation
            data_points = len(price_history.get('timestamps', []))
            log_success(f"Alpha Vantage data for {ticker}: {data_points} data points")

            log_data_operation(
                operation="fetch",
                data_type="stock",
                details={
                    "ticker": ticker,
                    "source": "Alpha Vantage",
                    "function": function,
                    "data_points": data_points
                },
                success=True
            )

            return price_history

        except Exception as e:
            # Log the error with enhanced logging
            log_error(f"Error fetching from Alpha Vantage for {ticker}: {str(e)}")

            # Log the exception with detailed context
            log_exception(
                e,
                context={
                    "function": "_fetch_from_alpha_vantage",
                    "ticker": ticker,
                    "api": "Alpha Vantage"
                }
            )

            return {"error": f"Failed to fetch from Alpha Vantage: {str(e)}"}

    def _fetch_from_coingecko(self, ticker: str, days: int) -> Dict[str, Any]:
        """
        Fetch price history from CoinGecko.

        Args:
            ticker: Cryptocurrency ticker symbol
            days: Number of days of history to fetch

        Returns:
            Dictionary containing price history data
        """
        try:
            # Convert ticker to CoinGecko ID
            coin_id = self._convert_ticker_to_coingecko_id(ticker)

            # Prepare request parameters
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily" if days > 90 else None
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Make the request
            url = f"{COINGECKO_URL}/coins/{coin_id}/market_chart"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse the CoinGecko data
            price_history = self._parse_coingecko_data(data)

            # Print debug info
            print(f"CoinGecko data for {ticker} ({coin_id}): {len(price_history.get('timestamps', []))} data points")

            return price_history

        except Exception as e:
            print(f"Error fetching from CoinGecko for {ticker}: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to fetch from CoinGecko: {str(e)}"}

    def _parse_alpha_vantage_data(self, data: Dict[str, Any], function: str) -> Dict[str, Any]:
        """
        Parse Alpha Vantage time series data.

        Args:
            data: Alpha Vantage API response
            function: Alpha Vantage function used

        Returns:
            Dictionary containing price history data
        """
        try:
            # Determine the time series key based on the function
            if function == "TIME_SERIES_DAILY":
                time_series_key = "Time Series (Daily)"
            elif function == "TIME_SERIES_WEEKLY":
                time_series_key = "Weekly Time Series"
            elif function == "TIME_SERIES_MONTHLY":
                time_series_key = "Monthly Time Series"
            else:
                return {"error": f"Unsupported function: {function}"}

            # Get the time series data
            time_series = data.get(time_series_key, {})

            if not time_series:
                return {"error": "No time series data found"}

            # Extract timestamps and prices
            timestamps = []
            prices = []
            volumes = []
            opens = []
            highs = []
            lows = []
            closes = []

            for date_str, values in time_series.items():
                try:
                    # Convert date string to timestamp
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    timestamp = date.timestamp() * 1000  # Convert to milliseconds

                    # Extract values
                    open_price = float(values.get("1. open", 0))
                    high_price = float(values.get("2. high", 0))
                    low_price = float(values.get("3. low", 0))
                    close_price = float(values.get("4. close", 0))
                    volume = float(values.get("5. volume", 0))

                    # Append to lists
                    timestamps.append(timestamp)
                    prices.append(close_price)
                    volumes.append(volume)
                    opens.append(open_price)
                    highs.append(high_price)
                    lows.append(low_price)
                    closes.append(close_price)

                except (ValueError, TypeError) as e:
                    print(f"Error parsing Alpha Vantage data for date {date_str}: {str(e)}")
                    continue

            # Sort by timestamp (oldest first)
            sorted_data = sorted(zip(timestamps, prices, volumes, opens, highs, lows, closes))

            if not sorted_data:
                return {"error": "No valid data points found"}

            # Unzip the sorted data
            timestamps, prices, volumes, opens, highs, lows, closes = zip(*sorted_data)

            # Create the price history dictionary
            price_history = {
                "timestamps": list(timestamps),
                "prices": list(prices),
                "volumes": list(volumes),
                "open": list(opens),
                "high": list(highs),
                "low": list(lows),
                "close": list(closes)
            }

            return price_history

        except Exception as e:
            print(f"Error parsing Alpha Vantage data: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to parse Alpha Vantage data: {str(e)}"}

    def _parse_coingecko_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse CoinGecko market chart data.

        Args:
            data: CoinGecko API response

        Returns:
            Dictionary containing price history data
        """
        try:
            # Check if required data is present
            if "prices" not in data:
                return {"error": "No price data found in CoinGecko response"}

            # Initialize lists for processed data
            timestamps = []
            prices = []
            volumes = []

            # Process timestamps and prices
            for entry in data.get("prices", []):
                if len(entry) >= 2:
                    try:
                        timestamps.append(float(entry[0]))
                        prices.append(float(entry[1]))
                    except (ValueError, TypeError) as e:
                        print(f"Error converting price data: {e}, value: {entry}")
                        continue

            # Process volumes separately
            for entry in data.get("total_volumes", []):
                if len(entry) >= 2:
                    try:
                        volumes.append(float(entry[1]))
                    except (ValueError, TypeError) as e:
                        print(f"Error converting volume data: {e}, value: {entry}")
                        volumes.append(0)  # Use 0 as fallback

            # Ensure volumes list is the same length as prices list
            while len(volumes) < len(prices):
                volumes.append(0)

            # Create synthetic OHLC data for candlestick charts
            # This is a simple approximation since CoinGecko doesn't provide OHLC data directly
            # Initialize with the same length as prices to avoid array length mismatch errors
            n = len(prices)
            opens = [0] * n
            highs = [0] * n
            lows = [0] * n
            closes = [0] * n

            # Use a simple algorithm to generate synthetic OHLC data
            for i in range(n):
                if i > 0:
                    # Open is the previous close
                    opens[i] = prices[i-1]
                    # Close is the current price
                    closes[i] = prices[i]
                    # High is the max of open and close plus a small random factor
                    highs[i] = max(prices[i-1], prices[i]) * 1.005
                    # Low is the min of open and close minus a small random factor
                    lows[i] = min(prices[i-1], prices[i]) * 0.995
                else:
                    # For the first point, use the same value for all
                    opens[i] = prices[i]
                    closes[i] = prices[i]
                    highs[i] = prices[i] * 1.005
                    lows[i] = prices[i] * 0.995

            # Create the price history dictionary
            price_history = {
                "timestamps": timestamps,
                "prices": prices,
                "volumes": volumes[:len(timestamps)],  # Ensure volumes matches timestamps length
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes
            }

            return price_history

        except Exception as e:
            print(f"Error parsing CoinGecko data: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to parse CoinGecko data: {str(e)}"}

    def _convert_ticker_to_coingecko_id(self, ticker: str) -> str:
        """
        Convert common cryptocurrency ticker symbols to CoinGecko IDs.

        Args:
            ticker: Cryptocurrency ticker symbol

        Returns:
            CoinGecko coin ID
        """
        # Common mappings
        mapping = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "usdt": "tether",
            "usdc": "usd-coin",
            "bnb": "binancecoin",
            "xrp": "ripple",
            "ada": "cardano",
            "doge": "dogecoin",
            "sol": "solana",
            "dot": "polkadot",
            "shib": "shiba-inu",
            "matic": "matic-network",
            "avax": "avalanche-2",
            "link": "chainlink",
            "uni": "uniswap",
            "ltc": "litecoin",
            "atom": "cosmos",
            "etc": "ethereum-classic",
            "xlm": "stellar",
            "algo": "algorand"
        }

        # Normalize ticker
        ticker_lower = ticker.lower()

        # Return mapped ID if available, otherwise use the ticker as is
        return mapping.get(ticker_lower, ticker_lower)

    def _period_to_alpha_vantage_params(self, period: str) -> Tuple[str, str]:
        """
        Convert period to Alpha Vantage function and outputsize.

        Args:
            period: Time period

        Returns:
            Tuple of (function, outputsize)
        """
        # Default values
        function = "TIME_SERIES_DAILY"
        outputsize = "compact"  # compact = 100 data points, full = 20+ years

        # Map period to function and outputsize
        if period in ["1d", "5d", "1mo"]:
            function = "TIME_SERIES_DAILY"
            outputsize = "compact"
        elif period in ["3mo", "6mo", "1y"]:
            function = "TIME_SERIES_DAILY"
            outputsize = "full"
        elif period in ["2y", "5y"]:
            function = "TIME_SERIES_WEEKLY"
            outputsize = "full"
        elif period in ["10y", "ytd", "max"]:
            function = "TIME_SERIES_MONTHLY"
            outputsize = "full"

        return function, outputsize

    def _period_to_days(self, period: str) -> int:
        """
        Convert period string to number of days.

        Args:
            period: Time period

        Returns:
            Number of days
        """
        # Map period to days
        if period == "1d":
            return 1
        elif period == "5d":
            return 5
        elif period == "1mo":
            return 30
        elif period == "3mo":
            return 90
        elif period == "6mo":
            return 180
        elif period == "1y":
            return 365
        elif period == "2y":
            return 730
        elif period == "5y":
            return 1825
        elif period == "10y":
            return 3650
        elif period == "ytd":
            # Calculate days from start of year to now
            now = datetime.now()
            start_of_year = datetime(now.year, 1, 1)
            return (now - start_of_year).days
        elif period == "max":
            return 3650  # 10 years as a reasonable maximum
        else:
            return 30  # Default to 1 month

    def _days_to_period(self, days: int) -> str:
        """
        Convert number of days to period string.

        Args:
            days: Number of days

        Returns:
            Period string
        """
        # Map days to period
        if days <= 1:
            return "1d"
        elif days <= 5:
            return "5d"
        elif days <= 30:
            return "1mo"
        elif days <= 90:
            return "3mo"
        elif days <= 180:
            return "6mo"
        elif days <= 365:
            return "1y"
        elif days <= 730:
            return "2y"
        elif days <= 1825:
            return "5y"
        else:
            return "max"

    def _get_extended_period(self, period: str) -> str:
        """
        Get an extended period for fallback.

        Args:
            period: Original time period

        Returns:
            Extended time period
        """
        # Map period to extended period
        period_map = {
            "1d": "5d",
            "5d": "1mo",
            "1mo": "3mo",
            "3mo": "6mo",
            "6mo": "1y",
            "1y": "2y",
            "2y": "5y",
            "5y": "10y",
            "10y": "max",
            "ytd": "1y",
            "max": "max"
        }

        return period_map.get(period, period)

    def _has_error(self, data: Dict[str, Any]) -> bool:
        """
        Check if data contains an error.

        Args:
            data: Data dictionary

        Returns:
            True if data contains an error, False otherwise
        """
        return "error" in data

    def _has_sufficient_data(self, data: Dict[str, Any], min_points: int = 10) -> bool:
        """
        Check if data has sufficient data points.

        Args:
            data: Data dictionary
            min_points: Minimum number of data points required

        Returns:
            True if data has sufficient data points, False otherwise
        """
        return (
            "timestamps" in data and
            "prices" in data and
            len(data["timestamps"]) >= min_points and
            len(data["prices"]) >= min_points
        )


# Create a singleton instance
enhanced_data_fetcher = EnhancedDataFetcher()

# Convenience functions
def fetch_price_history(ticker: str, asset_type: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Fetch price history for any asset type.

    Args:
        ticker: Asset ticker symbol
        asset_type: Asset type (stock, crypto, reit, etf)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    Returns:
        Dictionary containing price history data
    """
    return enhanced_data_fetcher.fetch_price_history(ticker, asset_type, period)

def fetch_stock_price_history(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Fetch stock price history.

    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    Returns:
        Dictionary containing price history data
    """
    return enhanced_data_fetcher.fetch_stock_price_history(ticker, period)

def fetch_crypto_price_history(ticker: str, days: int = 30) -> Dict[str, Any]:
    """
    Fetch cryptocurrency price history.

    Args:
        ticker: Cryptocurrency ticker symbol
        days: Number of days of history to fetch

    Returns:
        Dictionary containing price history data
    """
    return enhanced_data_fetcher.fetch_crypto_price_history(ticker, days)
