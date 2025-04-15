"""
API wrapper classes for the AI Finance Dashboard.
This module provides wrapper classes for various financial APIs.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from typing import Dict, Any, List

from utils.logger import log_api_call, log_error, log_info, log_success, log_warning
from utils.serialization import make_json_serializable

class YahooFinanceAPI:
    """Wrapper class for Yahoo Finance API with caching and error handling."""

    def __init__(self):
        """Initialize the Yahoo Finance API wrapper."""
        # Initialize cache
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = 3600  # Cache duration in seconds (1 hour)

    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get stock data from Yahoo Finance with caching.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock data
        """
        # Normalize ticker to uppercase
        ticker = ticker.upper()

        # Check cache first
        current_time = time.time()
        if ticker in self._cache and self._cache_expiry.get(ticker, 0) > current_time:
            log_info(f"Using cached data for {ticker}")
            return self._cache[ticker]

        try:
            log_info(f"Fetching fresh data for {ticker} from Yahoo Finance")
            log_api_call("Yahoo Finance", f"stock/{ticker}", {"period": "1y"}, success=True)

            # Get stock data from yfinance
            stock = yf.Ticker(ticker)

            # Get basic info with timeout handling
            try:
                info = stock.info
                if not info or len(info) == 0:
                    log_warning(f"No info data returned for {ticker}")
                    info = {"shortName": ticker}
            except Exception as info_error:
                log_error(f"Error fetching info for {ticker}: {str(info_error)}")
                log_api_call("Yahoo Finance", f"stock/{ticker}/info", None, success=False, error=str(info_error))
                info = {"shortName": ticker}

            # Get historical data with timeout handling
            try:
                hist = stock.history(period="1y")
                if hist.empty:
                    log_warning(f"No historical data returned for {ticker}")
                else:
                    log_info(f"Successfully fetched historical data for {ticker}: {hist.shape[0]} rows")
            except Exception as hist_error:
                log_error(f"Error fetching history for {ticker}: {str(hist_error)}")
                log_api_call("Yahoo Finance", f"stock/{ticker}/history", {"period": "1y"}, success=False, error=str(hist_error))
                hist = pd.DataFrame()

            # Calculate additional metrics (following deprecated version)
            if not hist.empty and len(hist) > 0:
                try:
                    price_change_1y = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
                except Exception as calc_error:
                    log_warning(f"Error calculating metrics for {ticker}: {str(calc_error)}")
                    price_change_1y = 0
                    volatility = 0
            else:
                price_change_1y = 0
                volatility = 0

            # Get analyst recommendations
            try:
                recommendations = stock.recommendations
                rec_summary = recommendations.groupby('To Grade').size().to_dict() if not recommendations.empty else {}
            except Exception as rec_error:
                log_warning(f"Error fetching recommendations for {ticker}: {str(rec_error)}")
                rec_summary = {}

            # Compile results (following deprecated version's structure)
            result = {
                "ticker": ticker,
                "company_name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "eps": info.get("trailingEps", 0),
                "beta": info.get("beta", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "price_change_1y": price_change_1y,
                "volatility": volatility,
                "analyst_recommendations": rec_summary,
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "target_price": info.get("targetMeanPrice", 0),
                "target_upside": ((info.get("targetMeanPrice", 0) / info.get("currentPrice", 1)) - 1) * 100 if info.get("currentPrice") else 0,
                "recommendation_key": info.get("recommendationKey", ""),
                # Store the original info for reference
                "info": info,
                # Add historical data for charts - convert to dict before serialization
                "history": hist
            }

            # Make the entire result JSON-serializable
            try:
                serialized_result = make_json_serializable(result)
                log_info(f"Successfully serialized data for {ticker}")
            except Exception as serialize_error:
                log_error(f"Error serializing data for {ticker}: {str(serialize_error)}")
                # Try to serialize without the history data
                result_without_history = {k: v for k, v in result.items() if k != 'history'}
                result_without_history['history'] = {}
                serialized_result = make_json_serializable(result_without_history)
                serialized_result['error'] = f"History serialization error: {str(serialize_error)}"

            # Cache the serialized result
            self._cache[ticker] = serialized_result
            self._cache_expiry[ticker] = current_time + self._cache_duration

            log_success(f"Successfully fetched data for {ticker}")
            return serialized_result
        except Exception as e:
            error_msg = f"Failed to fetch data for {ticker}: {str(e)}"
            log_error(error_msg)
            log_api_call("Yahoo Finance", f"stock/{ticker}", {"period": "1y"}, success=False, error=str(e))
            return {"error": error_msg, "ticker": ticker}

    def get_historical_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get historical price data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Dictionary with historical data (serialized DataFrame)
        """
        # Normalize ticker to uppercase
        ticker = ticker.upper()

        # Check cache first
        cache_key = f"hist_{ticker}_{period}"
        current_time = time.time()
        if cache_key in self._cache and self._cache_expiry.get(cache_key, 0) > current_time:
            log_info(f"Using cached historical data for {ticker} ({period})")
            return self._cache[cache_key]

        try:
            log_info(f"Fetching historical data for {ticker} ({period}) from Yahoo Finance")
            log_api_call("Yahoo Finance", f"stock/{ticker}/history", {"period": period}, success=True)

            # Get stock data from yfinance
            stock = yf.Ticker(ticker)

            # Get historical data
            hist = stock.history(period=period)

            if hist.empty:
                log_warning(f"No historical data available for {ticker} ({period})")
                # Return an empty dictionary with metadata
                empty_result = {
                    "_metadata": {
                        "empty": True,
                        "ticker": ticker,
                        "period": period
                    },
                    "Open": [],
                    "High": [],
                    "Low": [],
                    "Close": [],
                    "Volume": [],
                    "index": []
                }

                # Cache the empty result
                self._cache[cache_key] = empty_result
                self._cache_expiry[cache_key] = current_time + self._cache_duration

                return empty_result

            # Calculate additional metrics (following deprecated version)
            if len(hist) > 0:
                try:
                    # Add some technical indicators
                    # 1. Simple Moving Averages
                    if len(hist) >= 20:
                        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
                    if len(hist) >= 50:
                        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
                    if len(hist) >= 200:
                        hist['SMA200'] = hist['Close'].rolling(window=200).mean()

                    # 2. Relative Strength Index (RSI) - simplified
                    if len(hist) >= 14:
                        delta = hist['Close'].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        hist['RSI'] = 100 - (100 / (1 + rs))
                except Exception as calc_error:
                    log_warning(f"Error calculating technical indicators for {ticker}: {str(calc_error)}")

            # Make the result JSON-serializable
            try:
                serialized_hist = make_json_serializable(hist)

                # Add metadata to help with processing
                serialized_hist["_metadata"] = {
                    "empty": False,
                    "ticker": ticker,
                    "period": period,
                    "row_count": len(serialized_hist.get("index", [])),
                    "serialized": True
                }

                log_info(f"Successfully serialized historical data for {ticker} ({period})")
            except Exception as serialize_error:
                log_error(f"Error serializing historical data for {ticker}: {str(serialize_error)}")
                # Create a minimal serialized version with just the essential data
                try:
                    minimal_hist = {
                        "_metadata": {
                            "empty": False,
                            "ticker": ticker,
                            "period": period,
                            "error": str(serialize_error),
                            "minimal": True
                        },
                        "index": [idx.isoformat() if hasattr(idx, 'isoformat') else str(idx) for idx in hist.index],
                        "Open": hist['Open'].tolist(),
                        "High": hist['High'].tolist(),
                        "Low": hist['Low'].tolist(),
                        "Close": hist['Close'].tolist(),
                        "Volume": hist['Volume'].tolist()
                    }
                    serialized_hist = minimal_hist
                    log_warning(f"Using minimal serialization for {ticker} due to error")
                except Exception as minimal_error:
                    log_error(f"Even minimal serialization failed for {ticker}: {str(minimal_error)}")
                    serialized_hist = {
                        "_metadata": {
                            "empty": True,
                            "ticker": ticker,
                            "period": period,
                            "error": f"Serialization failed: {str(serialize_error)}, Minimal failed: {str(minimal_error)}"
                        },
                        "Open": [],
                        "High": [],
                        "Low": [],
                        "Close": [],
                        "Volume": [],
                        "index": []
                    }

            # Cache the serialized result
            self._cache[cache_key] = serialized_hist
            self._cache_expiry[cache_key] = current_time + self._cache_duration

            log_success(f"Successfully fetched historical data for {ticker} ({period}): {len(serialized_hist.get('index', []))} rows")
            return serialized_hist
        except Exception as e:
            error_msg = f"Error fetching historical data for {ticker}: {str(e)}"
            log_error(error_msg)
            log_api_call("Yahoo Finance", f"stock/{ticker}/history", {"period": period}, success=False, error=str(e))

            # Return an error dictionary with empty data arrays
            error_result = {
                "_metadata": {
                    "empty": True,
                    "ticker": ticker,
                    "period": period,
                    "error": error_msg
                },
                "Open": [],
                "High": [],
                "Low": [],
                "Close": [],
                "Volume": [],
                "index": []
            }
            return error_result

class CoinGeckoAPI:
    """Wrapper class for CoinGecko API with caching and error handling."""

    def __init__(self, api_key: str = None):
        """
        Initialize the CoinGecko API wrapper.

        Args:
            api_key: CoinGecko API key (optional)
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {}

        if api_key:
            self.headers["x-cg-pro-api-key"] = api_key

        # Initialize cache
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = 1800  # Cache duration in seconds (30 minutes)

    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get cryptocurrency data from CoinGecko with caching.

        Args:
            coin_id: Cryptocurrency ID (e.g., bitcoin, ethereum)

        Returns:
            Dictionary with cryptocurrency data
        """
        # Normalize coin_id to lowercase
        coin_id = coin_id.lower()

        # Handle common ticker to ID conversions
        coin_id_map = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "doge": "dogecoin",
            "xrp": "ripple",
            "ada": "cardano",
            "dot": "polkadot",
            "ltc": "litecoin"
        }

        if coin_id in coin_id_map:
            coin_id = coin_id_map[coin_id]

        # Check cache first
        cache_key = f"coin_data_{coin_id}"
        current_time = time.time()
        if cache_key in self._cache and self._cache_expiry.get(cache_key, 0) > current_time:
            log_info(f"Using cached data for {coin_id}")
            return self._cache[cache_key]

        try:
            log_info(f"Fetching fresh data for {coin_id} from CoinGecko")
            # Get coin data
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false"
            }

            log_api_call("CoinGecko", f"coins/{coin_id}", params, success=True)
            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Make the result JSON-serializable
                serialized_result = make_json_serializable(result)

                # Cache the serialized result
                self._cache[cache_key] = serialized_result
                self._cache_expiry[cache_key] = current_time + self._cache_duration

                log_success(f"Successfully fetched data for {coin_id}")
                return serialized_result
            else:
                error_msg = f"API error: {response.status_code}"
                log_error(f"CoinGecko API error for {coin_id}: {error_msg}")
                log_api_call("CoinGecko", f"coins/{coin_id}", params, success=False, error=error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to fetch data for {coin_id}: {str(e)}"
            log_error(error_msg)
            log_api_call("CoinGecko", f"coins/{coin_id}", params, success=False, error=str(e))
            return {"error": error_msg}

    def get_market_chart(self, coin_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get market chart data from CoinGecko with caching.

        Args:
            coin_id: Cryptocurrency ID (e.g., bitcoin, ethereum)
            days: Number of days of data to retrieve

        Returns:
            Dictionary with market chart data
        """
        # Normalize coin_id to lowercase
        coin_id = coin_id.lower()

        # Handle common ticker to ID conversions
        coin_id_map = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "doge": "dogecoin",
            "xrp": "ripple",
            "ada": "cardano",
            "dot": "polkadot",
            "ltc": "litecoin"
        }

        if coin_id in coin_id_map:
            coin_id = coin_id_map[coin_id]

        # Check cache first
        cache_key = f"market_chart_{coin_id}_{days}"
        current_time = time.time()
        if cache_key in self._cache and self._cache_expiry.get(cache_key, 0) > current_time:
            log_info(f"Using cached market chart data for {coin_id} ({days} days)")
            return self._cache[cache_key]

        try:
            log_info(f"Fetching fresh market chart data for {coin_id} ({days} days) from CoinGecko")
            # Get market chart data
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily" if days > 90 else None
            }

            log_api_call("CoinGecko", f"coins/{coin_id}/market_chart", params, success=True)
            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Make the result JSON-serializable
                serialized_result = make_json_serializable(result)

                # Cache the serialized result
                self._cache[cache_key] = serialized_result
                self._cache_expiry[cache_key] = current_time + self._cache_duration

                log_success(f"Successfully fetched market chart data for {coin_id} ({days} days)")
                return serialized_result
            else:
                error_msg = f"API error: {response.status_code}"
                log_error(f"CoinGecko market chart API error for {coin_id}: {error_msg}")
                log_api_call("CoinGecko", f"coins/{coin_id}/market_chart", params, success=False, error=error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to fetch market chart data for {coin_id}: {str(e)}"
            log_error(error_msg)
            log_api_call("CoinGecko", f"coins/{coin_id}/market_chart", params, success=False, error=str(e))
            return {"error": error_msg}

    def get_coin_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all cryptocurrencies from CoinGecko with caching.

        Returns:
            List of dictionaries with cryptocurrency data
        """
        # Check cache first
        cache_key = "coin_list"
        current_time = time.time()
        if cache_key in self._cache and self._cache_expiry.get(cache_key, 0) > current_time:
            log_info("Using cached coin list from CoinGecko")
            return self._cache[cache_key]

        try:
            log_info("Fetching fresh coin list from CoinGecko")
            # Get coin list
            url = f"{self.base_url}/coins/list"

            log_api_call("CoinGecko", "coins/list", {}, success=True)
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Make the result JSON-serializable
                serialized_result = make_json_serializable(result)

                # Cache the serialized result - longer cache time for this rarely changing data
                self._cache[cache_key] = serialized_result
                self._cache_expiry[cache_key] = current_time + (self._cache_duration * 12)  # 6 hours

                log_success(f"Successfully fetched coin list with {len(result)} coins")
                return serialized_result
            else:
                error_msg = f"API error: {response.status_code}"
                log_error(f"CoinGecko coin list API error: {error_msg}")
                log_api_call("CoinGecko", "coins/list", {}, success=False, error=error_msg)
                return []
        except Exception as e:
            error_msg = f"Failed to fetch coin list from CoinGecko: {str(e)}"
            log_error(error_msg)
            log_api_call("CoinGecko", "coins/list", {}, success=False, error=str(e))
            return []
