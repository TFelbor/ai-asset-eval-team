"""
CoinGecko API integration module.
Provides functionality to interact with the CoinGecko API.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime, timedelta
import time
from app.utils.logger import api_logger, log_api_call


class CoinGeckoAPI:
    """CoinGecko API wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CoinGecko API wrapper.

        Args:
            api_key: CoinGecko API key for Pro API (optional, free tier doesn't require a key)
        """
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_base_url = "https://pro-api.coingecko.com/api/v3"
        self.rate_limit_remaining = 50  # Default rate limit for free tier
        self.rate_limit_reset_at = datetime.now()

    @log_api_call(level=logging.INFO)
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, use_free_api: bool = False) -> Dict[str, Any]:
        """
        Make a request to the CoinGecko API with rate limiting and caching.

        Args:
            endpoint: API endpoint
            params: Query parameters
            use_free_api: Force using the free API endpoint

        Returns:
            API response as dictionary
        """
        from app.utils.logger import api_logger
        from cache_manager import CacheManager

        # Create a cache key based on endpoint and params
        cache_key = f"{endpoint}:{json.dumps(params) if params else ''}:{use_free_api}"

        # Initialize cache manager for CoinGecko API
        cache = CacheManager(cache_type="coingecko")

        # Try to get from cache first
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            api_logger.debug(f"Using cached response for {endpoint}")
            return cached_response

        # Check if we need to wait for rate limit reset
        if self.rate_limit_remaining <= 1 and datetime.now() < self.rate_limit_reset_at:
            wait_time = (self.rate_limit_reset_at - datetime.now()).total_seconds()
            if wait_time > 0:
                api_logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                # Instead of sleeping, return a rate limit error
                return {"error": "Rate limit exceeded", "retry_after": int(wait_time)}

        try:
            # Determine URL based on whether API key is provided and use_free_api flag
            if self.api_key and not use_free_api:
                # Use Pro API for endpoints that require authentication
                if endpoint.startswith("coins/") and "/market_chart" in endpoint and params and params.get("days", 0) <= 30:
                    url = f"{self.pro_base_url}/{endpoint}"
                    headers = {"x-cg-pro-api-key": self.api_key}
                    api_logger.info(f"Using Pro API for {endpoint}")
                else:
                    # Use regular API with API key in header for other endpoints
                    url = f"{self.base_url}/{endpoint}"
                    headers = {"x-cg-pro-api-key": self.api_key}
            else:
                # Use free tier
                url = f"{self.base_url}/{endpoint}"
                headers = {}

                # Adjust parameters for free tier limitations
                if endpoint.startswith("coins/") and "/market_chart" in endpoint and params:
                    # Free tier has limitations on historical data
                    if params.get("days", 0) > 30:
                        params["days"] = 30  # Limit to 30 days for free tier
                    if params.get("interval") == "hourly":
                        params["interval"] = "daily"  # Use daily interval for free tier

            api_logger.info(f"Making request to {url} with params {params}")

            response = requests.get(url, params=params, headers=headers)

            # Update rate limit info from headers
            if "x-ratelimit-remaining" in response.headers:
                self.rate_limit_remaining = int(response.headers["x-ratelimit-remaining"])
                api_logger.debug(f"Rate limit remaining: {self.rate_limit_remaining}")
            if "x-ratelimit-reset" in response.headers:
                reset_seconds = int(response.headers["x-ratelimit-reset"])
                self.rate_limit_reset_at = datetime.now() + timedelta(seconds=reset_seconds)
                api_logger.debug(f"Rate limit resets at: {self.rate_limit_reset_at}")

            response.raise_for_status()
            result = response.json()

            # Cache the successful response
            cache.set(cache_key, result)

            return result
        except requests.exceptions.HTTPError as e:
            api_logger.error(f"HTTP error in CoinGecko API request: {str(e)}")

            # If we're using the Pro API and get a 400 or 401 error, try the free API
            if not use_free_api and self.api_key and (e.response.status_code == 400 or e.response.status_code == 401):
                api_logger.warning(f"Pro API error, falling back to free API for {endpoint}")
                return self._make_request(endpoint, params, use_free_api=True)

            if e.response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(e.response.headers.get("retry-after", 60))
                api_logger.warning(f"Rate limit exceeded. Retry after {retry_after} seconds.")
                # Instead of retrying immediately, return a cached response or error
                return {"error": "Rate limit exceeded", "retry_after": retry_after}
            return {"error": str(e), "status_code": e.response.status_code}
        except requests.exceptions.ConnectionError as e:
            api_logger.error(f"Connection error in CoinGecko API request: {str(e)}")
            # If we're using the Pro API, try the free API
            if not use_free_api and self.api_key:
                api_logger.warning(f"Connection error with Pro API, falling back to free API for {endpoint}")
                return self._make_request(endpoint, params, use_free_api=True)
            return {"error": f"Connection error: {str(e)}"}
        except requests.exceptions.Timeout as e:
            api_logger.error(f"Timeout in CoinGecko API request: {str(e)}")
            # If we're using the Pro API, try the free API
            if not use_free_api and self.api_key:
                api_logger.warning(f"Timeout with Pro API, falling back to free API for {endpoint}")
                return self._make_request(endpoint, params, use_free_api=True)
            return {"error": f"Request timed out: {str(e)}"}
        except json.JSONDecodeError as e:
            api_logger.error(f"JSON decode error in CoinGecko API response: {str(e)}")
            # If we're using the Pro API, try the free API
            if not use_free_api and self.api_key:
                api_logger.warning(f"JSON decode error with Pro API, falling back to free API for {endpoint}")
                return self._make_request(endpoint, params, use_free_api=True)
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            api_logger.error(f"Unexpected error in CoinGecko API request: {str(e)}")
            # If we're using the Pro API, try the free API
            if not use_free_api and self.api_key:
                api_logger.warning(f"Unexpected error with Pro API, falling back to free API for {endpoint}")
                return self._make_request(endpoint, params, use_free_api=True)
            return {"error": str(e)}

    def get_coin_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all coins.

        Returns:
            List of coins with id, symbol, and name
        """
        response = self._make_request("coins/list")
        if isinstance(response, list):
            return response
        return []

    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific coin.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin', 'ethereum')

        Returns:
            Comprehensive coin data
        """
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false"
        }
        return self._make_request(f"coins/{coin_id}", params)

    def get_coin_market_data(self, coin_ids: List[str], vs_currency: str = "usd") -> List[Dict[str, Any]]:
        """
        Get market data for specified coins.

        Args:
            coin_ids: List of coin IDs
            vs_currency: Currency to compare against (default: 'usd')

        Returns:
            Market data for specified coins
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d,1y"
        }
        response = self._make_request("coins/markets", params)
        if isinstance(response, list):
            return response
        return []

    def get_coin_price_history(self, coin_id: str, vs_currency: str = "usd", days: Union[int, str] = "max") -> Dict[str, Any]:
        """
        Get historical price data for a coin.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin') or ticker (e.g., 'BTC')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of data to retrieve, or 'max' for all data

        Returns:
            Historical price data
        """
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

        # Check if input is a common ticker
        if coin_id_lower in common_tickers:
            coin_id = common_tickers[coin_id_lower]
            print(f"Converted ticker {coin_id_lower} to coin ID {coin_id}")

        # Set parameters
        params = {
            "vs_currency": vs_currency,
            "days": days
        }

        # Set interval based on days
        if isinstance(days, str) and days == "max":
            params["interval"] = "daily"
        elif isinstance(days, int) and days > 90:
            params["interval"] = "daily"
        else:
            params["interval"] = "hourly"

        print(f"Fetching price history for {coin_id} with params: {params}")

        # Make the request - try free API first to avoid Pro API rate limits
        result = self._make_request(f"coins/{coin_id}/market_chart", params, use_free_api=True)

        # Check for errors
        if isinstance(result, dict) and "error" in result:
            print(f"Error fetching price history for {coin_id}: {result['error']}")
            # Try with a different interval if the error might be related to that
            if "interval" in params and params["interval"] == "hourly":
                print("Retrying with daily interval...")
                params["interval"] = "daily"
                result = self._make_request(f"coins/{coin_id}/market_chart", params, use_free_api=True)

        return result

    def get_global_data(self) -> Dict[str, Any]:
        """
        Get global cryptocurrency market data.

        Returns:
            Global market data
        """
        return self._make_request("global")

    def get_defi_global_data(self) -> Dict[str, Any]:
        """
        Get global DeFi market data.

        Returns:
            Global DeFi market data
        """
        return self._make_request("global/decentralized_finance_defi")

    def get_exchanges(self) -> List[Dict[str, Any]]:
        """
        Get list of exchanges.

        Returns:
            List of exchanges
        """
        response = self._make_request("exchanges")
        if isinstance(response, list):
            return response
        return []

    def get_exchange_data(self, exchange_id: str) -> Dict[str, Any]:
        """
        Get data for a specific exchange.

        Args:
            exchange_id: Exchange ID (e.g., 'binance')

        Returns:
            Exchange data
        """
        return self._make_request(f"exchanges/{exchange_id}")

    def get_trending_coins(self) -> Dict[str, Any]:
        """
        Get trending coins (searched most on CoinGecko in the last 24 hours).

        Returns:
            Trending coins data
        """
        return self._make_request("search/trending")

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search for coins, categories, and markets.

        Args:
            query: Search query

        Returns:
            Search results
        """
        params = {"query": query}
        return self._make_request("search", params)

    def get_coin_ohlc(self, coin_id: str, vs_currency: str = "usd", days: int = 7) -> List[List[float]]:
        """
        Get OHLC (Open, High, Low, Close) data for a coin.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days (1, 7, 14, 30, 90, 180, 365)

        Returns:
            OHLC data as list of [timestamp, open, high, low, close]
        """
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        response = self._make_request(f"coins/{coin_id}/ohlc", params)
        if isinstance(response, list):
            return response
        return []

    def analyze_coin(self, coin_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a coin.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin')

        Returns:
            Comprehensive analysis data
        """
        try:
            # Get basic coin data
            coin_data = self.get_coin_data(coin_id)
            if "error" in coin_data:
                return coin_data

            # Get price history for volatility calculation
            price_history = self.get_coin_price_history(coin_id, days=30)

            # Calculate volatility (standard deviation of daily returns)
            prices = price_history.get("prices", [])
            daily_returns = []

            if len(prices) > 1:
                for i in range(1, len(prices)):
                    daily_return = (prices[i][1] - prices[i-1][1]) / prices[i-1][1]
                    daily_returns.append(daily_return)

                import numpy as np
                volatility = np.std(daily_returns) * np.sqrt(365) * 100  # Annualized volatility
            else:
                volatility = 0

            # Get market data for additional metrics
            market_data = coin_data.get("market_data", {})

            # Calculate market dominance
            global_data = self.get_global_data()
            total_market_cap = global_data.get("data", {}).get("total_market_cap", {}).get("usd", 0)
            market_cap = market_data.get("market_cap", {}).get("usd", 0)
            market_dominance = (market_cap / total_market_cap * 100) if total_market_cap else 0

            # Compile analysis results
            analysis = {
                "id": coin_id,
                "name": coin_data.get("name", ""),
                "symbol": coin_data.get("symbol", "").upper(),
                "current_price": market_data.get("current_price", {}).get("usd", 0),
                "market_cap": market_cap,
                "market_cap_rank": market_data.get("market_cap_rank", 0),
                "market_dominance": market_dominance,
                "trading_volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "max_supply": market_data.get("max_supply", 0),
                "total_supply": market_data.get("total_supply", 0),
                "supply_percentage": (market_data.get("circulating_supply", 0) / market_data.get("max_supply", 1) * 100) if market_data.get("max_supply") else 0,
                "all_time_high": market_data.get("ath", {}).get("usd", 0),
                "all_time_high_date": market_data.get("ath_date", {}).get("usd", ""),
                "all_time_high_change_percentage": market_data.get("ath_change_percentage", {}).get("usd", 0),
                "all_time_low": market_data.get("atl", {}).get("usd", 0),
                "all_time_low_date": market_data.get("atl_date", {}).get("usd", ""),
                "all_time_low_change_percentage": market_data.get("atl_change_percentage", {}).get("usd", 0),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h", 0),
                "price_change_percentage_7d": market_data.get("price_change_percentage_7d", 0),
                "price_change_percentage_30d": market_data.get("price_change_percentage_30d", 0),
                "price_change_percentage_1y": market_data.get("price_change_percentage_1y", 0),
                "volatility_30d": volatility,
                "market_sentiment": "Bullish" if market_data.get("price_change_percentage_7d", 0) > 0 else "Bearish",
                "developer_activity": {
                    "github_stars": coin_data.get("developer_data", {}).get("stars", 0),
                    "github_subscribers": coin_data.get("developer_data", {}).get("subscribers", 0),
                    "github_contributors": coin_data.get("developer_data", {}).get("contributors", 0),
                    "github_commits_4_weeks": coin_data.get("developer_data", {}).get("commit_count_4_weeks", 0),
                },
                "community_data": {
                    "twitter_followers": coin_data.get("community_data", {}).get("twitter_followers", 0),
                    "reddit_subscribers": coin_data.get("community_data", {}).get("reddit_subscribers", 0),
                    "telegram_channel_user_count": coin_data.get("community_data", {}).get("telegram_channel_user_count", 0),
                },
                "liquidity_score": coin_data.get("liquidity_score", 0),
                "public_interest_score": coin_data.get("public_interest_score", 0),
            }

            return analysis
        except Exception as e:
            return {"error": str(e), "coin_id": coin_id}
