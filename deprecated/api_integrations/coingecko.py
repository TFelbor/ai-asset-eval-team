"""
CoinGecko API wrapper for the AI Finance Dashboard.
This module provides a standardized way to interact with the CoinGecko API.
"""
import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime, timedelta
import time
from app.utils.logger import api_logger, log_api_call
from cache_manager import CacheManager


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
        self.cache = CacheManager(cache_type="coingecko")

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
        # Create a cache key based on endpoint and params
        cache_key = f"{endpoint}:{json.dumps(params) if params else ''}:{use_free_api}"

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            api_logger.debug(f"Using cached data for {endpoint}")
            return cached_data

        # Check rate limits
        now = datetime.now()
        if self.rate_limit_remaining <= 1 and now < self.rate_limit_reset_at:
            wait_time = (self.rate_limit_reset_at - now).total_seconds() + 1
            api_logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
            time.sleep(wait_time)

        # Determine which base URL to use
        if use_free_api or not self.api_key:
            url = f"{self.base_url}/{endpoint}"
            headers = {}
        else:
            url = f"{self.pro_base_url}/{endpoint}"
            headers = {"X-CG-Pro-API-Key": self.api_key}

        # Make the request
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)

            # Update rate limit info from headers
            if "X-RateLimit-Remaining" in response.headers:
                self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in response.headers:
                reset_time = int(response.headers["X-RateLimit-Reset"])
                self.rate_limit_reset_at = datetime.fromtimestamp(reset_time)

            # Handle response
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                self.cache.set(cache_key, data)
                return data
            elif response.status_code == 429:
                # Rate limit exceeded
                api_logger.warning("Rate limit exceeded. Switching to free API.")
                if not use_free_api and self.api_key:
                    # Try again with free API
                    return self._make_request(endpoint, params, use_free_api=True)
                else:
                    # Wait and retry
                    time.sleep(60)
                    return self._make_request(endpoint, params, use_free_api)
            else:
                api_logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
        except Exception as e:
            api_logger.error(f"Request error: {str(e)}")
            return {"error": f"Request error: {str(e)}"}

    def get_coin_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all coins with their ids.

        Returns:
            List of coins with id, symbol, and name
        """
        return self._make_request("coins/list")

    def get_coin_by_id(self, coin_id: str) -> Dict[str, Any]:
        """
        Get current data for a coin by id.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin')

        Returns:
            Coin data
        """
        return self._make_request(f"coins/{coin_id}", {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        })

    def get_coin_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get current data for a coin by symbol.

        Args:
            symbol: Coin symbol (e.g., 'btc')

        Returns:
            Coin data
        """
        # First, get the coin ID from the symbol
        coin_id = self.get_coin_id_from_symbol(symbol)
        if not coin_id:
            return {"error": f"Coin with symbol '{symbol}' not found"}

        # Then get the coin data
        return self.get_coin_by_id(coin_id)

    def get_coin_id_from_symbol(self, symbol: str) -> Optional[str]:
        """
        Get coin ID from symbol.

        Args:
            symbol: Coin symbol (e.g., 'btc')

        Returns:
            Coin ID or None if not found
        """
        # Normalize symbol
        symbol = symbol.lower()

        # Get list of coins
        coins = self.get_coin_list()

        # Find coin by symbol
        for coin in coins:
            if coin.get("symbol", "").lower() == symbol:
                return coin.get("id")

        return None

    def get_coin_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 30) -> Dict[str, Any]:
        """
        Get historical market data for a coin.

        Args:
            coin_id: Coin ID (e.g., 'bitcoin')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of data to retrieve (default: 30)

        Returns:
            Historical market data
        """
        return self._make_request(f"coins/{coin_id}/market_chart", {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily"
        })

    def get_coin_market_chart_by_symbol(self, symbol: str, vs_currency: str = "usd", days: int = 30) -> Dict[str, Any]:
        """
        Get historical market data for a coin by symbol.

        Args:
            symbol: Coin symbol (e.g., 'btc')
            vs_currency: Currency to compare against (default: 'usd')
            days: Number of days of data to retrieve (default: 30)

        Returns:
            Historical market data
        """
        # First, get the coin ID from the symbol
        coin_id = self.get_coin_id_from_symbol(symbol)
        if not coin_id:
            return {"error": f"Coin with symbol '{symbol}' not found"}

        # Then get the market chart
        return self.get_coin_market_chart(coin_id, vs_currency, days)

    def get_global_data(self) -> Dict[str, Any]:
        """
        Get cryptocurrency global data.

        Returns:
            Global cryptocurrency data
        """
        return self._make_request("global")

    def get_trending_coins(self) -> Dict[str, Any]:
        """
        Get trending coins (top-7 trending coins on CoinGecko).

        Returns:
            Trending coins data
        """
        return self._make_request("search/trending")

    def search_coins(self, query: str) -> Dict[str, Any]:
        """
        Search for coins, categories, and markets.

        Args:
            query: Search query

        Returns:
            Search results
        """
        return self._make_request("search", {"query": query})
