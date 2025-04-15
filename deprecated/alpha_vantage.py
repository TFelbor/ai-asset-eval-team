"""
AlphaVantage API integration module.
Provides functionality to interact with the AlphaVantage API.
"""

import os
import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
from config.settings import ALPHA_VANTAGE_API_KEY


class AlphaVantageAPI:
    """AlphaVantage API wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AlphaVantage API wrapper.

        Args:
            api_key: AlphaVantage API key (optional, will use config or environment variable if not provided)
        """
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not self.api_key:
            print("Warning: AlphaVantage API key not provided. Set ALPHA_VANTAGE_API_KEY in config or environment variable.")

        self.base_url = "https://www.alphavantage.co/query"

    def _make_request(self, function: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the AlphaVantage API.

        Args:
            function: API function to call
            params: Additional parameters

        Returns:
            API response as dictionary
        """
        try:
            params["function"] = function
            params["apikey"] = self.api_key

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            if "Information" in data and "Please consider optimizing your API call frequency" in data["Information"]:
                return {"error": "API rate limit exceeded", "data": data}

            return data
        except Exception as e:
            return {"error": str(e)}

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Company overview data
        """
        return self._make_request("OVERVIEW", {"symbol": symbol})

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Income statement data
        """
        return self._make_request("INCOME_STATEMENT", {"symbol": symbol})

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Balance sheet data
        """
        return self._make_request("BALANCE_SHEET", {"symbol": symbol})

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Cash flow data
        """
        return self._make_request("CASH_FLOW", {"symbol": symbol})

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Earnings data
        """
        return self._make_request("EARNINGS", {"symbol": symbol})

    def get_time_series_daily(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """
        Get daily time series data.

        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (last 100 data points) or 'full' (up to 20 years of data)

        Returns:
            Daily time series data
        """
        return self._make_request("TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": outputsize})

    def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current quote data
        """
        return self._make_request("GLOBAL_QUOTE", {"symbol": symbol})

    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Get sector performance data.

        Returns:
            Sector performance data
        """
        return self._make_request("SECTOR", {})

    def get_economic_indicator(self, indicator: str) -> Dict[str, Any]:
        """
        Get economic indicator data.

        Args:
            indicator: Economic indicator (e.g., 'REAL_GDP', 'CPI', 'UNEMPLOYMENT')

        Returns:
            Economic indicator data
        """
        return self._make_request(indicator, {})

    def get_technical_indicator(self, symbol: str, indicator: str, interval: str = "daily",
                               time_period: int = 14, series_type: str = "close") -> Dict[str, Any]:
        """
        Get technical indicator data.

        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator (e.g., 'SMA', 'EMA', 'RSI', 'MACD')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')
            time_period: Number of data points used to calculate the indicator
            series_type: Price type ('close', 'open', 'high', 'low')

        Returns:
            Technical indicator data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        }
        return self._make_request(indicator, params)

    def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        Get forex exchange rate.

        Args:
            from_currency: From currency code (e.g., 'USD')
            to_currency: To currency code (e.g., 'JPY')

        Returns:
            Forex exchange rate data
        """
        return self._make_request("CURRENCY_EXCHANGE_RATE", {
            "from_currency": from_currency,
            "to_currency": to_currency
        })
