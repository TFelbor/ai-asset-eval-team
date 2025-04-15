"""
Alpha Vantage API integration module.
Provides functionality to interact with the Alpha Vantage API.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime, timedelta
import time
import os
from app.utils.logger import api_logger, log_api_call


class AlphaVantageAPI:
    """Alpha Vantage API wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage API wrapper.

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            api_logger.warning("Alpha Vantage API key not provided. API calls will fail.")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_per_min = 5  # Free tier limit
        self.last_call_time = datetime.now() - timedelta(minutes=1)
        self.calls_this_minute = 0

    @log_api_call(level=logging.INFO)
    def _make_request(self, function: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Alpha Vantage API with rate limiting.

        Args:
            function: API function
            params: Additional query parameters

        Returns:
            API response as dictionary
        """
        # Rate limiting
        now = datetime.now()
        if (now - self.last_call_time).total_seconds() < 60:
            self.calls_this_minute += 1
            if self.calls_this_minute >= self.rate_limit_per_min:
                wait_time = 60 - (now - self.last_call_time).total_seconds()
                api_logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                self.calls_this_minute = 0
                self.last_call_time = datetime.now()
        else:
            self.calls_this_minute = 1
            self.last_call_time = now

        # Prepare request parameters
        request_params = {
            "function": function,
            "apikey": self.api_key
        }
        if params:
            request_params.update(params)

        # Make the request
        try:
            response = requests.get(self.base_url, params=request_params)
            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                api_logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {"error": data["Error Message"]}
            if "Information" in data and "call frequency" in data["Information"]:
                api_logger.warning(f"Alpha Vantage rate limit warning: {data['Information']}")
                return {"error": data["Information"]}

            return data
        except requests.exceptions.RequestException as e:
            api_logger.error(f"Request to Alpha Vantage failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            api_logger.error(f"Failed to parse Alpha Vantage response: {str(e)}")
            return {"error": f"Failed to parse response: {str(e)}"}

    def get_stock_time_series(self, symbol: str, interval: str = "daily", outputsize: str = "compact") -> Dict[str, Any]:
        """
        Get stock time series data.

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (daily, weekly, monthly)
            outputsize: Output size (compact or full)

        Returns:
            Dictionary containing time series data
        """
        function = f"TIME_SERIES_{interval.upper()}"
        params = {
            "symbol": symbol,
            "outputsize": outputsize
        }
        return self._make_request(function, params)

    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time stock quote.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing quote data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("GLOBAL_QUOTE", params)

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing company overview data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("OVERVIEW", params)

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing income statement data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("INCOME_STATEMENT", params)

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing balance sheet data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("BALANCE_SHEET", params)

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing cash flow data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("CASH_FLOW", params)

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing earnings data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("EARNINGS", params)

    def search_ticker(self, keywords: str) -> Dict[str, Any]:
        """
        Search for ticker symbols.

        Args:
            keywords: Search keywords

        Returns:
            Dictionary containing search results
        """
        params = {
            "keywords": keywords
        }
        return self._make_request("SYMBOL_SEARCH", params)

    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Get sector performance data.

        Returns:
            Dictionary containing sector performance data
        """
        return self._make_request("SECTOR")

    def get_economic_indicator(self, indicator: str) -> Dict[str, Any]:
        """
        Get economic indicator data.

        Args:
            indicator: Economic indicator (GDP, INFLATION, etc.)

        Returns:
            Dictionary containing economic indicator data
        """
        return self._make_request(indicator)

    def get_technical_indicator(self, symbol: str, indicator: str, interval: str = "daily", time_period: int = 14, series_type: str = "close") -> Dict[str, Any]:
        """
        Get technical indicator data.

        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator (SMA, EMA, RSI, etc.)
            interval: Time interval (daily, weekly, monthly)
            time_period: Time period
            series_type: Series type (close, open, high, low)

        Returns:
            Dictionary containing technical indicator data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        }
        return self._make_request(indicator, params)

    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing stock data
        """
        # Get quote data
        quote_data = self.get_stock_quote(symbol)
        if "error" in quote_data:
            return quote_data

        # Get company overview
        overview_data = self.get_company_overview(symbol)
        if "error" in overview_data:
            return overview_data

        # Get time series data
        time_series_data = self.get_stock_time_series(symbol)
        if "error" in time_series_data:
            return time_series_data

        # Get technical indicators
        rsi_data = self.get_technical_indicator(symbol, "RSI")
        macd_data = self.get_technical_indicator(symbol, "MACD")
        
        # Combine all data
        result = {
            "symbol": symbol,
            "quote": quote_data.get("Global Quote", {}),
            "overview": overview_data,
            "time_series": time_series_data,
            "technical_indicators": {
                "rsi": rsi_data.get("Technical Analysis: RSI", {}),
                "macd": macd_data.get("Technical Analysis: MACD", {})
            }
        }
        
        return result
