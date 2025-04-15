"""
Alpha Vantage API client for financial data.
This module provides a standardized way to interact with the Alpha Vantage API.
"""
from typing import Dict, Any, List, Optional, Union
from core.api.base_client import BaseAPIClient
from config import settings as config

class AlphaVantageClient(BaseAPIClient):
    """Alpha Vantage API client."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage API client.

        Args:
            api_key: Alpha Vantage API key (optional)
        """
        # Use the API key from config if not provided
        if api_key is None:
            api_key = config.ALPHA_VANTAGE_API_KEY

        # Initialize base client
        super().__init__(
            base_url="https://www.alphavantage.co/query",
            api_key=api_key,
            cache_type="alphavantage",
            timeout=15,
            max_retries=3,
            retry_delay=2
        )

    def _make_request(
        self,
        function: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to the Alpha Vantage API.

        Args:
            function: API function to call
            params: Additional parameters
            use_cache: Whether to use cache

        Returns:
            API response as dictionary
        """
        # Prepare request parameters
        if params is None:
            params = {}

        params["function"] = function

        # Make the request using the base client
        response = super()._make_request(
            endpoint="",
            params=params,
            use_cache=use_cache,
            use_api_key=True
        )

        # Check for Alpha Vantage-specific error messages
        if "Error Message" in response:
            return {"error": response["Error Message"]}

        if "Information" in response and "call frequency" in response["Information"]:
            return {"error": response["Information"], "rate_limited": True}

        return response

    def get_stock_time_series(
        self,
        symbol: str,
        interval: str = "daily",
        outputsize: str = "compact"
    ) -> Dict[str, Any]:
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
        Get company overview.

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
        Get income statement.

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
        Get balance sheet.

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
        Get cash flow.

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
        Get earnings.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing earnings data
        """
        params = {
            "symbol": symbol
        }
        return self._make_request("EARNINGS", params)

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close"
    ) -> Dict[str, Any]:
        """
        Get technical indicator data.

        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator (e.g., 'SMA', 'EMA', 'RSI', 'MACD')
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period: Number of data points used to calculate the indicator
            series_type: Price type (close, open, high, low)

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
            indicator: Economic indicator (e.g., 'REAL_GDP', 'CPI', 'UNEMPLOYMENT')

        Returns:
            Dictionary containing economic indicator data
        """
        return self._make_request(indicator)

    def get_comprehensive_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing comprehensive stock data
        """
        # Get quote data
        quote_data = self.get_stock_quote(symbol)

        # Get company overview
        overview_data = self.get_company_overview(symbol)

        # Get time series data
        time_series_data = self.get_stock_time_series(symbol)

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
