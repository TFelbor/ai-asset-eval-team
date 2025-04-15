"""
Yahoo Finance API client for stock, ETF, and REIT data.
This module provides a standardized way to interact with Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Any
from core.api.base_client import BaseAPIClient

# Import enhanced logging
from core.utils.logger import (
    log_info, log_error, log_success, log_warning, log_debug,
    log_api_call, log_data_operation, log_exception, PerformanceTimer
)

class YahooFinanceClient(BaseAPIClient):
    """Yahoo Finance API client."""

    def __init__(self):
        """Initialize the Yahoo Finance API client."""
        # Initialize base client with a dummy base URL since yfinance doesn't use REST API
        super().__init__(
            base_url="",
            cache_type="yahoo_finance",
            timeout=15,
            max_retries=3,
            retry_delay=2
        )

    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing stock data
        """
        # Create cache key
        cache_key = f"stock_data_{ticker}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get stock data using yfinance
            log_info(f"Fetching stock data for {ticker} from Yahoo Finance")
            with PerformanceTimer(f"yahoo_finance_fetch_{ticker}"):
                stock = yf.Ticker(ticker)
                info = stock.info

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_stock_data/{ticker}",
                success=True,
                response=f"Retrieved data for {ticker}"
            )

            # Check if we got valid data
            if not info or "symbol" not in info:
                error_msg = f"Could not find stock with ticker {ticker}"
                log_error(error_msg)
                return {"error": error_msg, "ticker": ticker}

            # Get historical data
            hist = stock.history(period="1mo")

            # Convert historical data to serializable format
            history_data = {}
            if not hist.empty:
                # Import the serialization utility
                from core.utils.serialization import make_json_serializable

                # Use the serialization utility to convert the DataFrame
                history_data = make_json_serializable(hist)

            # Calculate additional metrics
            market_cap = info.get("marketCap", 0)
            pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))
            pb_ratio = info.get("priceToBook", 0)
            dividend_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
            beta = info.get("beta", 0)

            # Format the data
            result = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "market_cap": self._format_market_cap(market_cap),
                "pe": pe_ratio,
                "pb": pb_ratio,
                "dividend_yield": f"{dividend_yield:.2f}%",
                "beta": beta,
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "history": history_data,
                "raw": info
            }

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched data from Yahoo Finance for {ticker}")
            log_data_operation(
                operation="process",
                data_type="stock",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "data_points": len(hist) if isinstance(hist, pd.DataFrame) else 0
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "source": "yahoo_finance", "method": "get_stock_data"})
            return {"error": str(e), "ticker": ticker}

    def get_etf_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive ETF data from Yahoo Finance.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary containing ETF data
        """
        # Create cache key
        cache_key = f"etf_data_{ticker}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get ETF data using yfinance
            log_info(f"Fetching ETF data for {ticker} from Yahoo Finance")
            with PerformanceTimer(f"yahoo_finance_fetch_etf_{ticker}"):
                etf = yf.Ticker(ticker)
                info = etf.info

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_etf_data/{ticker}",
                success=True,
                response=f"Retrieved ETF data for {ticker}"
            )

            # Check if we got valid data
            if not info or "symbol" not in info:
                return {"error": f"Could not find ETF with ticker {ticker}", "ticker": ticker}

            # Get historical data
            hist = etf.history(period="1mo")

            # Convert historical data to serializable format
            history_data = {}
            if not hist.empty:
                # Import the serialization utility
                from core.utils.serialization import make_json_serializable

                # Use the serialization utility to convert the DataFrame
                history_data = make_json_serializable(hist)

            # Calculate additional metrics
            aum = info.get("totalAssets", 0)
            expense_ratio = info.get("annualReportExpenseRatio", 0) * 100 if info.get("annualReportExpenseRatio") else 0
            ytd_return = info.get("ytdReturn", 0) * 100 if info.get("ytdReturn") else 0
            three_year_return = info.get("threeYearAverageReturn", 0) * 100 if info.get("threeYearAverageReturn") else 0

            # Get top holdings
            holdings = []
            for i in range(1, 11):
                holding_key = f"holdings{i}Symbol"
                if holding_key in info and info[holding_key]:
                    holdings.append(info[holding_key])

            # Get sector allocation
            sector_allocation = {}
            for key, value in info.items():
                if key.startswith("sector") and key != "sectorWeightings" and isinstance(value, (int, float)) and value > 0:
                    sector_name = key.replace("sector", "")
                    sector_allocation[sector_name] = value * 100

            # Format the data
            result = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "category": info.get("category", ""),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "aum": self._format_market_cap(aum),
                "expense_ratio": f"{expense_ratio:.2f}%",
                "expense_ratio_value": expense_ratio / 100,
                "ytd_return": f"{ytd_return:.2f}%",
                "ytd_return_value": ytd_return,
                "three_year_return": f"{three_year_return:.2f}%",
                "three_year_return_value": three_year_return,
                "top_holdings": holdings,
                "sector_allocation": sector_allocation,
                "history": history_data,
                "raw": info
            }

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched ETF data for {ticker}")

            # Log data operation
            log_data_operation(
                operation="process",
                data_type="etf",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "data_fields": list(result.keys())
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "source": "yahoo_finance", "method": "get_etf_data"})
            return {"error": str(e), "ticker": ticker}

    def get_reit_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive REIT data from Yahoo Finance.

        Args:
            ticker: REIT ticker symbol

        Returns:
            Dictionary containing REIT data
        """
        # Create cache key
        cache_key = f"reit_data_{ticker}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get REIT data using yfinance
            log_info(f"Fetching REIT data for {ticker} from Yahoo Finance")
            with PerformanceTimer(f"yahoo_finance_fetch_reit_{ticker}"):
                reit = yf.Ticker(ticker)
                info = reit.info

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_reit_data/{ticker}",
                success=True,
                response=f"Retrieved REIT data for {ticker}"
            )

            # Check if we got valid data
            if not info or "symbol" not in info:
                return {"error": f"Could not find REIT with ticker {ticker}", "ticker": ticker}

            # Get historical data
            hist = reit.history(period="1mo")

            # Convert historical data to serializable format
            history_data = {}
            if not hist.empty:
                # Import the serialization utility
                from core.utils.serialization import make_json_serializable

                # Use the serialization utility to convert the DataFrame
                history_data = make_json_serializable(hist)

            # Calculate additional metrics
            market_cap = info.get("marketCap", 0)
            dividend_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
            beta = info.get("beta", 0)

            # Estimate FFO (Funds From Operations) - a key REIT metric
            # This is a simplified calculation and may not be accurate
            net_income = info.get("netIncomeToCommon", 0)
            depreciation = info.get("totalAssets", 0) * 0.03  # Rough estimate
            ffo = net_income + depreciation

            # Calculate price to FFO
            price_to_ffo = info.get("marketCap", 0) / ffo if ffo else 0

            # Format the data
            result = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "property_type": info.get("industry", ""),
                "market_cap": self._format_market_cap(market_cap),
                "dividend_yield": f"{dividend_yield:.2f}%",
                "price_to_ffo": price_to_ffo,
                "funds_from_operations": ffo,
                "debt_to_equity": info.get("debtToEquity", 0),
                "beta": beta,
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "history": history_data,
                "raw": info
            }

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched REIT data for {ticker}")

            # Log data operation
            log_data_operation(
                operation="process",
                data_type="reit",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "data_fields": list(result.keys())
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "source": "yahoo_finance", "method": "get_reit_data"})
            return {"error": str(e), "ticker": ticker}

    def get_price_history(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get price history for a ticker.

        Args:
            ticker: Ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Time interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            Dictionary containing price history
        """
        # Create cache key
        cache_key = f"price_history_{ticker}_{period}_{interval}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get price history using yfinance
            log_info(f"Fetching price history for {ticker} with period {period} and interval {interval}")
            with PerformanceTimer(f"yahoo_finance_price_history_{ticker}"):
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period, interval=interval)

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_price_history/{ticker}",
                params={"period": period, "interval": interval},
                success=True,
                response=f"Retrieved {len(hist) if not hist.empty else 0} data points"
            )

            if hist.empty:
                error_msg = f"No price history available for {ticker}"
                log_warning(error_msg)
                return {"error": error_msg}

            # Convert to price history format using the serialization utility
            from core.utils.serialization import make_json_serializable
            result = make_json_serializable(hist)

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched price history for {ticker} with {len(hist)} data points")
            log_debug(f"Price history for {ticker}: {period} period, {interval} interval")

            # Log data operation
            log_data_operation(
                operation="process",
                data_type="price_history",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "period": period,
                    "interval": interval,
                    "data_points": len(hist)
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "period": period, "interval": interval})
            return {"error": str(e)}

    def get_historical_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get historical price data for a ticker.
        This method is used by the comparison tool.

        Args:
            ticker: Ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Dictionary containing historical price data
        """
        # Create cache key
        cache_key = f"historical_data_{ticker}_{period}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get historical data using yfinance
            log_info(f"Fetching historical data for {ticker} with period {period} from Yahoo Finance")
            with PerformanceTimer(f"yahoo_finance_historical_data_{ticker}"):
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period)

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_historical_data/{ticker}",
                params={"period": period},
                success=True,
                response=f"Retrieved {len(hist) if not hist.empty else 0} data points"
            )

            if hist.empty:
                error_msg = f"No historical data available for {ticker}"
                log_warning(error_msg)
                return {"error": error_msg}

            # Convert to serializable format using the serialization utility
            from core.utils.serialization import make_json_serializable
            result = make_json_serializable(hist)

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched historical data for {ticker} with {len(hist)} data points")

            # Log data operation
            log_data_operation(
                operation="process",
                data_type="historical_data",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "period": period,
                    "data_points": len(hist)
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "period": period})
            return {"error": str(e)}

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial statements for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        # Create cache key
        cache_key = f"financial_statements_{ticker}"

        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get financial statements using yfinance
            log_info(f"Fetching financial statements for {ticker} from Yahoo Finance")
            with PerformanceTimer(f"yahoo_finance_financial_statements_{ticker}"):
                stock = yf.Ticker(ticker)

            log_api_call(
                api_name="Yahoo Finance",
                endpoint=f"get_financial_statements/{ticker}",
                success=True,
                response=f"Retrieved financial statements for {ticker}"
            )

            # Convert DataFrames to serializable format
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            result = {
                "income_statement": self._convert_df_to_dict(income_stmt),
                "balance_sheet": self._convert_df_to_dict(balance_sheet),
                "cash_flow": self._convert_df_to_dict(cashflow)
            }

            # Cache the result
            self._set_in_cache(cache_key, result)

            # Log success
            log_success(f"Successfully fetched financial statements for {ticker}")

            # Log data operation
            log_data_operation(
                operation="process",
                data_type="financial_statements",
                details={
                    "ticker": ticker,
                    "source": "Yahoo Finance",
                    "statements": ["income_statement", "balance_sheet", "cash_flow"]
                },
                success=True
            )

            return result
        except Exception as e:
            log_exception(e, context={"ticker": ticker, "source": "yahoo_finance", "method": "get_financial_statements"})
            return {"error": str(e)}

    def _format_market_cap(self, market_cap: float) -> str:
        """
        Format market cap in a human-readable format.

        Args:
            market_cap: Market cap value

        Returns:
            Formatted market cap string
        """
        if market_cap >= 1e12:
            return f"${market_cap / 1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap / 1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap / 1e6:.2f}M"
        elif market_cap > 0:
            return f"${market_cap / 1e3:.2f}K"
        else:
            return "$0"

    def _convert_df_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert a pandas DataFrame to a serializable dictionary.

        Args:
            df: DataFrame to convert

        Returns:
            Dictionary representation of the DataFrame
        """
        if df is None or df.empty:
            return {}

        try:
            # Import the serialization utility
            from utils.serialization import make_json_serializable

            # Use the serialization utility to convert the DataFrame
            return make_json_serializable(df)
        except Exception as e:
            log_exception(e, context={"source": "yahoo_finance", "method": "_dataframe_to_dict"})
            return {"error": str(e)}
