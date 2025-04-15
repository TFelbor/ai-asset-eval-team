"""
Unified data service for the AI Finance Dashboard.
This module provides a standardized way to fetch data for different asset types.
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import API clients lazily to avoid circular imports

# Import cache manager
from core.data.cache_manager import CacheManager

# Import serialization utility
from core.utils.serialization import make_json_serializable

class DataService:
    """Unified data service for all asset types."""

    def __init__(self):
        """Initialize the data service."""
        # Initialize cache managers first
        self.stock_cache = CacheManager(cache_type="stock")
        self.crypto_cache = CacheManager(cache_type="crypto")
        self.reit_cache = CacheManager(cache_type="reit")
        self.etf_cache = CacheManager(cache_type="etf")
        self.macro_cache = CacheManager(cache_type="macro")

        # Initialize API clients lazily to avoid circular imports
        self._coingecko = None
        self._yahoo_finance = None
        self._alpha_vantage = None
        self._news_api = None

    @property
    def coingecko(self):
        """Lazy-loaded CoinGeckoClient."""
        if self._coingecko is None:
            # Import here to avoid circular imports
            from core.api.coingecko_client import CoinGeckoClient
            self._coingecko = CoinGeckoClient()
        return self._coingecko

    @property
    def yahoo_finance(self):
        """Lazy-loaded YahooFinanceClient."""
        if self._yahoo_finance is None:
            # Import here to avoid circular imports
            from core.api.yahoo_finance_client import YahooFinanceClient
            self._yahoo_finance = YahooFinanceClient()
        return self._yahoo_finance

    @property
    def alpha_vantage(self):
        """Lazy-loaded AlphaVantageClient."""
        if self._alpha_vantage is None:
            # Import here to avoid circular imports
            from core.api.alphavantage_client import AlphaVantageClient
            self._alpha_vantage = AlphaVantageClient()
        return self._alpha_vantage

    @property
    def news_api(self):
        """Lazy-loaded NewsClient."""
        if self._news_api is None:
            # Import here to avoid circular imports
            from core.api.news_client import NewsClient
            self._news_api = NewsClient(api_key=None)  # Will use the key from config
        return self._news_api

    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Stock data dictionary
        """
        # Check cache first
        cached_data = self.stock_cache.get(ticker)
        if cached_data:
            return cached_data

        # Fetch data from Yahoo Finance
        data = self.yahoo_finance.get_stock_data(ticker)

        # Serialize and cache the data
        if "error" not in data:
            serialized_data = make_json_serializable(data)
            self.stock_cache.set(ticker, serialized_data)

        return data

    def get_crypto_data(self, coin_id: str) -> Dict[str, Any]:
        """
        Get comprehensive cryptocurrency data.

        Args:
            coin_id: Cryptocurrency ID or symbol

        Returns:
            Cryptocurrency data dictionary
        """
        # Check cache first
        cached_data = self.crypto_cache.get(coin_id)
        if cached_data:
            return cached_data

        # Fetch data from CoinGecko
        data = self.coingecko.get_coin_data(coin_id)

        # Serialize and cache the data
        if "error" not in data:
            serialized_data = make_json_serializable(data)
            self.crypto_cache.set(coin_id, serialized_data)

        return data

    def get_reit_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive REIT data.

        Args:
            ticker: REIT ticker symbol

        Returns:
            REIT data dictionary
        """
        # Check cache first
        cached_data = self.reit_cache.get(ticker)
        if cached_data:
            return cached_data

        # Fetch data from Yahoo Finance
        data = self.yahoo_finance.get_reit_data(ticker)

        # Serialize and cache the data
        if "error" not in data:
            serialized_data = make_json_serializable(data)
            self.reit_cache.set(ticker, serialized_data)

        return data

    def get_etf_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive ETF data.

        Args:
            ticker: ETF ticker symbol

        Returns:
            ETF data dictionary
        """
        # Check cache first
        cached_data = self.etf_cache.get(ticker)
        if cached_data:
            return cached_data

        # Fetch data from Yahoo Finance
        data = self.yahoo_finance.get_etf_data(ticker)

        # Serialize and cache the data
        if "error" not in data:
            serialized_data = make_json_serializable(data)
            self.etf_cache.set(ticker, serialized_data)

        return data

    def get_price_history(
        self,
        ticker: str,
        asset_type: str,
        period: str = "1mo"
    ) -> Dict[str, Any]:
        """
        Get price history for any asset type.

        Args:
            ticker: Asset ticker or symbol
            asset_type: Asset type (stock, crypto, reit, etf)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Price history dictionary
        """
        # Create cache key
        cache_key = f"{ticker}_{period}_price_history"

        # Get the appropriate cache manager
        if asset_type.lower() == "stock":
            cache = self.stock_cache
        elif asset_type.lower() == "crypto":
            cache = self.crypto_cache
        elif asset_type.lower() == "reit":
            cache = self.reit_cache
        elif asset_type.lower() == "etf":
            cache = self.etf_cache
        else:
            return {"error": f"Unsupported asset type: {asset_type}"}

        # Check cache first
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data

        # Fetch data based on asset type
        if asset_type.lower() == "crypto":
            # Convert period to days for CoinGecko
            days = self._period_to_days(period)
            data = self.coingecko.get_coin_price_history(ticker, days=days)

            # Format data with enhanced error handling
            if "prices" in data and "error" not in data:
                # Ensure all values are properly converted to appropriate types
                try:
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

                    # Create the price history dictionary
                    price_history = {
                        "timestamps": timestamps,
                        "prices": prices,
                        "volumes": volumes[:len(timestamps)]  # Ensure volumes matches timestamps length
                    }

                    # Print debug info
                    print(f"Processed {len(timestamps)} data points for cryptocurrency")
                    if timestamps:
                        print(f"Sample data - timestamp: {timestamps[0]}, price: {prices[0]}")

                except (ValueError, TypeError, IndexError) as e:
                    print(f"Error converting price history data: {e}")
                    # Return error with sample data for debugging
                    sample_prices = data.get("prices", [])[:3] if data.get("prices") else []
                    return {"error": f"Failed to convert price history data: {e}", "sample_data": sample_prices}

                # Serialize and cache the data
                serialized_data = make_json_serializable(price_history)
                cache.set(cache_key, serialized_data)

                return price_history
            else:
                return data
        else:
            # Use AlphaVantage for stocks, REITs, and ETFs
            try:
                # Determine the appropriate time series interval based on the period
                interval, outputsize = self._period_to_alphavantage_params(period)

                # Get time series data from AlphaVantage
                data = self.alpha_vantage.get_stock_time_series(
                    symbol=ticker,
                    interval=interval,
                    outputsize=outputsize
                )

                # Check for errors
                if "error" in data:
                    return data

                # Parse the AlphaVantage time series data
                price_history = self._parse_alphavantage_time_series(data, interval)

                if price_history and len(price_history.get("timestamps", [])) > 0:
                    # Serialize and cache the data
                    serialized_data = make_json_serializable(price_history)
                    cache.set(cache_key, serialized_data)

                    return price_history
                else:
                    # Fallback to Yahoo Finance if AlphaVantage data is empty or invalid
                    try:
                        import yfinance as yf
                        ticker_obj = yf.Ticker(ticker)
                        hist = ticker_obj.history(period=period)

                        if not hist.empty:
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

                            # Serialize and cache the data
                            serialized_data = make_json_serializable(price_history)
                            cache.set(cache_key, serialized_data)

                            return price_history
                        else:
                            return {"error": "No historical data available"}
                    except Exception as e:
                        return {"error": f"Failed to get data from fallback source: {str(e)}"}
            except Exception as e:
                return {"error": f"Failed to get historical data: {str(e)}"}

    def get_news(
        self,
        asset_type: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get news for a specific asset or general market news.

        Args:
            asset_type: Asset type (stock, crypto, reit, etf)
            ticker: Asset ticker or symbol
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        if asset_type is None or ticker is None:
            # Get general market news
            return self.news_api.get_market_news(limit=limit)

        # Get news for a specific asset
        if asset_type.lower() == "stock":
            return self.news_api.get_stock_news(ticker, limit=limit)
        elif asset_type.lower() == "crypto":
            return self.news_api.get_crypto_news(ticker, limit=limit)
        elif asset_type.lower() == "reit":
            return self.news_api.get_reit_news(ticker, limit=limit)
        elif asset_type.lower() == "etf":
            return self.news_api.get_etf_news(ticker, limit=limit)
        else:
            return []

    def get_macro_data(self) -> Dict[str, Any]:
        """
        Get macroeconomic data.

        Returns:
            Macroeconomic data dictionary
        """
        # Check cache first
        cached_data = self.macro_cache.get("macro_data")
        if cached_data:
            return cached_data

        # Fetch data from Alpha Vantage
        try:
            # Get GDP data
            gdp_data = self.alpha_vantage.get_economic_indicator("REAL_GDP")

            # Get inflation data
            inflation_data = self.alpha_vantage.get_economic_indicator("CPI")

            # Get unemployment data
            unemployment_data = self.alpha_vantage.get_economic_indicator("UNEMPLOYMENT")

            # Combine data
            macro_data = {
                "gdp": gdp_data.get("data", []),
                "inflation": inflation_data.get("data", []),
                "unemployment": unemployment_data.get("data", []),
                "gdp_outlook": self._calculate_gdp_outlook(gdp_data.get("data", [])),
                "inflation_risk": self._calculate_inflation_risk(inflation_data.get("data", [])),
                "unemployment_trend": self._calculate_unemployment_trend(unemployment_data.get("data", []))
            }

            # Serialize and cache the data
            serialized_data = make_json_serializable(macro_data)
            self.macro_cache.set("macro_data", serialized_data)

            return macro_data
        except Exception as e:
            return {"error": str(e)}

    def _period_to_days(self, period: str) -> int:
        """
        Convert a period string to number of days.

        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Number of days
        """
        period_map = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "ytd": int((datetime.now() - datetime(datetime.now().year, 1, 1)).total_seconds() / 86400),
            "max": 3650  # Default to 10 years for "max"
        }

        return period_map.get(period, 30)  # Default to 30 days

    def _period_to_alphavantage_params(self, period: str) -> Tuple[str, str]:
        """
        Convert a period string to AlphaVantage parameters.

        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            Tuple of (interval, outputsize)
        """
        # Map period to appropriate interval and outputsize
        if period in ["1d", "5d"]:
            return "daily", "compact"  # Last 100 data points
        elif period in ["1mo", "3mo"]:
            return "daily", "compact"  # Last 100 data points
        elif period in ["6mo", "1y"]:
            return "daily", "full"  # Full historical data
        elif period in ["2y", "5y", "10y", "max"]:
            return "weekly", "full"  # Weekly data for longer periods
        elif period == "ytd":
            # Determine if YTD is more or less than 100 days
            ytd_days = int((datetime.now() - datetime(datetime.now().year, 1, 1)).total_seconds() / 86400)
            if ytd_days <= 100:
                return "daily", "compact"
            else:
                return "daily", "full"
        else:
            return "daily", "compact"  # Default

    def _parse_alphavantage_time_series(self, data: Dict[str, Any], interval: str) -> Dict[str, Any]:
        """
        Parse AlphaVantage time series data into a standardized format.

        Args:
            data: AlphaVantage time series data
            interval: Time interval (daily, weekly, monthly)

        Returns:
            Standardized price history dictionary
        """
        # Determine the time series key based on the interval
        time_series_key = f"Time Series ({interval.capitalize()})"
        if interval == "daily":
            time_series_key = "Time Series (Daily)"
        elif interval == "weekly":
            time_series_key = "Weekly Time Series"
        elif interval == "monthly":
            time_series_key = "Monthly Time Series"

        # Check if the time series data exists
        if time_series_key not in data:
            return {"error": f"No {interval} time series data available"}

        # Get the time series data
        time_series = data[time_series_key]

        # Parse the time series data
        timestamps = []
        prices = []
        volumes = []
        opens = []
        highs = []
        lows = []
        closes = []

        # Sort dates in ascending order (oldest to newest)
        for date in sorted(time_series.keys()):
            entry = time_series[date]

            # Convert date string to timestamp (milliseconds)
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            timestamp = int(date_obj.timestamp() * 1000)  # Convert to milliseconds

            # Extract values
            open_price = float(entry.get("1. open", 0))
            high_price = float(entry.get("2. high", 0))
            low_price = float(entry.get("3. low", 0))
            close_price = float(entry.get("4. close", 0))
            volume = int(float(entry.get("5. volume", 0)))

            # Append to lists
            timestamps.append(timestamp)
            prices.append(close_price)  # Use close price as the main price
            volumes.append(volume)
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)

        # Create the price history dictionary with OHLC data
        price_history = {
            "timestamps": timestamps,
            "prices": prices,
            "volumes": volumes,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes
        }

        # Debug info
        print(f"AlphaVantage data parsed: {len(timestamps)} data points with OHLC data")

        return price_history

    def _calculate_gdp_outlook(self, gdp_data: List[Dict[str, Any]]) -> str:
        """
        Calculate GDP outlook based on recent GDP data.

        Args:
            gdp_data: List of GDP data points

        Returns:
            GDP outlook (Strong Growth, Moderate Growth, Stable, Slowdown, Contraction)
        """
        if not gdp_data or len(gdp_data) < 2:
            return "Stable"

        # Sort data by date
        sorted_data = sorted(gdp_data, key=lambda x: x.get("date", ""), reverse=True)

        # Get the two most recent GDP values
        recent_gdp = float(sorted_data[0].get("value", 0))
        previous_gdp = float(sorted_data[1].get("value", 0))

        # Calculate growth rate
        growth_rate = (recent_gdp - previous_gdp) / previous_gdp * 100

        # Determine outlook
        if growth_rate > 3:
            return "Strong Growth"
        elif growth_rate > 1:
            return "Moderate Growth"
        elif growth_rate > -1:
            return "Stable"
        elif growth_rate > -3:
            return "Slowdown"
        else:
            return "Contraction"

    def _calculate_inflation_risk(self, inflation_data: List[Dict[str, Any]]) -> str:
        """
        Calculate inflation risk based on recent inflation data.

        Args:
            inflation_data: List of inflation data points

        Returns:
            Inflation risk (Very Low, Low, Moderate, High, Very High)
        """
        if not inflation_data:
            return "Moderate"

        # Sort data by date
        sorted_data = sorted(inflation_data, key=lambda x: x.get("date", ""), reverse=True)

        # Get the most recent inflation value
        recent_inflation = float(sorted_data[0].get("value", 0))

        # Determine risk
        if recent_inflation < 1:
            return "Very Low"
        elif recent_inflation < 2:
            return "Low"
        elif recent_inflation < 4:
            return "Moderate"
        elif recent_inflation < 7:
            return "High"
        else:
            return "Very High"

    def _calculate_unemployment_trend(self, unemployment_data: List[Dict[str, Any]]) -> str:
        """
        Calculate unemployment trend based on recent unemployment data.

        Args:
            unemployment_data: List of unemployment data points

        Returns:
            Unemployment trend (Improving, Stable, Worsening)
        """
        if not unemployment_data or len(unemployment_data) < 3:
            return "Stable"

        # Sort data by date
        sorted_data = sorted(unemployment_data, key=lambda x: x.get("date", ""), reverse=True)

        # Get the three most recent unemployment values
        recent_values = [float(data.get("value", 0)) for data in sorted_data[:3]]

        # Calculate trend
        if recent_values[0] < recent_values[1] < recent_values[2]:
            return "Improving"
        elif recent_values[0] > recent_values[1] > recent_values[2]:
            return "Worsening"
        else:
            return "Stable"
