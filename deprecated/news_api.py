"""
News API integration for financial news.
"""
import requests
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
import time
from config.settings import NEWS_API_KEY

class NewsAPI:
    """Client for interacting with the News API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News API client.

        Args:
            api_key: API key for News API
        """
        self.api_key = api_key or NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit_remaining = 100  # Default rate limit
        self.rate_limit_reset_at = datetime.now()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the News API with rate limiting.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            API response as dictionary
        """
        # Check if we need to wait for rate limit reset
        if self.rate_limit_remaining <= 1 and datetime.now() < self.rate_limit_reset_at:
            wait_time = (self.rate_limit_reset_at - datetime.now()).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer

        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"X-Api-Key": self.api_key}

            response = requests.get(url, params=params, headers=headers)

            # Update rate limit info from headers
            if "X-RateLimit-Remaining" in response.headers:
                self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in response.headers:
                reset_timestamp = int(response.headers["X-RateLimit-Reset"])
                self.rate_limit_reset_at = datetime.fromtimestamp(reset_timestamp)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(e.response.headers.get("Retry-After", 60))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)  # Retry
            return {"status": "error", "code": e.response.status_code, "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_top_headlines(self, category: str = "business", country: str = "us", page_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get top headlines.

        Args:
            category: News category (business, technology, etc.)
            country: Country code (us, gb, etc.)
            page_size: Number of results to return

        Returns:
            List of news articles
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": page_size
        }

        response = self._make_request("top-headlines", params)

        if response.get("status") == "ok":
            return response.get("articles", [])
        return []

    def search_news(self, query: str, from_date: Optional[str] = None, to_date: Optional[str] = None,
                   sort_by: str = "publishedAt", page_size: int = 10) -> List[Dict[str, Any]]:
        """
        Search for news articles.

        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of results to return

        Returns:
            List of news articles
        """
        # If dates not provided, use last 7 days
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "sortBy": sort_by,
            "pageSize": page_size,
            "language": "en"
        }

        response = self._make_request("everything", params)

        if response.get("status") == "ok":
            return response.get("articles", [])
        return []

    def get_stock_news(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get news for a specific stock.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            List of news articles
        """
        # Get company name for better search results
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            company_name = stock.info.get("shortName", ticker)
        except:
            company_name = ticker

        # Create search query with ticker and company name
        query = f"{ticker} OR {company_name}"

        # Get news from the last N days
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.search_news(query, from_date=from_date, page_size=5)

    def get_crypto_news(self, coin: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get news for a specific cryptocurrency.

        Args:
            coin: Cryptocurrency name or symbol
            days: Number of days to look back

        Returns:
            List of news articles
        """
        # Map common symbols to full names for better search results
        coin_names = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "ada": "cardano",
            "dot": "polkadot",
            "doge": "dogecoin",
            "xrp": "ripple",
            "ltc": "litecoin"
        }

        # Get full name if available
        coin_name = coin_names.get(coin.lower(), coin)

        # Create search query
        query = f"{coin_name} OR {coin} cryptocurrency"

        # Get news from the last N days
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.search_news(query, from_date=from_date, page_size=5)

    def get_market_news(self) -> List[Dict[str, Any]]:
        """
        Get general market news.

        Returns:
            List of news articles
        """
        # Get top business headlines
        headlines = self.get_top_headlines(category="business", page_size=5)

        # Get market-specific news
        market_news = self.search_news("stock market OR financial markets", page_size=5)

        # Combine and return unique articles
        all_news = headlines + market_news

        # Remove duplicates based on title
        unique_news = []
        titles = set()

        for article in all_news:
            title = article.get("title")
            if title and title not in titles:
                titles.add(title)
                unique_news.append(article)

        return unique_news[:10]  # Return top 10 unique articles
