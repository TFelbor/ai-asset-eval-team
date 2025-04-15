"""
News API client for financial news.
This module provides a standardized way to interact with the News API.
"""
from typing import Dict, Any, List, Optional, Union
from core.api.base_client import BaseAPIClient
from config import settings as config
from datetime import datetime, timedelta

class NewsClient(BaseAPIClient):
    """News API client."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the News API client.

        Args:
            api_key: News API key (optional)
        """
        # Use the API key from config if not provided
        if api_key is None:
            api_key = config.NEWS_API_KEY

        # Initialize base client
        super().__init__(
            base_url="https://newsapi.org/v2",
            api_key=api_key,
            cache_type="news",
            timeout=15,
            max_retries=3,
            retry_delay=2
        )

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to the News API.

        Args:
            endpoint: API endpoint to call
            params: Additional parameters
            use_cache: Whether to use cache

        Returns:
            API response as dictionary
        """
        # Make the request using the base client
        response = super()._make_request(
            endpoint=endpoint,
            params=params,
            headers={"X-Api-Key": self.api_key},
            use_cache=use_cache,
            use_api_key=False  # API key is added to headers
        )

        # Check for News API-specific error messages
        if "status" in response and response["status"] == "error":
            return {"error": response.get("message", "Unknown error")}

        return response

    def get_top_headlines(
        self,
        country: str = "us",
        category: str = "business",
        page_size: int = 10,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get top headlines.

        Args:
            country: Country code (e.g., 'us', 'gb', 'de')
            category: News category (e.g., 'business', 'technology', 'science')
            page_size: Number of results per page (max 100)
            page: Page number

        Returns:
            List of news articles
        """
        params = {
            "country": country,
            "category": category,
            "pageSize": page_size,
            "page": page
        }

        response = self._make_request("top-headlines", params)

        if "error" in response:
            return []

        return response.get("articles", [])

    def search_news(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 10,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles.

        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code (e.g., 'en', 'de', 'fr')
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of results per page (max 100)
            page: Page number

        Returns:
            List of news articles
        """
        # Set default dates if not provided
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page
        }

        response = self._make_request("everything", params)

        if "error" in response:
            return []

        return response.get("articles", [])

    def get_sources(
        self,
        category: Optional[str] = None,
        language: str = "en",
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news sources.

        Args:
            category: News category (e.g., 'business', 'technology', 'science')
            language: Language code (e.g., 'en', 'de', 'fr')
            country: Country code (e.g., 'us', 'gb', 'de')

        Returns:
            List of news sources
        """
        params = {
            "language": language
        }

        if category:
            params["category"] = category

        if country:
            params["country"] = country

        response = self._make_request("sources", params)

        if "error" in response:
            return []

        return response.get("sources", [])

    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get general market news.

        Args:
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        # Get top business headlines
        headlines = self.get_top_headlines(category="business", page_size=limit)

        # Get market-specific news
        market_news = self.search_news("stock market OR financial markets", page_size=limit)

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

        return unique_news[:limit]  # Return top N unique articles

    def get_stock_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific stock.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        # Search for news about the stock
        query = f"{ticker} stock OR {ticker} company OR {ticker} earnings"
        return self.search_news(query, page_size=limit)

    def get_crypto_news(self, coin_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific cryptocurrency.

        Args:
            coin_id: Cryptocurrency ID or symbol
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        # Search for news about the cryptocurrency
        query = f"{coin_id} crypto OR {coin_id} cryptocurrency OR {coin_id} blockchain"
        return self.search_news(query, page_size=limit)

    def get_reit_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific REIT.

        Args:
            ticker: REIT ticker symbol
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        # Search for news about the REIT
        query = f"{ticker} REIT OR {ticker} real estate OR {ticker} property"
        return self.search_news(query, page_size=limit)

    def get_etf_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific ETF.

        Args:
            ticker: ETF ticker symbol
            limit: Maximum number of news articles to return

        Returns:
            List of news articles
        """
        # Search for news about the ETF
        query = f"{ticker} ETF OR {ticker} fund OR {ticker} index"
        return self.search_news(query, page_size=limit)
