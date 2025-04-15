"""
Base API client for all external API integrations.
This module provides a standardized way to interact with external APIs,
including error handling, rate limiting, and caching.
"""
import requests
import json
import time
from typing import Dict, Any, Optional

# Import enhanced logging
from core.utils.logger import (
    log_info, log_error, log_success, log_warning, log_debug,
    log_api_call, log_data_operation, log_exception, performance_timer
)

class BaseAPIClient:
    """Base class for all API clients."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        cache_type: str = "default",
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the base API client.

        Args:
            base_url: Base URL for the API
            api_key: API key (optional)
            cache_type: Type of cache to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_type = cache_type

        # Simple in-memory cache to avoid circular imports
        self._cache_data = {}
        self._cache_expiry = {}

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from the cache."""
        current_time = time.time()
        if key in self._cache_data and self._cache_expiry.get(key, 0) > current_time:
            return self._cache_data[key]
        return None

    def _set_in_cache(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """Set a value in the cache."""
        self._cache_data[key] = value
        self._cache_expiry[key] = time.time() + ttl

    def _clear_cache(self) -> None:
        """Clear the cache."""
        self._cache_data = {}
        self._cache_expiry = {}

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour default cache TTL
        use_api_key: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to the API with error handling and caching.

        Args:
            endpoint: API endpoint to call
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            headers: HTTP headers
            data: Request body for POST requests
            use_cache: Whether to use cache
            cache_ttl: Cache time-to-live in seconds
            use_api_key: Whether to use the API key

        Returns:
            API response as dictionary
        """
        # Prepare URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Prepare headers
        if headers is None:
            headers = {}

        # Add API key to headers or params if available and requested
        if self.api_key and use_api_key:
            # Default to adding API key to params, but subclasses can override this
            if params is None:
                params = {}
            params["api_key"] = self.api_key

        # Create cache key
        cache_key = f"{method}:{url}:{json.dumps(params) if params else ''}:{json.dumps(data) if data else ''}"

        # Try to get from cache first if using cache
        if use_cache:
            cached_response = self._get_from_cache(cache_key)
            if cached_response is not None:
                log_debug(f"Using cached response for {endpoint}")
                return cached_response

        # Make the request with retries
        retries = 0
        while retries <= self.max_retries:
            try:
                log_debug(f"Making {method} request to {url}")

                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                # Check for HTTP errors
                response.raise_for_status()

                # Parse JSON response
                result = response.json()

                # Cache successful response if using cache
                if use_cache:
                    self._set_in_cache(cache_key, result, ttl=cache_ttl)

                return result

            except requests.exceptions.HTTPError as e:
                log_error(f"HTTP error in API request: {str(e)}")

                # Handle rate limiting
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("retry-after", self.retry_delay))
                    log_warning(f"Rate limit exceeded. Retry after {retry_after} seconds.")

                    # If we've reached max retries, return error
                    if retries >= self.max_retries:
                        return {"error": "Rate limit exceeded", "retry_after": retry_after}

                    # Wait and retry
                    time.sleep(retry_after)
                    retries += 1
                    continue

                # For other HTTP errors, return error response
                return {
                    "error": str(e),
                    "status_code": e.response.status_code,
                    "response": e.response.text
                }

            except requests.exceptions.ConnectionError as e:
                log_error(f"Connection error in API request: {str(e)}")

                # If we've reached max retries, return error
                if retries >= self.max_retries:
                    return {"error": f"Connection error: {str(e)}"}

                # Wait and retry
                time.sleep(self.retry_delay)
                retries += 1
                continue

            except requests.exceptions.Timeout as e:
                log_error(f"Timeout in API request: {str(e)}")

                # If we've reached max retries, return error
                if retries >= self.max_retries:
                    return {"error": f"Request timed out: {str(e)}"}

                # Wait and retry
                time.sleep(self.retry_delay)
                retries += 1
                continue

            except json.JSONDecodeError as e:
                log_error(f"JSON decode error in API response: {str(e)}")
                return {"error": f"Invalid JSON response: {str(e)}"}

            except Exception as e:
                log_error(f"Unexpected error in API request: {str(e)}")
                return {"error": str(e)}

    def _is_error_response(self, response: Dict[str, Any]) -> bool:
        """
        Check if the response contains an error.

        Args:
            response: API response

        Returns:
            True if the response contains an error, False otherwise
        """
        return isinstance(response, dict) and "error" in response

    def _handle_error_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle error response.

        Args:
            response: API error response

        Returns:
            Standardized error response
        """
        if not self._is_error_response(response):
            return response

        # Log the error
        log_error(f"API error: {response.get('error')}")

        # Return standardized error response
        return {
            "error": response.get("error", "Unknown error"),
            "status_code": response.get("status_code", 500),
            "retry_after": response.get("retry_after")
        }
