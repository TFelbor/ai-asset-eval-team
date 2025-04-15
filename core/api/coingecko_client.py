"""
CoinGecko API client for cryptocurrency data.
This module provides a standardized way to interact with the CoinGecko API.
"""
from typing import Dict, Any, List, Optional, Union
from core.api.base_client import BaseAPIClient
from config import settings as config

class CoinGeckoClient(BaseAPIClient):
    """CoinGecko API client."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CoinGecko API client.

        Args:
            api_key: CoinGecko API key (optional, for Pro API)
        """
        # Use the API key from config if not provided
        if api_key is None:
            api_key = config.COINGECKO_API_KEY

        # Initialize base client
        super().__init__(
            base_url="https://api.coingecko.com/api/v3",
            api_key=api_key,
            cache_type="coingecko",
            timeout=15,
            max_retries=3,
            retry_delay=2
        )

        # Pro API base URL
        self.pro_base_url = "https://pro-api.coingecko.com/api/v3"

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        use_pro_api: bool = False  # Default to free API
    ) -> Dict[str, Any]:
        """
        Make a request to the CoinGecko API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters
            use_cache: Whether to use cache
            use_pro_api: Whether to use the Pro API

        Returns:
            API response as dictionary
        """
        # Use Pro API if available and requested
        if self.api_key and use_pro_api:
            # Save original base URL
            original_base_url = self.base_url

            # Set base URL to Pro API
            self.base_url = self.pro_base_url

            # Add API key to headers (not params)
            headers = {"x-cg-pro-api-key": self.api_key}

            # Ensure params is initialized
            if params is None:
                params = {}

            # Make request
            response = super()._make_request(
                endpoint=endpoint,
                params=params,
                headers=headers,
                use_cache=use_cache,
                use_api_key=False  # API key is already added to headers
            )

            # Restore original base URL
            self.base_url = original_base_url

            # If request failed and it's a Pro API error, try the free API
            if self._is_error_response(response):
                error = response.get("error", "")
                status_code = response.get("status_code", 0)

                if status_code in [400, 401, 403] or "api key" in error.lower():
                    # Try the free API
                    # Remove the API key from params if it was added
                    if params and "x_cg_pro_api_key" in params:
                        params.pop("x_cg_pro_api_key")

                    return super()._make_request(
                        endpoint=endpoint,
                        params=params,
                        use_cache=use_cache,
                        use_api_key=False
                    )

            return response
        else:
            # Use free API
            return super()._make_request(
                endpoint=endpoint,
                params=params,
                use_cache=use_cache,
                use_api_key=False
            )

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
        Get data for a specific coin.

        Args:
            coin_id: Coin ID or symbol

        Returns:
            Coin data
        """
        try:
            # Convert common symbols to IDs
            coin_id = self._convert_symbol_to_id(coin_id)

            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }

            # Try with free API first
            response = self._make_request(f"coins/{coin_id}", params, use_pro_api=False)

            # Check if there was an error
            if self._is_error_response(response):
                # Log the error
                print(f"Error fetching data for {coin_id} from CoinGecko: {response.get('error')}")
                return response

            return response
        except Exception as e:
            # Return a standardized error response
            return {"error": f"Failed to fetch data for {coin_id}: {str(e)}", "ticker": coin_id}

    def get_coin_price_history(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get price history for a specific coin.

        Args:
            coin_id: Coin ID or symbol
            vs_currency: Currency to get prices in
            days: Number of days of data to retrieve

        Returns:
            Price history data
        """
        try:
            # Convert common symbols to IDs
            coin_id = self._convert_symbol_to_id(coin_id)

            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": "daily" if days > 90 else None
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Try with free API first
            response = self._make_request(f"coins/{coin_id}/market_chart", params, use_pro_api=False)

            # Check if there was an error
            if self._is_error_response(response):
                # Log the error
                print(f"Error fetching price history for {coin_id} from CoinGecko: {response.get('error')}")
                return response

            return response
        except Exception as e:
            # Return a standardized error response
            return {"error": f"Failed to fetch price history for {coin_id}: {str(e)}", "ticker": coin_id}

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
            exchange_id: Exchange ID

        Returns:
            Exchange data
        """
        return self._make_request(f"exchanges/{exchange_id}")

    def get_trending(self) -> Dict[str, Any]:
        """
        Get trending coins.

        Returns:
            Trending coins data
        """
        return self._make_request("search/trending")

    def _convert_symbol_to_id(self, symbol: str) -> str:
        """
        Convert a common symbol to its corresponding ID.

        Args:
            symbol: Coin symbol

        Returns:
            Coin ID
        """
        # Handle None or empty string
        if not symbol:
            return "bitcoin"  # Default to bitcoin if no symbol provided

        # Common symbol to ID mappings
        symbol_to_id = {
            "BTC": "bitcoin",
            "BITCOIN": "bitcoin",
            "ETH": "ethereum",
            "ETHEREUM": "ethereum",
            "USDT": "tether",
            "BNB": "binancecoin",
            "SOL": "solana",
            "XRP": "ripple",
            "USDC": "usd-coin",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LTC": "litecoin",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "SHIB": "shiba-inu",
            # Add more common symbols
            "BCH": "bitcoin-cash",
            "XLM": "stellar",
            "EOS": "eos",
            "TRX": "tron",
            "XMR": "monero",
            "XTZ": "tezos",
            "ATOM": "cosmos",
            "VET": "vechain",
            "ALGO": "algorand",
            "NEO": "neo",
            "MIOTA": "iota",
            "DASH": "dash",
            "ZEC": "zcash",
            "ETC": "ethereum-classic",
            "FIL": "filecoin",
            "THETA": "theta-token",
            "AAVE": "aave",
            "SNX": "synthetix-network-token",
            "MKR": "maker",
            "COMP": "compound-governance-token",
            "YFI": "yearn-finance",
            "SUSHI": "sushi",
            "GRT": "the-graph",
            "1INCH": "1inch",
            "CAKE": "pancakeswap-token"
        }

        # Check if the symbol is in our mapping
        if symbol.upper() in symbol_to_id:
            coin_id = symbol_to_id[symbol.upper()]
            print(f"Converted {symbol} to {coin_id}")
            return coin_id

        # If not, return the original symbol (might be an ID already)
        print(f"Using original symbol (lowercased): {symbol.lower()}")
        return symbol.lower()
