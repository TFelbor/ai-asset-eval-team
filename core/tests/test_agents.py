"""
Unit tests for the agent classes.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents import FundamentalAnalyst, CryptoAnalyst, MacroSentimentAnalyst

class TestFundamentalAnalyst(unittest.TestCase):
    """Test the FundamentalAnalyst class."""
    
    @patch('yfinance.Ticker')
    @patch('agents.stock_cache.get')
    @patch('agents.stock_cache.set')
    def test_analyze_with_cache_hit(self, mock_cache_set, mock_cache_get, mock_yf_ticker):
        """Test the analyze method with a cache hit."""
        # Mock the cache hit
        mock_cache_get.return_value = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "current_price": "$150.00",
            "dcf": "$170.00",
            "pe": 25.0,
            "confidence": 80
        }
        
        # Create the analyst
        analyst = FundamentalAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("AAPL")
        
        # Check that the cache was used
        mock_cache_get.assert_called_once_with("stock_aapl")
        mock_yf_ticker.assert_not_called()
        mock_cache_set.assert_not_called()
        
        # Check the result
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["name"], "Apple Inc.")
        
    @patch('yfinance.Ticker')
    @patch('agents.stock_cache.get')
    @patch('agents.stock_cache.set')
    def test_analyze_with_cache_miss(self, mock_cache_set, mock_cache_get, mock_yf_ticker):
        """Test the analyze method with a cache miss."""
        # Mock the cache miss
        mock_cache_get.return_value = None
        
        # Mock the yfinance Ticker
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "shortName": "Apple Inc.",
            "currentPrice": 150.0,
            "previousClose": 149.0,
            "enterpriseValue": 2000000000000,
            "marketCap": 2500000000000,
            "trailingPE": 25.0,
            "priceToBook": 30.0,
            "dividendYield": 0.005,
            "beta": 1.2,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "earningsGrowth": 0.1
        }
        mock_yf_ticker.return_value = mock_ticker_instance
        
        # Create the analyst
        analyst = FundamentalAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("AAPL")
        
        # Check that the cache was checked and the result was cached
        mock_cache_get.assert_called_once_with("stock_aapl")
        mock_yf_ticker.assert_called_once_with("AAPL")
        mock_cache_set.assert_called_once()
        
        # Check the result
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["name"], "Apple Inc.")
        self.assertEqual(result["current_price"], "$150.00")
        self.assertEqual(result["sector"], "Technology")
        self.assertEqual(result["industry"], "Consumer Electronics")
        
    @patch('yfinance.Ticker')
    @patch('agents.stock_cache.get')
    def test_analyze_with_error(self, mock_cache_get, mock_yf_ticker):
        """Test the analyze method with an error."""
        # Mock the cache miss
        mock_cache_get.return_value = None
        
        # Mock the yfinance Ticker to raise an exception
        mock_yf_ticker.side_effect = Exception("Test exception")
        
        # Create the analyst
        analyst = FundamentalAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("AAPL")
        
        # Check the result
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["confidence"], 0)
        self.assertIn("error", result)
        
class TestCryptoAnalyst(unittest.TestCase):
    """Test the CryptoAnalyst class."""
    
    @patch('agents.coingecko_client.analyze_coin')
    @patch('agents.crypto_cache.get')
    @patch('agents.crypto_cache.set')
    def test_analyze_with_cache_hit(self, mock_cache_set, mock_cache_get, mock_analyze_coin):
        """Test the analyze method with a cache hit."""
        # Mock the cache hit
        mock_cache_get.return_value = {
            "coin": "btc",
            "name": "Bitcoin",
            "symbol": "BTC",
            "current_price": 50000,
            "mcap": "$1,000,000,000,000",
            "volatility": "Medium",
            "confidence": 85
        }
        
        # Create the analyst
        analyst = CryptoAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("btc")
        
        # Check that the cache was used
        mock_cache_get.assert_called_once_with("crypto_btc")
        mock_analyze_coin.assert_not_called()
        mock_cache_set.assert_not_called()
        
        # Check the result
        self.assertEqual(result["coin"], "btc")
        self.assertEqual(result["name"], "Bitcoin")
        
    @patch('agents.coingecko_client.get_coin_list')
    @patch('agents.coingecko_client.analyze_coin')
    @patch('agents.crypto_cache.get')
    @patch('agents.crypto_cache.set')
    def test_analyze_with_cache_miss(self, mock_cache_set, mock_cache_get, mock_analyze_coin, mock_get_coin_list):
        """Test the analyze method with a cache miss."""
        # Mock the cache miss
        mock_cache_get.return_value = None
        
        # Mock the coin list
        mock_get_coin_list.return_value = [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}
        ]
        
        # Mock the analyze_coin method
        mock_analyze_coin.return_value = {
            "id": "bitcoin",
            "name": "Bitcoin",
            "symbol": "BTC",
            "current_price": 50000,
            "market_cap": 1000000000000,
            "market_cap_rank": 1,
            "market_dominance": 40,
            "trading_volume_24h": 50000000000,
            "price_change_percentage_24h": 2.5,
            "price_change_percentage_7d": 5.0,
            "volatility_30d": 60,
            "all_time_high": 69000,
            "all_time_high_change_percentage": -27.5,
            "circulating_supply": 19000000,
            "max_supply": 21000000,
            "supply_percentage": 90.5
        }
        
        # Create the analyst
        analyst = CryptoAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("btc")
        
        # Check that the cache was checked and the result was cached
        mock_cache_get.assert_called_once_with("crypto_btc")
        mock_get_coin_list.assert_called_once()
        mock_analyze_coin.assert_called_once_with("bitcoin")
        mock_cache_set.assert_called_once()
        
        # Check the result
        self.assertEqual(result["coin"], "btc")
        self.assertEqual(result["name"], "Bitcoin")
        self.assertEqual(result["symbol"], "BTC")
        self.assertEqual(result["current_price"], 50000)
        self.assertEqual(result["mcap"], "$1,000,000,000,000")
        
    @patch('agents.coingecko_client.get_coin_list')
    @patch('agents.coingecko_client.analyze_coin')
    @patch('agents.crypto_cache.get')
    def test_analyze_with_error(self, mock_cache_get, mock_analyze_coin, mock_get_coin_list):
        """Test the analyze method with an error."""
        # Mock the cache miss
        mock_cache_get.return_value = None
        
        # Mock the coin list
        mock_get_coin_list.return_value = [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}
        ]
        
        # Mock the analyze_coin method to raise an exception
        mock_analyze_coin.side_effect = Exception("Test exception")
        
        # Create the analyst
        analyst = CryptoAnalyst()
        
        # Call the analyze method
        result = analyst.analyze("btc")
        
        # Check the result
        self.assertEqual(result["coin"], "btc")
        self.assertEqual(result["confidence"], 0)
        self.assertIn("error", result)
        
class TestMacroSentimentAnalyst(unittest.TestCase):
    """Test the MacroSentimentAnalyst class."""
    
    @patch('agents.macro_cache.get')
    @patch('agents.macro_cache.set')
    def test_analyze_with_cache_hit(self, mock_cache_set, mock_cache_get):
        """Test the analyze method with a cache hit."""
        # Mock the cache hit
        mock_cache_get.return_value = {
            "sentiment": 65,
            "fed_impact": "Neutral",
            "inflation_risk": "Moderate",
            "gdp_growth": "2.3%",
            "gdp_outlook": "Stable"
        }
        
        # Create the analyst
        analyst = MacroSentimentAnalyst()
        
        # Call the analyze method
        result = analyst.analyze()
        
        # Check that the cache was used
        mock_cache_get.assert_called_once_with("macro_sentiment")
        mock_cache_set.assert_not_called()
        
        # Check the result
        self.assertEqual(result["sentiment"], 65)
        self.assertEqual(result["fed_impact"], "Neutral")
        self.assertEqual(result["inflation_risk"], "Moderate")
        
    @patch('agents.macro_cache.get')
    @patch('agents.macro_cache.set')
    def test_analyze_with_cache_miss(self, mock_cache_set, mock_cache_get):
        """Test the analyze method with a cache miss."""
        # Mock the cache miss
        mock_cache_get.return_value = None
        
        # Create the analyst
        analyst = MacroSentimentAnalyst()
        
        # Call the analyze method
        result = analyst.analyze()
        
        # Check that the cache was checked and the result was cached
        mock_cache_get.assert_called_once_with("macro_sentiment")
        mock_cache_set.assert_called_once()
        
        # Check the result
        self.assertIn("sentiment", result)
        self.assertIn("fed_impact", result)
        self.assertIn("inflation_risk", result)
        self.assertIn("gdp_growth", result)
        self.assertIn("gdp_outlook", result)
        
if __name__ == "__main__":
    unittest.main()
