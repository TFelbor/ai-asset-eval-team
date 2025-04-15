"""
Direct data fetching implementation for the AI Finance Dashboard.
This module provides a more reliable approach for fetching price history data.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import traceback

def fetch_stock_price_history(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Fetch stock price history directly from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary containing price history data
    """
    try:
        # Fetch data from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        
        if hist.empty:
            return {"error": f"No historical data available for {ticker}"}
        
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
        
        # Print debug info
        print(f"Successfully fetched price history for {ticker} from Yahoo Finance")
        print(f"Data points: {len(price_history['timestamps'])}")
        
        return price_history
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {str(e)}")
        traceback.print_exc()
        return {"error": f"Failed to fetch price history: {str(e)}"}

def fetch_crypto_price_history(ticker: str, days: int = 30) -> Dict[str, Any]:
    """
    Fetch cryptocurrency price history directly from Yahoo Finance.
    
    Args:
        ticker: Cryptocurrency ticker symbol (e.g., BTC-USD)
        days: Number of days of history to fetch
        
    Returns:
        Dictionary containing price history data
    """
    try:
        # Convert ticker to Yahoo Finance format if needed
        if "-USD" not in ticker:
            ticker = f"{ticker}-USD"
        
        # Fetch data from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=f"{days}d")
        
        if hist.empty:
            return {"error": f"No historical data available for {ticker}"}
        
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
        
        # Print debug info
        print(f"Successfully fetched price history for {ticker} from Yahoo Finance")
        print(f"Data points: {len(price_history['timestamps'])}")
        
        return price_history
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {str(e)}")
        traceback.print_exc()
        return {"error": f"Failed to fetch price history: {str(e)}"}

def fetch_price_history(ticker: str, asset_type: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Fetch price history for any asset type.
    
    Args:
        ticker: Asset ticker symbol
        asset_type: Asset type (stock, crypto, reit, etf)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary containing price history data
    """
    try:
        # For crypto, use the crypto-specific function
        if asset_type.lower() == "crypto" or asset_type.lower() == "cryptocurrency":
            # Convert period to days
            days = 30  # Default
            if period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            elif period == "1mo":
                days = 30
            elif period == "3mo":
                days = 90
            elif period == "6mo":
                days = 180
            elif period == "1y":
                days = 365
            elif period == "2y":
                days = 730
            elif period == "5y":
                days = 1825
            
            return fetch_crypto_price_history(ticker, days)
        
        # For all other asset types, use the stock function
        return fetch_stock_price_history(ticker, period)
    except Exception as e:
        print(f"Error in fetch_price_history for {ticker} ({asset_type}): {str(e)}")
        traceback.print_exc()
        return {"error": f"Failed to fetch price history: {str(e)}"}
