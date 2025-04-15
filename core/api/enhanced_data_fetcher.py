"""
Enhanced data fetching module for financial analysis.
This module provides enhanced data fetching tools from multiple sources.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
import time
import os
import json
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from pycoingecko import CoinGeckoAPI
import pandas_datareader as pdr

class EnhancedDataFetcher:
    """
    A class that provides enhanced data fetching tools from multiple sources.
    """
    
    def __init__(self, alpha_vantage_api_key: str = None, coingecko_api_key: str = None):
        """
        Initialize the EnhancedDataFetcher class.
        
        Args:
            alpha_vantage_api_key: Alpha Vantage API key
            coingecko_api_key: CoinGecko API key
        """
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.coingecko_api_key = coingecko_api_key
        
        # Initialize API clients
        if alpha_vantage_api_key:
            self.alpha_vantage_ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
            self.alpha_vantage_fd = FundamentalData(key=alpha_vantage_api_key, output_format='pandas')
        
        self.coingecko = CoinGeckoAPI(api_key=coingecko_api_key)
        
        # Initialize cache
        self.cache = {}
        
    def get_stock_data(self, ticker: str, period: str = '1y', interval: str = '1d', 
                      source: str = 'yahoo') -> pd.DataFrame:
        """
        Get stock data from various sources.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Time interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            source: Data source ('yahoo', 'alpha_vantage', 'iex', 'fred')
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Check cache
            cache_key = f"stock_{ticker}_{period}_{interval}_{source}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 hour old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
                    return self.cache[cache_key]['data']
            
            # Fetch data based on source
            if source == 'yahoo':
                data = self._get_yahoo_stock_data(ticker, period, interval)
            elif source == 'alpha_vantage':
                data = self._get_alpha_vantage_stock_data(ticker, interval)
            elif source == 'iex':
                data = self._get_iex_stock_data(ticker, period)
            elif source == 'fred':
                data = self._get_fred_data(ticker)
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            # Cache data
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            raise ValueError(f"Stock data fetching failed: {str(e)}")
    
    def _get_yahoo_stock_data(self, ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            return data
        except Exception as e:
            raise ValueError(f"Yahoo Finance data fetching failed: {str(e)}")
    
    def _get_alpha_vantage_stock_data(self, ticker: str, interval: str = '1d') -> pd.DataFrame:
        """Get stock data from Alpha Vantage"""
        if not self.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key not provided")
            
        try:
            # Map interval to Alpha Vantage format
            av_interval = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '60m': '60min',
                '1h': '60min',
                '1d': 'daily',
                '1wk': 'weekly',
                '1mo': 'monthly'
            }.get(interval, 'daily')
            
            # Fetch data
            if av_interval in ['1min', '5min', '15min', '30min', '60min']:
                data, meta_data = self.alpha_vantage_ts.get_intraday(symbol=ticker, interval=av_interval, outputsize='full')
            elif av_interval == 'daily':
                data, meta_data = self.alpha_vantage_ts.get_daily(symbol=ticker, outputsize='full')
            elif av_interval == 'weekly':
                data, meta_data = self.alpha_vantage_ts.get_weekly(symbol=ticker)
            elif av_interval == 'monthly':
                data, meta_data = self.alpha_vantage_ts.get_monthly(symbol=ticker)
            
            # Rename columns
            data.columns = [col.lower().split(' ')[1] for col in data.columns]
            
            # Reset index
            data = data.reset_index()
            data = data.rename(columns={'index': 'date'})
            data = data.set_index('date')
            
            return data
        except Exception as e:
            raise ValueError(f"Alpha Vantage data fetching failed: {str(e)}")
    
    def _get_iex_stock_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """Get stock data from IEX Cloud"""
        try:
            # Map period to IEX format
            end_date = datetime.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=2*365)
            elif period == '5y':
                start_date = end_date - timedelta(days=5*365)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Fetch data
            data = pdr.get_data_iex(ticker, start=start_date, end=end_date)
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            return data
        except Exception as e:
            raise ValueError(f"IEX data fetching failed: {str(e)}")
    
    def _get_fred_data(self, ticker: str) -> pd.DataFrame:
        """Get data from FRED"""
        try:
            # Fetch data
            data = pdr.get_data_fred(ticker)
            
            # Rename column to 'close'
            data = data.rename(columns={ticker: 'close'})
            
            return data
        except Exception as e:
            raise ValueError(f"FRED data fetching failed: {str(e)}")
    
    def get_crypto_data(self, ticker: str, vs_currency: str = 'usd', days: int = 365, 
                       interval: str = 'daily', source: str = 'coingecko') -> pd.DataFrame:
        """
        Get cryptocurrency data from various sources.
        
        Args:
            ticker: Cryptocurrency ticker symbol
            vs_currency: Quote currency
            days: Number of days of data to fetch
            interval: Time interval ('daily', 'hourly', 'minutely')
            source: Data source ('coingecko', 'ccxt')
            
        Returns:
            DataFrame with cryptocurrency data
        """
        try:
            # Check cache
            cache_key = f"crypto_{ticker}_{vs_currency}_{days}_{interval}_{source}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 hour old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
                    return self.cache[cache_key]['data']
            
            # Fetch data based on source
            if source == 'coingecko':
                data = self._get_coingecko_crypto_data(ticker, vs_currency, days, interval)
            elif source == 'ccxt':
                data = self._get_ccxt_crypto_data(ticker, vs_currency, days, interval)
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            # Cache data
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            raise ValueError(f"Cryptocurrency data fetching failed: {str(e)}")
    
    def _get_coingecko_crypto_data(self, ticker: str, vs_currency: str = 'usd', 
                                  days: int = 365, interval: str = 'daily') -> pd.DataFrame:
        """Get cryptocurrency data from CoinGecko"""
        try:
            # Convert ticker to CoinGecko ID if needed
            coin_id = self._get_coingecko_coin_id(ticker)
            
            # Fetch market data
            market_data = self.coingecko.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days,
                interval=interval
            )
            
            # Extract price data
            prices = market_data['prices']
            volumes = market_data['total_volumes']
            market_caps = market_data['market_caps']
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['volume'] = [v[1] for v in volumes]
            df['market_cap'] = [m[1] for m in market_caps]
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Rename columns to match OHLCV format
            df = df.rename(columns={'price': 'close'})
            
            # Add placeholder columns for OHLC
            df['open'] = df['close'].shift(1)
            df['high'] = df['close']
            df['low'] = df['close']
            
            # Fill NaN values
            df = df.fillna(method='bfill')
            
            return df
        except Exception as e:
            raise ValueError(f"CoinGecko data fetching failed: {str(e)}")
    
    def _get_coingecko_coin_id(self, ticker: str) -> str:
        """Convert ticker to CoinGecko ID"""
        try:
            # Check if ticker is already a CoinGecko ID
            if ticker.lower() in ['bitcoin', 'ethereum', 'litecoin', 'ripple', 'cardano', 'polkadot']:
                return ticker.lower()
            
            # Map common tickers to CoinGecko IDs
            ticker_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'LTC': 'litecoin',
                'XRP': 'ripple',
                'ADA': 'cardano',
                'DOT': 'polkadot',
                'LINK': 'chainlink',
                'XLM': 'stellar',
                'DOGE': 'dogecoin',
                'UNI': 'uniswap',
                'AAVE': 'aave',
                'SOL': 'solana',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network',
                'ATOM': 'cosmos',
                'ALGO': 'algorand'
            }
            
            if ticker.upper() in ticker_map:
                return ticker_map[ticker.upper()]
            
            # If not found in map, search for coin
            coins = self.coingecko.get_coins_list()
            for coin in coins:
                if coin['symbol'].lower() == ticker.lower():
                    return coin['id']
                
            # If still not found, return as is
            return ticker.lower()
        except Exception as e:
            raise ValueError(f"CoinGecko coin ID conversion failed: {str(e)}")
    
    def _get_ccxt_crypto_data(self, ticker: str, vs_currency: str = 'usd', 
                             days: int = 365, interval: str = 'daily') -> pd.DataFrame:
        """Get cryptocurrency data from CCXT"""
        try:
            # Initialize exchange
            exchange = ccxt.binance()
            
            # Map interval to CCXT timeframe
            timeframe = {
                'minutely': '1m',
                'hourly': '1h',
                'daily': '1d'
            }.get(interval, '1d')
            
            # Format symbol
            symbol = f"{ticker.upper()}/{vs_currency.upper()}"
            
            # Calculate start time
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            # Create DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            return df
        except Exception as e:
            raise ValueError(f"CCXT data fetching failed: {str(e)}")
    
    def get_fundamental_data(self, ticker: str, data_type: str = 'overview', 
                            source: str = 'alpha_vantage') -> pd.DataFrame:
        """
        Get fundamental data for a stock.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of fundamental data ('overview', 'income', 'balance', 'cash', 'earnings')
            source: Data source ('alpha_vantage', 'yahoo')
            
        Returns:
            DataFrame with fundamental data
        """
        try:
            # Check cache
            cache_key = f"fundamental_{ticker}_{data_type}_{source}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 day old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(days=1):
                    return self.cache[cache_key]['data']
            
            # Fetch data based on source
            if source == 'alpha_vantage':
                data = self._get_alpha_vantage_fundamental_data(ticker, data_type)
            elif source == 'yahoo':
                data = self._get_yahoo_fundamental_data(ticker, data_type)
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            # Cache data
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            raise ValueError(f"Fundamental data fetching failed: {str(e)}")
    
    def _get_alpha_vantage_fundamental_data(self, ticker: str, data_type: str = 'overview') -> pd.DataFrame:
        """Get fundamental data from Alpha Vantage"""
        if not self.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key not provided")
            
        try:
            # Fetch data based on type
            if data_type == 'overview':
                data, meta_data = self.alpha_vantage_fd.get_company_overview(symbol=ticker)
            elif data_type == 'income':
                data, meta_data = self.alpha_vantage_fd.get_income_statement_annual(symbol=ticker)
            elif data_type == 'balance':
                data, meta_data = self.alpha_vantage_fd.get_balance_sheet_annual(symbol=ticker)
            elif data_type == 'cash':
                data, meta_data = self.alpha_vantage_fd.get_cash_flow_annual(symbol=ticker)
            elif data_type == 'earnings':
                data, meta_data = self.alpha_vantage_fd.get_earnings_annual(symbol=ticker)
            else:
                raise ValueError(f"Unknown fundamental data type: {data_type}")
            
            return data
        except Exception as e:
            raise ValueError(f"Alpha Vantage fundamental data fetching failed: {str(e)}")
    
    def _get_yahoo_fundamental_data(self, ticker: str, data_type: str = 'overview') -> pd.DataFrame:
        """Get fundamental data from Yahoo Finance"""
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            
            # Extract data based on type
            if data_type == 'overview':
                data = pd.DataFrame.from_dict(stock.info, orient='index').T
            elif data_type == 'income':
                data = stock.income_stmt
            elif data_type == 'balance':
                data = stock.balance_sheet
            elif data_type == 'cash':
                data = stock.cashflow
            elif data_type == 'earnings':
                data = stock.earnings
            else:
                raise ValueError(f"Unknown fundamental data type: {data_type}")
            
            return data
        except Exception as e:
            raise ValueError(f"Yahoo Finance fundamental data fetching failed: {str(e)}")
    
    def get_economic_data(self, indicator: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get economic data from FRED.
        
        Args:
            indicator: Economic indicator code
            start_date: Start date (format: 'YYYY-MM-DD')
            end_date: End date (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with economic data
        """
        try:
            # Check cache
            cache_key = f"economic_{indicator}_{start_date}_{end_date}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 day old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(days=1):
                    return self.cache[cache_key]['data']
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch data
            data = pdr.get_data_fred(indicator, start=start_date, end=end_date)
            
            # Rename column
            data = data.rename(columns={indicator: 'value'})
            
            # Cache data
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            raise ValueError(f"Economic data fetching failed: {str(e)}")
    
    def get_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a ticker.
        
        Args:
            ticker: Ticker symbol
            limit: Maximum number of news items to return
            
        Returns:
            List of news items
        """
        try:
            # Check cache
            cache_key = f"news_{ticker}_{limit}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 hour old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
                    return self.cache[cache_key]['data']
            
            # Fetch news from Yahoo Finance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Limit number of news items
            news = news[:limit]
            
            # Cache data
            self.cache[cache_key] = {
                'data': news,
                'timestamp': datetime.now()
            }
            
            return news
        except Exception as e:
            raise ValueError(f"News fetching failed: {str(e)}")
    
    def get_multiple_tickers_data(self, tickers: List[str], period: str = '1y', 
                                 interval: str = '1d', source: str = 'yahoo') -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Time period
            interval: Time interval
            source: Data source
            
        Returns:
            Dict of DataFrames with ticker data
        """
        try:
            # Check cache
            cache_key = f"multiple_{'-'.join(tickers)}_{period}_{interval}_{source}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 hour old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(hours=1):
                    return self.cache[cache_key]['data']
            
            # Fetch data for each ticker
            data = {}
            for ticker in tickers:
                try:
                    data[ticker] = self.get_stock_data(ticker, period, interval, source)
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {str(e)}")
            
            # Cache data
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            raise ValueError(f"Multiple tickers data fetching failed: {str(e)}")
    
    def get_etf_holdings(self, ticker: str) -> pd.DataFrame:
        """
        Get ETF holdings.
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            DataFrame with ETF holdings
        """
        try:
            # Check cache
            cache_key = f"etf_holdings_{ticker}"
            if cache_key in self.cache:
                # Check if cache is still valid (less than 1 day old)
                if datetime.now() - self.cache[cache_key]['timestamp'] < timedelta(days=1):
                    return self.cache[cache_key]['data']
            
            # Fetch ETF holdings from Yahoo Finance
            etf = yf.Ticker(ticker)
            holdings = etf.get_holdings()
            
            # Cache data
            self.cache[cache_key] = {
                'data': holdings,
                'timestamp': datetime.now()
            }
            
            return holdings
        except Exception as e:
            raise ValueError(f"ETF holdings fetching failed: {str(e)}")
    
    def get_market_index_data(self, index_ticker: str, period: str = '1y', 
                             interval: str = '1d') -> pd.DataFrame:
        """
        Get market index data.
        
        Args:
            index_ticker: Market index ticker symbol
            period: Time period
            interval: Time interval
            
        Returns:
            DataFrame with market index data
        """
        try:
            # Map common index names to tickers
            index_map = {
                'S&P500': '^GSPC',
                'DOW': '^DJI',
                'NASDAQ': '^IXIC',
                'RUSSELL2000': '^RUT',
                'VIX': '^VIX',
                'FTSE100': '^FTSE',
                'DAX': '^GDAXI',
                'NIKKEI': '^N225'
            }
            
            # Convert index name to ticker if needed
            if index_ticker in index_map:
                index_ticker = index_map[index_ticker]
            
            # Fetch data
            return self.get_stock_data(index_ticker, period, interval)
        except Exception as e:
            raise ValueError(f"Market index data fetching failed: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache = {}
