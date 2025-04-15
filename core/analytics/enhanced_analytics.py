"""
Enhanced analytics module incorporating advanced quantitative analysis tools
"""
import backtrader as bt
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import quantstats as qs
import riskfolio as rp
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging

# Fix imports by using relative paths from core
from ..api.yahoo_finance_client import YahooFinanceClient
from ..api.coingecko_client import CoinGeckoClient
from .ml_analysis import MLAnalyzer
from .advanced_metrics import AdvancedMetrics

logger = logging.getLogger(__name__)

class EnhancedAnalytics:
    def __init__(self):
        self.portfolio = None
        self.risk_metrics = None
        
    def optimize_portfolio(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Portfolio optimization using PyPortfolioOpt
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            
        Returns:
            Dict containing weights and performance metrics
        
        Raises:
            ValueError: If prices_df is empty or contains invalid data
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Optimize for maximum Sharpe ratio
            ef = EfficientFrontier(mu, S)
            weights = ef.maximum_sharpe()
            cleaned_weights = ef.clean_weights()
            
            # Get portfolio performance metrics
            expected_return, volatility, sharpe = ef.portfolio_performance()
            
            return {
                'weights': cleaned_weights,
                'performance': {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe
                }
            }
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise
    
    def calculate_risk_metrics(self, returns_series: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics using QuantStats
        
        Args:
            returns_series: Series of asset returns
            
        Returns:
            Dict of risk metrics
        """
        if returns_series.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            metrics = {
                'sharpe': qs.stats.sharpe(returns_series),
                'sortino': qs.stats.sortino(returns_series),
                'max_drawdown': qs.stats.max_drawdown(returns_series),
                'var': qs.stats.var(returns_series),
                'cvar': qs.stats.cvar(returns_series),
                'calmar': qs.stats.calmar(returns_series),
                'omega': qs.stats.omega(returns_series),
                'tail_ratio': qs.stats.tail_ratio(returns_series),
                'value_at_risk': qs.stats.value_at_risk(returns_series)
            }
            
            # Validate metrics
            return {k: v for k, v in metrics.items() if not (np.isnan(v) or np.isinf(v))}
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            raise
    
    def perform_technical_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Technical analysis using pandas-ta
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        if data.empty:
            raise ValueError("Empty price data provided")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        try:
            # Create custom strategy with error checking
            custom_strategy = ta.Strategy(
                name="Combined Strategy",
                description="RSI, MACD, and Bollinger Bands",
                ta=[
                    {"kind": "rsi", "length": 14},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "bbands", "length": 20, "std": 2}
                ]
            )
            
            # Calculate technical indicators
            data.ta.strategy(custom_strategy)
            
            # Verify indicators were calculated
            expected_columns = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Failed to calculate indicators: {missing_columns}")
                
            return data
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            raise

class AdvancedBacktesting(bt.Strategy):
    """Advanced backtesting strategy using backtrader"""
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('stop_loss', 0.02),  # 2% stop loss
        ('take_profit', 0.05),  # 5% take profit
        ('position_size', 0.1),  # 10% of portfolio per trade
    )
    
    def __init__(self):
        # Validate parameters
        if self.params.fast_period >= self.params.slow_period:
            raise ValueError("Fast period must be less than slow period")
            
        # Initialize indicators
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        
        # Additional indicators
        self.atr = bt.indicators.ATR(self.data)
        self.rsi = bt.indicators.RSI(self.data)
        
        # Trading state
        self.order = None
        self.stop_loss_order = None
        self.take_profit_order = None
        
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
        
    def next(self):
        """Main strategy logic"""
        # Don't enter if we have pending orders
        if self.order:
            return
            
        # Check for entry conditions
        if not self.position:
            if self.fast_ma > self.slow_ma and self.rsi < 70:
                # Calculate position size
                size = int(self.broker.get_cash() * self.params.position_size / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    # Set stop loss and take profit
                    stop_price = self.data.close[0] * (1 - self.params.stop_loss)
                    target_price = self.data.close[0] * (1 + self.params.take_profit)
                    self.stop_loss_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_price)
                    self.take_profit_order = self.sell(size=size, exectype=bt.Order.Limit, price=target_price)
                    
        # Check for exit conditions
        elif self.fast_ma < self.slow_ma or self.rsi > 70:
            self.close()
            
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
