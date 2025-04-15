"""
Advanced backtesting module for financial analysis.
This module provides advanced backtesting tools using backtrader.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import quantstats as qs
import empyrical as ep
from typing import Dict, List, Union, Optional, Tuple, Any
import os
import datetime

class AdvancedBacktesting:
    """
    A class that provides advanced backtesting tools.
    """
    
    def __init__(self, cash: float = 100000.0, commission: float = 0.001):
        """
        Initialize the AdvancedBacktesting class.
        
        Args:
            cash: Initial cash
            commission: Commission rate
        """
        self.cash = cash
        self.commission = commission
        self.cerebro = None
        self.results = None
        self.strategies = {}
        
    def register_strategy(self, strategy_class: bt.Strategy, name: str, params: Dict[str, Any] = None):
        """
        Register a strategy for backtesting.
        
        Args:
            strategy_class: Backtrader strategy class
            name: Strategy name
            params: Strategy parameters
        """
        self.strategies[name] = {
            'class': strategy_class,
            'params': params or {}
        }
    
    def prepare_data(self, data: pd.DataFrame, date_col: str = None, open_col: str = 'open', 
                    high_col: str = 'high', low_col: str = 'low', close_col: str = 'close', 
                    volume_col: str = 'volume', openinterest_col: str = None) -> bt.feeds.PandasData:
        """
        Prepare data for backtesting.
        
        Args:
            data: DataFrame with OHLCV data
            date_col: Date column name
            open_col: Open column name
            high_col: High column name
            low_col: Low column name
            close_col: Close column name
            volume_col: Volume column name
            openinterest_col: Open interest column name
            
        Returns:
            Backtrader data feed
        """
        if data.empty:
            raise ValueError("Empty data provided")
            
        try:
            # Create a copy of the data
            df = data.copy()
            
            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Map column names to backtrader names
            cols_map = {}
            if date_col:
                cols_map['datetime'] = date_col.lower()
            if open_col:
                cols_map['open'] = open_col.lower()
            if high_col:
                cols_map['high'] = high_col.lower()
            if low_col:
                cols_map['low'] = low_col.lower()
            if close_col:
                cols_map['close'] = close_col.lower()
            if volume_col:
                cols_map['volume'] = volume_col.lower()
            if openinterest_col:
                cols_map['openinterest'] = openinterest_col.lower()
            
            # Create a custom pandas data feed
            class CustomPandasData(bt.feeds.PandasData):
                params = {
                    'datetime': None if date_col is None else -1,
                    'open': -1 if open_col is None else df.columns.get_loc(open_col.lower()),
                    'high': -1 if high_col is None else df.columns.get_loc(high_col.lower()),
                    'low': -1 if low_col is None else df.columns.get_loc(low_col.lower()),
                    'close': -1 if close_col is None else df.columns.get_loc(close_col.lower()),
                    'volume': -1 if volume_col is None else df.columns.get_loc(volume_col.lower()),
                    'openinterest': -1 if openinterest_col is None else df.columns.get_loc(openinterest_col.lower())
                }
            
            # Create data feed
            data_feed = CustomPandasData(dataname=df)
            
            return data_feed
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")
    
    def run_backtest(self, data: Union[bt.feeds.PandasData, List[bt.feeds.PandasData]], 
                    strategy_name: str = None, strategy_params: Dict[str, Any] = None,
                    start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            data: Backtrader data feed or list of data feeds
            strategy_name: Strategy name (must be registered)
            strategy_params: Strategy parameters (overrides registered parameters)
            start_date: Start date for backtest (format: 'YYYY-MM-DD')
            end_date: End date for backtest (format: 'YYYY-MM-DD')
            
        Returns:
            Dict of backtest results
        """
        try:
            # Create a new cerebro engine
            self.cerebro = bt.Cerebro()
            
            # Set initial cash
            self.cerebro.broker.setcash(self.cash)
            
            # Set commission
            self.cerebro.broker.setcommission(commission=self.commission)
            
            # Add data
            if isinstance(data, list):
                for d in data:
                    self.cerebro.adddata(d)
            else:
                self.cerebro.adddata(data)
            
            # Set start and end dates if provided
            if start_date:
                self.cerebro.addfilter(bt.filters.DateFilter, fromdate=datetime.datetime.strptime(start_date, '%Y-%m-%d'))
            if end_date:
                self.cerebro.addfilter(bt.filters.DateFilter, todate=datetime.datetime.strptime(end_date, '%Y-%m-%d'))
            
            # Add strategy
            if strategy_name:
                if strategy_name not in self.strategies:
                    raise ValueError(f"Strategy '{strategy_name}' not registered")
                
                strategy_class = self.strategies[strategy_name]['class']
                params = self.strategies[strategy_name]['params'].copy()
                
                # Override parameters if provided
                if strategy_params:
                    params.update(strategy_params)
                
                self.cerebro.addstrategy(strategy_class, **params)
            
            # Add analyzers
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
            self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
            
            # Run the backtest
            self.results = self.cerebro.run()
            
            # Extract results
            strategy = self.results[0]
            
            # Get analyzer results
            sharpe = strategy.analyzers.sharpe.get_analysis()
            returns = strategy.analyzers.returns.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()
            trades = strategy.analyzers.trades.get_analysis()
            sqn = strategy.analyzers.sqn.get_analysis()
            time_return = strategy.analyzers.time_return.get_analysis()
            
            # Calculate additional metrics
            total_return = self.cerebro.broker.getvalue() / self.cash - 1
            
            # Compile results
            results = {
                'initial_cash': self.cash,
                'final_value': self.cerebro.broker.getvalue(),
                'total_return': total_return,
                'sharpe_ratio': sharpe.get('sharperatio', 0.0),
                'annual_return': returns.get('ravg', 0.0) * 252,  # Annualized return
                'max_drawdown': drawdown.get('max', {}).get('drawdown', 0.0),
                'max_drawdown_length': drawdown.get('max', {}).get('len', 0),
                'trades': {
                    'total': trades.get('total', 0),
                    'won': trades.get('won', 0),
                    'lost': trades.get('lost', 0),
                    'win_rate': trades.get('won', 0) / trades.get('total', 1) if trades.get('total', 0) > 0 else 0.0,
                    'avg_profit': trades.get('pnl', {}).get('average', 0.0),
                    'avg_win': trades.get('won', {}).get('pnl', {}).get('average', 0.0),
                    'avg_loss': trades.get('lost', {}).get('pnl', {}).get('average', 0.0),
                },
                'sqn': sqn.get('sqn', 0.0),
                'time_returns': {str(k): v for k, v in time_return.items()}
            }
            
            return results
        except Exception as e:
            raise ValueError(f"Backtest failed: {str(e)}")
    
    def plot_results(self, filename: str = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot backtest results.
        
        Args:
            filename: Output filename (if None, plot is displayed)
            figsize: Figure size
        """
        if self.cerebro is None or self.results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
            
        try:
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Plot results
            self.cerebro.plot(style='candlestick', barup='green', bardown='red', 
                             volup='green', voldown='red', 
                             fill_up=True, fill_down=True,
                             figsize=figsize)
            
            # Save figure if filename is provided
            if filename:
                plt.savefig(filename)
                plt.close(fig)
        except Exception as e:
            raise ValueError(f"Results plotting failed: {str(e)}")
    
    def generate_report(self, returns_series: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                       output_file: str = None) -> None:
        """
        Generate a performance report using quantstats.
        
        Args:
            returns_series: Series of strategy returns
            benchmark_returns: Series of benchmark returns (optional)
            output_file: Output file path (if None, report is displayed in browser)
        """
        if returns_series.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Generate report
            if benchmark_returns is not None:
                qs.reports.html(returns_series, benchmark_returns, output=output_file)
            else:
                qs.reports.html(returns_series, output=output_file)
        except Exception as e:
            raise ValueError(f"Report generation failed: {str(e)}")
    
    def extract_returns_series(self) -> pd.Series:
        """
        Extract returns series from backtest results.
        
        Returns:
            Series of strategy returns
        """
        if self.cerebro is None or self.results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
            
        try:
            # Get time returns from analyzer
            strategy = self.results[0]
            time_return = strategy.analyzers.time_return.get_analysis()
            
            # Convert to pandas Series
            returns = pd.Series({k: v for k, v in time_return.items()})
            
            return returns
        except Exception as e:
            raise ValueError(f"Returns extraction failed: {str(e)}")


# Define some common strategies

class MovingAverageCrossStrategy(bt.Strategy):
    """Moving Average Cross Strategy"""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('order_percentage', 0.95),
        ('stop_loss', 0.05),
        ('trail', False),
    )
    
    def __init__(self):
        # Initialize indicators
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Trading state
        self.order = None
        self.stop_order = None
        self.trail_order = None
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return
            
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        # Reset order
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        
    def next(self):
        # Check if an order is pending
        if self.order:
            return
            
        # Check if we are in the market
        if not self.position:
            # Buy signal
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                # Calculate position size
                size = int(self.broker.getcash() * self.params.order_percentage / self.data.close[0])
                # Create buy order
                self.order = self.buy(size=size)
                
                # Set stop loss if enabled
                if self.params.stop_loss > 0:
                    stop_price = self.data.close[0] * (1.0 - self.params.stop_loss)
                    self.stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_price)
                    
                # Set trailing stop if enabled
                if self.params.trail:
                    self.trail_order = self.sell(size=size, exectype=bt.Order.StopTrail, trailpercent=self.params.stop_loss)
        else:
            # Sell signal
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                # Close position
                self.order = self.close()
                
                # Cancel stop orders
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
                    
                if self.trail_order:
                    self.cancel(self.trail_order)
                    self.trail_order = None
                    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')


class RSIStrategy(bt.Strategy):
    """RSI Strategy"""
    
    params = (
        ('period', 14),
        ('overbought', 70),
        ('oversold', 30),
        ('order_percentage', 0.95),
        ('stop_loss', 0.05),
    )
    
    def __init__(self):
        # Initialize indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)
        
        # Trading state
        self.order = None
        self.stop_order = None
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return
            
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        # Reset order
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        
    def next(self):
        # Check if an order is pending
        if self.order:
            return
            
        # Check if we are in the market
        if not self.position:
            # Buy signal - RSI crosses below oversold level
            if self.rsi[0] < self.params.oversold:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                # Calculate position size
                size = int(self.broker.getcash() * self.params.order_percentage / self.data.close[0])
                # Create buy order
                self.order = self.buy(size=size)
                
                # Set stop loss if enabled
                if self.params.stop_loss > 0:
                    stop_price = self.data.close[0] * (1.0 - self.params.stop_loss)
                    self.stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_price)
        else:
            # Sell signal - RSI crosses above overbought level
            if self.rsi[0] > self.params.overbought:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                # Close position
                self.order = self.close()
                
                # Cancel stop order
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
                    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')


class BollingerBandsStrategy(bt.Strategy):
    """Bollinger Bands Strategy"""
    
    params = (
        ('period', 20),
        ('devfactor', 2),
        ('order_percentage', 0.95),
        ('stop_loss', 0.05),
    )
    
    def __init__(self):
        # Initialize indicators
        self.boll = bt.indicators.BollingerBands(self.data.close, period=self.params.period, devfactor=self.params.devfactor)
        
        # Trading state
        self.order = None
        self.stop_order = None
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return
            
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        # Reset order
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        
    def next(self):
        # Check if an order is pending
        if self.order:
            return
            
        # Check if we are in the market
        if not self.position:
            # Buy signal - price crosses below lower band
            if self.data.close[0] < self.boll.lines.bot[0]:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                # Calculate position size
                size = int(self.broker.getcash() * self.params.order_percentage / self.data.close[0])
                # Create buy order
                self.order = self.buy(size=size)
                
                # Set stop loss if enabled
                if self.params.stop_loss > 0:
                    stop_price = self.data.close[0] * (1.0 - self.params.stop_loss)
                    self.stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_price)
        else:
            # Sell signal - price crosses above upper band
            if self.data.close[0] > self.boll.lines.top[0]:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                # Close position
                self.order = self.close()
                
                # Cancel stop order
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None
                    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
