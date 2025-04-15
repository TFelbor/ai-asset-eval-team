"""
Backtesting module for financial strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import io
import base64

class BacktestStrategy:
    """Base class for backtesting strategies."""

    def __init__(self, name: str):
        """Initialize the strategy."""
        self.name = name
        self.positions = {}  # Current positions
        self.trades = []  # Trade history

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.

        Args:
            data: Historical price data

        Returns:
            Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run a backtest of the strategy.

        Args:
            data: Historical price data with 'Open', 'High', 'Low', 'Close' columns
            initial_capital: Starting capital

        Returns:
            Dictionary of backtest results
        """
        # Generate signals
        signals = self.generate_signals(data)

        # Initialize portfolio and tracking variables
        portfolio = pd.DataFrame(index=data.index)
        portfolio['signal'] = signals
        portfolio['price'] = data['Close']
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = initial_capital

        # Track positions and trades
        self.positions = {}
        self.trades = []

        # Simulate trading
        for i in range(1, len(portfolio)):
            # Update portfolio value
            portfolio.loc[portfolio.index[i], 'holdings'] = portfolio.loc[portfolio.index[i-1], 'holdings']
            portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']

            # Check for buy signal
            if portfolio.loc[portfolio.index[i], 'signal'] == 1 and portfolio.loc[portfolio.index[i-1], 'signal'] != 1:
                # Calculate shares to buy (use 95% of available cash)
                available_cash = portfolio.loc[portfolio.index[i], 'cash'] * 0.95
                price = portfolio.loc[portfolio.index[i], 'price']
                shares = int(available_cash / price)

                if shares > 0:
                    # Record trade
                    trade = {
                        'date': portfolio.index[i],
                        'type': 'buy',
                        'price': price,
                        'shares': shares,
                        'value': shares * price
                    }
                    self.trades.append(trade)

                    # Update portfolio
                    portfolio.loc[portfolio.index[i], 'cash'] -= shares * price
                    portfolio.loc[portfolio.index[i], 'holdings'] += shares

            # Check for sell signal
            elif portfolio.loc[portfolio.index[i], 'signal'] == -1 and portfolio.loc[portfolio.index[i-1], 'signal'] != -1:
                # Sell all shares
                shares = portfolio.loc[portfolio.index[i], 'holdings']
                price = portfolio.loc[portfolio.index[i], 'price']

                if shares > 0:
                    # Record trade
                    trade = {
                        'date': portfolio.index[i],
                        'type': 'sell',
                        'price': price,
                        'shares': shares,
                        'value': shares * price
                    }
                    self.trades.append(trade)

                    # Update portfolio
                    portfolio.loc[portfolio.index[i], 'cash'] += shares * price
                    portfolio.loc[portfolio.index[i], 'holdings'] = 0

            # Update total value
            portfolio.loc[portfolio.index[i], 'total'] = (
                portfolio.loc[portfolio.index[i], 'cash'] +
                portfolio.loc[portfolio.index[i], 'holdings'] * portfolio.loc[portfolio.index[i], 'price']
            )

        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()

        # Calculate performance metrics
        total_return = (portfolio['total'].iloc[-1] / initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std() if portfolio['returns'].std() > 0 else 0
        max_drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1).min()

        # Calculate benchmark returns (buy and hold)
        benchmark_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        benchmark_annual_return = (1 + benchmark_return) ** (252 / len(data)) - 1

        # Compile results
        results = {
            'strategy_name': self.name,
            'initial_capital': initial_capital,
            'final_value': portfolio['total'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_return,
            'benchmark_annual_return': benchmark_annual_return,
            'alpha': annual_return - benchmark_annual_return,
            'beta': portfolio['returns'].cov(data['Close'].pct_change()) / data['Close'].pct_change().var() if data['Close'].pct_change().var() > 0 else 1,
            'trades': self.trades,
            'num_trades': len(self.trades),
            'portfolio': portfolio
        }

        return results


class MovingAverageCrossStrategy(BacktestStrategy):
    """Moving average crossover strategy."""

    def __init__(self, short_window: int = 50, long_window: int = 200):
        """
        Initialize the strategy.

        Args:
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__(f"MA Cross ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossovers.

        Args:
            data: Historical price data

        Returns:
            Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()

        # Generate signals
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal

        return signals


class RSIStrategy(BacktestStrategy):
    """Relative Strength Index (RSI) strategy."""

    def __init__(self, window: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialize the strategy.

        Args:
            window: RSI calculation window
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__(f"RSI ({window}, {oversold}/{overbought})")
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI.

        Args:
            data: Historical price data

        Returns:
            Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals[rsi < self.oversold] = 1  # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought

        return signals


class MACDStrategy(BacktestStrategy):
    """Moving Average Convergence Divergence (MACD) strategy."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize the strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__(f"MACD ({fast_period}/{slow_period}/{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD.

        Args:
            data: Historical price data

        Returns:
            Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)

        # Calculate MACD
        fast_ema = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()

        # Generate signals
        signals[macd > signal_line] = 1  # Buy when MACD crosses above signal line
        signals[macd < signal_line] = -1  # Sell when MACD crosses below signal line

        return signals


def run_backtest(ticker: str, strategy_name: str, params: Dict[str, Any], period: str = "5y") -> Dict[str, Any]:
    """
    Run a backtest for a given ticker and strategy.

    Args:
        ticker: Stock ticker symbol
        strategy_name: Name of the strategy to use
        params: Strategy parameters
        period: Historical data period

    Returns:
        Dictionary of backtest results
    """
    try:
        # Get historical data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            return {"error": "No historical data available"}

        # Create strategy
        if strategy_name == "ma_cross":
            strategy = MovingAverageCrossStrategy(
                short_window=params.get("short_window", 50),
                long_window=params.get("long_window", 200)
            )
        elif strategy_name == "rsi":
            strategy = RSIStrategy(
                window=params.get("window", 14),
                overbought=params.get("overbought", 70),
                oversold=params.get("oversold", 30)
            )
        elif strategy_name == "macd":
            strategy = MACDStrategy(
                fast_period=params.get("fast_period", 12),
                slow_period=params.get("slow_period", 26),
                signal_period=params.get("signal_period", 9)
            )
        else:
            return {"error": f"Unknown strategy: {strategy_name}"}

        # Run backtest
        results = strategy.backtest(data)

        # Generate chart
        chart_data = generate_backtest_chart(data, results)

        # Add chart to results
        results["chart"] = chart_data

        # Add ticker and strategy info
        results["ticker"] = ticker
        results["strategy"] = strategy_name
        results["params"] = params

        return results
    except Exception as e:
        return {"error": str(e)}


def generate_backtest_chart(data: pd.DataFrame, results: Dict[str, Any]) -> str:
    """
    Generate a chart for backtest results.

    Args:
        data: Historical price data
        results: Backtest results

    Returns:
        Base64-encoded chart image
    """
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot price and portfolio value
        portfolio = results["portfolio"]
        ax1.plot(data.index, data["Close"], label="Price", color="gray", alpha=0.7)
        ax1.plot(portfolio.index, portfolio["total"] / results["initial_capital"] * data["Close"].iloc[0],
                label="Portfolio Value", color="blue")

        # Plot buy and sell signals
        buy_signals = portfolio[portfolio["signal"] == 1].index
        sell_signals = portfolio[portfolio["signal"] == -1].index

        ax1.scatter(buy_signals, data.loc[buy_signals, "Close"], marker="^", color="green", s=100, label="Buy")
        ax1.scatter(sell_signals, data.loc[sell_signals, "Close"], marker="v", color="red", s=100, label="Sell")

        # Format first subplot
        # Use strategy instead of strategy_name (which doesn't exist in results)
        strategy_name = results.get('strategy', 'Unknown')
        ticker = results.get('ticker', 'Unknown')
        ax1.set_title(f"{ticker} - {strategy_name} Backtest")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot drawdown
        drawdown = (portfolio["total"] / portfolio["total"].cummax() - 1) * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color="red", label="Drawdown %")

        # Format second subplot
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Format dates
        fig.autofmt_xdate()

        # Add performance metrics as text
        metrics_text = (
            f"Total Return: {results['total_return']*100:.2f}%\n"
            f"Annual Return: {results['annual_return']*100:.2f}%\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']*100:.2f}%\n"
            f"Benchmark Return: {results['benchmark_return']*100:.2f}%\n"
            f"Alpha: {results['alpha']*100:.2f}%\n"
            f"Number of Trades: {results['num_trades']}"
        )

        plt.figtext(0.01, 0.01, metrics_text, fontsize=10, va="bottom")

        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        # Encode as base64 and format as a data URL
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        img_str = f"data:image/png;base64,{img_str}"

        # Close figure to free memory
        plt.close(fig)

        return img_str
    except Exception as e:
        print(f"Error generating backtest chart: {str(e)}")
        return ""
