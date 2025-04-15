"""
Utility functions for trading strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional

def calculate_moving_average(prices: List[float], window: int) -> List[float]:
    """
    Calculate the moving average of a list of prices.

    Args:
        prices: List of prices
        window: Window size for the moving average

    Returns:
        List of moving averages
    """
    if len(prices) < window:
        return [np.nan] * len(prices)

    # Convert to numpy array for faster computation
    prices_array = np.array(prices)

    # Calculate the moving average
    ma = np.convolve(prices_array, np.ones(window)/window, mode='valid')

    # Pad the beginning with NaN values
    padding = [np.nan] * (len(prices) - len(ma))

    return padding + ma.tolist()

def calculate_rsi(prices: List[float], window: int = 14) -> List[float]:
    """
    Calculate the Relative Strength Index (RSI) of a list of prices.

    Args:
        prices: List of prices
        window: Window size for the RSI calculation

    Returns:
        List of RSI values
    """
    if len(prices) <= window:
        return [np.nan] * len(prices)

    # Convert to numpy array for faster computation
    prices_array = np.array(prices)

    # Calculate price changes
    deltas = np.diff(prices_array)

    # Calculate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gains and losses
    avg_gain = np.convolve(gains, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')

    # Calculate RS and RSI
    rs = avg_gain / np.where(avg_loss == 0, 0.001, avg_loss)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Pad the beginning with NaN values
    padding = [np.nan] * (len(prices) - len(rsi))

    return padding + rsi.tolist()

def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) of a list of prices.

    Args:
        prices: List of prices
        fast_period: Period for the fast EMA
        slow_period: Period for the slow EMA
        signal_period: Period for the signal line

    Returns:
        Dictionary containing MACD line, signal line, and histogram
    """
    if len(prices) <= slow_period:
        result = {
            'macd': [np.nan] * len(prices),
            'signal': [np.nan] * len(prices),
            'histogram': [np.nan] * len(prices)
        }
        return result

    # Convert to pandas Series for easier EMA calculation
    prices_series = pd.Series(prices)

    # Calculate fast and slow EMAs
    fast_ema = prices_series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices_series.ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    result = {
        'macd': macd_line.tolist(),
        'signal': signal_line.tolist(),
        'histogram': histogram.tolist()
    }

    return result

def get_strategy_description(strategy: str) -> str:
    """Get a description of a trading strategy."""
    if strategy == "ma_cross" or strategy == "Moving Average Crossover":
        return """
        The Moving Average Crossover strategy generates buy signals when the short-term moving average crosses above the long-term moving average, and sell signals when it crosses below.

        This strategy works well in trending markets but may generate false signals in sideways or choppy markets.
        """
    elif strategy == "rsi" or strategy == "RSI":
        return """
        The Relative Strength Index (RSI) strategy generates buy signals when the RSI falls below the oversold threshold, and sell signals when it rises above the overbought threshold.

        This strategy works well in range-bound markets but may underperform in strongly trending markets.
        """
    elif strategy == "macd" or strategy == "MACD":
        return """
        The Moving Average Convergence Divergence (MACD) strategy generates buy signals when the MACD line crosses above the signal line, and sell signals when it crosses below.

        This strategy aims to capture momentum and trend changes, and can work in both trending and range-bound markets.
        """
    else:
        return "Unknown strategy"
