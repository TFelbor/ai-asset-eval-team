"""
Technical indicators module for financial analysis.
This module provides a comprehensive set of technical indicators for financial analysis.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Union, Optional, Tuple

# Import enhanced logging
from core.utils.logger import (
    log_info, log_error, log_success, log_warning, log_debug,
    log_api_call, log_data_operation, log_analytics_operation
)

# Try to import TA-Lib, but make it optional
try:
    import talib
    TALIB_AVAILABLE = True
    log_info("TA-Lib is available and will be used for technical indicators")
except ImportError:
    TALIB_AVAILABLE = False
    log_info("TA-Lib is not available, using pandas_ta as fallback")

class TechnicalIndicators:
    """
    A class that provides technical indicators for financial analysis.
    Combines functionality from pandas_ta, TA-Lib, and custom implementations.
    """

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a comprehensive set of technical indicators.

        Args:
            data: DataFrame with OHLCV data (columns must include: open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators added
        """
        if data.empty:
            raise ValueError("Empty price data provided")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col.lower() in map(str.lower, data.columns) for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        # Ensure column names are lowercase
        data_lower = data.copy()
        data_lower.columns = [col.lower() for col in data_lower.columns]

        # Create a new DataFrame for indicators
        result = data.copy()

        # Add trend indicators
        result = TechnicalIndicators.add_trend_indicators(result)

        # Add momentum indicators
        result = TechnicalIndicators.add_momentum_indicators(result)

        # Add volatility indicators
        result = TechnicalIndicators.add_volatility_indicators(result)

        # Add volume indicators
        result = TechnicalIndicators.add_volume_indicators(result)

        # Add cycle indicators
        result = TechnicalIndicators.add_cycle_indicators(result)

        # Add pattern recognition
        result = TechnicalIndicators.add_pattern_recognition(result)

        # Clean up NaN values
        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    @staticmethod
    def add_trend_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Moving Averages
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)

        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)

        # Parabolic SAR
        df['psar'] = ta.psar(df['high'], df['low'], df['close'])

        # ADX - Average Directional Index
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df = pd.concat([df, adx], axis=1)

        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        df = pd.concat([df, ichimoku], axis=1)

        return df

    @staticmethod
    def add_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # RSI - Relative Strength Index
        df['rsi_14'] = ta.rsi(df['close'], length=14)

        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df = pd.concat([df, stoch], axis=1)

        # CCI - Commodity Channel Index
        df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

        # Williams %R
        df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)

        # ROC - Rate of Change
        df['roc_10'] = ta.roc(df['close'], length=10)

        # Awesome Oscillator
        df['ao'] = ta.ao(df['high'], df['low'])

        return df

    @staticmethod
    def add_volatility_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # ATR - Average True Range
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Keltner Channel
        keltner = ta.kc(df['high'], df['low'], df['close'], length=20)
        df = pd.concat([df, keltner], axis=1)

        # Historical Volatility
        df['volatility_30'] = ta.volatility(df['close'], length=30)

        return df

    @staticmethod
    def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # OBV - On-Balance Volume
        df['obv'] = ta.obv(df['close'], df['volume'])

        # Volume Weighted Average Price
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        # Accumulation/Distribution Line
        df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

        # Chaikin Money Flow
        df['cmf_20'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)

        # Money Flow Index
        df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

        return df

    @staticmethod
    def add_cycle_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add cycle indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        if TALIB_AVAILABLE:
            try:
                # Hilbert Transform - Sine Wave (requires TA-Lib)
                df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(df['close'])

                # Hilbert Transform - Dominant Cycle Period
                df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])

                # Hilbert Transform - Dominant Cycle Phase
                df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
            except Exception as e:
                logger.warning(f"Error calculating cycle indicators with TA-Lib: {str(e)}")
        else:
            # If TA-Lib is not available, use alternative implementations or skip
            logger.info("Skipping cycle indicators that require TA-Lib")
            # We could implement alternative cycle indicators here if needed

        return df

    @staticmethod
    def add_pattern_recognition(data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition indicators to the DataFrame"""
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        if TALIB_AVAILABLE:
            try:
                # Candlestick Patterns (requires TA-Lib)
                pattern_functions = [
                    ('cdl_doji', talib.CDLDOJI),
                    ('cdl_hammer', talib.CDLHAMMER),
                    ('cdl_shooting_star', talib.CDLSHOOTINGSTAR),
                    ('cdl_engulfing', talib.CDLENGULFING),
                    ('cdl_morning_star', talib.CDLMORNINGSTAR),
                    ('cdl_evening_star', talib.CDLEVENINGSTAR)
                ]

                for name, func in pattern_functions:
                    df[name] = func(df['open'], df['high'], df['low'], df['close'])
            except Exception as e:
                logger.warning(f"Error calculating pattern recognition with TA-Lib: {str(e)}")
        else:
            # If TA-Lib is not available, use pandas_ta for basic patterns
            logger.info("Using pandas_ta for basic pattern recognition")
            try:
                # Use pandas_ta for basic patterns
                # Check for doji pattern (simplified)
                df['cdl_doji'] = ((abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.1).astype(int)

                # Add more pattern recognition if needed
            except Exception as e:
                logger.warning(f"Error calculating pattern recognition with pandas_ta: {str(e)}")

        return df

    @staticmethod
    def get_support_resistance_levels(data: pd.DataFrame, window: int = 20, threshold: float = 0.05) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels using local minima and maxima.

        Args:
            data: DataFrame with price data
            window: Window size for finding local extrema
            threshold: Minimum percentage difference between levels

        Returns:
            Dictionary with support and resistance levels
        """
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Find local maxima and minima
        df['min'] = df['close'].rolling(window=window, center=True).min()
        df['max'] = df['close'].rolling(window=window, center=True).max()

        # Identify potential support levels (local minima)
        support_levels = []
        for i in range(window, len(df) - window):
            if df['close'].iloc[i] == df['min'].iloc[i] and df['close'].iloc[i] != df['close'].iloc[i-1]:
                support_levels.append(df['close'].iloc[i])

        # Identify potential resistance levels (local maxima)
        resistance_levels = []
        for i in range(window, len(df) - window):
            if df['close'].iloc[i] == df['max'].iloc[i] and df['close'].iloc[i] != df['close'].iloc[i-1]:
                resistance_levels.append(df['close'].iloc[i])

        # Filter out levels that are too close to each other
        support_levels = TechnicalIndicators._filter_levels(support_levels, threshold)
        resistance_levels = TechnicalIndicators._filter_levels(resistance_levels, threshold)

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    @staticmethod
    def _filter_levels(levels: List[float], threshold: float) -> List[float]:
        """Filter out levels that are too close to each other"""
        if not levels:
            return []

        levels = sorted(levels)
        filtered_levels = [levels[0]]

        for level in levels[1:]:
            if (level - filtered_levels[-1]) / filtered_levels[-1] > threshold:
                filtered_levels.append(level)

        return filtered_levels

    @staticmethod
    def calculate_pivot_points(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Calculate pivot points using different methods.

        Args:
            data: DataFrame with OHLC data
            method: Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')

        Returns:
            DataFrame with pivot points
        """
        df = data.copy()

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Get high, low, close for the period
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]

        pivot_points = {}

        if method == 'standard':
            # Standard pivot points
            pivot = (high + low + close) / 3
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)

            pivot_points = {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }

        elif method == 'fibonacci':
            # Fibonacci pivot points
            pivot = (high + low + close) / 3
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)

            pivot_points = {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                'r1': r1,
                'r2': r2,
                'r3': r3
            }

        elif method == 'woodie':
            # Woodie pivot points
            pivot = (high + low + 2 * close) / 4
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)

            pivot_points = {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                'r1': r1,
                'r2': r2
            }

        elif method == 'camarilla':
            # Camarilla pivot points
            pivot = (high + low + close) / 3
            s1 = close - 1.1 * (high - low) / 12
            s2 = close - 1.1 * (high - low) / 6
            s3 = close - 1.1 * (high - low) / 4
            s4 = close - 1.1 * (high - low) / 2
            r1 = close + 1.1 * (high - low) / 12
            r2 = close + 1.1 * (high - low) / 6
            r3 = close + 1.1 * (high - low) / 4
            r4 = close + 1.1 * (high - low) / 2

            pivot_points = {
                'pivot': pivot,
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                'r4': r4
            }

        elif method == 'demark':
            # DeMark pivot points
            if close < open_price:
                pivot = high + 2 * low + close
            elif close > open_price:
                pivot = 2 * high + low + close
            else:
                pivot = high + low + 2 * close

            pivot = pivot / 4
            s1 = pivot / 2 - high
            r1 = pivot / 2 - low

            pivot_points = {
                'pivot': pivot,
                's1': s1,
                'r1': r1
            }

        return pd.DataFrame(pivot_points, index=[0])
