"""
Machine learning analysis module for financial predictions
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import logging

# Define functions for feature engineering
def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    from .technical_indicators import TechnicalIndicators
    return TechnicalIndicators.calculate_all_indicators(data)

def calculate_volume_features(data):
    """Calculate volume-based features"""
    from .technical_indicators import TechnicalIndicators
    return TechnicalIndicators.add_volume_indicators(data)

def calculate_fractional_differentiation(series, d=0.5, thres=1e-5):
    """Calculate fractionally differentiated series"""
    # Simple implementation of fractional differentiation
    import numpy as np
    import pandas as pd

    # Get weights
    weights = [1.0]
    for k in range(1, len(series)):
        weights.append(weights[-1] * (d - k + 1) / k)

    # Determine cutoff point
    weights_above_threshold = np.where(np.abs(weights) > thres)[0]
    if len(weights_above_threshold) > 0:
        cutoff = weights_above_threshold[-1]
    else:
        cutoff = len(weights)

    # Apply weights
    weights = weights[:cutoff+1]

    # Calculate differentiated series
    diff_series = pd.Series(index=series.index)
    for i in range(len(weights), len(series)):
        diff_series.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1])

    return diff_series

logger = logging.getLogger(__name__)

class MLAnalyzer:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize ML Analyzer

        Args:
            n_estimators: Number of trees in RandomForest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning analysis

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        if data.empty:
            raise ValueError("Empty data provided")

        try:
            # Ensure column names are lowercase
            data_lower = data.copy()
            data_lower.columns = [col.lower() for col in data_lower.columns]

            # Create a new DataFrame for features
            features = pd.DataFrame(index=data_lower.index)

            # Fractionally differentiated features
            features['fd_close'] = calculate_fractional_differentiation(
                data_lower['close'],
                d=0.5,
                thres=1e-5
            )

            # Add technical indicators
            try:
                # Calculate technical indicators
                tech_data = calculate_technical_indicators(data_lower)

                # Select key indicators to avoid too many features
                key_indicators = [
                    'sma_20', 'sma_50', 'ema_12', 'ema_26',
                    'macd', 'macds', 'macdh',
                    'bbl_20_2.0', 'bbm_20_2.0', 'bbu_20_2.0',
                    'rsi_14', 'cci_20', 'willr_14', 'roc_10',
                    'atr_14', 'volatility_30', 'obv', 'cmf_20', 'mfi_14'
                ]

                # Add selected indicators to features
                for indicator in key_indicators:
                    if indicator in tech_data.columns:
                        features[indicator] = tech_data[indicator]
            except Exception as e:
                logger.warning(f"Error calculating technical indicators: {str(e)}")

            # Add price-based features
            features['returns'] = data_lower['close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['log_return'] = np.log(data_lower['close']).diff()

            # Add rolling statistics
            for window in [5, 10, 20, 50]:
                features[f'close_rolling_mean_{window}'] = data_lower['close'].rolling(window=window).mean()
                features[f'close_rolling_std_{window}'] = data_lower['close'].rolling(window=window).std()
                features[f'close_rolling_min_{window}'] = data_lower['close'].rolling(window=window).min()
                features[f'close_rolling_max_{window}'] = data_lower['close'].rolling(window=window).max()

            # Add lagged features
            for lag in range(1, 6):
                features[f'close_lag_{lag}'] = data_lower['close'].shift(lag)
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)

            # Add volume features if available
            if 'volume' in data_lower.columns:
                features['volume_change'] = data_lower['volume'].pct_change()
                features['volume_rolling_mean_5'] = data_lower['volume'].rolling(window=5).mean()
                features['volume_rolling_std_5'] = data_lower['volume'].rolling(window=5).std()

                # Add price-volume features
                features['price_volume_ratio'] = data_lower['close'] / (data_lower['volume'] + 1)  # Add 1 to avoid division by zero
                features['price_volume_trend'] = (data_lower['close'] - data_lower['close'].shift(1)) * data_lower['volume']

            # Clean up
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill')
            features = features.dropna()

            return features
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            raise

    def train_model(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Train the prediction model with time series cross-validation

        Args:
            features: DataFrame of features
            target: Series of target values

        Returns:
            Dict of performance metrics
        """
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []

            for train_idx, test_idx in tscv.split(features):
                X_train = features.iloc[train_idx]
                X_test = features.iloc[test_idx]
                y_train = target.iloc[train_idx]
                y_test = target.iloc[test_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # Train and evaluate
                self.model.fit(X_train_scaled, y_train)
                y_pred = self.model.predict(X_test_scaled)
                cv_scores.append({
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                })

            # Store feature importance
            self.feature_importance = dict(zip(
                features.columns,
                self.model.feature_importances_
            ))

            # Calculate average metrics
            avg_metrics = {
                'mean_mse': np.mean([s['mse'] for s in cv_scores]),
                'mean_r2': np.mean([s['r2'] for s in cv_scores]),
                'std_mse': np.std([s['mse'] for s in cv_scores]),
                'std_r2': np.std([s['r2'] for s in cv_scores])
            }

            return avg_metrics
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Make predictions with confidence metrics

        Args:
            features: DataFrame of features

        Returns:
            Tuple of (predictions, confidence_metrics)
        """
        try:
            # Scale features
            X_scaled = self.scaler.transform(features)

            # Make predictions
            predictions = self.model.predict(X_scaled)

            # Calculate prediction intervals using tree variance
            tree_predictions = np.array([tree.predict(X_scaled)
                                      for tree in self.model.estimators_])
            confidence_metrics = {
                'mean': predictions,
                'std': np.std(tree_predictions, axis=0),
                'lower_bound': np.percentile(tree_predictions, 5, axis=0),
                'upper_bound': np.percentile(tree_predictions, 95, axis=0)
            }

            return predictions, confidence_metrics
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
