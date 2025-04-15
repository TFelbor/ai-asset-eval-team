"""
Machine learning module for financial predictions.
This module provides machine learning tools for financial predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from typing import Dict, List, Union, Optional, Tuple, Any

class MLPredictor:
    """
    A class that provides machine learning tools for financial predictions.
    """

    def __init__(self):
        """Initialize the MLPredictor class"""
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_importance = None
        self.best_params = None

    def prepare_features(self, data: pd.DataFrame, target_col: str = 'close', n_lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning.

        Args:
            data: DataFrame with financial data
            target_col: Target column name
            n_lags: Number of lagged features to create

        Returns:
            Tuple of (features_df, target_series)
        """
        if data.empty:
            raise ValueError("Empty data provided")

        try:
            # Create a copy of the data
            df = data.copy()

            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]

            # Create lagged features
            for lag in range(1, n_lags + 1):
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

            # Create rolling statistics
            for window in [5, 10, 20]:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
                df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
                df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()

            # Create returns
            df[f'{target_col}_return_1d'] = df[target_col].pct_change()
            df[f'{target_col}_return_5d'] = df[target_col].pct_change(5)

            # Create target variable (next day's price)
            df['target'] = df[target_col].shift(-1)

            # Drop rows with NaN values
            df = df.dropna()

            # Separate features and target
            X = df.drop(['target', target_col], axis=1)
            y = df['target']

            return X, y
        except Exception as e:
            raise ValueError(f"Feature preparation failed: {str(e)}")

    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest',
                   cv_splits: int = 5, tune_hyperparams: bool = True) -> Dict[str, float]:
        """
        Train a machine learning model.

        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 'elastic_net', 'svr', 'mlp', 'lstm')
            cv_splits: Number of cross-validation splits
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Dict of performance metrics
        """
        try:
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            # Scale features
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

            # Select model
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                } if tune_hyperparams else {}

            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                } if tune_hyperparams else {}

            elif model_type == 'linear':
                model = LinearRegression()
                param_grid = {}

            elif model_type == 'ridge':
                model = Ridge(random_state=42)
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                } if tune_hyperparams else {}

            elif model_type == 'lasso':
                model = Lasso(random_state=42)
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0]
                } if tune_hyperparams else {}

            elif model_type == 'elastic_net':
                model = ElasticNet(random_state=42)
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                } if tune_hyperparams else {}

            elif model_type == 'svr':
                model = SVR()
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['linear', 'rbf']
                } if tune_hyperparams else {}

            elif model_type == 'mlp':
                model = MLPRegressor(random_state=42, max_iter=1000)
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                } if tune_hyperparams else {}

            elif model_type == 'lstm':
                # For LSTM, we need to reshape the data
                return self._train_lstm_model(X, y, cv_splits)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Tune hyperparameters if requested
            if tune_hyperparams and param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
                grid_search.fit(X_scaled, y_scaled)
                model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
            else:
                model.fit(X_scaled, y_scaled)
                self.best_params = {}

            # Store the trained model
            self.model = model

            # Calculate cross-validation scores
            cv_scores = []
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                cv_scores.append({
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })

            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = dict(zip(X.columns, model.feature_importances_))

            # Calculate average metrics
            avg_metrics = {
                'mean_mse': np.mean([s['mse'] for s in cv_scores]),
                'mean_rmse': np.mean([s['rmse'] for s in cv_scores]),
                'mean_mae': np.mean([s['mae'] for s in cv_scores]),
                'mean_r2': np.mean([s['r2'] for s in cv_scores]),
                'std_mse': np.std([s['mse'] for s in cv_scores]),
                'std_rmse': np.std([s['rmse'] for s in cv_scores]),
                'std_mae': np.std([s['mae'] for s in cv_scores]),
                'std_r2': np.std([s['r2'] for s in cv_scores])
            }

            return avg_metrics
        except Exception as e:
            raise ValueError(f"Model training failed: {str(e)}")

    def _train_lstm_model(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> Dict[str, float]:
        """
        Train an LSTM model.

        Args:
            X: Feature DataFrame
            y: Target Series
            cv_splits: Number of cross-validation splits

        Returns:
            Dict of performance metrics
        """
        try:
            # Scale features and target
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

            # Reshape data for LSTM [samples, time steps, features]
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)

            # Initialize metrics
            cv_scores = []

            for train_idx, test_idx in tscv.split(X_reshaped):
                X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                # Create and compile LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(1, X.shape[1])),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mse')

                # Train model
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

                # Make predictions
                y_pred = model.predict(X_test).flatten()

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                cv_scores.append({
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })

            # Store the last model
            self.model = model

            # Calculate average metrics
            avg_metrics = {
                'mean_mse': np.mean([s['mse'] for s in cv_scores]),
                'mean_rmse': np.mean([s['rmse'] for s in cv_scores]),
                'mean_mae': np.mean([s['mae'] for s in cv_scores]),
                'mean_r2': np.mean([s['r2'] for s in cv_scores]),
                'std_mse': np.std([s['mse'] for s in cv_scores]),
                'std_rmse': np.std([s['rmse'] for s in cv_scores]),
                'std_mae': np.std([s['mae'] for s in cv_scores]),
                'std_r2': np.std([s['r2'] for s in cv_scores])
            }

            return avg_metrics
        except Exception as e:
            raise ValueError(f"LSTM model training failed: {str(e)}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make predictions with confidence metrics.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, confidence_metrics)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Scale features
            X_scaled = self.scaler_X.transform(X)

            # Make predictions based on model type
            if isinstance(self.model, (RandomForestRegressor, GradientBoostingRegressor)):
                # For ensemble models, we can get prediction intervals
                predictions = self.model.predict(X_scaled)

                # Calculate prediction intervals using tree variance
                if isinstance(self.model, RandomForestRegressor):
                    tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                    std = np.std(tree_predictions, axis=0)
                    lower_bound = np.percentile(tree_predictions, 5, axis=0)
                    upper_bound = np.percentile(tree_predictions, 95, axis=0)
                else:  # GradientBoostingRegressor
                    # For GBR, we use a simpler approach
                    std = np.ones_like(predictions) * np.sqrt(self.model.loss_.get_init_raw_predictions(X_scaled.shape[0], self.model.init_).reshape(-1))
                    lower_bound = predictions - 1.96 * std
                    upper_bound = predictions + 1.96 * std

                # Inverse transform predictions
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
                lower_bound = self.scaler_y.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                upper_bound = self.scaler_y.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

                confidence_metrics = {
                    'mean': predictions,
                    'std': std,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            elif isinstance(self.model, Sequential):  # LSTM model
                # Reshape data for LSTM [samples, time steps, features]
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

                # Make predictions
                predictions = self.model.predict(X_reshaped).flatten()

                # Inverse transform predictions
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

                # For LSTM, we don't have built-in uncertainty estimates
                # Use a simple approach based on training error
                std = np.ones_like(predictions) * 0.1  # Placeholder
                lower_bound = predictions - 1.96 * std
                upper_bound = predictions + 1.96 * std

                confidence_metrics = {
                    'mean': predictions,
                    'std': std,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            else:
                # For other models, make simple predictions
                predictions = self.model.predict(X_scaled)

                # Inverse transform predictions
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

                # Use a simple approach for confidence intervals
                std = np.ones_like(predictions) * 0.1  # Placeholder
                lower_bound = predictions - 1.96 * std
                upper_bound = predictions + 1.96 * std

                confidence_metrics = {
                    'mean': predictions,
                    'std': std,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            return predictions, confidence_metrics
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def plot_feature_importance(self, top_n: int = 10, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train a model that supports feature importance.")

        try:
            # Sort feature importance
            sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

            # Select top N features
            top_features = sorted_importance[:top_n]

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot feature importance
            features = [f[0] for f in top_features]
            importance = [f[1] for f in top_features]

            ax.barh(range(len(features)), importance, align='center')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance')

            plt.tight_layout()

            return fig
        except Exception as e:
            raise ValueError(f"Feature importance plotting failed: {str(e)}")

    def plot_predictions(self, X: pd.DataFrame, y: pd.Series, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot predictions vs actual values.

        Args:
            X: Feature DataFrame
            y: Target Series
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Make predictions
            predictions, confidence = self.predict(X)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot actual values
            ax.plot(y.index, y.values, label='Actual', color='blue')

            # Plot predictions
            ax.plot(y.index, predictions, label='Predicted', color='red', linestyle='--')

            # Plot confidence intervals
            ax.fill_between(y.index,
                           confidence['lower_bound'],
                           confidence['upper_bound'],
                           color='red', alpha=0.2, label='95% Confidence Interval')

            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Predictions vs Actual Values')
            ax.legend()

            plt.tight_layout()

            return fig
        except Exception as e:
            raise ValueError(f"Predictions plotting failed: {str(e)}")

    def plot_residuals(self, X: pd.DataFrame, y: pd.Series, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot residuals.

        Args:
            X: Feature DataFrame
            y: Target Series
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Make predictions
            predictions, _ = self.predict(X)

            # Calculate residuals
            residuals = y.values - predictions

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=figsize)

            # Plot residuals over time
            axes[0].plot(y.index, residuals, color='green')
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Residual')
            axes[0].set_title('Residuals Over Time')

            # Plot residual histogram
            axes[1].hist(residuals, bins=30, color='green', alpha=0.7)
            axes[1].axvline(x=0, color='red', linestyle='--')
            axes[1].set_xlabel('Residual')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residual Distribution')

            plt.tight_layout()

            return fig
        except Exception as e:
            raise ValueError(f"Residuals plotting failed: {str(e)}")

    def forecast_future(self, X: pd.DataFrame, y: pd.Series, periods: int = 30) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast future values.

        Args:
            X: Feature DataFrame
            y: Target Series
            periods: Number of periods to forecast

        Returns:
            Tuple of (forecast_series, confidence_df)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Create a copy of the data
            X_future = X.iloc[-1:].copy()
            last_date = y.index[-1]

            # Generate future dates with proper handling of different date types
            try:
                # If it's a pandas Timestamp
                if isinstance(last_date, pd.Timestamp):
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
                # If it's a string, try to convert to datetime
                elif isinstance(last_date, str):
                    start_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                    future_dates = pd.date_range(start=start_date, periods=periods)
                # If it's a number or other type, use a simple range index
                else:
                    future_dates = pd.RangeIndex(start=1, stop=periods + 1)
            except Exception as e:
                # Fallback to simple integer index if date handling fails
                print(f"Error creating forecast dates: {str(e)}. Using integer index instead.")
                future_dates = pd.RangeIndex(start=1, stop=periods + 1)

            # Initialize forecast arrays
            forecasts = np.zeros(periods)
            lower_bounds = np.zeros(periods)
            upper_bounds = np.zeros(periods)

            # Get the target column (assuming it's the first part of the lag columns)
            target_col = [col for col in X.columns if '_lag_1' in col][0].replace('_lag_1', '')

            # Iteratively forecast future values
            for i in range(periods):
                # Make prediction for the next step
                pred, conf = self.predict(X_future)

                # Store forecast and confidence intervals
                forecasts[i] = pred[0]
                lower_bounds[i] = conf['lower_bound'][0]
                upper_bounds[i] = conf['upper_bound'][0]

                # Update features for the next step
                for lag in range(1, 6):  # Assuming 5 lags
                    lag_col = f'{target_col}_lag_{lag}'
                    if lag_col in X_future.columns:
                        if lag == 1:
                            X_future[lag_col] = pred[0]
                        else:
                            prev_lag_col = f'{target_col}_lag_{lag-1}'
                            if prev_lag_col in X_future.columns:
                                X_future[lag_col] = X_future[prev_lag_col]

                # Update rolling statistics (simplified)
                # In a real implementation, you would need to update all features properly

            # Create forecast series and confidence DataFrame
            forecast_series = pd.Series(forecasts, index=future_dates, name='forecast')
            confidence_df = pd.DataFrame({
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            }, index=future_dates)

            return forecast_series, confidence_df
        except Exception as e:
            raise ValueError(f"Future forecasting failed: {str(e)}")

    def plot_forecast(self, X: pd.DataFrame, y: pd.Series, periods: int = 30,
                     history_periods: int = 60, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot forecast.

        Args:
            X: Feature DataFrame
            y: Target Series
            periods: Number of periods to forecast
            history_periods: Number of historical periods to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        try:
            # Generate forecast
            forecast, confidence = self.forecast_future(X, y, periods)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot historical data
            history = y.iloc[-history_periods:]
            ax.plot(history.index, history.values, label='Historical', color='blue')

            # Plot forecast
            ax.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')

            # Plot confidence intervals
            ax.fill_between(forecast.index,
                           confidence['lower_bound'],
                           confidence['upper_bound'],
                           color='red', alpha=0.2, label='95% Confidence Interval')

            # Add vertical line to separate historical data and forecast
            ax.axvline(x=y.index[-1], color='green', linestyle='-', alpha=0.5)

            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Forecast')
            ax.legend()

            plt.tight_layout()

            return fig
        except Exception as e:
            raise ValueError(f"Forecast plotting failed: {str(e)}")
