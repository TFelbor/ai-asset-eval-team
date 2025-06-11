"""
Machine Learning Analysis UI component for the AI Finance Dashboard.
This module provides UI components for ML-based financial analysis with educational features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

# Import enhanced logging
from core.utils.logger import log_error

# Import services lazily to improve startup performance
from core.data.data_service import DataService

# Initialize data service (lightweight)
data_service = DataService()

# Educational content for ML models
ML_MODEL_EXPLANATIONS = {
    "linear": {
        "name": "Linear Regression",
        "description": "A simple model that assumes a linear relationship between inputs and outputs. Good for understanding basic trends.",
        "strengths": ["Simple to understand", "Fast to train", "Works well with linear relationships"],
        "weaknesses": ["Cannot capture complex patterns", "Sensitive to outliers", "Assumes linear relationships"],
        "complexity": "Low",
        "use_case": "Good for initial analysis and establishing baselines"
    },
    "ridge": {
        "name": "Ridge Regression",
        "description": "Linear regression with regularization to prevent overfitting. Better for data with many features.",
        "strengths": ["Handles multicollinearity well", "Reduces overfitting", "Still relatively simple"],
        "weaknesses": ["Still assumes linear relationships", "May underfit complex data"],
        "complexity": "Low",
        "use_case": "When you have many correlated features"
    },
    "lasso": {
        "name": "Lasso Regression",
        "description": "Linear regression with L1 regularization that can eliminate irrelevant features.",
        "strengths": ["Feature selection capability", "Reduces overfitting", "Creates sparse models"],
        "weaknesses": ["May eliminate useful features", "Still assumes linear relationships"],
        "complexity": "Low",
        "use_case": "When you want to identify the most important features"
    },
    "elastic_net": {
        "name": "Elastic Net",
        "description": "Combines Ridge and Lasso regularization for balanced feature selection and regularization.",
        "strengths": ["Balanced regularization", "Handles correlated features", "Some feature selection"],
        "weaknesses": ["Requires tuning of multiple parameters", "Still assumes linear relationships"],
        "complexity": "Medium",
        "use_case": "When you want benefits of both Ridge and Lasso"
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of decision trees that can capture non-linear patterns and feature interactions.",
        "strengths": ["Captures non-linear relationships", "Robust to outliers", "Provides feature importance"],
        "weaknesses": ["Black box model", "Can overfit with too many trees", "Computationally intensive"],
        "complexity": "Medium",
        "use_case": "General-purpose model that works well for many financial applications"
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Sequential ensemble that builds trees to correct errors of previous trees. Often very accurate.",
        "strengths": ["High accuracy", "Captures complex patterns", "Provides feature importance"],
        "weaknesses": ["Prone to overfitting", "Computationally intensive", "Requires careful tuning"],
        "complexity": "High",
        "use_case": "When you need high accuracy and have time to tune parameters"
    },
    "svr": {
        "name": "Support Vector Regression",
        "description": "Uses a kernel function to project data into higher dimensions to find better relationships.",
        "strengths": ["Works well with non-linear data", "Robust to outliers", "Effective in high dimensions"],
        "weaknesses": ["Slow on large datasets", "Difficult to interpret", "Sensitive to parameter selection"],
        "complexity": "Medium",
        "use_case": "When dealing with complex non-linear relationships"
    },
    "mlp": {
        "name": "Multi-layer Perceptron",
        "description": "A type of neural network that can learn complex patterns through multiple layers of neurons.",
        "strengths": ["Can model highly complex relationships", "Flexible architecture", "Good with large datasets"],
        "weaknesses": ["Requires large amounts of data", "Black box model", "Computationally intensive"],
        "complexity": "High",
        "use_case": "When you have lots of data and need to capture complex patterns"
    },
    "lstm": {
        "name": "Long Short-Term Memory",
        "description": "A recurrent neural network designed to capture long-term dependencies in time series data.",
        "strengths": ["Excellent for sequential data", "Captures long-term patterns", "Remembers important information"],
        "weaknesses": ["Very data-hungry", "Slow to train", "Complex to tune properly"],
        "complexity": "Very High",
        "use_case": "For complex time series with long-term dependencies"
    }
}

# Educational content for metrics
METRIC_EXPLANATIONS = {
    "R² Score": "R-squared measures how well the model explains the variance in the data. Values range from 0 to 1, with 1 being perfect prediction. A value of 0.7 means the model explains 70% of the variance in the target variable.",
    "RMSE": "Root Mean Squared Error measures the average magnitude of prediction errors. Lower values indicate better accuracy. It's in the same units as the target variable, making it interpretable.",
    "MAE": "Mean Absolute Error is the average absolute difference between predicted and actual values. Like RMSE, lower is better, but MAE is less sensitive to outliers.",
    "Feature Count": "The number of input variables used by the model. More features can capture more information but may lead to overfitting if not properly regularized."
}

# Educational content for technical indicators
TECHNICAL_INDICATOR_EXPLANATIONS = {
    "sma_20": "Simple Moving Average (20-day) - The average price over the last 20 days, smoothing out short-term fluctuations.",
    "sma_50": "Simple Moving Average (50-day) - The average price over the last 50 days, often used to identify medium-term trends.",
    "sma_200": "Simple Moving Average (200-day) - The average price over the last 200 days, commonly used to identify long-term trends.",
    "ema_12": "Exponential Moving Average (12-day) - Similar to SMA but gives more weight to recent prices.",
    "ema_26": "Exponential Moving Average (26-day) - A longer-term EMA often used with the 12-day EMA to generate MACD signals.",
    "macd": "Moving Average Convergence Divergence - The difference between two EMAs, used to identify momentum changes.",
    "rsi_14": "Relative Strength Index (14-day) - Measures the speed and change of price movements on a scale of 0-100. Values above 70 suggest overbought conditions, while values below 30 suggest oversold conditions.",
    "bbands": "Bollinger Bands - Consists of a middle band (SMA) with upper and lower bands (standard deviations). Used to identify volatility and potential price breakouts.",
    "volume": "Trading volume - The number of shares or contracts traded. High volume often confirms strong price movements."
}

def display_educational_tooltip(label: str, explanation: str, icon: str = "ℹ️"):
    """Display a label with an educational tooltip."""
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(f"**{label}**")
    with col2:
        st.markdown(f"<span title='{explanation}'>{icon}</span>", unsafe_allow_html=True)

def display_model_info_card(model_type: str):
    """Display an educational card with information about the selected model."""
    if model_type not in ML_MODEL_EXPLANATIONS:
        return

    model_info = ML_MODEL_EXPLANATIONS[model_type]

    with st.expander(f"📚 Learn about {model_info['name']}", expanded=False):
        st.markdown(f"### {model_info['name']}")
        st.markdown(model_info['description'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Strengths")
            for strength in model_info['strengths']:
                st.markdown(f"✅ {strength}")

        with col2:
            st.markdown("#### Limitations")
            for weakness in model_info['weaknesses']:
                st.markdown(f"⚠️ {weakness}")

        st.markdown(f"**Complexity:** {model_info['complexity']}")
        st.markdown(f"**Best used when:** {model_info['use_case']}")

def render_ml_analysis_ui(ticker: str, asset_type: str, price_history: Dict[str, Any] = None):
    """
    Render the ML analysis UI component.

    Args:
        ticker: Asset ticker symbol
        asset_type: Type of asset (stock, crypto, reit, etf)
        price_history: Optional pre-loaded price history data
    """
    # Add ML Basics educational section at the top
    with st.expander("📚 Machine Learning Basics - Click to Learn More", expanded=False):
        st.markdown("### What is Machine Learning in Finance?")
        st.markdown("""
        Machine learning (ML) is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed.
        In finance, ML helps identify patterns in market data that might predict future price movements.

        **Key concepts in this analysis:**

        * **Models** - Mathematical algorithms that learn patterns from historical data
        * **Features** - Input variables (like price, volume, technical indicators) used to make predictions
        * **Target** - What we're trying to predict (usually future price or returns)
        * **Training** - The process of teaching the model using historical data
        * **Prediction** - Using the trained model to forecast future values
        * **Metrics** - Measurements of how well the model performs
        """)

        st.markdown("### Understanding the Analysis Process")
        st.markdown("""
        1. **Data Preparation**: Historical price data is collected and processed
        2. **Feature Engineering**: Creating useful inputs for the model (like technical indicators)
        3. **Model Training**: The algorithm learns patterns from historical data
        4. **Evaluation**: Testing how well the model performs on unseen data
        5. **Forecasting**: Predicting future values based on the trained model
        """)

        st.markdown("### Limitations to Keep in Mind")
        st.markdown("""
        * **Past ≠ Future**: Historical patterns don't guarantee future performance
        * **Market Complexity**: Financial markets are influenced by countless factors
        * **Black Swan Events**: Unexpected major events can invalidate predictions
        * **Model Simplicity**: All models are simplifications of reality

        **Always use ML predictions as one of many tools in your investment decision process, not as the sole basis for decisions.**
        """)

    # Header is already provided in dashboard.py, so we only show the info message
    st.info("💡 **Performance Tip:** ML analysis can be resource-intensive. If you're experiencing slow performance, enable lightweight mode below for faster analysis with simpler models.")

    # Get price history data if not provided
    if not price_history:
        price_history = data_service.get_price_history(ticker, asset_type, "1y")

    # Check if we have valid price data
    if "error" in price_history:
        st.error(f"Error fetching price data: {price_history['error']}")
        return

    if not price_history.get("timestamps") or not price_history.get("prices"):
        st.error("Insufficient price data for ML analysis")
        return

    # Add option for lightweight mode
    lightweight_mode = st.checkbox("Enable lightweight mode (faster on older computers)", value=True,
                                help="Uses simpler models and fewer features for better performance on older hardware")

    # Lazy import ML components to improve startup performance
    with st.spinner("Loading ML components..."):
        try:
            from core.analytics.ml_predictions import MLPredictor
            from core.analytics.technical_indicators import TechnicalIndicators

            # Initialize ML predictor only when needed
            ml_predictor = MLPredictor()
        except ImportError as e:
            if "tensorflow" in str(e).lower():
                st.error("TensorFlow is not installed. Some ML models (LSTM) will not be available.")
                st.info("To enable all ML features, install TensorFlow: `pip install tensorflow`")
                # Still try to import without TensorFlow
                from core.analytics.ml_predictions import MLPredictor
                from core.analytics.technical_indicators import TechnicalIndicators
                ml_predictor = MLPredictor()
            else:
                st.error(f"Error loading ML components: {str(e)}")
                return

    # Get data arrays from price history
    timestamps = price_history.get("timestamps", [])
    opens = price_history.get("open", price_history.get("opens", []))
    highs = price_history.get("high", price_history.get("highs", []))
    lows = price_history.get("low", price_history.get("lows", []))
    closes = price_history.get("close", price_history.get("prices", []))
    volumes = price_history.get("volume", price_history.get("volumes", []))

    # Check if we have valid data
    if not timestamps or not closes:
        st.error("Missing essential price data (timestamps or prices)")
        return

    # Ensure all arrays have the same length as timestamps
    n = len(timestamps)

    # Pad or truncate arrays to match timestamp length
    if len(opens) != n:
        st.warning(f"Open prices array length mismatch: {len(opens)} vs {n}. Adjusting...")
        if len(opens) > n:
            opens = opens[:n]  # Truncate
        else:
            # Pad with the last value or zeros
            last_value = opens[-1] if opens else 0
            opens = opens + [last_value] * (n - len(opens))

    if len(highs) != n:
        st.warning(f"High prices array length mismatch: {len(highs)} vs {n}. Adjusting...")
        if len(highs) > n:
            highs = highs[:n]  # Truncate
        else:
            # Pad with the last value or zeros
            last_value = highs[-1] if highs else 0
            highs = highs + [last_value] * (n - len(highs))

    if len(lows) != n:
        st.warning(f"Low prices array length mismatch: {len(lows)} vs {n}. Adjusting...")
        if len(lows) > n:
            lows = lows[:n]  # Truncate
        else:
            # Pad with the last value or zeros
            last_value = lows[-1] if lows else 0
            lows = lows + [last_value] * (n - len(lows))

    if len(closes) != n:
        st.warning(f"Close prices array length mismatch: {len(closes)} vs {n}. Adjusting...")
        if len(closes) > n:
            closes = closes[:n]  # Truncate
        else:
            # Pad with the last value or zeros
            last_value = closes[-1] if closes else 0
            closes = closes + [last_value] * (n - len(closes))

    if len(volumes) != n:
        st.warning(f"Volumes array length mismatch: {len(volumes)} vs {n}. Adjusting...")
        if len(volumes) > n:
            volumes = volumes[:n]  # Truncate
        else:
            # Pad with zeros
            volumes = volumes + [0] * (n - len(volumes))

    # Now create DataFrame with arrays of equal length
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    # Ensure all required columns exist
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0  # Default volume to 0 if not available
            else:
                # For OHLC, use close price if specific column not available
                df[col] = df["close"]

    # Display data overview
    with st.expander("📊 Data Overview", expanded=False):
        st.dataframe(df.tail(10))

        # Display basic statistics
        st.markdown("### Basic Statistics")
        st.dataframe(df.describe())

    # ML Analysis Options
    st.markdown("### ML Analysis Options")

    col1, col2 = st.columns(2)

    with col1:
        # Model selection - offer different options based on lightweight mode
        if lightweight_mode:
            model_type = st.selectbox(
                "Select Model Type",
                ["linear", "ridge", "random_forest"],
                index=0,
                help="Select the type of machine learning model to use for prediction (limited options in lightweight mode)"
            )
        else:
            model_type = st.selectbox(
                "Select Model Type",
                ["random_forest", "gradient_boosting", "linear", "ridge", "lasso", "elastic_net", "svr", "mlp", "lstm"],
                index=0,
                help="Select the type of machine learning model to use for prediction"
            )

        # Display educational information about the selected model
        display_model_info_card(model_type)

        # Target variable with educational tooltip
        st.markdown("")
        target_col = st.selectbox(
            "Target Variable",
            ["close", "return_1d"],
            index=0,
            help="Select the target variable to predict: 'close' for price or 'return_1d' for daily returns"
        )

        # Add educational explanation for target variable
        if target_col == "close":
            st.info("💡 **Price Prediction**: Forecasting the actual closing price of the asset. Useful for understanding absolute price levels.")
        else:
            st.info("💡 **Returns Prediction**: Forecasting the percentage change in price. Often more stable to predict than absolute prices and useful for comparing different assets.")

    with col2:
        # Forecast horizon - limit in lightweight mode
        if lightweight_mode:
            forecast_periods = st.slider(
                "Forecast Horizon (Days)",
                min_value=1,
                max_value=30,  # Limit to 30 days in lightweight mode
                value=14,
                help="Number of days to forecast into the future (limited in lightweight mode)"
            )
        else:
            forecast_periods = st.slider(
                "Forecast Horizon (Days)",
                min_value=1,
                max_value=90,
                value=30,
                help="Number of days to forecast into the future"
            )

        # Feature engineering options - limit in lightweight mode
        if lightweight_mode:
            n_lags = st.slider(
                "Number of Lags",
                min_value=1,
                max_value=5,  # Limit to 5 lags in lightweight mode
                value=3,
                help="Number of lagged features to create (limited in lightweight mode)"
            )
        else:
            n_lags = st.slider(
                "Number of Lags",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of lagged features to create"
            )

    # Advanced options - limit options in lightweight mode
    with st.expander("Advanced Options", expanded=False):
        # Add educational section about advanced options
        st.markdown("""#### Understanding Advanced Options
        These settings control the complexity and behavior of the machine learning process. Adjusting them can improve model performance but may increase computation time.
        """)

        if lightweight_mode:
            st.info("Advanced options are limited in lightweight mode for better performance.")
            tune_hyperparams = False
            cv_splits = 3
            add_technical_indicators = st.checkbox(
                "Add Basic Technical Indicators",
                value=False,
                help="Add a few basic technical indicators as features (may slow down analysis)"
            )

            # Add educational tooltip for technical indicators in lightweight mode
            if add_technical_indicators:
                st.markdown("""<div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <b>Technical Indicators Used:</b><br>
                • <b>SMA (20-day)</b>: Simple Moving Average - average price over 20 days<br>
                • <b>RSI (14-day)</b>: Relative Strength Index - momentum oscillator measuring speed and change of price movements
                </div>""", unsafe_allow_html=True)
        else:
            tune_hyperparams = st.checkbox(
                "Tune Hyperparameters",
                value=False,
                help="Automatically tune model hyperparameters (takes longer)"
            )

            # Add educational tooltip for hyperparameter tuning
            if tune_hyperparams:
                st.markdown("""<div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <b>What are Hyperparameters?</b><br>
                Hyperparameters are configuration settings that control the learning process. The system will automatically test different combinations to find optimal settings for your data.
                </div>""", unsafe_allow_html=True)

            cv_splits = st.slider(
                "Cross-Validation Splits",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of cross-validation splits"
            )

            # Add educational tooltip for cross-validation
            st.markdown("""<div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <b>Cross-Validation</b> divides your data into multiple subsets to test model performance more thoroughly. More splits provide better validation but take longer to compute.
            </div>""", unsafe_allow_html=True)

            add_technical_indicators = st.checkbox(
                "Add Technical Indicators",
                value=True,
                help="Add technical indicators as features"
            )

            # Add educational tooltip for technical indicators
            if add_technical_indicators:
                with st.expander("Learn about Technical Indicators", expanded=False):
                    st.markdown("""Technical indicators are mathematical calculations based on price, volume, or open interest of a security. They help identify patterns and predict future price movements.

                    **Key indicators used in this analysis:**
                    """)

                    for indicator, explanation in list(TECHNICAL_INDICATOR_EXPLANATIONS.items())[:5]:  # Show first 5 indicators
                        st.markdown(f"**{indicator}**: {explanation}")

                    st.markdown("[Click here to learn more about technical indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)")

    # Run ML analysis button
    if st.button("Run ML Analysis", type="primary"):
        with st.spinner("Running ML analysis..."):
            try:
                # Prepare data - use different approaches based on lightweight mode
                if add_technical_indicators:
                    if lightweight_mode:
                        # Add only basic technical indicators in lightweight mode
                        # Calculate SMA and RSI only
                        df['sma_20'] = df['close'].rolling(window=20).mean()

                        # Simple RSI calculation
                        delta = df['close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        rs = gain / loss
                        df['rsi_14'] = 100 - (100 / (1 + rs))
                    else:
                        # Add full technical indicators in normal mode
                        try:
                            df = TechnicalIndicators.calculate_all_indicators(df)
                        except Exception as e:
                            st.warning(f"Some technical indicators could not be calculated: {str(e)}")
                            # Fallback to basic indicators
                            df['sma_20'] = df['close'].rolling(window=20).mean()

                            # Simple RSI calculation
                            delta = df['close'].diff()
                            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                            rs = gain / loss
                            df['rsi_14'] = 100 - (100 / (1 + rs))

                # Prepare features and target
                if target_col == "return_1d":
                    # Calculate returns
                    df["return_1d"] = df["close"].pct_change()

                # Use different feature preparation based on mode
                if lightweight_mode:
                    # Simplified feature preparation for lightweight mode
                    # Create a smaller feature set
                    X = pd.DataFrame(index=df.index)

                    # Add basic features
                    X['close'] = df['close']
                    X['volume'] = df.get('volume', 0)

                    # Add minimal lagged features
                    for lag in range(1, min(n_lags, 3) + 1):  # Limit to max 3 lags in lightweight mode
                        X[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

                    # Add minimal technical indicators if available
                    if 'sma_20' in df.columns:
                        X['sma_20'] = df['sma_20']
                    if 'rsi_14' in df.columns:
                        X['rsi_14'] = df['rsi_14']

                    # Create target variable
                    y = df[target_col].shift(-1)  # Predict next day's value

                    # Drop rows with NaN values
                    valid_idx = X.dropna().index.intersection(y.dropna().index)
                    X = X.loc[valid_idx]
                    y = y.loc[valid_idx]
                else:
                    # Full feature preparation in normal mode
                    X, y = ml_predictor.prepare_features(df, target_col=target_col, n_lags=n_lags)

                # Train model with appropriate settings
                metrics = ml_predictor.train_model(
                    X, y,
                    model_type=model_type,
                    cv_splits=cv_splits,
                    tune_hyperparams=tune_hyperparams
                )

                # Make predictions on training data
                predictions, confidence = ml_predictor.predict(X)

                # Generate forecast - use different approach in lightweight mode
                if lightweight_mode:
                    # Simplified forecast generation for lightweight mode
                    with st.spinner("Generating forecast (lightweight mode)..."):
                        # Get the last date in the data
                        last_date = X.index[-1]

                        # Create future dates - ensure we're using datetime objects
                        try:
                            # If last_date is already a datetime/timestamp
                            if hasattr(last_date, 'to_pydatetime'):
                                start_date = last_date + pd.Timedelta(days=1)
                            else:
                                # If it's a string or other format, try to convert
                                start_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)

                            future_dates = pd.date_range(start=start_date, periods=forecast_periods)
                        except Exception as e:
                            st.warning(f"Error creating forecast dates: {str(e)}")
                            # Fallback to simple integer index
                            future_dates = pd.RangeIndex(start=1, stop=forecast_periods+1)

                        # Make a single prediction and extend it with a simple trend
                        last_prediction, _ = ml_predictor.predict(X.iloc[-1:].copy())
                        base_prediction = last_prediction[0]

                        # Calculate a simple trend based on recent data
                        recent_values = y.iloc[-30:] if len(y) >= 30 else y
                        trend = 0
                        if len(recent_values) > 1:
                            trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)

                        # Generate forecast with trend
                        forecast_values = [base_prediction + trend * i for i in range(forecast_periods)]
                        forecast = pd.Series(forecast_values, index=future_dates)

                        # Generate simple confidence intervals
                        std = recent_values.std() if len(recent_values) > 1 else y.std()
                        lower_bound = [val - 1.96 * std for val in forecast_values]
                        upper_bound = [val + 1.96 * std for val in forecast_values]

                        forecast_conf = pd.DataFrame({
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }, index=future_dates)
                else:
                    # Full forecast generation in normal mode
                    forecast, forecast_conf = ml_predictor.forecast_future(X, y, periods=forecast_periods)

                # Display results
                display_ml_results(
                    ticker,
                    asset_type,
                    df,
                    X,
                    y,
                    metrics,
                    predictions,
                    confidence,
                    forecast,
                    forecast_conf,
                    model_type,
                    target_col,
                    ml_predictor
                )

            except Exception as e:
                st.error(f"Error in ML analysis: {str(e)}")
                st.exception(e)

def display_ml_results(
    ticker: str,
    asset_type: str,  # Used in future extensions for asset-specific explanations
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: Dict[str, float],
    predictions: np.ndarray,
    confidence: Dict[str, Any],
    forecast: pd.Series,
    forecast_conf: pd.DataFrame,
    model_type: str,
    target_col: str,
    ml_predictor: Any
):
    """
    Display ML analysis results.

    Args:
        ticker: Asset ticker symbol
        asset_type: Type of asset
        data: Original data DataFrame
        X: Feature DataFrame
        y: Target Series
        metrics: Model performance metrics
        predictions: Predictions array
        confidence: Confidence metrics dictionary
        forecast: Forecast Series
        forecast_conf: Forecast confidence DataFrame
        model_type: Type of model used
        target_col: Target column name
    """
    # Display model performance metrics with educational content
    st.markdown("### Model Performance")

    # Add educational section about model metrics
    with st.expander("📚 Understanding Model Performance Metrics", expanded=False):
        st.markdown("""
        Model performance metrics help you evaluate how well the model is performing. Here's what each metric means:

        * **R² Score (Coefficient of Determination)**: Measures how well the model explains the variance in the data
          * Range: 0 to 1 (higher is better)
          * 0.7 means the model explains 70% of the variance in the target variable
          * Values close to 1 indicate a good fit

        * **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors
          * Lower values indicate better accuracy
          * In the same units as the target variable (e.g., dollars for price predictions)
          * More sensitive to large errors than MAE

        * **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
          * Lower values indicate better accuracy
          * Less sensitive to outliers than RMSE
          * Also in the same units as the target variable

        * **Feature Count**: Number of input variables used by the model
          * More features can capture more information
          * Too many features can lead to overfitting
        """)

    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        r2_value = metrics.get('mean_r2', 0)
        st.metric("R² Score", f"{r2_value:.4f}")
        # Add color-coded interpretation
        if r2_value > 0.7:
            st.markdown("<span style='color:green'>Strong fit</span>", unsafe_allow_html=True)
        elif r2_value > 0.5:
            st.markdown("<span style='color:orange'>Moderate fit</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>Weak fit</span>", unsafe_allow_html=True)

    with col2:
        rmse_value = metrics.get('mean_rmse', 0)
        st.metric("RMSE", f"{rmse_value:.4f}")
        st.markdown("<span style='font-size:0.8em'>Lower is better</span>", unsafe_allow_html=True)

    with col3:
        mae_value = metrics.get('mean_mae', 0)
        st.metric("MAE", f"{mae_value:.4f}")
        st.markdown("<span style='font-size:0.8em'>Lower is better</span>", unsafe_allow_html=True)

    with col4:
        if model_type in ["random_forest", "gradient_boosting"]:
            st.metric("Feature Count", f"{len(X.columns)}")
            if len(X.columns) > 20:
                st.markdown("<span style='font-size:0.8em'>High complexity</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size:0.8em'>Moderate complexity</span>", unsafe_allow_html=True)
        else:
            st.metric("Model Type", model_type.replace("_", " ").title())
            complexity = ML_MODEL_EXPLANATIONS.get(model_type, {}).get("complexity", "Medium")
            st.markdown(f"<span style='font-size:0.8em'>{complexity} complexity</span>", unsafe_allow_html=True)

    # Display predictions vs actual with educational content
    st.markdown("### Predictions vs Actual")

    # Add educational explanation for the predictions chart
    st.markdown("""
    <div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
    <b>How to Read This Chart:</b> This chart shows how well the model's predictions (red dashed line) match the actual values (blue line).
    The shaded area represents the 95% confidence interval - we expect actual values to fall within this range 95% of the time.
    Closer alignment between predicted and actual lines indicates better model performance.
    </div>
    """, unsafe_allow_html=True)

    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        "actual": y,
        "predicted": predictions[:len(y)],
        "lower_bound": confidence["lower_bound"][:len(y)],
        "upper_bound": confidence["upper_bound"][:len(y)]
    })

    # Create a plotly figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["actual"],
        mode="lines",
        name="Actual",
        line=dict(color="#4f46e5", width=2)
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df["predicted"],
        mode="lines",
        name="Predicted",
        line=dict(color="#ef4444", width=2, dash="dash")
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=results_df.index.tolist() + results_df.index.tolist()[::-1],
        y=results_df["upper_bound"].tolist() + results_df["lower_bound"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.2)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        name="95% Confidence Interval"
    ))

    # Update layout
    fig.update_layout(
        title=f"{ticker.upper()} - {target_col.replace('_', ' ').title()} Prediction",
        xaxis_title="Date",
        yaxis_title=target_col.replace("_", " ").title(),
        height=500,
        template="plotly_dark",
        hovermode="x unified"
    )

    # Display the figure
    st.plotly_chart(fig)

    # Display forecast with educational content
    st.markdown("### Future Forecast")

    # Add educational explanation for the forecast chart
    st.markdown("""
    <div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
    <b>Understanding the Forecast:</b> This chart shows the model's prediction for future values (red dashed line) based on historical data (blue line).
    The green vertical line marks where historical data ends and forecasting begins. The shaded area represents the 95% confidence interval,
    which widens over time as uncertainty increases. Remember that all forecasts are estimates and actual results may vary significantly.
    </div>
    """, unsafe_allow_html=True)

    # Create a plotly figure for forecast
    fig_forecast = go.Figure()

    # Add historical values (last 60 days)
    historical = data[target_col].iloc[-60:]

    fig_forecast.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode="lines",
        name="Historical",
        line=dict(color="#4f46e5", width=2)
    ))

    # Add forecast values
    fig_forecast.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines",
        name="Forecast",
        line=dict(color="#ef4444", width=2, dash="dash")
    ))

    # Add forecast confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=forecast_conf["upper_bound"].tolist() + forecast_conf["lower_bound"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.2)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        name="95% Confidence Interval"
    ))

    # Add vertical line to separate historical and forecast
    try:
        # Get the last timestamp from historical data
        last_timestamp = historical.index[-1]

        # Add vertical line at the exact timestamp (no conversion needed)
        fig_forecast.add_vline(
            x=last_timestamp,  # Use the timestamp directly
            line_dash="dash",
            line_color="green",
            annotation_text="Forecast Start",
            annotation_position="top right"
        )
    except Exception as e:
        # Log the error but don't show warning to user
        log_error(f"Could not add forecast separation line: {str(e)}")
        # Continue without the line

    # Update layout
    fig_forecast.update_layout(
        title=f"{ticker.upper()} - {target_col.replace('_', ' ').title()} Forecast ({len(forecast)} days)",
        xaxis_title="Date",
        yaxis_title=target_col.replace("_", " ").title(),
        height=500,
        template="plotly_dark",
        hovermode="x unified"
    )

    # Display the figure
    st.plotly_chart(fig_forecast)

    # Display feature importance if available with educational content
    if hasattr(ml_predictor, "feature_importance") and ml_predictor.feature_importance:
        st.markdown("### Feature Importance")

        # Add educational explanation for feature importance
        st.markdown("""
        <div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
        <b>What is Feature Importance?</b> This chart shows which input variables (features) have the most influence on the model's predictions.
        Features with higher importance scores have a stronger effect on the prediction outcome. This can help you understand what factors
        are driving the price or returns of this asset according to the model. Technical indicators, price patterns, and volume metrics often
        rank highly for financial assets.
        </div>
        """, unsafe_allow_html=True)

        # Get feature importance
        feature_importance = ml_predictor.feature_importance

        # Sort feature importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Select top 15 features
        top_features = sorted_importance[:15]

        # Create a DataFrame
        importance_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])

        # Create a plotly figure
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Feature Importance",
            color="Importance",
            color_continuous_scale="Viridis"
        )

        # Update layout
        fig_importance.update_layout(
            height=500,
            template="plotly_dark",
            yaxis=dict(autorange="reversed")
        )

        # Display the figure
        st.plotly_chart(fig_importance)

    # Display residuals analysis with educational content
    st.markdown("### Residuals Analysis")

    # Add educational explanation for residuals analysis
    with st.expander("📚 Understanding Residuals Analysis", expanded=False):
        st.markdown("""
        Residuals are the differences between actual values and predicted values. Analyzing residuals helps assess model quality and identify potential issues.

        **What to look for:**

        * **Random scatter around zero**: Ideally, residuals should be randomly distributed around zero with no clear pattern
        * **No trend or curve**: A trend in residuals suggests the model is missing important patterns in the data
        * **Normal distribution**: Residuals should follow a bell-shaped distribution centered at zero
        * **Consistent spread**: The spread of residuals should be similar across all predicted values

        **Common issues revealed by residuals:**

        * **Heteroscedasticity**: Increasing or decreasing spread of residuals across predicted values
        * **Autocorrelation**: Pattern in residuals over time, suggesting time-dependent relationships not captured by the model
        * **Non-linearity**: Curved pattern in residuals, suggesting non-linear relationships not captured by the model
        * **Outliers**: Extreme residual values that may be influencing the model excessively
        """)

    st.markdown("""
    <div style='background-color: rgba(79, 70, 229, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
    <b>Reading the Residuals Plot:</b> This scatter plot shows the difference between actual and predicted values.
    Ideally, points should be randomly scattered around the horizontal line at zero with no clear pattern.
    Any visible patterns may indicate that the model is missing important relationships in the data.
    </div>
    """, unsafe_allow_html=True)

    # Calculate residuals
    residuals = y - predictions[:len(y)]

    # Create a plotly figure for residuals
    fig_residuals = go.Figure()

    # Add residuals scatter plot
    fig_residuals.add_trace(go.Scatter(
        x=y,
        y=residuals,
        mode="markers",
        marker=dict(
            color=residuals,
            colorscale="RdBu",
            colorbar=dict(title="Residual"),
            size=8,
            opacity=0.7
        ),
        name="Residuals"
    ))

    # Add horizontal line at y=0
    fig_residuals.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
        line_width=2
    )

    # Update layout
    fig_residuals.update_layout(
        title="Residuals vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Residuals",
        height=500,
        template="plotly_dark"
    )

    # Display the figure
    st.plotly_chart(fig_residuals)

    # Display residuals distribution
    fig_dist = go.Figure()

    # Add histogram
    fig_dist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color="#4f46e5",
        opacity=0.7,
        name="Residuals Distribution"
    ))

    # Update layout
    fig_dist.update_layout(
        title="Residuals Distribution",
        xaxis_title="Residual",
        yaxis_title="Frequency",
        height=400,
        template="plotly_dark"
    )

    # Display the figure
    st.plotly_chart(fig_dist)

    # Display model details
    with st.expander("Model Details", expanded=False):
        # Display model type and parameters
        st.markdown(f"**Model Type:** {model_type.replace('_', ' ').title()}")

        # Display best parameters if available
        if hasattr(ml_predictor, "best_params") and ml_predictor.best_params:
            st.markdown("**Best Parameters:**")
            for param, value in ml_predictor.best_params.items():
                st.markdown(f"- {param}: {value}")

        # Display detailed metrics
        st.markdown("**Detailed Metrics:**")
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })
        st.dataframe(metrics_df)

        # Display feature list
        st.markdown("**Features Used:**")
        st.dataframe(pd.DataFrame({"Feature": X.columns}))
