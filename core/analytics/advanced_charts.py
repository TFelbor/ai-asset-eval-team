"""
Advanced chart types for the AI Finance Dashboard.
This module provides advanced chart types for financial analysis.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

def create_candlestick_chart(ticker: str, price_history: Dict[str, Any]) -> go.Figure:
    """
    Create a candlestick chart with volume.

    Args:
        ticker: Asset ticker symbol
        price_history: Dictionary with price history data

    Returns:
        Plotly figure object
    """
    # Extract data from price_history
    timestamps = price_history.get("timestamps", [])
    opens = price_history.get("opens", [])
    highs = price_history.get("highs", [])
    lows = price_history.get("lows", [])
    closes = price_history.get("closes", [])
    volumes = price_history.get("volumes", [])

    # Check if we have OHLC data
    if not all([timestamps, opens, highs, lows, closes]):
        # If we only have close prices, create synthetic OHLC data
        if timestamps and "prices" in price_history:
            # Ensure all prices are floats
            try:
                closes = [float(p) for p in price_history["prices"]]
                # Create synthetic OHLC data
                opens = [closes[0]] + closes[:-1]
                highs = [max(float(o), float(c)) * (1 + np.random.uniform(0, 0.005)) for o, c in zip(opens, closes)]
                lows = [min(float(o), float(c)) * (1 - np.random.uniform(0, 0.005)) for o, c in zip(opens, closes)]
            except (ValueError, TypeError) as e:
                # Log the error
                print(f"Error converting price data to floats: {str(e)}")
                # Not enough valid data for a candlestick chart
                fig = go.Figure()
                fig.add_annotation(text="Error processing price data for candlestick chart", showarrow=False, font_size=20)
                return fig
        else:
            # Not enough data for a candlestick chart
            fig = go.Figure()
            fig.add_annotation(text="Not enough data for candlestick chart", showarrow=False, font_size=20)
            return fig

    # Convert timestamps to datetime if they're not already
    try:
        if timestamps and not isinstance(timestamps[0], datetime):
            if isinstance(timestamps[0], str):
                # Handle different string formats
                try:
                    timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                except ValueError:
                    # Try a different format
                    from dateutil import parser
                    timestamps = [parser.parse(ts) for ts in timestamps]
            elif isinstance(timestamps[0], (int, float)):
                # Convert milliseconds to seconds if needed
                if timestamps[0] > 1e10:  # Likely milliseconds
                    timestamps = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
                else:
                    timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
    except Exception as e:
        print(f"Error converting timestamps: {str(e)}")
        # Create a fallback figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error processing timestamps: {str(e)}", showarrow=False, font_size=20)
        return fig

    # Create figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name=ticker,
        increasing_line_color='#10b981',  # Green
        decreasing_line_color='#ef4444'   # Red
    ))

    # Add volume bars if available
    if volumes:
        # Create a secondary y-axis for volume
        fig.add_trace(go.Bar(
            x=timestamps,
            y=volumes,
            name='Volume',
            marker_color='rgba(79, 70, 229, 0.3)',
            yaxis='y2'
        ))

        # Update layout for dual y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_technical_chart(ticker: str, price_history: Dict[str, Any], indicators: List[str] = None) -> go.Figure:
    """
    Create a technical analysis chart with indicators.

    Args:
        ticker: Asset ticker symbol
        price_history: Dictionary with price history data
        indicators: List of indicators to include (e.g., ["rsi", "macd", "bollinger"])

    Returns:
        Plotly figure object
    """
    # Default indicators if none provided
    if indicators is None:
        indicators = ["rsi", "macd", "bollinger"]

    # Extract data from price_history
    timestamps = price_history.get("timestamps", [])
    prices = price_history.get("prices", [])

    # Check if we have enough data
    if not timestamps or not prices or len(prices) < 30:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for technical analysis", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime if they're not already
    if not isinstance(timestamps[0], datetime):
        if isinstance(timestamps[0], str):
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        elif isinstance(timestamps[0], (int, float)):
            timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]

    # Create DataFrame for calculations
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices
    })

    # Create subplots based on selected indicators
    subplot_count = 1 + sum(1 for ind in indicators if ind in ["rsi", "macd"])
    subplot_heights = [0.6] + [0.2] * (subplot_count - 1)

    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=subplot_heights,
        subplot_titles=[f"{ticker} - Price"] + [ind.upper() for ind in indicators if ind in ["rsi", "macd"]]
    )

    # Add price trace to main subplot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='#4f46e5', width=2)
        ),
        row=1, col=1
    )

    # Add indicators
    current_row = 2

    # Add Bollinger Bands to price chart
    if "bollinger" in indicators:
        # Calculate Bollinger Bands
        window = 20
        df['sma'] = df['price'].rolling(window=window).mean()
        df['std'] = df['price'].rolling(window=window).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)

        # Add SMA
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma'],
                mode='lines',
                name=f'SMA ({window})',
                line=dict(color='#f59e0b', width=1.5)
            ),
            row=1, col=1
        )

        # Add upper band
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['upper_band'],
                mode='lines',
                name='Upper Band',
                line=dict(color='#10b981', width=1, dash='dash')
            ),
            row=1, col=1
        )

        # Add lower band
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['lower_band'],
                mode='lines',
                name='Lower Band',
                line=dict(color='#ef4444', width=1, dash='dash')
            ),
            row=1, col=1
        )

    # Add RSI
    if "rsi" in indicators:
        # Calculate RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Add RSI trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#4f46e5', width=1.5)
            ),
            row=current_row, col=1
        )

        # Add overbought/oversold lines
        fig.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=70,
            x1=df['timestamp'].iloc[-1],
            y1=70,
            line=dict(color="#ef4444", width=1, dash="dash"),
            row=current_row, col=1
        )

        fig.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=30,
            x1=df['timestamp'].iloc[-1],
            y1=30,
            line=dict(color="#10b981", width=1, dash="dash"),
            row=current_row, col=1
        )

        # Update y-axis range
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        current_row += 1

    # Add MACD
    if "macd" in indicators:
        # Calculate MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']

        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='#4f46e5', width=1.5)
            ),
            row=current_row, col=1
        )

        # Add signal line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#f59e0b', width=1.5)
            ),
            row=current_row, col=1
        )

        # Add histogram
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['histogram'],
                name='Histogram',
                marker_color=np.where(df['histogram'] >= 0, '#10b981', '#ef4444')
            ),
            row=current_row, col=1
        )

        current_row += 1

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_price_volume_chart(ticker: str, price_history: Dict[str, Any]) -> go.Figure:
    """
    Create a price and volume chart.

    Args:
        ticker: Asset ticker symbol
        price_history: Dictionary with price history data

    Returns:
        Plotly figure object
    """
    # Extract data from price_history
    timestamps = price_history.get("timestamps", [])
    prices = price_history.get("prices", [])
    volumes = price_history.get("volumes", [])

    # Check if we have enough data
    if not timestamps or not prices:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for price & volume chart", showarrow=False, font_size=20)
        return fig

    # Convert timestamps to datetime if they're not already
    if not isinstance(timestamps[0], datetime):
        if isinstance(timestamps[0], str):
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        elif isinstance(timestamps[0], (int, float)):
            timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='#4f46e5', width=2)
    ))

    # Add volume bars if available
    if volumes:
        # Add volume bars on secondary y-axis
        fig.add_trace(go.Bar(
            x=timestamps,
            y=volumes,
            name='Volume',
            marker_color='rgba(79, 70, 229, 0.3)',
            yaxis='y2'
        ))

        # Update layout for dual y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} - Price & Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_correlation_matrix(tickers: List[str], price_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create a correlation matrix for multiple assets.

    Args:
        tickers: List of asset ticker symbols
        price_data: Dictionary with price data for each ticker

    Returns:
        Plotly figure object
    """
    # Check if we have enough data
    if not tickers or len(tickers) < 2 or not price_data:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for correlation matrix", showarrow=False, font_size=20)
        return fig

    # Create DataFrame with returns for each ticker
    returns_df = pd.DataFrame()

    for ticker in tickers:
        if ticker in price_data:
            prices = price_data[ticker].get("prices", [])
            if prices:
                # Calculate returns
                returns = [0]
                for i in range(1, len(prices)):
                    returns.append((prices[i] / prices[i-1]) - 1)

                # Add to DataFrame
                returns_df[ticker] = returns

    # Check if we have enough data
    if returns_df.empty or returns_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for correlation matrix", showarrow=False, font_size=20)
        return fig

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    # Update layout
    fig.update_layout(
        title="Correlation Matrix",
        height=600,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig

# Import proper make_subplots from plotly
from plotly.subplots import make_subplots

def create_prediction_chart(data: pd.DataFrame) -> go.Figure:
    """Create prediction visualization chart"""
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        name='Historical Price',
        line=dict(color='#4f46e5')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['predictions'],
        name='Predicted Price',
        line=dict(color='#10b981', dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['upper_bound'],
        fill=None,
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['lower_bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(16, 185, 129, 0)',
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Price Predictions with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified"
    )
    
    return fig

def create_feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    """Create feature importance visualization"""
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    fig = go.Figure(go.Bar(
        x=[x[1] for x in sorted_features],
        y=[x[0] for x in sorted_features],
        orientation='h',
        marker_color='#4f46e5'
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template="plotly_dark",
        height=400 + len(feature_importance) * 20
    )
    
    return fig
