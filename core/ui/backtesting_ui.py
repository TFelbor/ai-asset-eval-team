"""
Backtesting UI component for the AI Finance Dashboard.
This module provides a Streamlit UI for backtesting trading strategies.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.utils.strategy_utils import get_strategy_description
from core.analytics.backtesting import run_backtest

def render_backtesting_ui():
    """Render the backtesting UI component."""
    st.markdown('<h2 class="subheader">🧪 Strategy Backtesting</h2>', unsafe_allow_html=True)

    # Add a button to return to the main dashboard
    if st.button("Return to Home", key="return_home_backtesting"):
        # Reset all view flags to return to the landing page
        st.session_state.show_comparison = False
        st.session_state.show_docs = False
        st.session_state.show_backtesting = False
        st.session_state.show_news = False
        st.session_state.last_analysis = None
        st.rerun()

    # Add a description
    st.markdown("""
    Backtest trading strategies on historical data to evaluate their performance.
    Select a stock, choose a strategy, and configure parameters to see how the strategy would have performed.
    """)

    # Create columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # Stock ticker input
        ticker = st.text_input("Stock Ticker", "AAPL")

        # Strategy selection
        strategy = st.selectbox(
            "Trading Strategy",
            ["Moving Average Crossover", "RSI", "MACD"]
        )

        # Display strategy description
        st.info(get_strategy_description(strategy))

        # Time period selection
        period = st.selectbox(
            "Time Period",
            ["1y", "2y", "3y", "5y", "10y"],
            index=2
        )

    with col2:
        # Strategy-specific parameters
        if strategy == "Moving Average Crossover":
            st.subheader("Strategy Parameters")
            short_window = st.slider("Short Window", 5, 100, 50, 5)
            long_window = st.slider("Long Window", 50, 300, 200, 10)

            # Ensure short window is less than long window
            if short_window >= long_window:
                st.warning("Short window must be less than long window. Adjusting...")
                short_window = min(short_window, long_window - 10)

            strategy_params = {
                "short_window": short_window,
                "long_window": long_window
            }
            strategy_name = "ma_cross"

        elif strategy == "RSI":
            st.subheader("Strategy Parameters")
            rsi_window = st.slider("RSI Window", 5, 30, 14, 1)
            rsi_overbought = st.slider("Overbought Level", 60, 90, 70, 5)
            rsi_oversold = st.slider("Oversold Level", 10, 40, 30, 5)

            strategy_params = {
                "window": rsi_window,
                "overbought": rsi_overbought,
                "oversold": rsi_oversold
            }
            strategy_name = "rsi"

        elif strategy == "MACD":
            st.subheader("Strategy Parameters")
            macd_fast = st.slider("Fast Period", 5, 20, 12, 1)
            macd_slow = st.slider("Slow Period", 15, 40, 26, 1)
            macd_signal = st.slider("Signal Period", 5, 15, 9, 1)

            strategy_params = {
                "fast_period": macd_fast,
                "slow_period": macd_slow,
                "signal_period": macd_signal
            }
            strategy_name = "macd"

    # Initial capital input
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

    # Run backtest button
    run_button = st.button("Run Backtest", key="run_backtest_button")

    if run_button:
        with st.spinner(f"Running {strategy} backtest on {ticker}..."):
            try:
                # Update strategy params with initial capital
                strategy_params["initial_capital"] = initial_capital

                # Run the backtest
                results = run_backtest(ticker, strategy_name, strategy_params, period)

                if "error" in results:
                    st.error(f"Error running backtest: {results['error']}")
                else:
                    # Display results
                    display_backtest_results(results, ticker, strategy)
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

def display_backtest_results(results, ticker, strategy_name):
    """Display backtest results."""
    # Create tabs for different sections
    tabs = st.tabs(["Performance", "Trades", "Chart", "Raw Data"])

    with tabs[0]:  # Performance tab
        st.markdown(f"### {ticker} - {strategy_name} Performance")

        # Create metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{results['total_return']*100:.2f}%")
            st.metric("Initial Capital", f"${results['initial_capital']:,.2f}")
            st.metric("Final Value", f"${results['final_value']:,.2f}")
        with col2:
            st.metric("Annual Return", f"{results['annual_return']*100:.2f}%")
            st.metric("Benchmark Return", f"{results['benchmark_return']*100:.2f}%")
            st.metric("Alpha", f"{results['alpha']*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{results['max_drawdown']*100:.2f}%")
            st.metric("Beta", f"{results['beta']:.2f}")

        # Create performance chart
        portfolio = results.get("portfolio", pd.DataFrame())
        if not portfolio.empty:
            fig = go.Figure()

            # Add portfolio value line
            fig.add_trace(go.Scatter(
                x=portfolio.index,
                y=portfolio["total"],
                mode="lines",
                name="Strategy",
                line=dict(color="#4f46e5", width=2)
            ))

            # Add benchmark line
            if "benchmark" in portfolio.columns:
                fig.add_trace(go.Scatter(
                    x=portfolio.index,
                    y=portfolio["benchmark"],
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="#f59e0b", width=2, dash="dash")
                ))

            # Update layout
            fig.update_layout(
                title=f"{ticker} - {strategy_name} Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                template="plotly_dark",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:  # Trades tab
        st.markdown(f"### {ticker} - {strategy_name} Trades")

        # Display trade statistics
        st.metric("Number of Trades", results["num_trades"])

        # Display trade list
        trades = results.get("trades", [])
        if trades:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)

            # Format the DataFrame
            if not trades_df.empty:
                # Format date column
                if "date" in trades_df.columns:
                    trades_df["date"] = pd.to_datetime(trades_df["date"])

                # Format numeric columns
                numeric_cols = ["price", "shares", "value"]
                for col in numeric_cols:
                    if col in trades_df.columns:
                        trades_df[col] = trades_df[col].round(2)

                # Display the DataFrame
                st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades were executed during the backtest period.")

    with tabs[2]:  # Chart tab
        st.markdown(f"### {ticker} - {strategy_name} Chart")

        # Display the chart
        if "chart" in results:
            # If chart is a base64 string, decode and display
            if isinstance(results["chart"], str):
                if results["chart"].startswith("data:image"):
                    st.image(results["chart"])
                elif results["chart"]:
                    # Try to display it anyway, assuming it's a valid base64 string
                    try:
                        img_str = f"data:image/png;base64,{results['chart']}"
                        st.image(img_str)
                    except Exception as e:
                        st.warning(f"Could not display chart: {str(e)}")
                else:
                    st.warning("No chart data available.")
            else:
                st.warning("Chart data is not in the expected format.")
        else:
            # Create a chart from portfolio data
            portfolio = results.get("portfolio", pd.DataFrame())
            if not portfolio.empty:
                fig = go.Figure()

                # Add price line
                fig.add_trace(go.Scatter(
                    x=portfolio.index,
                    y=portfolio["price"],
                    mode="lines",
                    name="Price",
                    line=dict(color="#10b981", width=1)
                ))

                # Add buy signals
                buy_signals = portfolio[portfolio["signal"] == 1].index
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals,
                        y=portfolio.loc[buy_signals, "price"],
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(color="#10b981", size=10, symbol="triangle-up")
                    ))

                # Add sell signals
                sell_signals = portfolio[portfolio["signal"] == -1].index
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals,
                        y=portfolio.loc[sell_signals, "price"],
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(color="#ef4444", size=10, symbol="triangle-down")
                    ))

                # Update layout
                fig.update_layout(
                    title=f"{ticker} - {strategy_name} Signals",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    template="plotly_dark",
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No portfolio data available to create chart.")

    with tabs[3]:  # Raw Data tab
        st.markdown(f"### {ticker} - {strategy_name} Raw Data")

        # Display raw results
        with st.expander("Raw Results"):
            # Create a copy of results without large objects
            display_results = results.copy()
            if "portfolio" in display_results:
                display_results["portfolio"] = "DataFrame (too large to display)"
            if "chart" in display_results:
                display_results["chart"] = "Base64 image data (too large to display)"

            st.json(display_results)

# Strategy descriptions moved to utils/strategy_utils.py
