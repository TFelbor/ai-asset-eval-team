"""
Comparison UI component for the AI Finance Dashboard.
This module provides a Streamlit UI for comparing different assets.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

from core.teams.comparison_team import ComparisonTeam
from core.analytics.optimized_charts import generate_comparison_chart
from core.analytics.cross_asset_charts import generate_cross_asset_chart
from core.api.yahoo_finance_client import YahooFinanceClient
from core.api.coingecko_client import CoinGeckoClient

def render_comparison_ui():
    """Render the comparison UI component."""
    st.markdown('<h2 class="subheader">🔄 Asset Comparison</h2>', unsafe_allow_html=True)

    # Create tabs for different comparison types
    tabs = st.tabs(["Same Asset Type", "Cross-Asset Type"])

    with tabs[0]:  # Same Asset Type comparison
        render_same_asset_comparison()

    with tabs[1]:  # Cross-Asset Type comparison
        render_cross_asset_comparison()

def render_same_asset_comparison():
    """Render the UI for comparing assets of the same type."""
    # Add a button to return to the main dashboard
    if st.button("Return to Home", key="return_home_same_asset"):
        # Reset all view flags to return to the landing page
        st.session_state.show_comparison = False
        st.session_state.show_docs = False
        st.session_state.show_backtesting = False
        st.session_state.show_news = False
        st.session_state.last_analysis = None
        st.rerun()

    st.markdown("### Compare Multiple Assets of the Same Type")
    st.markdown("Select multiple assets of the same type to compare their performance and metrics.")

    # Asset type selection
    asset_type = st.selectbox(
        "Asset Type",
        ["Stock", "Cryptocurrency", "ETF", "REIT"],
        key="same_asset_type"
    )

    # Initialize tickers list if not in session state
    if 'comparison_tickers' not in st.session_state:
        st.session_state.comparison_tickers = []

    # Create columns for input and buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Add ticker input
        new_ticker = st.text_input("Add Ticker/Symbol", key="same_asset_new_ticker")

    with col2:
        # Add button
        if st.button("Add to Comparison", key="same_asset_add_button"):
            if new_ticker and new_ticker.upper() not in [t.upper() for t in st.session_state.comparison_tickers]:
                st.session_state.comparison_tickers.append(new_ticker.upper())
                st.rerun()

    with col3:
        # Clear button
        if st.button("Clear All", key="same_asset_clear_button"):
            st.session_state.comparison_tickers = []
            st.rerun()

    # Display selected tickers
    if st.session_state.comparison_tickers:
        st.markdown("### Selected Assets")

        # Create columns for the selected tickers
        cols = st.columns(min(len(st.session_state.comparison_tickers), 4))

        # Create a list to store tickers to remove
        tickers_to_remove = []

        for i, ticker in enumerate(st.session_state.comparison_tickers):
            with cols[i % 4]:
                st.markdown(f'<div class="card">{ticker}</div>', unsafe_allow_html=True)
                if st.button(f"Remove", key=f"same_asset_remove_{ticker}"):
                    tickers_to_remove.append(ticker)

        # Remove tickers outside the loop to avoid modifying the list during iteration
        for ticker in tickers_to_remove:
            st.session_state.comparison_tickers.remove(ticker)
            st.rerun()

        # Comparison options
        st.markdown("### Comparison Options")

        col1, col2 = st.columns(2)

        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Price", "Performance", "Correlation"],
                key="same_asset_chart_type"
            )

        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"],
                key="same_asset_time_period"
            )

        # Generate comparison button
        compare_button = st.button("Generate Comparison", key="same_asset_compare_button")

        if compare_button and len(st.session_state.comparison_tickers) > 0:
            with st.spinner("Generating comparison..."):
                try:
                    # Convert time period to days for API calls
                    days = 30  # Default to 30 days
                    if time_period == "3 Months":
                        days = 90
                    elif time_period == "6 Months":
                        days = 180
                    elif time_period == "1 Year":
                        days = 365
                    elif time_period == "3 Years":
                        days = 1095
                    elif time_period == "5 Years":
                        days = 1825

                    # Initialize API based on asset type
                    price_data = {}

                    if asset_type.lower() == "stock" or asset_type.lower() == "etf" or asset_type.lower() == "reit":
                        # Use Yahoo Finance API for stocks, ETFs, and REITs
                        yahoo_client = YahooFinanceClient()

                        # Get data for each ticker
                        for ticker in st.session_state.comparison_tickers:
                            try:
                                # Convert time period to yfinance format
                                period_map = {
                                    "1 month": "1mo",
                                    "3 months": "3mo",
                                    "6 months": "6mo",
                                    "1 year": "1y",
                                    "3 years": "3y",
                                    "5 years": "5y"
                                }
                                period = period_map.get(time_period.lower(), "1mo")

                                # Get historical data
                                hist_data = yahoo_client.get_historical_data(ticker, period=period)

                                if isinstance(hist_data, pd.DataFrame) and not hist_data.empty:
                                    # Convert DataFrame to price history format
                                    price_data[ticker] = {
                                        "timestamps": hist_data.index.tolist(),
                                        "prices": hist_data["Close"].tolist(),
                                        "volumes": hist_data["Volume"].tolist() if "Volume" in hist_data.columns else []
                                    }
                                elif isinstance(hist_data, dict):
                                    # Handle case where hist_data is already a dictionary
                                    if "Close" in hist_data and len(hist_data["Close"]) > 0:
                                        price_data[ticker] = {
                                            "timestamps": hist_data.get("index", []),
                                            "prices": hist_data.get("Close", []),
                                            "volumes": hist_data.get("Volume", []) if "Volume" in hist_data else []
                                        }
                            except Exception as e:
                                st.error(f"Error fetching data for {ticker}: {str(e)}")

                    elif asset_type.lower() == "cryptocurrency":
                        # Use CoinGecko API for cryptocurrencies
                        coingecko_client = CoinGeckoClient()

                        # Get data for each ticker
                        for ticker in st.session_state.comparison_tickers:
                            try:
                                # Get market data using the correct method
                                market_data = coingecko_client.get_coin_price_history(ticker, days=days)

                                if market_data and "prices" in market_data:
                                    # Extract price and volume data
                                    timestamps = [datetime.fromtimestamp(price[0]/1000) for price in market_data["prices"]]
                                    prices = [price[1] for price in market_data["prices"]]
                                    volumes = [volume[1] for volume in market_data["total_volumes"]] if "total_volumes" in market_data else []

                                    # Store in price_data
                                    price_data[ticker] = {
                                        "timestamps": timestamps,
                                        "prices": prices,
                                        "volumes": volumes
                                    }
                            except Exception as e:
                                st.error(f"Error fetching data for {ticker}: {str(e)}")

                    # Generate comparison chart
                    if price_data:
                        if chart_type.lower() == "price":
                            fig = generate_comparison_chart(
                                st.session_state.comparison_tickers,
                                price_data,
                                "absolute"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif chart_type.lower() == "performance":
                            fig = generate_comparison_chart(
                                st.session_state.comparison_tickers,
                                price_data,
                                "normalized"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif chart_type.lower() == "correlation":
                            fig = generate_comparison_chart(
                                st.session_state.comparison_tickers,
                                price_data,
                                "correlation"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Generate comparison metrics
                        generate_comparison_metrics(st.session_state.comparison_tickers, price_data, asset_type)
                    else:
                        st.warning("No data available for the selected assets.")

                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")
    else:
        st.info("Add at least two assets to compare.")

def render_cross_asset_comparison():
    """Render the UI for comparing assets of different types."""
    # Add a button to return to the main dashboard
    if st.button("Return to Home", key="return_home_cross_asset"):
        # Reset all view flags to return to the landing page
        st.session_state.show_comparison = False
        st.session_state.show_docs = False
        st.session_state.show_backtesting = False
        st.session_state.show_news = False
        st.session_state.last_analysis = None
        st.rerun()

    st.markdown("### Compare Assets of Different Types")
    st.markdown("Select assets of different types to compare their performance and characteristics.")

    # Create columns for different asset types
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("#### Stock")
        stock_ticker = st.text_input("Stock Ticker", key="cross_asset_stock")

    with col2:
        st.markdown("#### Cryptocurrency")
        crypto_ticker = st.text_input("Crypto Symbol/ID", key="cross_asset_crypto")

    with col3:
        st.markdown("#### ETF")
        etf_ticker = st.text_input("ETF Ticker", key="cross_asset_etf")

    with col4:
        st.markdown("#### REIT")
        reit_ticker = st.text_input("REIT Ticker", key="cross_asset_reit")

    # Time period selection (for UI consistency, not used in the API call)
    st.selectbox(
        "Time Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"],
        key="cross_asset_time_period"
    )

    # Generate comparison button
    compare_button = st.button("Generate Comparison", key="cross_asset_compare_button")

    if compare_button:
        # Check if at least two assets are selected
        selected_assets = []
        if stock_ticker:
            selected_assets.append(("stock", stock_ticker))
        if crypto_ticker:
            selected_assets.append(("crypto", crypto_ticker))
        if etf_ticker:
            selected_assets.append(("etf", etf_ticker))
        if reit_ticker:
            selected_assets.append(("reit", reit_ticker))

        if len(selected_assets) < 2:
            st.warning("Please select at least two different assets to compare.")
        else:
            with st.spinner("Generating cross-asset comparison..."):
                try:
                    # Initialize comparison team
                    comparison_team = ComparisonTeam()

                    # Prepare assets dictionary for cross-asset comparison
                    assets = {}
                    for i, (asset_type, ticker) in enumerate(selected_assets):
                        asset_id = f"asset{i+1}"
                        assets[asset_id] = (ticker, asset_type)

                    # Generate comparison using the new cross-asset comparison method
                    comparison_result = comparison_team.compare_cross_asset(assets)

                    # Display comparison results
                    display_cross_asset_comparison(comparison_result, selected_assets)

                except Exception as e:
                    st.error(f"Error generating cross-asset comparison: {str(e)}")

def generate_comparison_metrics(tickers, price_data, _):
    """Generate and display comparison metrics for assets of the same type."""
    st.markdown("### Comparison Metrics")

    # Calculate metrics for each ticker
    metrics = {}

    for ticker in tickers:
        if ticker in price_data:
            prices = price_data[ticker].get("prices", [])
            timestamps = price_data[ticker].get("timestamps", [])

            if prices and len(prices) > 1:
                # Calculate returns
                total_return = (prices[-1] / prices[0] - 1) * 100

                # Calculate volatility (annualized standard deviation of returns)
                returns = []
                for i in range(1, len(prices)):
                    daily_return = (prices[i] / prices[i-1]) - 1
                    returns.append(daily_return)

                volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized

                # Calculate max drawdown
                max_drawdown = 0
                peak = prices[0]

                for price in prices:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                # Store metrics
                metrics[ticker] = {
                    "current_price": f"${prices[-1]:.2f}",
                    "total_return": f"{total_return:.2f}%",
                    "volatility": f"{volatility:.2f}%",
                    "max_drawdown": f"{max_drawdown*100:.2f}%",
                    "start_date": timestamps[0].strftime("%Y-%m-%d") if isinstance(timestamps[0], datetime) else "N/A",
                    "end_date": timestamps[-1].strftime("%Y-%m-%d") if isinstance(timestamps[-1], datetime) else "N/A"
                }

    # Display metrics in a table
    if metrics:
        # Create DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

        # Display the table
        st.dataframe(metrics_df, use_container_width=True)

        # Create a bar chart for total return
        fig = go.Figure()

        # Extract total returns (remove % sign and convert to float)
        returns = {ticker: float(metrics[ticker]["total_return"].replace("%", "")) for ticker in metrics}

        # Add bars
        fig.add_trace(go.Bar(
            x=list(returns.keys()),
            y=list(returns.values()),
            marker_color=['#10b981' if val >= 0 else '#ef4444' for val in returns.values()]
        ))

        # Update layout
        fig.update_layout(
            title="Total Return Comparison",
            xaxis_title="Asset",
            yaxis_title="Total Return (%)",
            height=400,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create a bar chart for volatility
        fig = go.Figure()

        # Extract volatilities (remove % sign and convert to float)
        volatilities = {ticker: float(metrics[ticker]["volatility"].replace("%", "")) for ticker in metrics}

        # Add bars
        fig.add_trace(go.Bar(
            x=list(volatilities.keys()),
            y=list(volatilities.values()),
            marker_color='#4f46e5'
        ))

        # Update layout
        fig.update_layout(
            title="Volatility Comparison",
            xaxis_title="Asset",
            yaxis_title="Volatility (% annualized)",
            height=400,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics available for the selected assets.")

def display_cross_asset_comparison(comparison_result, _):
    """Display cross-asset comparison results."""
    # Check for errors
    if "error" in comparison_result:
        st.error(f"Error in comparison: {comparison_result['error']}")
        return

    # Display asset information
    st.markdown("### Assets Being Compared")

    asset_info = comparison_result.get("asset_info", {})
    if asset_info:
        # Create a DataFrame for asset info
        asset_data = []
        for asset_id, info in asset_info.items():
            asset_data.append({
                "Asset ID": asset_id,
                "Ticker": info.get("ticker", ""),
                "Name": info.get("name", ""),
                "Type": info.get("asset_type", "").upper()
            })

        asset_df = pd.DataFrame(asset_data)
        st.dataframe(asset_df, use_container_width=True)

    # Add chart options
    st.markdown("### Comparison Charts")

    # Create tabs for different chart types
    chart_tabs = st.tabs(["Price", "Normalized", "Correlation", "Allocation"])

    with chart_tabs[0]:  # Price chart
        st.markdown("#### Price Comparison")
        try:
            fig = generate_cross_asset_chart(
                "price",
                asset_info,
                comparison_result.get("price_histories", {})
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate price chart: {str(e)}")

    with chart_tabs[1]:  # Normalized chart
        st.markdown("#### Normalized Price Comparison (% Change)")
        try:
            fig = generate_cross_asset_chart(
                "normalized",
                asset_info,
                comparison_result.get("price_histories", {})
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate normalized chart: {str(e)}")

    with chart_tabs[2]:  # Correlation chart
        st.markdown("#### Correlation Matrix")
        try:
            fig = generate_cross_asset_chart(
                "correlation",
                asset_info,
                comparison_result.get("price_histories", {}),
                comparison_result.get("correlation_matrix")
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate correlation chart: {str(e)}")

    with chart_tabs[3]:  # Allocation chart
        st.markdown("#### Recommended Allocation")
        try:
            fig = generate_cross_asset_chart(
                "allocation",
                asset_info,
                comparison_result.get("price_histories", {}),
                portfolio_allocation=comparison_result.get("portfolio_allocation")
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate allocation chart: {str(e)}")

    # Display comparison analysis
    st.markdown("### Comparison Analysis")

    comparison = comparison_result.get("comparison", {})
    if comparison:
        # Create tabs for different aspects of the comparison
        tabs = st.tabs(["Performance", "Risk", "Diversification", "Outlook"])

        with tabs[0]:
            st.markdown("#### Performance Analysis")
            st.markdown(comparison.get("performance", "No performance analysis available."))

            # Display performance data
            performance_data = comparison_result.get("performance_data", {})
            if performance_data:
                # Create a DataFrame for performance metrics
                perf_data = []
                for asset_id, metrics in performance_data.items():
                    asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
                    perf_data.append({
                        "Asset": asset_name,
                        "Total Return": metrics.get("total_return", "N/A"),
                        "Volatility": metrics.get("volatility", "N/A"),
                        "Current Price": metrics.get("current_price", "N/A"),
                        "Start Price": metrics.get("start_price", "N/A")
                    })

                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)

                # Create a bar chart for total returns
                try:
                    fig = go.Figure()

                    # Extract total returns (remove % sign and convert to float)
                    returns = {}
                    for asset_id, metrics in performance_data.items():
                        asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
                        try:
                            return_str = metrics.get("total_return", "0%")
                            return_val = float(return_str.replace("%", ""))
                            returns[asset_name] = return_val
                        except (ValueError, TypeError):
                            continue

                    if returns:
                        # Add bars
                        fig.add_trace(go.Bar(
                            x=list(returns.keys()),
                            y=list(returns.values()),
                            marker_color=['#10b981' if val >= 0 else '#ef4444' for val in returns.values()]
                        ))

                        # Update layout
                        fig.update_layout(
                            title="Total Return Comparison",
                            xaxis_title="Asset",
                            yaxis_title="Total Return (%)",
                            height=400,
                            template="plotly_dark"
                        )

                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create performance chart: {str(e)}")

        with tabs[1]:
            st.markdown("#### Risk Analysis")
            st.markdown(comparison.get("risk", "No risk analysis available."))

            # Display correlation matrix if available
            correlation_matrix = comparison_result.get("correlation_matrix")
            if correlation_matrix:
                st.markdown("#### Correlation Matrix")

                # Convert correlation matrix to DataFrame
                corr_df = pd.DataFrame(correlation_matrix)

                # Replace asset IDs with asset names in the index and columns
                asset_names = {asset_id: info.get("name", asset_id) for asset_id, info in asset_info.items()}
                corr_df.index = [asset_names.get(idx, idx) for idx in corr_df.index]
                corr_df.columns = [asset_names.get(col, col) for col in corr_df.columns]

                # Display the correlation matrix
                st.dataframe(corr_df, use_container_width=True)

                # Create a heatmap
                try:
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.index,
                        colorscale='Viridis',
                        zmin=-1,
                        zmax=1
                    ))

                    fig.update_layout(
                        title="Correlation Matrix",
                        height=500,
                        template="plotly_dark"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create correlation heatmap: {str(e)}")

        with tabs[2]:
            st.markdown("#### Diversification Benefits")
            st.markdown(comparison.get("diversification", "No diversification analysis available."))

        with tabs[3]:
            st.markdown("#### Investment Outlook")
            st.markdown(comparison.get("outlook", "No investment outlook available."))
    else:
        st.info("No comparison analysis available.")

    # Display asset ranking
    st.markdown("### Asset Ranking")

    ranking = comparison_result.get("ranking", [])
    if ranking:
        # Create a DataFrame for the ranking
        ranking_data = []
        for rank_info in ranking:
            asset_id = rank_info.get("asset_id", "")
            asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
            asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")

            ranking_data.append({
                "Rank": rank_info.get("rank", 0),
                "Asset": f"{asset_name} ({asset_ticker})",
                "Rationale": rank_info.get("rationale", "")
            })

        # Sort by rank
        ranking_df = pd.DataFrame(ranking_data).sort_values("Rank")

        # Display the ranking
        st.dataframe(ranking_df, use_container_width=True)
    else:
        st.info("No ranking available.")

    # Display asset analysis
    st.markdown("### Asset Analysis")

    asset_analysis = comparison_result.get("asset_analysis", {})
    if asset_analysis:
        # Create tabs for each asset
        asset_tabs = st.tabs([asset_info.get(asset_id, {}).get("name", asset_id) for asset_id in asset_analysis.keys()])

        for i, (asset_id, analysis) in enumerate(asset_analysis.items()):
            with asset_tabs[i]:
                asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
                asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")
                asset_type = asset_info.get(asset_id, {}).get("asset_type", "").upper()

                st.markdown(f"#### {asset_name} ({asset_ticker}) - {asset_type}")

                # Display strengths
                st.markdown("**Strengths:**")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for strength in strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.info("No strengths listed.")

                # Display weaknesses
                st.markdown("**Weaknesses:**")
                weaknesses = analysis.get("weaknesses", [])
                if weaknesses:
                    for weakness in weaknesses:
                        st.markdown(f"- {weakness}")
                else:
                    st.info("No weaknesses listed.")
    else:
        st.info("No asset analysis available.")

    # Display portfolio allocation recommendations
    st.markdown("### Portfolio Allocation Recommendations")

    portfolio_allocation = comparison_result.get("portfolio_allocation", {})
    if portfolio_allocation:
        # Display overall recommendation
        st.markdown("#### Overall Recommendation")
        st.markdown(portfolio_allocation.get("recommendation", "No overall recommendation available."))

        # Display allocation percentages
        allocations = portfolio_allocation.get("allocations", [])
        if allocations:
            # Create a DataFrame for the allocations
            allocation_data = []
            for allocation in allocations:
                asset_id = allocation.get("asset_id", "")
                asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
                asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")

                allocation_data.append({
                    "Asset": f"{asset_name} ({asset_ticker})",
                    "Percentage": f"{allocation.get('percentage', 0)}%",
                    "Rationale": allocation.get("rationale", "")
                })

            # Display the allocations
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, use_container_width=True)

            # Create a pie chart for the allocations
            try:
                fig = go.Figure(data=[go.Pie(
                    labels=[f"{asset_info.get(alloc['asset_id'], {}).get('name', alloc['asset_id'])} ({asset_info.get(alloc['asset_id'], {}).get('ticker', '')})" for alloc in allocations],
                    values=[alloc.get("percentage", 0) for alloc in allocations],
                    hole=.3
                )])

                fig.update_layout(
                    title="Recommended Portfolio Allocation",
                    height=500,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create allocation chart: {str(e)}")
        else:
            st.info("No allocation percentages available.")
    else:
        st.info("No portfolio allocation recommendations available.")
