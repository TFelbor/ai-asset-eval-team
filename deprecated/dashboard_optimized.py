"""
AI Finance Dashboard - Optimized Version
This is the main Streamlit dashboard file for the AI Finance Dashboard.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import time

# Import unified services
from app.services.data_service import DataService
from core.utils.server_manager import ServerManager
from analytics.direct_charts import generate_direct_chart

# Import analysis teams
from teams.stock_team import StockAnalysisTeam
from teams.crypto_team import CryptoAnalysisTeam
from teams.reit_team import REITAnalysisTeam
from teams.etf_team import ETFAnalysisTeam

# Initialize services
data_service = DataService()

# Function to safely shutdown the Streamlit server
def shutdown_server():
    """Safely shutdown the Streamlit server"""
    return ServerManager.shutdown()

# Set page config
st.set_page_config(
    page_title="AI Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def apply_custom_css():
    """Apply custom CSS to the dashboard"""
    st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 4px;
        border: 1px solid #4f46e5;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        border: 1px solid #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stTab {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stTab[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4f46e5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Sidebar
def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        st.title("AI Finance Dashboard")
        st.markdown("---")

        # Asset type selection
        asset_type = st.selectbox(
            "Asset Type",
            ["Stock", "Cryptocurrency", "REIT", "ETF"],
            key="asset_type"
        )

        # Ticker input
        ticker = st.text_input("Ticker Symbol", key="ticker")

        # Analyze button
        analyze_button = st.button("Analyze", key="analyze_button")

        st.markdown("---")

        # Navigation
        st.subheader("Navigation")

        if st.button("Home"):
            st.session_state.page = "home"
            st.experimental_rerun()

        if st.button("Comparison"):
            st.session_state.page = "comparison"
            st.experimental_rerun()

        if st.button("Documentation"):
            st.session_state.page = "docs"
            st.experimental_rerun()

        st.markdown("---")

        # Server management
        st.subheader("Server Management")

        if st.button("Shutdown Server"):
            if shutdown_server():
                st.success("Server shutdown initiated. You will need to restart the server to use the dashboard again.")
                st.info("After the server shuts down, you'll see a page with instructions on how to restart it.")
                time.sleep(1)

        st.markdown("---")

        # Footer
        st.caption("AI Finance Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

    return asset_type, ticker, analyze_button

# Generate chart options based on asset type
def generate_stock_charts(_=None):
    """Generate chart options for stock analysis"""
    return [
        {"type": "price", "title": "Price Chart"},
        {"type": "candlestick", "title": "Candlestick Chart"},
        {"type": "metrics", "title": "Key Metrics Chart"},
        {"type": "technical", "title": "Technical Analysis"}
    ]

def generate_crypto_charts(_=None):
    """Generate chart options for cryptocurrency analysis"""
    return [
        {"type": "price", "title": "Price Chart"},
        {"type": "candlestick", "title": "Candlestick Chart"},
        {"type": "performance", "title": "Performance Chart"},
        {"type": "volume", "title": "Volume Chart"}
    ]

def generate_reit_charts(_=None):
    """Generate chart options for REIT analysis"""
    return [
        {"type": "price", "title": "Price Chart"},
        {"type": "candlestick", "title": "Candlestick Chart"},
        {"type": "metrics", "title": "Key Metrics Chart"},
        {"type": "dividend", "title": "Dividend History"}
    ]

def generate_etf_charts(_=None):
    """Generate chart options for ETF analysis"""
    return [
        {"type": "price", "title": "Price Chart"},
        {"type": "allocation", "title": "Sector Allocation"},
        {"type": "performance", "title": "Performance Comparison"}
    ]

# Generate insights based on analysis
def generate_stock_insights(report):
    """Generate insights for stock analysis"""
    stock_data = report.get("stock", {})

    insights = [
        f"**Current Price**: {stock_data.get('current_price', '$0')}",
        f"**Market Cap**: {stock_data.get('market_cap', '$0')}",
        f"**P/E Ratio**: {stock_data.get('pe', 0):.2f}",
        f"**P/B Ratio**: {stock_data.get('pb', 0):.2f}",
        f"**Dividend Yield**: {stock_data.get('dividend_yield', '0%')}",
        f"**Beta**: {stock_data.get('beta', 0):.2f}",
        f"**52-Week High**: ${stock_data.get('52w_high', 0):.2f}",
        f"**52-Week Low**: ${stock_data.get('52w_low', 0):.2f}",
        f"**Intrinsic Value**: {stock_data.get('intrinsic_value', '$0')}",
        f"**Upside Potential**: {stock_data.get('upside_potential', '0%')}",
        f"**Recommendation**: {report.get('recommendation', 'Unknown')}",
        f"**Confidence Score**: {stock_data.get('confidence', 0)}/100"
    ]

    return insights

def generate_crypto_insights(report):
    """Generate insights for cryptocurrency analysis"""
    # Debug the report structure
    print(f"Crypto report keys: {report.keys() if isinstance(report, dict) else 'Not a dict'}")

    # Check if 'cryptocurrency' key exists, if not, try 'crypto'
    if "cryptocurrency" in report:
        crypto_data = report.get("cryptocurrency", {})
    elif "crypto" in report:
        crypto_data = report.get("crypto", {})
    else:
        # If neither key exists, use an empty dict
        crypto_data = {}

    print(f"Crypto data keys: {crypto_data.keys() if isinstance(crypto_data, dict) else 'Not a dict'}")

    # Format values with proper error handling
    def safe_get(data, key, default, formatter=lambda x: x):
        try:
            value = data.get(key, default)
            return formatter(value)
        except Exception as e:
            print(f"Error formatting {key}: {e}")
            return default

    insights = [
        f"**Current Price**: {safe_get(crypto_data, 'current_price', '$0')}",
        f"**Market Cap**: {safe_get(crypto_data, 'market_cap', '$0')}",
        f"**24h Trading Volume**: {safe_get(crypto_data, 'total_volume', '$0')}",
        f"**24h Price Change**: {safe_get(crypto_data, 'price_change_24h', '0%')}",
        f"**7d Price Change**: {safe_get(crypto_data, 'price_change_7d', '0%')}",
        f"**30d Price Change**: {safe_get(crypto_data, 'price_change_30d', '0%')}",
        f"**All-Time High**: {safe_get(crypto_data, 'all_time_high', '$0')}",
        f"**All-Time Low**: {safe_get(crypto_data, 'all_time_low', '$0')}",
        f"**Potential**: {safe_get(crypto_data, 'potential', '0%')}",
        f"**Volatility**: {safe_get(crypto_data, 'volatility', 'Unknown')}",
        f"**Recommendation**: {safe_get(report, 'recommendation', 'Unknown')}",
        f"**Confidence Score**: {safe_get(crypto_data, 'confidence', '0')}/100"
    ]

    return insights

def generate_reit_insights(report):
    """Generate insights for REIT analysis"""
    reit_data = report.get("reit", {})

    insights = [
        f"**Current Price**: {reit_data.get('current_price', '$0')}",
        f"**Property Type**: {reit_data.get('property_type', 'Commercial')}",
        f"**Market Cap**: {reit_data.get('market_cap', '$0')}",
        f"**Dividend Yield**: {reit_data.get('dividend_yield', '0%')}",
        f"**Price to FFO**: {reit_data.get('price_to_ffo', 0):.2f}",
        f"**Debt to Equity**: {reit_data.get('debt_to_equity', 0):.2f}",
        f"**Beta**: {reit_data.get('beta', 0):.2f}",
        f"**52-Week High**: ${reit_data.get('52w_high', 0):.2f}",
        f"**52-Week Low**: ${reit_data.get('52w_low', 0):.2f}",
        f"**Recommendation**: {report.get('recommendation', 'Unknown')}",
        f"**Confidence Score**: {reit_data.get('confidence', 0)}/100"
    ]

    return insights

def generate_etf_insights(report):
    """Generate insights for ETF analysis"""
    etf_data = report.get("etf", {})

    insights = [
        f"**Current Price**: {etf_data.get('current_price', '$0')}",
        f"**Category**: {etf_data.get('category', 'Unknown')}",
        f"**AUM**: {etf_data.get('aum', '$0')}",
        f"**Expense Ratio**: {etf_data.get('expense_ratio', '0%')}",
        f"**YTD Return**: {etf_data.get('ytd_return', '0%')}",
        f"**1-Year Return**: {etf_data.get('one_year_return', '0%')}",
        f"**3-Year Return**: {etf_data.get('three_year_return', '0%')}",
        f"**Top Holdings**: {', '.join(etf_data.get('top_holdings', [])[:5])}",
        f"**Recommendation**: {report.get('recommendation', 'Unknown')}",
        f"**Confidence Score**: {etf_data.get('confidence', 0)}/100"
    ]

    return insights

# Render analysis page
def render_analysis_page(asset_type, ticker, analyze_button):
    """Render the analysis page"""
    st.title(f"{asset_type} Analysis")

    # Convert asset type to lowercase for consistency
    asset_type_lower = asset_type.lower()

    # Show loading spinner
    if analyze_button:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                response = {}

                if asset_type_lower == "stock":
                    # Use StockAnalysisTeam
                    team = StockAnalysisTeam()
                    report = team.analyze(ticker)
                    response = {
                        "report": report,
                        "insights": generate_stock_insights(report),
                        "charts": generate_stock_charts(ticker)
                    }

                elif asset_type_lower == "cryptocurrency":
                    # Use CryptoAnalysisTeam
                    team = CryptoAnalysisTeam()
                    report = team.analyze(ticker)
                    response = {
                        "report": report,
                        "insights": generate_crypto_insights(report),
                        "charts": generate_crypto_charts(ticker)
                    }

                elif asset_type_lower == "reit":
                    # Use REITAnalysisTeam
                    team = REITAnalysisTeam()
                    report = team.analyze(ticker)
                    response = {
                        "report": report,
                        "insights": generate_reit_insights(report),
                        "charts": generate_reit_charts(ticker)
                    }

                elif asset_type_lower == "etf":
                    # Use ETFAnalysisTeam
                    team = ETFAnalysisTeam()
                    report = team.analyze(ticker)
                    response = {
                        "report": report,
                        "insights": generate_etf_insights(report),
                        "charts": generate_etf_charts(ticker)
                    }

                # Cache the results
                st.session_state.last_analysis = response
                st.session_state.last_ticker = ticker
                st.session_state.last_asset_type = asset_type
                st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                return

    # Check if we have analysis results
    if hasattr(st.session_state, "last_analysis") and st.session_state.last_analysis:
        # Get the last analysis results
        last_analysis = st.session_state.last_analysis
        last_ticker = st.session_state.last_ticker
        last_asset_type = st.session_state.last_asset_type
        last_update_time = st.session_state.last_update_time

        # Display the results
        st.subheader(f"{last_ticker} ({last_asset_type})")
        st.caption(f"Last updated: {last_update_time}")

        # Create columns for insights and charts
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display insights
            st.subheader("Key Insights")
            for insight in last_analysis.get("insights", []):
                st.markdown(insight)

        with col2:
            # Display charts
            st.subheader("Charts")

            # Display charts with radio buttons instead of tabs
            chart_options = last_analysis.get("charts", [])
            if chart_options:
                # Create a radio button for chart selection
                chart_titles = [chart["title"] for chart in chart_options]
                selected_chart_index = st.radio(
                    "Select Chart",
                    options=range(len(chart_titles)),
                    format_func=lambda i: chart_titles[i],
                    horizontal=True
                )

                # Get the selected chart type
                chart_type = chart_options[selected_chart_index]["type"]

                # Add a separator
                st.markdown("---")

                # Generate chart using the direct chart generator
                asset_type_key = last_asset_type.lower()
                if asset_type_key in last_analysis["report"]:
                    asset_data = last_analysis["report"][asset_type_key]

                    # Create a container for the chart
                    chart_container = st.container()
                    with chart_container:
                        st.subheader(f"{chart_titles[selected_chart_index]}")

                        # Generate and render the chart directly
                        generate_direct_chart(last_ticker, asset_data, chart_type)

        # Display news if available
        if "news" in last_analysis.get("report", {}).get(last_asset_type.lower(), {}):
            st.subheader("Latest News")
            news_items = last_analysis["report"][last_asset_type.lower()].get("news", [])

            for news in news_items[:5]:  # Show top 5 news items
                st.markdown(f"**{news.get('title', '')}**")
                st.caption(f"{news.get('source', {}).get('name', '')} - {news.get('publishedAt', '')}")
                st.markdown(f"{news.get('description', '')}")
                if news.get("url"):
                    st.markdown(f"[Read more]({news.get('url')})")
                st.markdown("---")

        # Display analysis if available
        if "analysis" in last_analysis.get("report", {}).get(last_asset_type.lower(), {}):
            st.subheader("AI Analysis")
            analysis = last_analysis["report"][last_asset_type.lower()].get("analysis", {})

            # Create tabs for different analysis sections
            if analysis:
                analysis_tabs = st.tabs(["Summary", "Strengths", "Weaknesses", "Opportunities", "Threats"])

                with analysis_tabs[0]:
                    st.markdown("### Fundamental Analysis")
                    st.markdown(analysis.get("fundamental_analysis", "No fundamental analysis available."))

                    st.markdown("### Technical Outlook")
                    st.markdown(analysis.get("technical_outlook", "No technical outlook available."))

                    st.markdown("### Investment Thesis")
                    st.markdown(analysis.get("investment_thesis", "No investment thesis available."))

                with analysis_tabs[1]:
                    strengths = analysis.get("strengths", [])
                    if strengths:
                        for strength in strengths:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("No strengths identified.")

                with analysis_tabs[2]:
                    weaknesses = analysis.get("weaknesses", [])
                    if weaknesses:
                        for weakness in weaknesses:
                            st.markdown(f"- {weakness}")
                    else:
                        st.markdown("No weaknesses identified.")

                with analysis_tabs[3]:
                    opportunities = analysis.get("opportunities", [])
                    if opportunities:
                        for opportunity in opportunities:
                            st.markdown(f"- {opportunity}")
                    else:
                        st.markdown("No opportunities identified.")

                with analysis_tabs[4]:
                    threats = analysis.get("threats", [])
                    if threats:
                        for threat in threats:
                            st.markdown(f"- {threat}")
                    else:
                        st.markdown("No threats identified.")

# Render comparison page
def render_comparison_page():
    """Render the comparison page"""
    st.title("Asset Comparison")

    # Initialize session state for comparison tickers if not exists
    if "comparison_tickers" not in st.session_state:
        st.session_state.comparison_tickers = []

    # Create columns for inputs
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Asset type selection
        comparison_asset_type = st.selectbox(
            "Asset Type",
            ["Stock", "Cryptocurrency", "REIT", "ETF"],
            key="comparison_asset_type"
        )

    with col2:
        # Ticker input
        comparison_ticker = st.text_input("Add Ticker", key="comparison_ticker")

    with col3:
        # Add button
        if st.button("Add to Comparison"):
            if comparison_ticker and comparison_ticker not in st.session_state.comparison_tickers:
                st.session_state.comparison_tickers.append(comparison_ticker)

    # Display current comparison tickers
    st.subheader("Comparison Tickers")

    if not st.session_state.comparison_tickers:
        st.info("No tickers added to comparison. Add tickers above.")
    else:
        # Create columns for ticker list and remove buttons
        cols = st.columns(4)

        for i, ticker in enumerate(st.session_state.comparison_tickers):
            col_index = i % 4
            with cols[col_index]:
                if st.button(f"Remove {ticker}", key=f"remove_{ticker}"):
                    st.session_state.comparison_tickers.remove(ticker)
                    st.experimental_rerun()
                st.write(ticker)

    # Comparison options
    if st.session_state.comparison_tickers:
        st.subheader("Comparison Options")

        # Create columns for chart type and time period
        col1, col2 = st.columns(2)

        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Price", "Performance", "Correlation"],
                key="comparison_chart_type"
            )

        with col2:
            time_period = st.selectbox(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"],
                key="comparison_time_period"
            )

        # Generate comparison button
        compare_button = st.button("Generate Comparison")

        if compare_button and len(st.session_state.comparison_tickers) > 0:
            with st.spinner("Generating comparison..."):
                # Initialize data service
                data_service = DataService()
                price_data = {}

                # Map time period to period string
                period_map = {
                    "1 Month": "1mo",
                    "3 Months": "3mo",
                    "6 Months": "6mo",
                    "1 Year": "1y",
                    "3 Years": "3y",
                    "5 Years": "5y"
                }
                period = period_map.get(time_period, "1mo")

                # Get data for each ticker
                for ticker in st.session_state.comparison_tickers:
                    try:
                        # Get price history data
                        price_history = data_service.get_price_history(
                            ticker,
                            comparison_asset_type.lower(),
                            period
                        )

                        if price_history and "timestamps" in price_history and "prices" in price_history:
                            price_data[ticker] = price_history
                    except Exception as e:
                        st.error(f"Error fetching data for {ticker}: {str(e)}")

                # Generate comparison chart
                if price_data:
                    # Use the direct comparison chart
                    from analytics.direct_charts import generate_direct_comparison
                    generate_direct_comparison(st.session_state.comparison_tickers, price_data)

    # Add a button to return to home
    if st.button("Return to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# Render documentation page
def render_docs_page():
    """Render the documentation page"""
    st.title("Documentation")

    # Create tabs for different documentation sections
    tabs = st.tabs(["Overview", "Usage Guide", "Asset Types", "Analysis Methods", "Code Structure"])

    with tabs[0]:
        st.header("AI Finance Dashboard")
        st.markdown("""
        The AI Finance Dashboard is a comprehensive tool for analyzing various financial assets using artificial intelligence.
        It provides in-depth analysis of stocks, cryptocurrencies, REITs, and ETFs, helping you make informed investment decisions.

        ### Key Features

        - **AI-Powered Analysis**: Utilizes advanced AI models to analyze financial assets
        - **Multiple Asset Types**: Supports stocks, cryptocurrencies, REITs, and ETFs
        - **Interactive Charts**: Visualize asset performance with interactive charts
        - **Comparison Tool**: Compare multiple assets side by side
        - **Real-Time Data**: Access up-to-date financial data
        """)

    with tabs[1]:
        st.header("Usage Guide")
        st.markdown("""
        ### Analyzing an Asset

        1. Select an asset type from the sidebar (Stock, Cryptocurrency, REIT, ETF)
        2. Enter the ticker symbol in the input field
        3. Click the "Analyze" button
        4. View the analysis results, including key insights, charts, and AI analysis

        ### Comparing Assets

        1. Navigate to the Comparison page using the sidebar
        2. Select an asset type
        3. Enter ticker symbols one by one and click "Add to Comparison"
        4. Choose a chart type and time period
        5. Click "Generate Comparison" to view the comparison chart
        """)

    with tabs[2]:
        st.header("Asset Types")
        st.markdown("""
        ### Stocks

        Stocks represent ownership in a company. The dashboard analyzes stocks based on:

        - Financial metrics (P/E ratio, P/B ratio, dividend yield)
        - Technical indicators
        - Intrinsic value calculation
        - Market sentiment

        ### Cryptocurrencies

        Cryptocurrencies are digital assets. The dashboard analyzes cryptocurrencies based on:

        - Price trends
        - Market capitalization
        - Trading volume
        - Volatility
        - Market sentiment

        ### REITs (Real Estate Investment Trusts)

        REITs are companies that own, operate, or finance income-generating real estate. The dashboard analyzes REITs based on:

        - Dividend yield
        - Price to FFO (Funds From Operations)
        - Debt to equity ratio
        - Property type
        - Market sentiment

        ### ETFs (Exchange-Traded Funds)

        ETFs are baskets of securities that trade on exchanges. The dashboard analyzes ETFs based on:

        - Expense ratio
        - Sector allocation
        - Performance metrics
        - Top holdings
        - Market sentiment
        """)

    with tabs[3]:
        st.header("Analysis Methods")
        st.markdown("""
        ### Fundamental Analysis

        Fundamental analysis evaluates an asset's intrinsic value by examining related economic, financial, and other qualitative and quantitative factors. The dashboard uses:

        - Financial ratios
        - Growth metrics
        - Valuation models
        - Industry comparisons

        ### Technical Analysis

        Technical analysis evaluates assets based on market activity, such as price movements and volume. The dashboard uses:

        - Price trends
        - Moving averages
        - Relative Strength Index (RSI)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands

        ### AI Analysis

        The dashboard uses AI models to provide comprehensive analysis, including:

        - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
        - Investment thesis
        - Risk assessment
        - Recommendation with confidence score
        """)

    with tabs[4]:
        st.header("Code Structure")
        st.markdown("""
        The AI Finance Dashboard is organized into several modules:

        ### Server Management

        - `app/server_manager.py`: Central server management class
        - `run_dashboard.py`: Main entry point for starting the dashboard

        ### Data Services

        - `app/services/data_service.py`: Unified data service for all asset types
        - `cache_manager.py`: Caching mechanism for improved performance

        ### Analysis Teams

        - `teams/base_team.py`: Base class for all analysis teams
        - `teams/stock_team.py`: Stock analysis team
        - `teams/crypto_team.py`: Cryptocurrency analysis team
        - `teams/reit_team.py`: REIT analysis team
        - `teams/etf_team.py`: ETF analysis team

        ### Chart Generation

        - `analytics/chart_generator.py`: Unified chart generation for all asset types
        - `analytics/optimized_charts.py`: Optimized chart generation functions

        ### API Integrations

        - `api_integrations/base_client.py`: Base class for all API clients
        - `api_integrations/yahoo_finance_client.py`: Yahoo Finance API client
        - `api_integrations/coingecko_client.py`: CoinGecko API client
        - `api_integrations/alphavantage_client.py`: Alpha Vantage API client
        - `api_integrations/news_client.py`: News API client
        """)

    # Add a button to return to home
    if st.button("Return to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# Main function
def main():
    """Main function"""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Render sidebar
    asset_type, ticker, analyze_button = render_sidebar()

    # Render page based on session state
    if st.session_state.page == "home":
        render_analysis_page(asset_type, ticker, analyze_button)
    elif st.session_state.page == "comparison":
        render_comparison_page()
    elif st.session_state.page == "docs":
        render_docs_page()

if __name__ == "__main__":
    main()
