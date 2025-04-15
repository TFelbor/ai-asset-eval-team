"""
AI Finance Dashboard - Main Entry Point
This is the main entry point for the AI Finance Dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import time
import signal
import subprocess
import threading
import atexit

# Add the parent directory to the path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logging
from core.utils.logger import log_info, log_error, log_success, log_warning

# Import analysis teams
from core.teams.stock_team import StockAnalysisTeam
from core.teams.crypto_team import CryptoAnalysisTeam
from core.teams.reit_team import REITAnalysisTeam
from core.teams.etf_team import ETFAnalysisTeam

# Import chart generation
from core.analytics.optimized_charts import (
    generate_stock_chart,
    generate_crypto_chart,
    generate_reit_chart,
    generate_etf_chart
)

# Import direct chart rendering, enhanced data fetching, and specialized charts
from core.analytics.direct_charts import (
    generate_direct_chart,
    generate_direct_comparison
)
from core.analytics.enhanced_data_fetcher import fetch_price_history
from core.analytics.specialized_charts import (
    create_etf_sector_allocation_chart,
    create_key_metrics_chart,
    create_dividend_history_chart,
    create_performance_comparison_chart
)

# Import API clients
from core.api.yahoo_finance_client import YahooFinanceClient
from core.api.coingecko_client import CoinGeckoClient

# Import UI components
from core.ui.backtesting_ui import render_backtesting_ui
from core.ui.comparison_ui import render_comparison_ui
from core.ui.news_ui import render_news_ui

# Import from the original dashboard.py
# This is a placeholder - the actual implementation will be copied from dashboard.py
def main():
    """Main function to run the dashboard."""
    st.set_page_config(
        page_title="AI Finance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI Finance Dashboard")
    st.write("This is a placeholder for the main dashboard. The actual implementation will be copied from dashboard.py.")

if __name__ == "__main__":
    main()
