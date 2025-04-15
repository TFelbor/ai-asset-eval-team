# Deprecated Files

This directory contains files that are no longer used in the AI Finance Dashboard. They have been replaced by more optimized and standardized implementations.

## Deprecated Files

- `main.py`: Legacy FastAPI server entry point, replaced by Streamlit-only implementation
- `run.py`: Legacy FastAPI server startup script, replaced by unified server manager
- `run_streamlit.py`: Old Streamlit startup script, replaced by unified server manager
- `start_dashboard.py`: Old dashboard startup script, replaced by unified server manager
- `start_dashboard_with_loading.py`: Old loading screen implementation, replaced by unified server manager
- `run_dashboard_with_loading.sh`: Old loading screen shell script, replaced by unified server manager
- `analytics/advanced_metrics.py`: Old chart generation functions, replaced by unified chart generator
- `api_integrations/alpha_vantage.py`: Old Alpha Vantage API client, replaced by unified API client
- `api_integrations/alphavantage.py`: Duplicate Alpha Vantage API client, replaced by unified API client
- `api_integrations/coingecko.py`: Old CoinGecko API client, replaced by unified API client
- `api_integrations/news_api.py`: Old News API client, replaced by unified API client
- `api_integrations/yahoo_finance.py`: Old Yahoo Finance API client, replaced by unified API client
- `teams.py`: Old analysis teams implementation, replaced by modular analysis teams

## New Implementations

The functionality provided by these files has been replaced by:

- `app/server_manager.py`: Unified server management
- `run_dashboard.py`: Unified dashboard entry point
- `run_dashboard.sh`: Unified shell script
- `analytics/chart_generator.py`: Unified chart generation
- `api_integrations/base_client.py`: Base API client class
- `api_integrations/*_client.py`: Standardized API clients
- `teams/*_team.py`: Modular analysis teams

## Note

These files are kept for reference purposes only and should not be used in new code.
