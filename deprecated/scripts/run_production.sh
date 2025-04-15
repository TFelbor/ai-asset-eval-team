#!/bin/bash
# Script to run the Streamlit dashboard in production mode

# Ensure the script is executable with: chmod +x run_production.sh

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Theme configuration
export STREAMLIT_THEME_BASE="dark"
export STREAMLIT_THEME_PRIMARY_COLOR="#4f46e5"
export STREAMLIT_THEME_BACKGROUND_COLOR="#1e1e1e"
export STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#2d2d2d"
export STREAMLIT_THEME_TEXT_COLOR="#ffffff"

# Logging configuration
export STREAMLIT_LOGGER_LEVEL="info"

# Set production mode
export PRODUCTION=true

# Run the Streamlit app
echo "Starting AI Finance Dashboard in production mode..."
streamlit run dashboard.py
