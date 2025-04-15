#!/bin/bash
# Unified script to run the AI Finance Dashboard
# This script replaces all previous startup scripts with a single entry point

# Ensure the script is executable with: chmod +x run_dashboard.sh

# Set up environment variables if needed
# export ALPHA_VANTAGE_API_KEY="your_key_here"
# export COINGECKO_API_KEY="your_key_here"
# export NEWS_API_KEY="your_key_here"

# Parse arguments
BROWSER_FLAG=""
LOADING_FLAG=""
DEBUG_FLAG=""
PORT_FLAG=""
THEME_FLAG=""

# Process command line arguments
for arg in "$@"; do
  case $arg in
    --browser)
      BROWSER_FLAG="--browser"
      ;;
    --loading)
      LOADING_FLAG="--loading"
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      ;;
    --port=*)
      PORT_FLAG="--port=${arg#*=}"
      ;;
    --theme=*)
      THEME_FLAG="--theme=${arg#*=}"
      ;;
  esac
done

# Run the dashboard directly with Streamlit
streamlit run core/dashboard.py $BROWSER_FLAG $LOADING_FLAG $DEBUG_FLAG $PORT_FLAG $THEME_FLAG
