#!/bin/bash
# Script to run the Streamlit dashboard with a loading screen

# Ensure the script is executable with: chmod +x run_dashboard_with_loading.sh

# Set up environment variables if needed
# export ALPHA_VANTAGE_API_KEY="your_key_here"
# export COINGECKO_API_KEY="your_key_here"
# export NEWS_API_KEY="your_key_here"

# Create or update the server status file
mkdir -p static
echo "starting" > static/server_status.txt

# Run the dashboard with loading screen
python start_dashboard_with_loading.py --browser "$@"
