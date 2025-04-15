#!/bin/bash
# Unified script to run the AI Finance Dashboard
# This script replaces all previous startup scripts with a single entry point

# Ensure the script is executable with: chmod +x run.sh

# Set up environment variables if needed
# export ALPHA_VANTAGE_API_KEY="your_key_here"
# export COINGECKO_API_KEY="your_key_here"
# export NEWS_API_KEY="your_key_here"

# Run the unified Python runner with all arguments passed to this script
python run.py "$@"
