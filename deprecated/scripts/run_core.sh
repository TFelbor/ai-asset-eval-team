#!/bin/bash
# Script to run the Streamlit dashboard from the core directory

# Ensure the script is executable with: chmod +x run_core.sh

# Set environment variables to control browser behavior
export STREAMLIT_SERVER_HEADLESS=true

# Check if --browser flag is passed
if [[ " $* " == *" --browser "* ]]; then
    # If --browser flag is passed, open the browser manually after a short delay
    streamlit run core/main.py "$@" &
    sleep 2
    open http://localhost:8501
else
    # Otherwise, just run Streamlit without opening browser
    streamlit run core/main.py "$@"
fi
