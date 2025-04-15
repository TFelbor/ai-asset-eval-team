#!/usr/bin/env python
"""
Run script for the Streamlit financial analysis dashboard.
"""
import os
import sys
import argparse
import streamlit.web.cli as stcli
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import configuration
from config import settings as config
from app.utils.logger import app_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Analysis Dashboard (Streamlit)")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,  # Default Streamlit port
        help="Port to run the Streamlit server on (default: 8501)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.DEBUG,
        help="Run in debug mode"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    # Ensure required directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    app_logger.info("Using simulated real-time data for price updates")

    # Start Streamlit app
    app_logger.info(f"Starting Streamlit server on port {args.port}...")
    
    # Set Streamlit configuration via environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "dashboard.py"]
    stcli.main()

if __name__ == "__main__":
    main()
