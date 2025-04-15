#!/usr/bin/env python
"""
Run script for the financial analysis dashboard using FastAPI (DEPRECATED).

NOTE: This script is deprecated. Please use run_streamlit.py instead.
"""
import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import configuration
from config import settings as config
from app.utils.logger import app_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Analysis Dashboard")
    parser.add_argument(
        "--host",
        type=str,
        default=config.HOST,
        help=f"Host to run the server on (default: {config.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.PORT,
        help=f"Port to run the server on (default: {config.PORT})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.DEBUG,
        help="Run in debug mode"
    )
    # WebSocket server no longer used
    # parser.add_argument(
    #     "--ws-port",
    #     type=int,
    #     default=8001,
    #     help="Port for WebSocket server (default: 8001)"
    # )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    # Ensure required directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Print deprecation warning
    app_logger.warning("DEPRECATED: This FastAPI server is deprecated and will be removed in a future version.")
    app_logger.warning("Please use 'python run_streamlit.py' or 'make run-streamlit' instead.")

    # No WebSocket server needed - using simulated data instead
    app_logger.info("Using simulated real-time data for price updates")

    # Start FastAPI server
    app_logger.info(f"Starting FastAPI server on {args.host}:{args.port}...")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )

if __name__ == "__main__":
    main()
