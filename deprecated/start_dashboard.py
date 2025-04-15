#!/usr/bin/env python
"""
Streamlit dashboard startup script with enhanced configuration options.
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
    parser = argparse.ArgumentParser(description="AI Finance Dashboard (Streamlit)")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,  # Default Streamlit port
        help="Port to run the Streamlit server on (default: 8501)"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        default=False,
        help="Open browser automatically"
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default="dark",
        help="UI theme (default: dark)"
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

    app_logger.info("Starting AI Finance Dashboard with Streamlit")

    # Set Streamlit configuration via environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"  # Always set to true to prevent duplicate browser tabs

    # Browser opening will be handled manually if requested

    # Set theme
    if args.theme == "light":
        os.environ["STREAMLIT_THEME_BASE"] = "light"
        # Override theme colors for light mode
        os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] = "#4f46e5"
        os.environ["STREAMLIT_THEME_BACKGROUND_COLOR"] = "#ffffff"
        os.environ["STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR"] = "#f3f4f6"
        os.environ["STREAMLIT_THEME_TEXT_COLOR"] = "#111827"
    else:
        os.environ["STREAMLIT_THEME_BASE"] = "dark"
        # Override theme colors for dark mode
        os.environ["STREAMLIT_THEME_PRIMARY_COLOR"] = "#4f46e5"
        os.environ["STREAMLIT_THEME_BACKGROUND_COLOR"] = "#1e1e1e"
        os.environ["STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR"] = "#2d2d2d"
        os.environ["STREAMLIT_THEME_TEXT_COLOR"] = "#ffffff"

    # Set debug mode
    if args.debug:
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
        app_logger.setLevel("DEBUG")
        app_logger.debug("Debug mode enabled")

    # Run the Streamlit app
    app_logger.info(f"Starting Streamlit server on port {args.port}...")

    # Start Streamlit
    sys.argv = ["streamlit", "run", "dashboard.py"]

    # If browser flag is set, open browser manually after a short delay
    if args.browser:
        import threading
        import webbrowser
        import time

        def open_browser():
            time.sleep(2)  # Wait for Streamlit to start
            webbrowser.open(f"http://localhost:{args.port}")

        # Start browser in a separate thread
        threading.Thread(target=open_browser).start()

    # Start Streamlit
    stcli.main()

if __name__ == "__main__":
    main()
