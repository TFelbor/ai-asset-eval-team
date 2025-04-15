#!/usr/bin/env python3
"""
Unified dashboard runner for the AI Finance Dashboard.
This script provides a single entry point for running the dashboard with various options.
"""
import os
import sys
import time
import subprocess
import argparse
import webbrowser
from pathlib import Path

def main():
    """Main function to run the dashboard."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the AI Finance Dashboard")
    parser.add_argument("--browser", action="store_true", help="Open browser automatically")
    parser.add_argument("--loading", action="store_true", help="Show loading screen")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the dashboard on")
    parser.add_argument("--theme", type=str, default="dark", help="Theme to use (dark or light)")
    args = parser.parse_args()

    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_THEME"] = args.theme
    if args.debug:
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"

    # Construct the command to run Streamlit
    cmd = [
        "streamlit", "run", "core/dashboard.py",
        "--server.port", str(args.port),
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false"
    ]

    # Run Streamlit
    process = subprocess.Popen(cmd)

    # Open browser if requested
    if args.browser:
        # Wait a moment for Streamlit to start
        time.sleep(2)
        webbrowser.open(f"http://localhost:{args.port}")

    # Wait for the process to complete
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down the dashboard...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
