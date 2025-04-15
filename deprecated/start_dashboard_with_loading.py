#!/usr/bin/env python
"""
Streamlit dashboard startup script with enhanced configuration options and loading screen.
"""
import os
import sys
import argparse
import streamlit.web.cli as stcli
from pathlib import Path
import threading
import time
import http.server
import socketserver
import webbrowser

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import configuration
from config import settings as config
from app.utils.logger import app_logger

# Define a simple HTTP server to serve the loading page
class LoadingPageHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Check if the server is running
        server_status_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "server_status.txt")
        
        # If server is closed, show the server_closed.html page
        if os.path.exists(server_status_path):
            with open(server_status_path, 'r') as f:
                status = f.read().strip()
                if status == "closed":
                    self.path = '/static/server_closed.html'
        
        # If path is root, serve the loading page
        if self.path == '/':
            self.path = '/static/loading.html'
        
        # Try to serve the requested file
        try:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())
    
    def log_message(self, format, *args):
        # Suppress log messages
        return

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
    parser.add_argument(
        "--loading-port",
        type=int,
        default=8500,
        help="Port to run the loading page server on (default: 8500)"
    )
    return parser.parse_args()

def start_loading_server(port):
    """Start a simple HTTP server to serve the loading page."""
    try:
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Create or update server status file
        with open(os.path.join("static", "server_status.txt"), "w") as f:
            f.write("starting")
        
        # Start the HTTP server
        with socketserver.TCPServer(("", port), LoadingPageHandler) as httpd:
            app_logger.info(f"Loading page server started on port {port}")
            httpd.serve_forever()
    except Exception as e:
        app_logger.error(f"Error starting loading page server: {e}")

def main():
    """Main entry point."""
    args = parse_args()

    # Ensure required directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)

    app_logger.info("Starting AI Finance Dashboard with Streamlit")

    # Start the loading page server in a separate thread
    loading_server_thread = threading.Thread(
        target=start_loading_server,
        args=(args.loading_port,),
        daemon=True
    )
    loading_server_thread.start()
    app_logger.info(f"Loading page available at http://localhost:{args.loading_port}")

    # Open browser to loading page if requested
    if args.browser:
        webbrowser.open(f"http://localhost:{args.loading_port}")

    # Set Streamlit configuration via environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"  # Always set to true to prevent duplicate browser tabs

    # Set debug mode
    if args.debug:
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
        app_logger.setLevel("DEBUG")
        app_logger.debug("Debug mode enabled")

    # Run the Streamlit app
    app_logger.info(f"Starting Streamlit server on port {args.port}...")

    # Update server status file
    with open(os.path.join("static", "server_status.txt"), "w") as f:
        f.write("running")

    # Start Streamlit
    sys.argv = ["streamlit", "run", "dashboard.py"]
    stcli.main()

if __name__ == "__main__":
    main()
