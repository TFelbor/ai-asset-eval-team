"""
Unified server management module for the AI Finance Dashboard.
This module provides a centralized way to start, stop, and manage the Streamlit server.
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
import signal
import subprocess
from typing import Optional, Dict, Any, Callable

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import configuration
from config import settings as config
from app.utils.logger import app_logger

# Define a simple HTTP server to serve the loading page
class LoadingPageHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Check if the server is running
        server_status_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "server_status.txt")
        
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

class ServerManager:
    """Unified server manager for the AI Finance Dashboard."""
    
    def __init__(self):
        """Initialize the server manager."""
        self.loading_server = None
        self.loading_server_thread = None
        self.streamlit_port = 8501
        self.loading_port = 8500
        self.debug = config.DEBUG
        self.theme = "dark"
        self.browser = False
        self.status_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "server_status.txt")
    
    def parse_args(self) -> argparse.Namespace:
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
        parser.add_argument(
            "--loading",
            action="store_true",
            default=False,
            help="Show loading screen while server starts"
        )
        return parser.parse_args()
    
    def setup(self, args: Optional[argparse.Namespace] = None) -> None:
        """
        Set up the server manager with the given arguments.
        
        Args:
            args: Command line arguments (optional)
        """
        if args is None:
            args = self.parse_args()
        
        # Set up server parameters
        self.streamlit_port = args.port
        self.loading_port = args.loading_port
        self.debug = args.debug
        self.theme = args.theme
        self.browser = args.browser
        self.show_loading = args.loading
        
        # Ensure required directories exist
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs("static", exist_ok=True)
        
        # Set up server status
        with open(self.status_file_path, "w") as f:
            f.write("starting")
    
    def start_loading_server(self) -> None:
        """Start a simple HTTP server to serve the loading page."""
        try:
            # Start the HTTP server
            with socketserver.TCPServer(("", self.loading_port), LoadingPageHandler) as httpd:
                self.loading_server = httpd
                app_logger.info(f"Loading page server started on port {self.loading_port}")
                httpd.serve_forever()
        except Exception as e:
            app_logger.error(f"Error starting loading page server: {e}")
    
    def open_browser_to_loading(self) -> None:
        """Open the browser to the loading page."""
        if self.browser:
            webbrowser.open(f"http://localhost:{self.loading_port}")
    
    def open_browser_to_streamlit(self) -> None:
        """Open the browser to the Streamlit app."""
        if self.browser:
            def _open_browser():
                time.sleep(2)  # Wait for Streamlit to start
                webbrowser.open(f"http://localhost:{self.streamlit_port}")
            
            # Start browser in a separate thread
            threading.Thread(target=_open_browser).start()
    
    def set_streamlit_config(self) -> None:
        """Set Streamlit configuration via environment variables."""
        # Set port
        os.environ["STREAMLIT_SERVER_PORT"] = str(self.streamlit_port)
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"  # Always set to true to prevent duplicate browser tabs
        
        # Set theme
        if self.theme == "light":
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
        if self.debug:
            os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
            app_logger.setLevel("DEBUG")
            app_logger.debug("Debug mode enabled")
    
    def start_streamlit(self) -> None:
        """Start the Streamlit server."""
        app_logger.info(f"Starting Streamlit server on port {self.streamlit_port}...")
        
        # Update server status
        with open(self.status_file_path, "w") as f:
            f.write("running")
        
        # Start Streamlit
        sys.argv = ["streamlit", "run", "dashboard.py"]
        stcli.main()
    
    def start(self) -> None:
        """Start the server with the current configuration."""
        app_logger.info("Starting AI Finance Dashboard with Streamlit")
        
        if self.show_loading:
            # Start the loading page server in a separate thread
            self.loading_server_thread = threading.Thread(
                target=self.start_loading_server,
                daemon=True
            )
            self.loading_server_thread.start()
            app_logger.info(f"Loading page available at http://localhost:{self.loading_port}")
            
            # Open browser to loading page if requested
            self.open_browser_to_loading()
        elif self.browser:
            # Open browser to Streamlit directly
            self.open_browser_to_streamlit()
        
        # Set Streamlit configuration
        self.set_streamlit_config()
        
        # Start Streamlit
        self.start_streamlit()
    
    @staticmethod
    def shutdown(callback: Optional[Callable] = None) -> bool:
        """
        Safely shutdown the Streamlit server.
        
        Args:
            callback: Optional callback function to execute before shutdown
            
        Returns:
            True if shutdown was initiated, False otherwise
        """
        try:
            # Get the current process ID
            pid = os.getpid()
            
            # Create a function to perform the actual shutdown after a delay
            def delayed_shutdown(seconds=3):
                # Display a countdown message
                for i in range(seconds, 0, -1):
                    if i == seconds:
                        print(f"\nShutting down server in {seconds} seconds...")
                    print(f"{i}...")
                    time.sleep(1)
                print("Server shutdown initiated.")
                
                # Create a file to indicate server is closed (for loading screen detection)
                try:
                    status_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "server_status.txt")
                    with open(status_file_path, "w") as f:
                        f.write("closed")
                except Exception as status_err:
                    print(f"Failed to write server status file: {status_err}")
                
                # Execute callback if provided
                if callback:
                    try:
                        callback()
                    except Exception as callback_err:
                        print(f"Error executing shutdown callback: {callback_err}")
                
                # On Unix-like systems (Linux, macOS)
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception as e:
                    print(f"Unix-style shutdown failed: {e}")
                    
                    # Fallback for Windows
                    try:
                        if sys.platform == 'win32':
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)])
                        else:
                            os.kill(pid, signal.SIGKILL)  # Last resort
                    except Exception as e2:
                        print(f"Fallback shutdown failed: {e2}")
            
            # Start the delayed shutdown in a separate thread
            shutdown_thread = threading.Thread(target=delayed_shutdown)
            shutdown_thread.daemon = True  # Thread will exit when main thread exits
            shutdown_thread.start()
            
            return True
        except Exception as e:
            print(f"Error during shutdown: {e}")
            return False

def main():
    """Main entry point for the server manager."""
    # Create server manager
    manager = ServerManager()
    
    # Parse arguments and set up
    args = manager.parse_args()
    manager.setup(args)
    
    # Start the server
    manager.start()

if __name__ == "__main__":
    main()
