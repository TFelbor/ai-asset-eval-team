#!/usr/bin/env python
"""
Test script to verify that the Streamlit app works correctly.
"""
import sys
import os
import subprocess
import time
import requests
import webbrowser
from pathlib import Path

def main():
    """Main entry point."""
    print("Testing Streamlit app...")
    
    # Start the Streamlit app in the background
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "dashboard.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the Streamlit app to start
    print("Waiting for Streamlit app to start...")
    time.sleep(5)
    
    # Check if the Streamlit app is running
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("Streamlit app is running!")
            print("You can access it at: http://localhost:8501")
            
            # Open the Streamlit app in the default browser
            webbrowser.open("http://localhost:8501")
            
            # Keep the app running for a while
            print("Press Ctrl+C to stop the Streamlit app...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping Streamlit app...")
        else:
            print(f"Streamlit app returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to Streamlit app!")
    finally:
        # Stop the Streamlit app
        streamlit_process.terminate()
        streamlit_process.wait()
        print("Streamlit app stopped.")

if __name__ == "__main__":
    main()
