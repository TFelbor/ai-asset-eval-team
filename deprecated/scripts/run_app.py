#!/usr/bin/env python3
"""
Simple script to run the dashboard directly.
"""
import os
import sys
import subprocess

def main():
    """Run the dashboard."""
    # Add the current directory to the Python path
    sys.path.insert(0, os.getcwd())
    
    # Run streamlit with the dashboard
    subprocess.run(["streamlit", "run", "core/dashboard.py"])

if __name__ == "__main__":
    main()
