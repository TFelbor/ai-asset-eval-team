#!/usr/bin/env python
"""
Script to check if the Streamlit app is running correctly.
"""
import sys
import os
import argparse
import requests
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check if the Streamlit app is running correctly")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8501",
        help="URL of the Streamlit app (default: http://localhost:8501)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds (default: 5)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=False,
        help="Run continuously (default: False)"
    )
    return parser.parse_args()

def check_streamlit(url, timeout):
    """Check if the Streamlit app is running correctly."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, "Streamlit app is running correctly"
        else:
            return False, f"Streamlit app returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Failed to connect to Streamlit app"
    except requests.exceptions.Timeout:
        return False, "Connection to Streamlit app timed out"
    except Exception as e:
        return False, f"Error checking Streamlit app: {str(e)}"

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.continuous:
        print(f"Checking Streamlit app at {args.url} every {args.interval} seconds...")
        while True:
            success, message = check_streamlit(args.url, args.timeout)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if success:
                print(f"[{timestamp}] {message}")
            else:
                print(f"[{timestamp}] ERROR: {message}")
            time.sleep(args.interval)
    else:
        print(f"Checking Streamlit app at {args.url}...")
        success, message = check_streamlit(args.url, args.timeout)
        if success:
            print(message)
            sys.exit(0)
        else:
            print(f"ERROR: {message}")
            sys.exit(1)

if __name__ == "__main__":
    main()
