"""
Test script to verify the serialization and processing of Yahoo Finance data.
"""
import pandas as pd
import yfinance as yf
import json
from utils.serialization import make_json_serializable
from core.api.api_wrappers import YahooFinanceAPI

def test_serialization():
    """Test the serialization of Yahoo Finance data."""
    print("Testing serialization of Yahoo Finance data...")

    # Get stock data directly from yfinance
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")

    # Print original DataFrame info
    print(f"\nOriginal DataFrame:")
    print(f"Type: {type(hist)}")
    print(f"Empty: {hist.empty}")
    print(f"Shape: {hist.shape}")
    print(f"Columns: {hist.columns.tolist()}")

    # Serialize the DataFrame
    serialized_hist = make_json_serializable(hist)

    # Print serialized data info
    print(f"\nSerialized data:")
    print(f"Type: {type(serialized_hist)}")
    print(f"Keys: {list(serialized_hist.keys())}")

    # Check if columns are directly accessible
    if "Close" in serialized_hist:
        print(f"Close data length: {len(serialized_hist['Close'])}")
        print(f"First close value: {serialized_hist['Close'][0]}")
    else:
        print("Close data not found in serialized data")

    # Save to JSON file for inspection
    with open("test_serialized_data.json", "w") as f:
        json.dump(serialized_hist, f, indent=2)

    print("\nSerialized data saved to test_serialized_data.json")

    return serialized_hist

def test_api_wrapper():
    """Test the YahooFinanceAPI wrapper."""
    print("\nTesting YahooFinanceAPI wrapper...")

    # Initialize API wrapper
    api = YahooFinanceAPI()

    # Get stock data
    ticker = "AAPL"
    stock_data = api.get_stock_data(ticker)

    # Print stock data info
    print(f"\nStock data:")
    print(f"Type: {type(stock_data)}")
    print(f"Keys: {list(stock_data.keys())}")

    # Check if history is properly serialized
    if "history" in stock_data:
        hist = stock_data["history"]
        print(f"History type: {type(hist)}")
        print(f"History keys: {list(hist.keys()) if isinstance(hist, dict) else 'Not a dict'}")

        # Check if columns are directly accessible
        if isinstance(hist, dict) and "Close" in hist:
            print(f"Close data length: {len(hist['Close'])}")
            print(f"First close value: {hist['Close'][0]}")
        else:
            print("Close data not found in history")
    else:
        print("History not found in stock data")

    # Get historical data directly
    hist_data = api.get_historical_data(ticker, period="1mo")

    # Print historical data info
    print(f"\nHistorical data:")
    print(f"Type: {type(hist_data)}")
    print(f"Keys: {list(hist_data.keys())}")

    # Check if metadata is present
    if "_metadata" in hist_data:
        print(f"Metadata: {hist_data['_metadata']}")
    else:
        print("Metadata not found in historical data")

    # Check if columns are directly accessible
    if "Close" in hist_data:
        print(f"Close data length: {len(hist_data['Close'])}")
        print(f"First close value: {hist_data['Close'][0]}")
    else:
        print("Close data not found in historical data")

    # Save to JSON file for inspection
    with open("test_api_data.json", "w") as f:
        json.dump(hist_data, f, indent=2)

    print("\nAPI data saved to test_api_data.json")

    return hist_data

def test_processing(serialized_data):
    """Test the processing of serialized data."""
    print("\nTesting processing of serialized data...")

    # Process the serialized data as if it were in the dashboard
    if isinstance(serialized_data, dict):
        # Check if it's an empty result or has metadata indicating it's empty
        if not serialized_data or ("_metadata" in serialized_data and serialized_data["_metadata"].get("empty", False)):
            print("Empty data detected")
            price_history = {"timestamps": [], "prices": [], "volumes": []}
        # Check if there's an error in the metadata
        elif "_metadata" in serialized_data and "error" in serialized_data["_metadata"]:
            print(f"Error detected: {serialized_data['_metadata']['error']}")
            price_history = {"timestamps": [], "prices": [], "volumes": []}
        # Handle case where serialized_data is a dictionary with data
        elif "Close" in serialized_data and len(serialized_data["Close"]) > 0:
            price_history = {
                "timestamps": serialized_data.get("index", []),
                "prices": serialized_data.get("Close", []),
                "volumes": serialized_data.get("Volume", []),
                "open": serialized_data.get("Open", []),
                "high": serialized_data.get("High", []),
                "low": serialized_data.get("Low", []),
                "close": serialized_data.get("Close", [])
            }
            print(f"Successfully processed serialized data: {len(price_history['timestamps'])} data points")
        else:
            # Create empty price history for any other case
            price_history = {"timestamps": [], "prices": [], "volumes": []}
            print("No valid data found in dictionary")
    else:
        # Create empty price history
        price_history = {"timestamps": [], "prices": [], "volumes": []}
        print("Not a dictionary")

    # Print price history info
    print(f"\nPrice history:")
    print(f"Timestamps length: {len(price_history['timestamps'])}")
    print(f"Prices length: {len(price_history['prices'])}")
    print(f"Volumes length: {len(price_history['volumes'])}")

    if len(price_history['timestamps']) > 0:
        print(f"First timestamp: {price_history['timestamps'][0]}")
        print(f"First price: {price_history['prices'][0]}")

    return price_history

if __name__ == "__main__":
    # Test serialization
    serialized_data = test_serialization()

    # Test API wrapper
    api_data = test_api_wrapper()

    # Test processing with serialized data
    test_processing(serialized_data)

    # Test processing with API data
    test_processing(api_data)

    print("\nAll tests completed!")
