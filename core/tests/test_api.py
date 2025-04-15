import pandas as pd
import yfinance as yf
from utils.serialization import make_json_serializable
import json

def test_yahoo_finance_api():
    """Test the Yahoo Finance API with serialization."""
    print("Testing Yahoo Finance API with serialization...")
    
    # Get stock data from yfinance
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    
    # Get historical data
    hist = stock.history(period="1mo")
    
    # Print original DataFrame
    print("\nOriginal DataFrame:")
    print(f"Type: {type(hist)}")
    print(f"Empty: {hist.empty}")
    print(f"Shape: {hist.shape}")
    
    # Serialize the DataFrame
    serialized_hist = make_json_serializable(hist)
    
    # Print serialized data
    print("\nSerialized data:")
    print(f"Type: {type(serialized_hist)}")
    print(f"Keys: {serialized_hist.keys()}")
    
    # Check if 'Close' is in the serialized data
    if "Close" in serialized_hist:
        print(f"Close data length: {len(serialized_hist['Close'])}")
    else:
        print("Close data not found in serialized data")
    
    # Save to JSON file for inspection
    with open("test_data.json", "w") as f:
        json.dump(serialized_hist, f, indent=2)
    
    print("\nSerialized data saved to test_data.json")
    
    # Test our empty check logic
    print("\nTesting empty check logic:")
    
    # Case 1: DataFrame
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        print("Case 1: DataFrame is not empty - PASS")
    else:
        print("Case 1: DataFrame is empty - FAIL")
    
    # Case 2: Dictionary with Close data
    if isinstance(serialized_hist, dict) and "Close" in serialized_hist and len(serialized_hist["Close"]) > 0:
        print("Case 2: Dictionary has Close data - PASS")
    else:
        print("Case 2: Dictionary does not have Close data - FAIL")
    
    # Create price history from both formats
    print("\nCreating price history from both formats:")
    
    # From DataFrame
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        price_history_df = {
            "timestamps": hist.index.tolist(),
            "prices": hist["Close"].tolist(),
            "volumes": hist["Volume"].tolist(),
        }
        print(f"From DataFrame - timestamps length: {len(price_history_df['timestamps'])}")
    
    # From Dictionary
    if isinstance(serialized_hist, dict) and "Close" in serialized_hist and len(serialized_hist["Close"]) > 0:
        price_history_dict = {
            "timestamps": serialized_hist.get("index", []),
            "prices": serialized_hist.get("Close", []),
            "volumes": serialized_hist.get("Volume", []),
        }
        print(f"From Dictionary - timestamps length: {len(price_history_dict['timestamps'])}")
    
    # Compare the results
    if len(price_history_df['timestamps']) == len(price_history_dict['timestamps']):
        print("Both methods produce same length data - PASS")
    else:
        print(f"Length mismatch: DataFrame={len(price_history_df['timestamps'])}, Dict={len(price_history_dict['timestamps'])} - FAIL")

if __name__ == "__main__":
    test_yahoo_finance_api()
