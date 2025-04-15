import yfinance as yf
import pandas as pd

# Get stock data
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get historical data
hist = stock.history(period="1mo")

# Print info
print(f"Type: {type(hist)}")
print(f"Empty: {hist.empty}")
print(f"Shape: {hist.shape}")

# Convert to dictionary
hist_dict = hist.to_dict()

# Try to access .empty on the dictionary (this should fail)
try:
    print(f"Dict empty: {hist_dict.empty}")
except AttributeError as e:
    print(f"Error: {e}")

# Our fix should check the type before accessing .empty
if isinstance(hist, pd.DataFrame) and hist.empty:
    print("DataFrame is empty")
elif isinstance(hist_dict, dict):
    print("This is a dictionary, not a DataFrame")
