"""
Test script to verify the serialization of timestamp objects.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
from utils.serialization import make_json_serializable, serialize_response

def test_timestamp_serialization():
    """Test the serialization of timestamp objects."""
    print("Testing timestamp serialization...")

    # Create a DataFrame with a DatetimeIndex
    dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'A': np.random.randn(5),
        'B': np.random.randn(5)
    }, index=dates)

    # Print original DataFrame info
    print(f"\nOriginal DataFrame:")
    print(f"Type: {type(df)}")
    print(f"Index type: {type(df.index)}")
    print(f"First index: {df.index[0]}")
    print(f"Index type: {type(df.index[0])}")

    # Create a dictionary with various timestamp types
    timestamp_dict = {
        'pandas_timestamp': pd.Timestamp('2025-01-01'),
        'datetime': datetime(2025, 1, 1),
        'numpy_datetime64': np.datetime64('2025-01-01'),
        'datetime_index': df.index,
        'timestamp_series': pd.Series(dates),
        'dataframe': df
    }

    # Try to serialize the dictionary
    try:
        print("\nSerializing dictionary with timestamp objects...")
        serialized_dict = make_json_serializable(timestamp_dict)
        print("Successfully serialized dictionary with timestamp objects")

        # Check the serialized values
        for key, value in serialized_dict.items():
            print(f"{key}: {type(value)} - {value}")

        # Try to convert to JSON
        print("\nConverting to JSON...")
        json_str = json.dumps(serialized_dict)
        print("Successfully converted to JSON")

        # Save to file for inspection
        with open('test_timestamp.json', 'w') as f:
            f.write(json_str)
        print("Saved to test_timestamp.json")

    except Exception as e:
        print(f"\nError serializing timestamp dictionary: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test the serialize_response function with a problematic response
    test_response = {
        'normal_data': 'This is normal data',
        'timestamp': pd.Timestamp('2025-01-01'),
        'nested': {
            'timestamp': pd.Timestamp('2025-01-02'),
            'normal': 'This is normal nested data'
        },
        'dataframe': df
    }

    try:
        serialized_response = serialize_response(test_response)
        print("\nSuccessfully serialized response with timestamp objects")

        # Try to convert to JSON
        json_str = json.dumps(serialized_response)
        print("Successfully converted response to JSON")

        # Save to file for inspection
        with open('test_response.json', 'w') as f:
            f.write(json_str)
        print("Saved to test_response.json")

    except Exception as e:
        print(f"\nError serializing response: {str(e)}")

if __name__ == "__main__":
    test_timestamp_serialization()
