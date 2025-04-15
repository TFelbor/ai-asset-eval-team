"""
Utility functions for data serialization.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

def dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame to a JSON-serializable dictionary.

    Args:
        df: The DataFrame to convert

    Returns:
        A dictionary representation of the DataFrame with column data directly accessible
    """
    if df is None or df.empty:
        return {}

    # Create a base result dictionary
    result = {}

    # Add metadata
    result["_metadata"] = {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "index_type": str(type(df.index))
    }

    # Add index as a list (handling datetime index specially)
    if isinstance(df.index, pd.DatetimeIndex):
        result["index"] = [idx.isoformat() for idx in df.index]
    else:
        result["index"] = df.index.tolist()

    # Add each column directly to the root for easy access
    for col in df.columns:
        # Convert numpy values to Python native types
        values = df[col].tolist()
        result[col] = values

    return result

def make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.

    Args:
        obj: The object to convert

    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_dict(obj)
    elif isinstance(obj, pd.Series):
        # Handle Series with special care for timestamp values
        if pd.api.types.is_datetime64_any_dtype(obj):
            # If it's a datetime series, convert each value to ISO format string
            return {str(k): v.isoformat() if hasattr(v, 'isoformat') else str(v)
                    for k, v in obj.items()}
        else:
            # Otherwise use the standard to_dict method
            return obj.to_dict()
    elif isinstance(obj, pd.DatetimeIndex):
        # Handle DatetimeIndex objects explicitly
        return [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in obj]
    elif isinstance(obj, np.ndarray):
        # Handle numpy arrays
        if np.issubdtype(obj.dtype, np.datetime64):
            # For datetime arrays, convert to strings
            return [str(ts) for ts in obj]
        else:
            # For other arrays, convert to list
            return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        # Handle pandas Timestamp objects specifically
        try:
            return obj.isoformat()
        except:
            return str(obj)
    elif isinstance(obj, datetime):
        # Handle standard datetime objects
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):
        # Handle any other datetime-like objects
        try:
            return obj.isoformat()
        except:
            return str(obj)
    elif isinstance(obj, str):
        # Try to convert string to numeric if it looks like a number
        try:
            # Check if it's a number with a decimal point
            if '.' in obj:
                return float(obj)
            # Check if it's an integer
            elif obj.isdigit() or (obj.startswith('-') and obj[1:].isdigit()):
                return int(obj)
            # Otherwise keep it as a string
            else:
                return obj
        except (ValueError, TypeError):
            return obj
    else:
        # Try to convert to a basic type
        try:
            return str(obj)
        except:
            return repr(obj)

def serialize_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a response dictionary to make it JSON-serializable.

    Args:
        response: The response dictionary to serialize

    Returns:
        A JSON-serializable version of the response
    """
    try:
        # Try to serialize the entire response at once
        return make_json_serializable(response)
    except Exception as e:
        # If that fails, try to serialize each key individually
        try:
            serialized_response = {}
            for key, value in response.items():
                try:
                    serialized_response[key] = make_json_serializable(value)
                except Exception as key_error:
                    # If a specific key fails, include an error message for that key
                    serialized_response[key] = {"error": f"Failed to serialize {key}: {str(key_error)}"}
            return serialized_response
        except Exception as fallback_error:
            # If all else fails, return a generic error
            return {"error": f"Serialization error: {str(e)}", "fallback_error": str(fallback_error)}
