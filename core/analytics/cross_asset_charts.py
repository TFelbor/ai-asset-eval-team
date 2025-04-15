"""
Cross-asset chart generation for the AI Finance Dashboard.
This module provides specialized chart rendering functions for cross-asset comparison.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

def create_cross_asset_price_chart(
    asset_info: Dict[str, Dict[str, Any]],
    price_histories: Dict[str, Dict[str, Any]]
) -> go.Figure:
    """
    Create a price chart comparing multiple assets of different types.

    Args:
        asset_info: Dictionary mapping asset_ids to asset information
        price_histories: Dictionary mapping asset_ids to price history dictionaries

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add a line for each asset
    for asset_id, price_history in price_histories.items():
        # Skip if no price history
        if not price_history or "timestamps" not in price_history or "prices" not in price_history:
            continue

        timestamps = price_history["timestamps"]
        prices = price_history["prices"]

        # Skip if no data
        if not timestamps or not prices:
            continue

        # Get asset info
        asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
        asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")
        asset_type = asset_info.get(asset_id, {}).get("asset_type", "").upper()

        # Add line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name=f"{asset_name} ({asset_ticker}) - {asset_type}",
            hovertemplate="%{y:$.2f}<extra>%{x|%b %d, %Y}</extra>"
        ))

    # Update layout
    fig.update_layout(
        title="Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

def create_cross_asset_normalized_chart(
    asset_info: Dict[str, Dict[str, Any]],
    price_histories: Dict[str, Dict[str, Any]]
) -> go.Figure:
    """
    Create a normalized price chart (percentage change) comparing multiple assets of different types.

    Args:
        asset_info: Dictionary mapping asset_ids to asset information
        price_histories: Dictionary mapping asset_ids to price history dictionaries

    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Add a line for each asset
    for asset_id, price_history in price_histories.items():
        # Skip if no price history
        if not price_history or "timestamps" not in price_history or "prices" not in price_history:
            continue

        timestamps = price_history["timestamps"]
        prices = price_history["prices"]

        # Skip if no data
        if not timestamps or not prices or len(timestamps) != len(prices):
            continue

        # Get asset info
        asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
        asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")
        asset_type = asset_info.get(asset_id, {}).get("asset_type", "").upper()

        # Calculate percentage change from first day
        base_price = prices[0]
        if base_price == 0:
            continue  # Skip if base price is zero to avoid division by zero

        normalized_prices = [(price / base_price - 1) * 100 for price in prices]

        # Add line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=normalized_prices,
            mode='lines',
            name=f"{asset_name} ({asset_ticker}) - {asset_type}",
            hovertemplate="%{y:.2f}%<extra>%{x|%b %d, %Y}</extra>"
        ))

    # Update layout
    fig.update_layout(
        title="Normalized Price Comparison (% Change)",
        xaxis_title="Date",
        yaxis_title="% Change",
        height=500,
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

def create_cross_asset_correlation_matrix(
    asset_info: Dict[str, Dict[str, Any]],
    correlation_matrix: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Create a correlation matrix heatmap for multiple assets of different types.

    Args:
        asset_info: Dictionary mapping asset_ids to asset information
        correlation_matrix: Correlation matrix as a nested dictionary

    Returns:
        Plotly figure object
    """
    # Convert correlation matrix to lists for plotting
    asset_ids = list(correlation_matrix.keys())
    
    # Create labels with asset name and type
    labels = []
    for asset_id in asset_ids:
        asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
        asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")
        asset_type = asset_info.get(asset_id, {}).get("asset_type", "").upper()
        labels.append(f"{asset_name} ({asset_ticker}) - {asset_type}")
    
    # Extract correlation values
    z_values = []
    for asset_id1 in asset_ids:
        row = []
        for asset_id2 in asset_ids:
            row.append(correlation_matrix[asset_id1][asset_id2])
        z_values.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        hovertemplate="Correlation: %{z:.2f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=600,
        width=800,
        template="plotly_dark"
    )
    
    return fig

def create_cross_asset_allocation_chart(
    asset_info: Dict[str, Dict[str, Any]],
    allocations: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create a pie chart showing recommended portfolio allocation.

    Args:
        asset_info: Dictionary mapping asset_ids to asset information
        allocations: List of allocation dictionaries with asset_id and percentage

    Returns:
        Plotly figure object
    """
    # Extract labels and values
    labels = []
    values = []
    
    for allocation in allocations:
        asset_id = allocation.get("asset_id", "")
        percentage = allocation.get("percentage", 0)
        
        asset_name = asset_info.get(asset_id, {}).get("name", asset_id)
        asset_ticker = asset_info.get(asset_id, {}).get("ticker", "")
        asset_type = asset_info.get(asset_id, {}).get("asset_type", "").upper()
        
        labels.append(f"{asset_name} ({asset_ticker}) - {asset_type}")
        values.append(percentage)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Bold)
    )])
    
    # Update layout
    fig.update_layout(
        title="Recommended Portfolio Allocation",
        height=600,
        template="plotly_dark"
    )
    
    return fig

def generate_cross_asset_chart(
    chart_type: str,
    asset_info: Dict[str, Dict[str, Any]],
    price_histories: Dict[str, Dict[str, Any]],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    portfolio_allocation: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Generate a chart for cross-asset comparison based on chart type.

    Args:
        chart_type: Type of chart to generate (price, normalized, correlation, allocation)
        asset_info: Dictionary mapping asset_ids to asset information
        price_histories: Dictionary mapping asset_ids to price history dictionaries
        correlation_matrix: Optional correlation matrix as a nested dictionary
        portfolio_allocation: Optional portfolio allocation dictionary

    Returns:
        Plotly figure object
    """
    if chart_type.lower() == "price":
        return create_cross_asset_price_chart(asset_info, price_histories)
    
    elif chart_type.lower() == "normalized":
        return create_cross_asset_normalized_chart(asset_info, price_histories)
    
    elif chart_type.lower() == "correlation" and correlation_matrix:
        return create_cross_asset_correlation_matrix(asset_info, correlation_matrix)
    
    elif chart_type.lower() == "allocation" and portfolio_allocation:
        allocations = portfolio_allocation.get("allocations", [])
        if allocations:
            return create_cross_asset_allocation_chart(asset_info, allocations)
    
    # Default to a simple placeholder chart
    fig = go.Figure()
    fig.add_annotation(text=f"Chart type '{chart_type}' not available or missing data", showarrow=False, font_size=20)
    return fig
