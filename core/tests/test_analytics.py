"""
Tests for the analytics modules.
"""
import pytest
import numpy as np
from analytics.advanced_metrics import AdvancedAnalytics

def test_calculate_volatility():
    """Test the volatility calculation."""
    # Test with a simple price series
    prices = [100, 102, 99, 101, 103]
    volatility = AdvancedAnalytics.calculate_volatility(prices)
    
    # Volatility should be positive
    assert volatility > 0
    
    # Test with empty list
    assert AdvancedAnalytics.calculate_volatility([]) == 0.0
    
    # Test with single value
    assert AdvancedAnalytics.calculate_volatility([100]) == 0.0

def test_calculate_sharpe_ratio():
    """Test the Sharpe ratio calculation."""
    # Test with positive returns
    returns = [0.01, 0.02, -0.01, 0.015, 0.005]
    sharpe = AdvancedAnalytics.calculate_sharpe_ratio(returns)
    
    # Sharpe ratio should be a float
    assert isinstance(sharpe, float)
    
    # Test with empty list
    assert AdvancedAnalytics.calculate_sharpe_ratio([]) == 0.0
    
    # Test with single value
    assert AdvancedAnalytics.calculate_sharpe_ratio([0.01]) == 0.0

def test_calculate_beta():
    """Test the beta calculation."""
    # Test with correlated returns
    stock_returns = [0.01, 0.02, -0.01, 0.015, 0.005]
    market_returns = [0.005, 0.015, -0.005, 0.01, 0.0]
    
    beta = AdvancedAnalytics.calculate_beta(stock_returns, market_returns)
    
    # Beta should be a float
    assert isinstance(beta, float)
    
    # Test with mismatched lengths
    assert AdvancedAnalytics.calculate_beta([0.01, 0.02], [0.005]) == 1.0
    
    # Test with empty lists
    assert AdvancedAnalytics.calculate_beta([], []) == 1.0
