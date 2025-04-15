"""
Utils package for the AI Finance Dashboard.
This package contains utility functions for the dashboard.
"""

# Import utility functions
from core.utils.logger import (
    # Basic logging functions
    log_info, log_error, log_success, log_warning, log_debug,
    # Advanced logging functions
    log_api_call, log_data_operation, log_analytics_operation, log_exception,
    # Performance tracking
    performance_timer,
    # Logger instances
    app_logger, api_logger, data_logger, cache_logger
)
from core.utils.serialization import make_json_serializable
from core.utils.strategy_utils import get_strategy_description, calculate_moving_average, calculate_rsi, calculate_macd

__all__ = [
    # Basic logging functions
    'log_info',
    'log_error',
    'log_success',
    'log_warning',
    'log_debug',
    # Advanced logging functions
    'log_api_call',
    'log_data_operation',
    'log_analytics_operation',
    'log_exception',
    # Performance tracking
    'performance_timer',
    # Logger instances
    'app_logger',
    'api_logger',
    'data_logger',
    'cache_logger',
    # Serialization
    'make_json_serializable',
    # Strategy utilities
    'get_strategy_description',
    'calculate_moving_average',
    'calculate_rsi',
    'calculate_macd'
]