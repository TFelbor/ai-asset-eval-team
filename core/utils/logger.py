"""
Logging configuration for the financial analysis dashboard.
"""
import os
import sys
import logging
import time
import functools
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Callable

# Try to import config, but provide fallbacks if not available
try:
    from core.config import settings as config
except ImportError:
    try:
        import config
    except ImportError:
        # Create a minimal config fallback
        class Config:
            LOG_DIR = Path("core/logs")
            LOG_LEVEL = "INFO"
            LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        config = Config()

# Try to import loguru, but provide fallbacks if not available
try:
    import sys
    from loguru import logger

    # Configure Loguru
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Add file handlers for different log types
    log_dir = config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Main application log
    logger.add(
        log_dir / "app.log",
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "app" in record["name"],
        level="INFO"
    )

    # API log
    logger.add(
        log_dir / "api.log",
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "api" in record["name"],
        level="INFO"
    )

    # Data log
    logger.add(
        log_dir / "data.log",
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "data" in record["name"],
        level="INFO"
    )

    # Cache log
    logger.add(
        log_dir / "cache.log",
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "cache" in record["name"],
        level="INFO"
    )

    LOGURU_AVAILABLE = True
    print("\033[92m✓ Loguru configured successfully.\033[0m")

except ImportError:
    LOGURU_AVAILABLE = False
    print("\033[93m⚠ Loguru not available, using standard logging instead.\033[0m")
    print("\033[94mℹ Run 'pip install loguru' for enhanced logging features.\033[0m")

def setup_logger(name, log_file=None, level=None):
    """
    Set up a logger with the specified name and configuration.

    Args:
        name: Name of the logger
        log_file: Path to the log file (optional)
        level: Logging level (optional)

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Set level from parameter, config, or default to INFO
    if level is None:
        level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)

        # Create rotating file handler (10 MB max size, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Create main application logger
app_logger = setup_logger(
    'financial_dashboard',
    log_file=config.LOG_DIR / 'app.log'
)

# Create API logger
api_logger = setup_logger(
    'api',
    log_file=config.LOG_DIR / 'api.log'
)

# Create data logger
data_logger = setup_logger(
    'data',
    log_file=config.LOG_DIR / 'data.log'
)

# Create cache logger
cache_logger = setup_logger(
    'cache',
    log_file=config.LOG_DIR / 'cache.log'
)

# Convenience functions for colored console output
def log_success(message):
    """Log a success message with green color in console."""
    if LOGURU_AVAILABLE:
        logger.success(f"✓ {message}")
    else:
        print(f"\033[92m✓ {message}\033[0m")
        app_logger.info(f"SUCCESS: {message}")

def log_warning(message):
    """Log a warning message with yellow color in console."""
    if LOGURU_AVAILABLE:
        logger.warning(f"⚠ {message}")
    else:
        print(f"\033[93m⚠ {message}\033[0m")
        app_logger.warning(message)

def log_error(message):
    """Log an error message with red color in console."""
    if LOGURU_AVAILABLE:
        logger.error(f"✗ {message}")
    else:
        print(f"\033[91m✗ {message}\033[0m")
        app_logger.error(message)

def log_info(message):
    """Log an info message with blue color in console."""
    if LOGURU_AVAILABLE:
        logger.info(f"ℹ {message}")
    else:
        print(f"\033[94mℹ {message}\033[0m")
        app_logger.info(message)

def log_debug(message):
    """Log a debug message."""
    if LOGURU_AVAILABLE:
        logger.debug(message)
    else:
        app_logger.debug(message)

def log_api_call(api_name, endpoint, params=None, success=True, response=None, error=None, request_time=None):
    """Log an API call with details."""
    if LOGURU_AVAILABLE:
        # Use structured logging with Loguru
        context = {
            "api_name": api_name,
            "endpoint": endpoint,
            "params": params,
            "success": success
        }

        if request_time is not None:
            context["request_time"] = f"{request_time:.3f}s"

        if response is not None:
            # Truncate response if it's too large
            if isinstance(response, dict) and len(str(response)) > 500:
                context["response"] = "<truncated>"
            else:
                context["response"] = response

        if error is not None:
            context["error"] = str(error)

        if success:
            logger.bind(**context).info("API Call")
        else:
            logger.bind(**context).error("API Error")
    else:
        # Fall back to standard logging
        if success:
            api_logger.info(f"API Call: {api_name} - {endpoint} - Params: {params}")
            if response:
                api_logger.debug(f"Response: {response}")
        else:
            api_logger.error(f"API Error: {api_name} - {endpoint} - Params: {params} - Error: {error}")

def log_data_operation(operation, data_type, details=None, success=True, error=None, data_size=None):
    """Log a data operation with details."""
    if LOGURU_AVAILABLE:
        # Use structured logging with Loguru
        context = {
            "operation": operation,
            "data_type": data_type,
            "success": success
        }

        if details is not None:
            context["details"] = details

        if data_size is not None:
            context["data_size"] = data_size

        if error is not None:
            context["error"] = str(error)

        if success:
            logger.bind(**context).info("Data Operation")
        else:
            logger.bind(**context).error("Data Error")
    else:
        # Fall back to standard logging
        if success:
            data_logger.info(f"Data Operation: {operation} - Type: {data_type} - Details: {details}")
        else:
            data_logger.error(f"Data Error: {operation} - Type: {data_type} - Details: {details} - Error: {error}")

def log_exception(exception, context=None, level="ERROR"):
    """Log an exception with enhanced details."""
    if LOGURU_AVAILABLE:
        # Use structured logging with Loguru
        ctx = context or {}
        ctx["exception_type"] = type(exception).__name__
        ctx["exception"] = str(exception)

        # Get traceback information
        tb_str = traceback.format_exc()
        ctx["traceback"] = tb_str

        # Log with appropriate level
        log_func = getattr(logger.bind(**ctx), level.lower(), logger.bind(**ctx).error)
        log_func(f"Exception: {str(exception)}")
    else:
        # Fall back to standard logging
        error_msg = f"Exception: {type(exception).__name__}: {str(exception)}"
        if context:
            error_msg += f" - Context: {context}"
        app_logger.error(error_msg)
        app_logger.error(traceback.format_exc())

def log_analytics_operation(operation, asset_type, ticker, details=None, execution_time=None, metrics=None):
    """Log an analytics operation with details."""
    if LOGURU_AVAILABLE:
        # Use structured logging with Loguru
        context = {
            "operation": operation,
            "asset_type": asset_type,
            "ticker": ticker
        }

        if details is not None:
            context["details"] = details

        if execution_time is not None:
            context["execution_time"] = f"{execution_time:.3f}s"

        if metrics is not None:
            context["metrics"] = metrics

        logger.bind(**context).info("Analytics Operation")
    else:
        # Fall back to standard logging
        metrics_str = f" - Metrics: {metrics}" if metrics else ""
        time_str = f" - Time: {execution_time:.3f}s" if execution_time else ""
        app_logger.info(f"Analytics: {operation} - {asset_type} - {ticker}{time_str}{metrics_str}")

def performance_timer(category="general"):
    """Decorator to measure and log function execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log performance data
                if LOGURU_AVAILABLE:
                    logger.bind(
                        category=category,
                        function=func.__name__,
                        execution_time=f"{execution_time:.3f}s",
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    ).debug(f"Performance: {func.__name__} took {execution_time:.3f}s")
                else:
                    app_logger.debug(f"Performance: {category} - {func.__name__} took {execution_time:.3f}s")

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                if LOGURU_AVAILABLE:
                    logger.bind(
                        category=category,
                        function=func.__name__,
                        execution_time=f"{execution_time:.3f}s",
                        error=str(e)
                    ).error(f"Error in {func.__name__}: {str(e)}")
                else:
                    app_logger.error(f"Error in {func.__name__}: {str(e)} (took {execution_time:.3f}s)")
                raise
        return wrapper
    return decorator


class PerformanceTimer:
    """Context manager for measuring and logging execution time."""
    def __init__(self, operation_name, category="general"):
        self.operation_name = operation_name
        self.category = category
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        if exc_type is None:
            # Operation completed successfully
            if LOGURU_AVAILABLE:
                logger.bind(
                    category=self.category,
                    operation=self.operation_name,
                    execution_time=f"{execution_time:.3f}s"
                ).debug(f"Performance: {self.operation_name} took {execution_time:.3f}s")
            else:
                app_logger.debug(f"Performance: {self.category} - {self.operation_name} took {execution_time:.3f}s")
        else:
            # Operation failed
            if LOGURU_AVAILABLE:
                logger.bind(
                    category=self.category,
                    operation=self.operation_name,
                    error=str(exc_val)
                ).error(f"Error in {self.operation_name}: {str(exc_val)}")
            else:
                app_logger.error(f"Error in {self.category} - {self.operation_name}: {str(exc_val)}")

def log_cache_operation(func=None):
    """Decorator to log cache operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = kwargs.get('key') or (args[1] if len(args) > 1 else 'unknown')
            cache_type = args[0].cache_type if hasattr(args[0], 'cache_type') else 'unknown'

            try:
                result = func(*args, **kwargs)
                operation = func.__name__

                # Create context for structured logging
                context = {
                    "cache_type": cache_type,
                    "cache_key": cache_key,
                    "operation": operation
                }

                if operation == 'get':
                    if result is not None:
                        if LOGURU_AVAILABLE:
                            logger.bind(**context).info(f"Cache hit: {cache_type}:{cache_key}")
                        else:
                            cache_logger.info(f"Cache hit: {cache_type}:{cache_key}")
                    else:
                        if LOGURU_AVAILABLE:
                            logger.bind(**context).info(f"Cache miss: {cache_type}:{cache_key}")
                        else:
                            cache_logger.info(f"Cache miss: {cache_type}:{cache_key}")
                elif operation == 'set':
                    if LOGURU_AVAILABLE:
                        logger.bind(**context).info(f"Cache set: {cache_type}:{cache_key}")
                    else:
                        cache_logger.info(f"Cache set: {cache_type}:{cache_key}")
                elif operation == 'delete':
                    if LOGURU_AVAILABLE:
                        logger.bind(**context).info(f"Cache delete: {cache_type}:{cache_key}")
                    else:
                        cache_logger.info(f"Cache delete: {cache_type}:{cache_key}")
                return result
            except Exception as e:
                if LOGURU_AVAILABLE:
                    logger.bind(
                        cache_type=cache_type,
                        cache_key=cache_key,
                        operation=func.__name__,
                        error=str(e)
                    ).error(f"Cache error: {cache_type}:{cache_key} - {str(e)}")
                else:
                    cache_logger.error(f"Cache error: {cache_type}:{cache_key} - {str(e)}")
                raise
        return wrapper

    if func:
        return decorator(func)
    return decorator
