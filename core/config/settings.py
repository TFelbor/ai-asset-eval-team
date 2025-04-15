"""
Configuration settings for the financial analysis dashboard.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "TGG73CRDOH6JEJLI")  # Get from https://www.alphavantage.co/support/#api-key
# CoinGecko API key - set to None to use free API by default
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", None)  # Using free API by default
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "f4222cd00cfd4f7ba8c79d421f7271df")  # Get from https://newsapi.org

# Cache settings
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_EXPIRY = {
    "stock": 3600,  # 1 hour
    "crypto": 1800,  # 30 minutes
    "macro": 14400,  # 4 hours
    "reit": 3600,  # 1 hour
    "etf": 3600,  # 1 hour
}

# Server settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# Data sources
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
STOCK_FUNDAMENTALS_CSV = DATA_DIR / "stock_fundamentals.csv"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default values
DEFAULT_SECTOR_PE = 25.0
DEFAULT_CONFIDENCE_THRESHOLD = 70  # Threshold for Buy recommendation

# Chart settings
class ChartSettings:
    COLORS = {
        "primary": "#2196F3",
        "secondary": "#4CAF50",
        "accent": "#FF9800",
        "danger": "#F44336",
        "success": "#4CAF50",
        "info": "#2196F3",
        "warning": "#FF9800"
    }

    DIMENSIONS = {
        "default_width": 800,
        "default_height": 500,
        "large_width": 900,
        "large_height": 600
    }

    TEMPLATES = {
        "default": "plotly",
        "dark": "plotly_dark"
    }
    # Additional chart settings can be added here

class Settings:
    # Base directory
    BASE_DIR = BASE_DIR

    # API Keys
    ALPHA_VANTAGE_API_KEY = "TGG73CRDOH6JEJLI"
    COINGECKO_API_KEY = None  # Using free API by default
    NEWS_API_KEY = "f4222cd00cfd4f7ba8c79d421f7271df"

    # Cache settings
    CACHE_DIR = CACHE_DIR
    CACHE_EXPIRY = CACHE_EXPIRY

    # Server settings
    HOST = HOST
    PORT = PORT
    DEBUG = DEBUG

    # Data sources
    DATA_DIR = DATA_DIR
    TEMP_DIR = TEMP_DIR
    STOCK_FUNDAMENTALS_CSV = STOCK_FUNDAMENTALS_CSV

    # Logging
    LOG_DIR = LOG_DIR
    LOG_LEVEL = LOG_LEVEL
    LOG_FORMAT = LOG_FORMAT

    # Default values
    DEFAULT_SECTOR_PE = DEFAULT_SECTOR_PE
    DEFAULT_CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD

# Initialize settings
settings = Settings()
chart_settings = ChartSettings()

# WebSocket settings (no longer used - using simulated data instead)
# WS_HOST = "0.0.0.0"
# WS_PORT = 8001
# WS_UPDATE_INTERVAL = 5  # seconds
