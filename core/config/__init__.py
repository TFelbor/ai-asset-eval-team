"""
Configuration package for the financial analysis dashboard.
"""
from config.settings import *

# Try to import local config if it exists
try:
    from config.local import *
except ImportError:
    pass
