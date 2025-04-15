"""
UI package for the AI Finance Dashboard.
This package contains UI components for the dashboard.
"""

from core.ui.backtesting_ui import render_backtesting_ui
from core.ui.comparison_ui import render_comparison_ui
from core.ui.news_ui import render_news_ui

__all__ = [
    'render_backtesting_ui',
    'render_comparison_ui',
    'render_news_ui'
]