"""
Teams package for the AI Finance Dashboard.
This package contains analysis teams for different asset types.
"""

# Import team classes directly
from core.teams.stock_team import StockAnalysisTeam
from core.teams.crypto_team import CryptoAnalysisTeam
from core.teams.reit_team import REITAnalysisTeam
from core.teams.etf_team import ETFAnalysisTeam
from core.teams.comparison_team import ComparisonTeam

__all__ = [
    'StockAnalysisTeam',
    'CryptoAnalysisTeam',
    'REITAnalysisTeam',
    'ETFAnalysisTeam',
    'ComparisonTeam'
]