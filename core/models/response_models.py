"""
Response data models for the Financial Analysis Dashboard API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChartLink(BaseModel):
    """Chart link model."""
    type: str
    url: str
    title: str


class NewsLink(BaseModel):
    """News link model."""
    url: str
    title: str


class AdvancedLink(BaseModel):
    """Advanced analytics link model."""
    url: str
    title: str


class AnalysisResponse(BaseModel):
    """Base analysis response model."""
    report: Dict[str, Any]
    insights: List[str]
    charts: Optional[List[ChartLink]] = None
    news: Optional[NewsLink] = None
    advanced: Optional[AdvancedLink] = None


class StockAnalysisResponse(AnalysisResponse):
    """Stock analysis response model."""
    pass


class CryptoAnalysisResponse(AnalysisResponse):
    """Cryptocurrency analysis response model."""
    pass


class REITAnalysisResponse(AnalysisResponse):
    """REIT analysis response model."""
    pass


class ETFAnalysisResponse(AnalysisResponse):
    """ETF analysis response model."""
    pass


class ComparisonResponse(BaseModel):
    """Comparison response model."""
    reports: Dict[str, Any]
    summary: str
