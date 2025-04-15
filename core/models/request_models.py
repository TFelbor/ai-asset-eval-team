from pydantic import BaseModel, Field, validator
from typing import Optional, List
import re

class TickerRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not re.match(r'^[A-Za-z0-9.-]+$', v):
            raise ValueError('Ticker must contain only letters, numbers, dots, or hyphens')
        return v.upper()

class BacktestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    strategy: str = Field(..., regex='^(ma_cross|rsi|macd)$')
    period: str = Field('5y', regex='^[0-9]+[dwmy]$')
    short_window: Optional[int] = Field(50, ge=5, le=200)
    long_window: Optional[int] = Field(200, ge=20, le=500)
    rsi_window: Optional[int] = Field(14, ge=2, le=50)
    rsi_overbought: Optional[int] = Field(70, ge=50, le=90)
    rsi_oversold: Optional[int] = Field(30, ge=10, le=50)
    macd_fast: Optional[int] = Field(12, ge=5, le=30)
    macd_slow: Optional[int] = Field(26, ge=10, le=50)
    macd_signal: Optional[int] = Field(9, ge=3, le=20)
    
    @validator('period')
    def validate_period(cls, v):
        # Ensure period is in valid format (e.g., 1d, 5y, 3m)
        if not re.match(r'^[0-9]+[dwmy]$', v):
            raise ValueError('Period must be in format like 1d, 5y, 3m, etc.')
        return v
