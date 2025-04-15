"""
Yahoo Finance API integration module.
Provides enhanced functionality on top of the yfinance library.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime, timedelta


class YahooFinanceAPI:
    """Enhanced Yahoo Finance API wrapper."""

    @staticmethod
    def get_stock_data(ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing stock data
        """
        try:
            # Print debug info
            print(f"Fetching data for {ticker} from Yahoo Finance")

            # Initialize ticker object
            stock = yf.Ticker(ticker)

            # Get info with error handling
            try:
                info = stock.info
                if not info or len(info) == 0:
                    print(f"Warning: No info data returned for {ticker}")
                    info = {}
            except Exception as info_error:
                print(f"Error fetching info for {ticker}: {str(info_error)}")
                info = {}

            # Get historical data with error handling
            try:
                hist = stock.history(period="1y")
                if hist.empty:
                    print(f"Warning: No historical data returned for {ticker}")
            except Exception as hist_error:
                print(f"Error fetching historical data for {ticker}: {str(hist_error)}")
                hist = pd.DataFrame()

            # Calculate additional metrics
            if not hist.empty and len(hist) > 0:
                price_change_1y = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
            else:
                price_change_1y = 0
                volatility = 0

            # Get analyst recommendations
            try:
                recommendations = stock.recommendations
                rec_summary = recommendations.groupby('To Grade').size().to_dict()
            except:
                rec_summary = {}

            # Compile results
            result = {
                "ticker": ticker,
                "company_name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "eps": info.get("trailingEps", 0),
                "beta": info.get("beta", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "price_change_1y": price_change_1y,
                "volatility": volatility,
                "analyst_recommendations": rec_summary,
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "target_price": info.get("targetMeanPrice", 0),
                "target_upside": ((info.get("targetMeanPrice", 0) / info.get("currentPrice", 1)) - 1) * 100 if info.get("currentPrice") else 0,
                "recommendation_key": info.get("recommendationKey", ""),
                # Add historical data for charts
                "history": hist
            }

            return result
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @staticmethod
    def get_financial_statements(ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        try:
            stock = yf.Ticker(ticker)

            return {
                "income_statement": stock.income_stmt,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def calculate_dcf(ticker: str, growth_rate: float = 0.15, discount_rate: float = 0.10,
                      terminal_growth: float = 0.03, years: int = 5) -> Dict[str, Any]:
        """
        Calculate Discounted Cash Flow valuation.

        Args:
            ticker: Stock ticker symbol
            growth_rate: Annual growth rate for cash flows
            discount_rate: Discount rate (WACC)
            terminal_growth: Terminal growth rate
            years: Number of years to project

        Returns:
            Dictionary with DCF valuation results
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get financial data
            financials = stock.cashflow

            # Use Free Cash Flow if available, otherwise use Operating Cash Flow
            if 'Free Cash Flow' in financials.index:
                latest_cash_flow = financials.loc['Free Cash Flow'].iloc[0]
            else:
                latest_cash_flow = financials.loc['Operating Cash Flow'].iloc[0]

            # Project future cash flows
            cash_flows = []
            for year in range(1, years + 1):
                cf = latest_cash_flow * ((1 + growth_rate) ** year)
                cash_flows.append(cf)

            # Calculate terminal value
            terminal_value = cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

            # Discount all cash flows
            discounted_cash_flows = []
            for i, cf in enumerate(cash_flows):
                dcf = cf / ((1 + discount_rate) ** (i + 1))
                discounted_cash_flows.append(dcf)

            # Discount terminal value
            discounted_terminal_value = terminal_value / ((1 + discount_rate) ** years)

            # Calculate enterprise value
            enterprise_value = sum(discounted_cash_flows) + discounted_terminal_value

            # Get debt and cash
            try:
                balance_sheet = stock.balance_sheet
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                cash = balance_sheet.loc['Cash'].iloc[0] if 'Cash' in balance_sheet.index else 0
            except:
                total_debt = info.get('totalDebt', 0)
                cash = info.get('totalCash', 0)

            # Calculate equity value
            equity_value = enterprise_value - total_debt + cash

            # Calculate per share value
            shares_outstanding = info.get('sharesOutstanding', 0)
            if shares_outstanding > 0:
                per_share_value = equity_value / shares_outstanding
            else:
                per_share_value = 0

            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            return {
                "ticker": ticker,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "dcf_per_share": per_share_value,
                "current_price": current_price,
                "upside_potential": ((per_share_value / current_price) - 1) * 100 if current_price else 0,
                "assumptions": {
                    "growth_rate": growth_rate,
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "projection_years": years
                }
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @staticmethod
    def get_etf_data(ticker: str) -> Dict[str, Any]:
        """
        Get ETF-specific data.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary with ETF information
        """
        try:
            etf = yf.Ticker(ticker)
            info = etf.info

            # Get holdings if available
            try:
                holdings = etf.holdings
                top_holdings = holdings.head(10).to_dict() if isinstance(holdings, pd.DataFrame) else {}
            except:
                top_holdings = {}

            return {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "category": info.get("category", ""),
                "asset_class": info.get("assetClass", ""),
                "net_assets": info.get("totalAssets", 0),
                "expense_ratio": info.get("annualReportExpenseRatio", 0),
                "inception_date": info.get("fundInceptionDate", ""),
                "yield": info.get("yield", 0) * 100 if info.get("yield") else 0,
                "ytd_return": info.get("ytdReturn", 0) * 100 if info.get("ytdReturn") else 0,
                "three_year_return": info.get("threeYearAverageReturn", 0) * 100 if info.get("threeYearAverageReturn") else 0,
                "five_year_return": info.get("fiveYearAverageReturn", 0) * 100 if info.get("fiveYearAverageReturn") else 0,
                "top_holdings": top_holdings,
                "sector_weights": info.get("sectorWeightings", []),
                "bond_ratings": info.get("bondRatings", [])
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @staticmethod
    def get_reit_data(ticker: str) -> Dict[str, Any]:
        """
        Get REIT-specific data.

        Args:
            ticker: REIT ticker symbol

        Returns:
            Dictionary with REIT information
        """
        try:
            reit = yf.Ticker(ticker)
            info = reit.info

            # Calculate additional REIT metrics
            financials = reit.income_stmt
            balance_sheet = reit.balance_sheet

            # Try to calculate FFO (Funds From Operations)
            try:
                net_income = financials.loc['Net Income'].iloc[0]
                depreciation = financials.loc['Depreciation'].iloc[0] if 'Depreciation' in financials.index else 0
                ffo = net_income + depreciation
            except:
                ffo = 0

            return {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "property_type": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "price_to_ffo": info.get("marketCap", 0) / ffo if ffo else 0,
                "funds_from_operations": ffo,
                "debt_to_equity": info.get("debtToEquity", 0),
                "beta": info.get("beta", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker}
