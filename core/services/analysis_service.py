"""
Analysis service for the Financial Analysis Dashboard.
"""
from typing import Dict, Any, List, Optional

from teams import (
    StockAnalysisTeam,
    CryptoAnalysisTeam,
    REITAnalysisTeam,
    ETFAnalysisTeam,
    ComparisonTeam
)


class AnalysisService:
    """Service for analyzing financial assets."""

    @staticmethod
    def analyze_stock(ticker: str) -> Dict[str, Any]:
        """
        Analyze a stock by ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analysis report
        """
        team = StockAnalysisTeam()
        report = team.analyze(ticker)

        # Generate insights based on analysis
        insights = AnalysisService._generate_stock_insights(report)

        # Add chart links
        chart_links = [
            {"type": "price", "url": f"/analyze/stock/chart/{ticker}?chart_type=price", "title": "Financial Health Dashboard"},
            {"type": "metrics", "url": f"/analyze/stock/chart/{ticker}?chart_type=metrics", "title": "Key Metrics Chart"},
            {"type": "history", "url": f"/analyze/stock/chart/{ticker}?chart_type=history", "title": "Price History"},
            {"type": "candlestick", "url": f"/analyze/stock/chart/{ticker}?chart_type=candlestick", "title": "Candlestick Chart"}
        ]

        # Add news link
        news_link = {"url": f"/news/stock/{ticker}", "title": "Latest News"}

        # Add advanced analytics link
        advanced_link = {"url": f"/advanced/stock/{ticker}", "title": "Advanced Analytics"}

        return {
            "report": report,
            "insights": insights,
            "charts": chart_links,
            "news": news_link,
            "advanced": advanced_link
        }

    @staticmethod
    def analyze_crypto(coin_id: str) -> Dict[str, Any]:
        """
        Analyze a cryptocurrency by coin ID.

        Args:
            coin_id: Cryptocurrency ID

        Returns:
            Analysis report
        """
        team = CryptoAnalysisTeam()
        report = team.analyze(coin_id)

        # Generate insights based on analysis
        insights = AnalysisService._generate_crypto_insights(report)

        # Add chart links
        chart_links = [
            {"type": "price", "url": f"/analyze/crypto/chart/{coin_id}?chart_type=price", "title": "Price Chart"},
            {"type": "performance", "url": f"/analyze/crypto/chart/{coin_id}?chart_type=performance", "title": "Performance Chart"},
            {"type": "volume", "url": f"/analyze/crypto/chart/{coin_id}?chart_type=volume", "title": "Volume Chart"}
        ]

        # Add news link
        news_link = {"url": f"/news/crypto/{coin_id}", "title": "Latest News"}

        # Add advanced analytics link
        advanced_link = {"url": f"/advanced/crypto/{coin_id}", "title": "Advanced Analytics"}

        return {
            "report": report,
            "insights": insights,
            "charts": chart_links,
            "news": news_link,
            "advanced": advanced_link
        }

    @staticmethod
    def analyze_reit(ticker: str) -> Dict[str, Any]:
        """
        Analyze a REIT by ticker symbol.

        Args:
            ticker: REIT ticker symbol

        Returns:
            Analysis report
        """
        team = REITAnalysisTeam()
        report = team.analyze(ticker)

        # Generate insights based on analysis
        insights = AnalysisService._generate_reit_insights(report)

        return {
            "report": report,
            "insights": insights
        }

    @staticmethod
    def analyze_etf(ticker: str) -> Dict[str, Any]:
        """
        Analyze an ETF by ticker symbol.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Analysis report
        """
        team = ETFAnalysisTeam()
        report = team.analyze(ticker)

        # Generate insights based on analysis
        insights = AnalysisService._generate_etf_insights(report)

        return {
            "report": report,
            "insights": insights
        }

    @staticmethod
    def compare_securities(
        stock: Optional[str] = None,
        crypto: Optional[str] = None,
        reit: Optional[str] = None,
        etf: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare different types of securities.

        Args:
            stock: Stock ticker symbol (optional)
            crypto: Cryptocurrency ID (optional)
            reit: REIT ticker symbol (optional)
            etf: ETF ticker symbol (optional)

        Returns:
            Comparison report
        """
        results = {}
        securities = {}

        # Analyze stock if provided
        if stock:
            stock_team = StockAnalysisTeam()
            stock_report = stock_team.analyze(stock)
            results["stock"] = stock_report
            securities["stock"] = stock_report["stock"]

        # Analyze crypto if provided
        if crypto:
            crypto_team = CryptoAnalysisTeam()
            crypto_report = crypto_team.analyze(crypto)
            results["crypto"] = crypto_report
            securities["crypto"] = crypto_report["crypto"]

        # Analyze REIT if provided
        if reit:
            reit_team = REITAnalysisTeam()
            reit_report = reit_team.analyze(reit)
            results["reit"] = reit_report
            securities["reit"] = reit_report["reit"]

        # Analyze ETF if provided
        if etf:
            etf_team = ETFAnalysisTeam()
            etf_report = etf_team.analyze(etf)
            results["etf"] = etf_report
            securities["etf"] = etf_report["etf"]

        # Use the ComparisonTeam to generate insights and recommendations
        comparison_team = ComparisonTeam()
        comparison_result = comparison_team.compare(securities)

        # Combine the individual reports with the comparison results
        response = {
            "reports": results,
            "insights": comparison_result.get("insights", []),
            "recommendations": comparison_result.get("recommendations", []),
            "comparison_data": comparison_result.get("comparison_data", {}),
            "summary": f"Compared {len(results)} different types of securities."
        }

        return response

    @staticmethod
    def _generate_stock_insights(report: Dict[str, Any]) -> List[str]:
        """
        Generate insights for a stock analysis.

        Args:
            report: Stock analysis report

        Returns:
            List of insights
        """
        try:
            # Extract data from the report
            stock_data = report.get('stock', {})
            macro_data = report.get('macro', {})

            if not stock_data:
                raise ValueError("No stock data available")

            # Ensure numeric values are properly formatted
            # Convert PE ratio to float if it's not already
            pe_ratio = stock_data.get('pe', 0)
            if pe_ratio is None:
                pe_ratio = 0
            elif isinstance(pe_ratio, str):
                try:
                    pe_ratio = float(pe_ratio.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    pe_ratio = 0

            # Convert sector PE to float if it's not already
            sector_pe = stock_data.get('sector_pe', 0)
            if sector_pe is None:
                sector_pe = 0
            elif isinstance(sector_pe, str):
                try:
                    sector_pe = float(sector_pe.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    sector_pe = 0

            # Convert P/B ratio to float if it's not already
            pb_ratio = stock_data.get('pb', 0)
            if pb_ratio is None:
                pb_ratio = 0
            elif isinstance(pb_ratio, str):
                try:
                    pb_ratio = float(pb_ratio.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    pb_ratio = 0

            # Convert beta to float if it's not already
            beta = stock_data.get('beta', 0)
            if beta is None:
                beta = 0
            elif isinstance(beta, str):
                try:
                    beta = float(beta.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    beta = 0

            # Convert overall score to float if it's not already
            overall_score = report.get('overall_score', 0)
            if overall_score is None:
                overall_score = 0
            elif isinstance(overall_score, str):
                try:
                    overall_score = float(overall_score.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    overall_score = 0

            # Use the enhanced data from FundamentalAnalyst
            insights = [
                f"{stock_data.get('name', 'Unknown')} ({stock_data.get('ticker', 'Unknown')}) in {stock_data.get('sector', 'Unknown')} sector, {stock_data.get('industry', 'Unknown')} industry.",
                f"Current price: {stock_data.get('current_price', '$0.00')} with upside potential of {stock_data.get('upside_potential', '0.00%')}.",
                f"P/E ratio is {pe_ratio:.1f} compared to sector average of {sector_pe:.1f}. P/B ratio: {pb_ratio:.2f}.",
                f"Market cap: {stock_data.get('market_cap', '$0')} with dividend yield of {stock_data.get('dividend_yield', '0.00%')} and beta of {beta:.2f}.",
                f"Market sentiment is {macro_data.get('sentiment', 0)}/100 with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {overall_score:.1f}/100 score."
            ]
        except Exception as e:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed, but there was an error generating insights: {str(e)}"]

        return insights

    @staticmethod
    def _generate_crypto_insights(report: Dict[str, Any]) -> List[str]:
        """
        Generate insights for a cryptocurrency analysis.

        Args:
            report: Cryptocurrency analysis report

        Returns:
            List of insights
        """
        try:
            crypto_data = report.get('crypto', {})
            macro_data = report.get('macro', {})

            if not crypto_data:
                raise ValueError("No cryptocurrency data available")

            # Use the real data from CoinGecko
            # Handle both pre-formatted strings and raw values
            current_price = crypto_data.get('current_price', '0')
            if not isinstance(current_price, str):
                current_price = f"${current_price:,.2f}"

            all_time_high = crypto_data.get('all_time_high', '0')
            if not isinstance(all_time_high, str):
                all_time_high = f"${all_time_high:,.2f}"

            all_time_high_change = crypto_data.get('all_time_high_change', '0%')
            if not isinstance(all_time_high_change, str):
                all_time_high_change = f"{all_time_high_change:.2f}%"
            elif not all_time_high_change.endswith('%'):
                all_time_high_change = f"{all_time_high_change}%"

            market_dominance = crypto_data.get('market_dominance', '0')
            if not isinstance(market_dominance, str):
                market_dominance = f"{market_dominance:.2f}%"
            elif not market_dominance.endswith('%'):
                market_dominance = f"{market_dominance}%"

            # Convert overall score to float if it's not already
            overall_score = report.get('overall_score', 0)
            if overall_score is None:
                overall_score = 0
            elif isinstance(overall_score, str):
                try:
                    overall_score = float(overall_score.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    overall_score = 0

            # Create insights with safe formatting
            insights = [
                f"Current price: {current_price} with market cap of {crypto_data.get('mcap', 'Unknown')}.",
                f"Market cap rank: #{crypto_data.get('market_cap_rank', 0)} with {market_dominance} market dominance.",
                f"24h price change: {crypto_data.get('price_change_24h', '0%')} with 7d change of {crypto_data.get('price_change_7d', '0%')}.",
                f"Volatility: {crypto_data.get('volatility', 'Unknown')} with 24h volume of {crypto_data.get('volume_24h', '$0')}.",
                f"All-time high: {all_time_high} ({all_time_high_change} from current price).",
                f"Supply: {crypto_data.get('circulating_supply', 'Unknown')} / {crypto_data.get('max_supply', 'Unlimited')} ({crypto_data.get('supply_percentage', 'N/A')} circulating).",
                f"Macroeconomic outlook: {macro_data.get('gdp_outlook', 'Stable')} with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {overall_score:.1f}/100 score."
            ]
        except Exception as e:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed, but there was an error generating insights: {str(e)}"]

        return insights

    @staticmethod
    def _generate_reit_insights(report: Dict[str, Any]) -> List[str]:
        """
        Generate insights for a REIT analysis.

        Args:
            report: REIT analysis report

        Returns:
            List of insights
        """
        try:
            reit_data = report.get('reit', {})
            macro_data = report.get('macro', {})

            if not reit_data:
                raise ValueError("No REIT data available")

            # Ensure numeric values are properly formatted
            # Convert market cap to float if it's not already
            market_cap = reit_data.get('market_cap', 0)
            if market_cap is None:
                market_cap = 0
            elif isinstance(market_cap, str):
                try:
                    # Remove currency symbols and commas
                    market_cap = float(market_cap.replace('$', '').replace(',', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3'))
                except (ValueError, TypeError):
                    market_cap = 0

            # Convert price to FFO to float if it's not already
            price_to_ffo = reit_data.get('price_to_ffo', 0)
            if price_to_ffo is None:
                price_to_ffo = 0
            elif isinstance(price_to_ffo, str):
                try:
                    price_to_ffo = float(price_to_ffo.replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    price_to_ffo = 0

            # Convert debt to equity to float if it's not already
            debt_to_equity = reit_data.get('debt_to_equity', 0)
            if debt_to_equity is None:
                debt_to_equity = 0
            elif isinstance(debt_to_equity, str):
                try:
                    debt_to_equity = float(debt_to_equity.replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    debt_to_equity = 0

            # Convert beta to float if it's not already
            beta = reit_data.get('beta', 0)
            if beta is None:
                beta = 0
            elif isinstance(beta, str):
                try:
                    beta = float(beta.replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    beta = 0

            # Convert overall score to float if it's not already
            overall_score = report.get('overall_score', 0)
            if overall_score is None:
                overall_score = 0
            elif isinstance(overall_score, str):
                try:
                    overall_score = float(overall_score.replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    overall_score = 0

            insights = [
                f"{reit_data.get('name', 'Unknown')} is a {reit_data.get('property_type', 'Commercial')} REIT with market cap of ${market_cap:,.0f}.",
                f"Dividend yield: {reit_data.get('dividend_yield', '0.0%')} with price to FFO ratio of {price_to_ffo:.2f}.",
                f"Debt to equity ratio: {debt_to_equity:.2f} with beta of {beta:.2f}.",
                f"Macroeconomic outlook: Stable with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {overall_score:.1f}/100 score."
            ]
        except Exception as e:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed, but there was an error generating insights: {str(e)}"]

        return insights

    @staticmethod
    def _generate_etf_insights(report: Dict[str, Any]) -> List[str]:
        """
        Generate insights for an ETF analysis.

        Args:
            report: ETF analysis report

        Returns:
            List of insights
        """
        try:
            etf_data = report.get('etf', {})
            macro_data = report.get('macro', {})

            if not etf_data:
                raise ValueError("No ETF data available")

            # Convert overall score to float if it's not already
            overall_score = report.get('overall_score', 0)
            if overall_score is None:
                overall_score = 0
            elif isinstance(overall_score, str):
                try:
                    overall_score = float(overall_score.replace('$', '').replace(',', '').replace('%', ''))
                except (ValueError, TypeError):
                    overall_score = 0

            insights = [
                f"{etf_data.get('name', 'Unknown')} is a {etf_data.get('category', 'Broad Market')} ETF in the {etf_data.get('asset_class', 'Equity')} asset class.",
                f"Expense ratio: {etf_data.get('expense_ratio', '0.0%')} with yield of {etf_data.get('yield', '0.0%')}.",
                f"YTD return: {etf_data.get('ytd_return', '0.0%')} with 3-year return of {etf_data.get('three_year_return', '0.0%')}.",
                f"Macroeconomic outlook: Stable with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {overall_score:.1f}/100 score."
            ]
        except Exception as e:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed, but there was an error generating insights: {str(e)}"]

        return insights
