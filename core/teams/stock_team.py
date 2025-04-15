"""
Stock analysis team for analyzing stocks.
This module provides a standardized way to analyze stocks.
"""
from typing import Dict, Any
from .base_team import BaseAnalysisTeam

class StockAnalysisTeam(BaseAnalysisTeam):
    """Stock analysis team."""

    def __init__(self):
        """Initialize the stock analysis team."""
        super().__init__(asset_type="stock")

    def _create_knowledge_base(self):
        """
        Create a knowledge base for the agent.

        Returns:
            Knowledge base
        """
        # Create a simple knowledge base
        kb = super()._create_knowledge_base()

        # Add stock-specific knowledge as a source
        kb.add_source({
            "type": "text",
            "content": """
            Stock Analysis Guidelines:

            1. Fundamental Analysis:
               - Evaluate P/E ratio relative to industry average and historical values
               - Assess P/B ratio to determine if stock is trading above or below book value
               - Analyze dividend yield and payout ratio for income potential
               - Review revenue and earnings growth trends
               - Examine debt-to-equity ratio for financial health

            2. Technical Analysis:
               - Identify key support and resistance levels
               - Evaluate moving averages (50-day, 200-day) for trend direction
               - Assess relative strength index (RSI) for overbought/oversold conditions
               - Look for chart patterns (head and shoulders, double tops/bottoms)
               - Consider volume trends to confirm price movements

            3. Valuation Methods:
               - Discounted Cash Flow (DCF) analysis
               - Comparable company analysis (using P/E, P/S, EV/EBITDA multiples)
               - Dividend Discount Model for dividend-paying stocks
               - Asset-based valuation for asset-heavy companies
               - Growth-adjusted valuation metrics (PEG ratio)

            4. Risk Assessment:
               - Beta as a measure of market risk
               - Industry-specific risks
               - Company-specific risks (management, competition, regulation)
               - Macroeconomic risks (interest rates, inflation, economic growth)

            5. Investment Recommendations:
               - Strong Buy: Significantly undervalued with strong growth prospects
               - Buy: Moderately undervalued with good growth prospects
               - Hold: Fairly valued with stable prospects
               - Sell: Moderately overvalued or deteriorating fundamentals
               - Strong Sell: Significantly overvalued or serious fundamental issues
            """
        })

        return kb

    def _get_asset_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get data for the stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Stock data
        """
        return self.data_service.get_stock_data(ticker)

    def _perform_analysis(
        self,
        ticker: str,
        asset_data: Dict[str, Any],
        macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform analysis on the stock.

        Args:
            ticker: Stock ticker symbol
            asset_data: Stock data
            macro_data: Macroeconomic data

        Returns:
            Analysis results
        """
        # Extract key metrics and ensure they are the correct types
        current_price = asset_data.get("current_price", 0)
        if isinstance(current_price, str):
            try:
                current_price = float(current_price.replace("$", "").replace(",", ""))
            except (ValueError, TypeError):
                current_price = 0

        pe_ratio = asset_data.get("pe", 0)
        if isinstance(pe_ratio, str):
            try:
                pe_ratio = float(pe_ratio)
            except (ValueError, TypeError):
                pe_ratio = 0

        pb_ratio = asset_data.get("pb", 0)
        if isinstance(pb_ratio, str):
            try:
                pb_ratio = float(pb_ratio)
            except (ValueError, TypeError):
                pb_ratio = 0

        dividend_yield = asset_data.get("dividend_yield", "0%")

        beta = asset_data.get("beta", 0)
        if isinstance(beta, str):
            try:
                beta = float(beta)
            except (ValueError, TypeError):
                beta = 0

        market_cap = asset_data.get("market_cap", "$0")

        # Get price history
        price_history = self.data_service.get_price_history(ticker, "stock", "1mo")

        # Calculate intrinsic value using multiple methods
        intrinsic_value = self._calculate_intrinsic_value(asset_data)

        # Calculate upside potential
        upside_potential = self._calculate_upside_potential(current_price, intrinsic_value)

        # Get news
        news = self.data_service.get_news("stock", ticker, limit=5)

        # Perform AI analysis
        ai_analysis = self._perform_ai_analysis(ticker, asset_data, macro_data, intrinsic_value)

        # Combine all analysis
        analysis = {
            "intrinsic_value": intrinsic_value,
            "upside_potential": upside_potential,
            "price_history": price_history,
            "news": news,
            "ai_analysis": ai_analysis,
            "raw_metrics": {
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield,
                "beta": beta,
                "market_cap": market_cap
            }
        }

        return analysis

    def _create_report(
        self,
        ticker: str,
        asset_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a report from the analysis.

        Args:
            ticker: Stock ticker symbol
            asset_data: Stock data
            macro_data: Macroeconomic data
            analysis: Analysis results

        Returns:
            Analysis report
        """
        # Calculate confidence score
        confidence = self._calculate_confidence(analysis)

        # Determine recommendation
        recommendation = self._determine_recommendation(analysis, confidence)

        # Create report
        report = {
            "stock": {
                "ticker": ticker,
                "name": asset_data.get("name", ""),
                "sector": asset_data.get("sector", ""),
                "industry": asset_data.get("industry", ""),
                "current_price": asset_data.get("current_price", 0),
                "market_cap": asset_data.get("market_cap", "$0"),
                "pe": asset_data.get("pe", 0),
                "pb": asset_data.get("pb", 0),
                "dividend_yield": asset_data.get("dividend_yield", "0%"),
                "beta": asset_data.get("beta", 0),
                "52w_high": asset_data.get("52w_high", 0),
                "52w_low": asset_data.get("52w_low", 0),
                "intrinsic_value": analysis["intrinsic_value"],
                "upside_potential": analysis["upside_potential"],
                "price_history": analysis["price_history"],
                "news": analysis["news"],
                "analysis": analysis["ai_analysis"],
                "confidence": confidence,
                "raw": asset_data.get("raw", {})
            },
            "recommendation": recommendation,
            "overall_score": confidence
        }

        return report

    def _calculate_intrinsic_value(self, asset_data: Dict[str, Any]) -> float:
        """
        Calculate intrinsic value using multiple methods.

        Args:
            asset_data: Stock data

        Returns:
            Intrinsic value
        """
        # Ensure current_price is a float
        current_price = asset_data.get("current_price", 0)
        if isinstance(current_price, str):
            try:
                current_price = float(current_price.replace("$", "").replace(",", ""))
            except (ValueError, TypeError):
                current_price = 0

        # Method 1: P/E-based valuation
        pe_ratio = asset_data.get("pe", 0)
        # Ensure pe_ratio is a float
        if isinstance(pe_ratio, str):
            try:
                pe_ratio = float(pe_ratio)
            except (ValueError, TypeError):
                pe_ratio = 0

        industry_avg_pe = 15  # Default industry average P/E
        pe_based_value = 0

        if pe_ratio > 0:
            pe_based_value = current_price * (industry_avg_pe / pe_ratio)

        # Method 2: P/B-based valuation
        pb_ratio = asset_data.get("pb", 0)
        # Ensure pb_ratio is a float
        if isinstance(pb_ratio, str):
            try:
                pb_ratio = float(pb_ratio)
            except (ValueError, TypeError):
                pb_ratio = 0

        industry_avg_pb = 2  # Default industry average P/B
        pb_based_value = 0

        if pb_ratio > 0:
            pb_based_value = current_price * (industry_avg_pb / pb_ratio)

        # Method 3: Dividend Discount Model (simplified)
        dividend_yield_str = asset_data.get("dividend_yield", "0%")
        dividend_yield = float(dividend_yield_str.replace("%", "")) / 100
        growth_rate = 0.03  # Assumed growth rate
        discount_rate = 0.08  # Assumed discount rate

        ddm_value = 0
        if discount_rate > growth_rate and dividend_yield > 0:
            annual_dividend = current_price * dividend_yield
            ddm_value = annual_dividend / (discount_rate - growth_rate)

        # Method 4: DCF (simplified)
        # This is a very simplified DCF calculation
        dcf_value = current_price * 1.1  # Assume 10% undervalued

        # Combine all methods with weights
        weights = {
            "pe": 0.25,
            "pb": 0.25,
            "ddm": 0.2,
            "dcf": 0.3
        }

        intrinsic_value = (
            weights["pe"] * pe_based_value +
            weights["pb"] * pb_based_value +
            weights["ddm"] * ddm_value +
            weights["dcf"] * dcf_value
        )

        # If intrinsic value is too low or zero, use current price
        if intrinsic_value < current_price * 0.5 or intrinsic_value == 0:
            intrinsic_value = current_price

        return round(intrinsic_value, 2)

    def _calculate_upside_potential(
        self,
        current_price: float,
        intrinsic_value: float
    ) -> str:
        """
        Calculate upside potential.

        Args:
            current_price: Current stock price
            intrinsic_value: Calculated intrinsic value

        Returns:
            Upside potential as a percentage string
        """
        # Ensure both values are floats
        try:
            if isinstance(current_price, str):
                current_price = float(current_price.replace("$", "").replace(",", ""))
            if isinstance(intrinsic_value, str):
                intrinsic_value = float(intrinsic_value.replace("$", "").replace(",", ""))

            current_price = float(current_price)
            intrinsic_value = float(intrinsic_value)
        except (ValueError, TypeError):
            return "0%"

        if current_price <= 0 or intrinsic_value <= 0:
            return "0%"

        upside = (intrinsic_value / current_price - 1) * 100
        return f"{upside:.2f}%"

    def _perform_ai_analysis(
        self,
        ticker: str,
        asset_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        intrinsic_value: float
    ) -> Dict[str, Any]:
        """
        Perform AI analysis on the stock.

        Args:
            ticker: Stock ticker symbol
            asset_data: Stock data
            macro_data: Macroeconomic data
            intrinsic_value: Calculated intrinsic value

        Returns:
            AI analysis results
        """
        # Prepare prompt for the agent
        prompt = f"""
        Analyze the stock {ticker} ({asset_data.get('name', '')}) based on the following data:

        Current Price: ${asset_data.get('current_price', 0)}
        Intrinsic Value: ${intrinsic_value}
        P/E Ratio: {asset_data.get('pe', 0)}
        P/B Ratio: {asset_data.get('pb', 0)}
        Dividend Yield: {asset_data.get('dividend_yield', '0%')}
        Beta: {asset_data.get('beta', 0)}
        Market Cap: {asset_data.get('market_cap', '$0')}
        52-Week High: ${asset_data.get('52w_high', 0)}
        52-Week Low: ${asset_data.get('52w_low', 0)}

        Macroeconomic Environment:
        GDP Outlook: {macro_data.get('gdp_outlook', 'Stable')}
        Inflation Risk: {macro_data.get('inflation_risk', 'Moderate')}
        Unemployment Trend: {macro_data.get('unemployment_trend', 'Stable')}

        Provide a comprehensive analysis including:
        1. Fundamental Analysis
        2. Technical Outlook
        3. Valuation Assessment
        4. Risk Factors
        5. Investment Thesis

        Format your response as a JSON object with the following structure:
        {{"fundamental_analysis": "Your analysis here", "technical_outlook": "Your analysis here", "valuation_assessment": "Your analysis here", "risk_factors": "Your analysis here", "investment_thesis": "Your analysis here", "strengths": ["Strength 1", "Strength 2"], "weaknesses": ["Weakness 1", "Weakness 2"], "opportunities": ["Opportunity 1", "Opportunity 2"], "threats": ["Threat 1", "Threat 2"]}}
        """

        # Get response from the agent
        response = self.agent.generate(prompt)

        # Parse JSON response
        try:
            import json
            analysis = json.loads(response)
            return analysis
        except Exception as e:
            # If JSON parsing fails, return a structured error
            return {
                "error": f"Failed to parse AI analysis: {str(e)}",
                "raw_response": response
            }

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> int:
        """
        Calculate confidence score based on analysis.

        Args:
            analysis: Analysis results

        Returns:
            Confidence score (0-100)
        """
        # Start with a base confidence
        confidence = 50

        # Adjust based on upside potential
        upside_str = analysis.get("upside_potential", "0%")
        upside = float(upside_str.replace("%", ""))

        if upside > 30:
            confidence += 20
        elif upside > 15:
            confidence += 10
        elif upside > 5:
            confidence += 5
        elif upside < -15:
            confidence -= 15
        elif upside < -5:
            confidence -= 10

        # Adjust based on AI analysis
        ai_analysis = analysis.get("ai_analysis", {})

        # Count strengths and weaknesses
        strengths = len(ai_analysis.get("strengths", []))
        weaknesses = len(ai_analysis.get("weaknesses", []))

        # Adjust confidence based on strengths vs weaknesses
        if strengths > weaknesses + 2:
            confidence += 15
        elif strengths > weaknesses:
            confidence += 10
        elif weaknesses > strengths + 2:
            confidence -= 15
        elif weaknesses > strengths:
            confidence -= 10

        # Ensure confidence is within bounds
        confidence = max(0, min(100, confidence))

        return confidence

    def _determine_recommendation(
        self,
        analysis: Dict[str, Any],
        confidence: int
    ) -> str:
        """
        Determine recommendation based on analysis and confidence.

        Args:
            analysis: Analysis results
            confidence: Confidence score

        Returns:
            Recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
        """
        # Get upside potential
        upside_str = analysis.get("upside_potential", "0%")
        upside = float(upside_str.replace("%", ""))

        # Determine recommendation based on upside and confidence
        if upside > 30 and confidence >= 70:
            return "Strong Buy"
        elif upside > 15 and confidence >= 60:
            return "Buy"
        elif upside > -10 and upside < 15:
            return "Hold"
        elif upside < -15 and confidence <= 40:
            return "Strong Sell"
        elif upside < -5 and confidence <= 50:
            return "Sell"
        else:
            return "Hold"
