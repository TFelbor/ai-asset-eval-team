"""
REIT analysis team for analyzing REITs.
This module provides a standardized way to analyze REITs.
"""
from typing import Dict, Any
from .base_team import BaseAnalysisTeam

class REITAnalysisTeam(BaseAnalysisTeam):
    """REIT analysis team."""

    def __init__(self):
        """Initialize the REIT analysis team."""
        super().__init__(asset_type="reit")

    def _create_knowledge_base(self):
        """
        Create a knowledge base for the agent.

        Returns:
            Knowledge base
        """
        # Create a simple knowledge base
        kb = super()._create_knowledge_base()

        # Add REIT-specific knowledge as a source
        kb.add_source({
            "type": "text",
            "content": """
            # REIT Analysis Guide

            ## What are REITs?
            Real Estate Investment Trusts (REITs) are companies that own, operate, or finance income-generating real estate across a range of property sectors. They allow individual investors to earn dividends from real estate investments without having to buy, manage, or finance any properties themselves.

            ## Key REIT Metrics

            ### 1. Funds From Operations (FFO)
            FFO is a key metric for evaluating REITs, as it provides a more accurate measure of a REIT's operating performance than net income. FFO adds back depreciation and amortization to net income and excludes gains or losses from property sales.

            Formula: FFO = Net Income + Depreciation + Amortization - Gains from Property Sales

            ### 2. Adjusted Funds From Operations (AFFO)
            AFFO is considered a more refined measure than FFO as it further adjusts for recurring capital expenditures and straight-line rent adjustments.

            Formula: AFFO = FFO - Recurring Capital Expenditures - Straight-Line Rent Adjustments

            ### 3. Net Asset Value (NAV)
            NAV represents the net market value of a REIT's assets minus its liabilities.

            Formula: NAV = Market Value of Assets - Liabilities

            ### 4. Dividend Yield
            Dividend yield is the annual dividend payment divided by the REIT's current share price.

            Formula: Dividend Yield = Annual Dividend / Current Share Price

            ### 5. Price to FFO (P/FFO)
            Similar to the P/E ratio for stocks, P/FFO is used to value REITs.

            Formula: P/FFO = Share Price / FFO per Share

            ### 6. Debt to EBITDA
            This ratio measures a REIT's ability to pay off its debt.

            Formula: Debt to EBITDA = Total Debt / EBITDA

            ### 7. Interest Coverage Ratio
            This ratio indicates how easily a REIT can pay interest on its outstanding debt.

            Formula: Interest Coverage Ratio = EBITDA / Interest Expense

            ### 8. Occupancy Rate
            The percentage of a REIT's rentable space that is occupied by tenants.

            Formula: Occupancy Rate = Occupied Space / Total Rentable Space

            ## REIT Property Types

            1. **Residential REITs**: Apartment buildings, student housing, manufactured homes, single-family homes
            2. **Retail REITs**: Shopping malls, shopping centers, outlet centers
            3. **Office REITs**: Office buildings in urban and suburban areas
            4. **Healthcare REITs**: Hospitals, medical centers, nursing facilities, retirement homes
            5. **Industrial REITs**: Warehouses, distribution centers, manufacturing facilities
            6. **Hotel & Resort REITs**: Hotels, resorts, and other lodging properties
            7. **Self-Storage REITs**: Self-storage facilities
            8. **Infrastructure REITs**: Fiber cables, wireless infrastructure, telecommunications towers
            9. **Data Center REITs**: Facilities that house servers and networking equipment
            10. **Diversified REITs**: Properties across multiple sectors

            ## REIT Analysis Framework

            ### 1. Property Portfolio Analysis
            - Property type and diversification
            - Geographic diversification
            - Tenant quality and lease terms
            - Occupancy rates and trends

            ### 2. Financial Analysis
            - FFO and AFFO growth
            - Dividend sustainability and growth
            - Balance sheet strength
            - Debt levels and maturity schedule

            ### 3. Management Analysis
            - Track record of management team
            - Alignment with shareholder interests
            - Capital allocation strategy
            - Development pipeline

            ### 4. Valuation Analysis
            - P/FFO relative to peers and historical average
            - Dividend yield relative to peers and historical average
            - NAV premium/discount
            - Implied cap rate

            ### 5. Market Analysis
            - Supply and demand dynamics in relevant markets
            - Regulatory environment
            - Interest rate environment
            - Economic trends affecting property types

            ## REIT Investment Considerations

            ### Strengths
            - Regular income through dividends
            - Liquidity compared to direct real estate investment
            - Professional management
            - Diversification benefits
            - Inflation hedge potential

            ### Weaknesses
            - Interest rate sensitivity
            - Property-specific risks
            - Economic sensitivity
            - Tax treatment of dividends
            - Potential for dilution through secondary offerings

            ## REIT Valuation Methods

            ### 1. Relative Valuation
            - P/FFO multiple compared to peers
            - Dividend yield compared to peers
            - Premium/discount to NAV

            ### 2. Discounted Cash Flow (DCF)
            - Project future FFO or AFFO
            - Discount to present value using appropriate discount rate
            - Sum of discounted cash flows plus terminal value

        ### 3. Net Asset Value (NAV)
        - Estimate market value of properties
        - Subtract liabilities
        - Divide by number of shares outstanding
            """

        })

        return kb

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a REIT.

        Args:
            ticker: REIT ticker symbol

        Returns:
            Analysis results
        """
        # Get REIT data
        reit_data = self.data_service.get_reit_data(ticker)

        # Check for errors
        if "error" in reit_data:
            return {"error": reit_data["error"]}

        # Get price history
        price_history = self.data_service.get_price_history(ticker, "reit", "1y")

        # Get news
        news = self.data_service.get_news("reit", ticker, limit=5)

        # Prepare context for the agent
        context = {
            "ticker": ticker,
            "name": reit_data.get("name", ticker),
            "property_type": reit_data.get("property_type", "Unknown"),
            "current_price": reit_data.get("current_price", "$0"),
            "market_cap": reit_data.get("market_cap", "$0"),
            "dividend_yield": reit_data.get("dividend_yield", "0%"),
            "price_to_ffo": reit_data.get("price_to_ffo", 0),
            "debt_to_equity": reit_data.get("debt_to_equity", 0),
            "beta": reit_data.get("beta", 0),
            "52w_high": reit_data.get("52w_high", 0),
            "52w_low": reit_data.get("52w_low", 0),
            "price_history": price_history,
            "news": news
        }

        # Generate analysis prompt
        prompt = f"""
        You are a REIT analyst tasked with analyzing {ticker} ({reit_data.get('name', ticker)}).

        Here is the data you have:
        - Current Price: {reit_data.get('current_price', '$0')}
        - Property Type: {reit_data.get('property_type', 'Unknown')}
        - Market Cap: {reit_data.get('market_cap', '$0')}
        - Dividend Yield: {reit_data.get('dividend_yield', '0%')}
        - Price to FFO: {reit_data.get('price_to_ffo', 0)}
        - Debt to Equity: {reit_data.get('debt_to_equity', 0)}
        - Beta: {reit_data.get('beta', 0)}
        - 52-Week High: ${reit_data.get('52w_high', 0)}
        - 52-Week Low: ${reit_data.get('52w_low', 0)}

        Based on this data and your knowledge of REITs, please provide:

        1. A comprehensive analysis of {ticker}, including:
           - Fundamental analysis
           - Technical outlook
           - Investment thesis

        2. A SWOT analysis:
           - Strengths
           - Weaknesses
           - Opportunities
           - Threats

        3. A recommendation (Buy, Hold, or Sell) with a confidence score (0-100)

        Format your response as a JSON object with the following structure:
        ```json
        {{"analysis": {{"fundamental_analysis": "Your fundamental analysis here", "technical_outlook": "Your technical analysis here", "investment_thesis": "Your investment thesis here", "strengths": ["Strength 1", "Strength 2"], "weaknesses": ["Weakness 1", "Weakness 2"], "opportunities": ["Opportunity 1", "Opportunity 2"], "threats": ["Threat 1", "Threat 2"]}}, "recommendation": "Buy/Hold/Sell", "confidence": 75}}
        ```

        Ensure your analysis is data-driven, balanced, and considers both the positives and negatives.
        """

        # Get analysis from agent
        response = self.agent.run(prompt, context=context)

        try:
            # Extract JSON from response
            import json
            import re

            # Find JSON pattern in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                analysis_result = json.loads(json_str)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    analysis_result = json.loads(json_str)
                else:
                    # Fallback to a simple structure
                    analysis_result = {
                        "analysis": {
                            "fundamental_analysis": "Analysis not available in structured format.",
                            "technical_outlook": "Analysis not available in structured format.",
                            "investment_thesis": "Analysis not available in structured format.",
                            "strengths": [],
                            "weaknesses": [],
                            "opportunities": [],
                            "threats": []
                        },
                        "recommendation": "Hold",
                        "confidence": 50
                    }

            # Combine all data
            result = {
                "ticker": ticker,
                "name": reit_data.get("name", ticker),
                "reit": reit_data,
                "price_history": price_history,
                "news": news,
                "analysis": analysis_result.get("analysis", {}),
                "recommendation": analysis_result.get("recommendation", "Hold"),
                "confidence": analysis_result.get("confidence", 50)
            }

            return result
        except Exception as e:
            # Return error with raw response
            return {
                "ticker": ticker,
                "error": f"Error parsing analysis: {str(e)}",
                "raw_response": response,
                "reit": reit_data,
                "price_history": price_history,
                "news": news,
                "recommendation": "Hold",
                "confidence": 0
            }