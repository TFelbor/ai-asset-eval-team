"""
ETF analysis team for analyzing ETFs.
This module provides a standardized way to analyze ETFs.
"""
from typing import Dict, Any
from .base_team import BaseAnalysisTeam

class ETFAnalysisTeam(BaseAnalysisTeam):
    """ETF analysis team."""

    def __init__(self):
        """Initialize the ETF analysis team."""
        super().__init__(asset_type="etf")

    def _create_knowledge_base(self):
        """
        Create a knowledge base for the agent.

        Returns:
            Knowledge base
        """
        # Create a simple knowledge base
        kb = super()._create_knowledge_base()

        # Add ETF-specific knowledge as a source
        kb.add_source({
            "type": "text",
            "content": """
        # ETF Analysis Guide

        ## What are ETFs?
        Exchange-Traded Funds (ETFs) are investment funds that trade on stock exchanges, much like stocks. They hold assets such as stocks, bonds, commodities, or a mix of these. ETFs offer investors a way to diversify their investments without having to buy the individual components.

        ## Key ETF Metrics

        ### 1. Expense Ratio
        The expense ratio represents the annual fee that the fund charges investors for managing the ETF. Lower expense ratios are generally better for investors as they reduce the drag on returns.

        Formula: Expense Ratio = Annual Fund Operating Expenses / Average Net Assets

        ### 2. Assets Under Management (AUM)
        AUM represents the total market value of the assets that an ETF manages. Larger AUM generally indicates more liquidity and stability.

        ### 3. Average Daily Volume
        The average number of shares traded daily. Higher volume typically means better liquidity and tighter bid-ask spreads.

        ### 4. Tracking Error
        Measures how closely an ETF follows its benchmark index. Lower tracking error indicates better index replication.

        Formula: Tracking Error = Standard Deviation of (ETF Return - Index Return)

        ### 5. Premium/Discount to NAV
        The difference between an ETF's market price and its Net Asset Value (NAV). Ideally, this should be minimal.

        Formula: Premium/Discount = (Market Price - NAV) / NAV

        ### 6. Yield
        The income returned on an investment in the ETF, typically expressed as a percentage.

        Formula: Yield = Annual Distributions / Current Share Price

        ### 7. Beta
        Measures an ETF's volatility compared to the market. A beta of 1 indicates the ETF moves with the market, less than 1 means lower volatility, and greater than 1 means higher volatility.

        ### 8. Sharpe Ratio
        Measures risk-adjusted return. Higher Sharpe ratios indicate better risk-adjusted performance.

        Formula: Sharpe Ratio = (ETF Return - Risk-Free Rate) / ETF Standard Deviation

        ## ETF Types

        1. **Index ETFs**: Track a specific index like the S&P 500 or NASDAQ
        2. **Sector ETFs**: Focus on specific sectors like technology, healthcare, or energy
        3. **Bond ETFs**: Invest in various types of bonds
        4. **Commodity ETFs**: Track commodities like gold, oil, or agricultural products
        5. **Currency ETFs**: Track currency pairs or baskets of currencies
        6. **Inverse ETFs**: Aim to profit from a decline in the underlying index
        7. **Leveraged ETFs**: Use financial derivatives to amplify returns
        8. **Actively Managed ETFs**: Managed by portfolio managers who make decisions about allocations
        9. **Smart Beta ETFs**: Follow indexes that use alternative weighting schemes
        10. **Thematic ETFs**: Focus on specific investment themes like ESG, robotics, or cybersecurity

        ## ETF Analysis Framework

        ### 1. Investment Objective Analysis
        - Alignment with investment goals
        - Benchmark index selection
        - Investment strategy (passive vs. active)
        - Geographic and sector exposure

        ### 2. Performance Analysis
        - Historical returns
        - Risk-adjusted returns
        - Performance relative to benchmark
        - Performance in different market conditions

        ### 3. Cost Analysis
        - Expense ratio
        - Trading costs (bid-ask spread)
        - Tax efficiency
        - Total cost of ownership

        ### 4. Portfolio Analysis
        - Holdings concentration
        - Sector allocation
        - Geographic allocation
        - Market cap distribution
        - Credit quality (for bond ETFs)

        ### 5. Technical Analysis
        - Liquidity assessment
        - Premium/discount to NAV
        - Creation/redemption mechanism efficiency
        - Trading volume trends

        ## ETF Investment Considerations

        ### Strengths
        - Diversification
        - Low costs compared to mutual funds
        - Intraday trading
        - Tax efficiency
        - Transparency of holdings

        ### Weaknesses
        - Potential tracking error
        - Some niche ETFs have high expense ratios
        - Liquidity concerns for smaller ETFs
        - Complexity of some specialized ETFs
        - Potential hidden costs in bid-ask spreads

        ## ETF Selection Process

        ### 1. Define Investment Objectives
        - Investment goals
        - Risk tolerance
        - Time horizon
        - Income needs

        ### 2. Screen ETFs
        - Asset class
        - Geographic focus
        - Sector focus
        - Investment style
        - Expense ratio

        ### 3. Compare Similar ETFs
        - Performance history
        - Expense ratio
        - Tracking error
        - Liquidity
        - Holdings

        ### 4. Analyze Structure and Management
        - Fund sponsor reputation
        - Fund age and size
        - Index methodology
        - Rebalancing frequency
        - Securities lending practices

        ### 5. Evaluate Costs
        - Expense ratio
        - Bid-ask spread
        - Premium/discount to NAV
        - Tax efficiency
        - Trading commissions
        """
        })

        return kb

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze an ETF.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Analysis results
        """
        # Get ETF data
        etf_data = self.data_service.get_etf_data(ticker)

        # Check for errors
        if "error" in etf_data:
            return {"error": etf_data["error"]}

        # Get price history
        price_history = self.data_service.get_price_history(ticker, "etf", "1y")

        # Get news
        news = self.data_service.get_news("etf", ticker, limit=5)

        # Prepare context for the agent
        context = {
            "ticker": ticker,
            "name": etf_data.get("name", ticker),
            "category": etf_data.get("category", "Unknown"),
            "current_price": etf_data.get("current_price", "$0"),
            "aum": etf_data.get("aum", "$0"),
            "expense_ratio": etf_data.get("expense_ratio", "0%"),
            "ytd_return": etf_data.get("ytd_return", "0%"),
            "one_year_return": etf_data.get("one_year_return", "0%"),
            "three_year_return": etf_data.get("three_year_return", "0%"),
            "five_year_return": etf_data.get("five_year_return", "0%"),
            "top_holdings": etf_data.get("top_holdings", []),
            "sector_allocation": etf_data.get("sector_allocation", {}),
            "beta": etf_data.get("beta", 0),
            "price_history": price_history,
            "news": news
        }

        # Generate analysis prompt
        prompt = f"""
        You are an ETF analyst tasked with analyzing {ticker} ({etf_data.get('name', ticker)}).

        Here is the data you have:
        - Current Price: {etf_data.get('current_price', '$0')}
        - Category: {etf_data.get('category', 'Unknown')}
        - AUM: {etf_data.get('aum', '$0')}
        - Expense Ratio: {etf_data.get('expense_ratio', '0%')}
        - YTD Return: {etf_data.get('ytd_return', '0%')}
        - 1-Year Return: {etf_data.get('one_year_return', '0%')}
        - 3-Year Return: {etf_data.get('three_year_return', '0%')}
        - 5-Year Return: {etf_data.get('five_year_return', '0%')}
        - Top Holdings: {', '.join(etf_data.get('top_holdings', [])[:5])}
        - Beta: {etf_data.get('beta', 0)}

        Sector Allocation:
        {chr(10).join([f"- {sector}: {allocation}%" for sector, allocation in etf_data.get('sector_allocation', {}).items()])}

        Based on this data and your knowledge of ETFs, please provide:

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
                "name": etf_data.get("name", ticker),
                "etf": etf_data,
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
                "etf": etf_data,
                "price_history": price_history,
                "news": news,
                "recommendation": "Hold",
                "confidence": 0
            }
