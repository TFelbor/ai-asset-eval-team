from agno.team.team import Team
from core.agents import FundamentalAnalyst, CryptoAnalyst, MacroSentimentAnalyst, ETFAnalyst, ComparisonAgent


class StockAnalysisTeam(Team):
    def __init__(self):
        super().__init__(
            name="StockTeam",
            members=[FundamentalAnalyst(), MacroSentimentAnalyst()],
            mode="coordinate",
            instructions="Merge fundamental and macro analysis.",
        )

    def analyze(self, ticker: str) -> dict:
        fundamental = self.members[0].analyze(ticker)
        macro = self.members[1].analyze()
        return {"stock": fundamental, "macro": macro, "recommendation": "Buy" if fundamental["confidence"] > 70 else "Hold", "overall_score": (fundamental["confidence"] * 0.7 + macro["sentiment"] * 0.3)}


class CryptoAnalysisTeam(Team):
    def __init__(self):
        super().__init__(
            name="CryptoTeam",
            members=[CryptoAnalyst(), MacroSentimentAnalyst()],
            mode="coordinate",
            instructions="Merge crypto and macro analysis.",
        )

    def analyze(self, coin_id: str) -> dict:
        crypto = self.members[0].analyze(coin_id)
        macro = self.members[1].analyze()
        return {"crypto": crypto, "macro": macro, "recommendation": "Buy" if crypto["confidence"] > 70 else "Hold", "overall_score": (crypto["confidence"] * 0.7 + macro["sentiment"] * 0.3)}


class REITAnalysisTeam(Team):
    def __init__(self):
        super().__init__(
            name="REITTeam",
            members=[FundamentalAnalyst(), MacroSentimentAnalyst()],  # Using FundamentalAnalyst as placeholder
            mode="coordinate",
            instructions="Merge REIT and macro analysis.",
        )

    def analyze(self, ticker: str) -> dict:
        # Using FundamentalAnalyst as a placeholder for REIT analysis
        # In a real implementation, you would have a dedicated REITAnalyst
        fundamental = self.members[0].analyze(ticker)
        macro = self.members[1].analyze()

        # Convert fundamental data to REIT-specific format
        reit_data = {
            "ticker": ticker,
            "name": f"{ticker} REIT",
            "property_type": "Commercial",  # Placeholder
            "market_cap": 5000000000,  # Placeholder
            "dividend_yield": "4.5%",  # Placeholder
            "price_to_ffo": 15.2,  # Placeholder
            "debt_to_equity": 1.2,  # Placeholder
            "beta": 0.8  # Placeholder
        }

        return {"reit": reit_data, "macro": macro, "recommendation": "Buy" if fundamental["confidence"] > 70 else "Hold", "overall_score": (fundamental["confidence"] * 0.6 + macro["sentiment"] * 0.4)}


class ETFAnalysisTeam(Team):
    def __init__(self):
        super().__init__(
            name="ETFTeam",
            members=[ETFAnalyst(), MacroSentimentAnalyst()],
            mode="coordinate",
            instructions="Merge ETF and macro analysis.",
        )

    def analyze(self, ticker: str) -> dict:
        # Use the dedicated ETFAnalyst for ETF analysis
        etf_analysis = self.members[0].analyze(ticker)
        macro = self.members[1].analyze()

        # Calculate overall score and recommendation
        confidence = etf_analysis.get("confidence", 50)
        sentiment = macro.get("sentiment", 50)
        overall_score = (confidence * 0.7 + sentiment * 0.3)

        # Determine recommendation based on overall score
        if overall_score >= 75:
            recommendation = "Strong Buy"
        elif overall_score >= 65:
            recommendation = "Buy"
        elif overall_score >= 45:
            recommendation = "Hold"
        elif overall_score >= 35:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"

        return {"etf": etf_analysis, "macro": macro, "recommendation": recommendation, "overall_score": overall_score}


class ComparisonTeam(Team):
    def __init__(self):
        super().__init__(
            name="ComparisonTeam",
            members=[ComparisonAgent()],
            mode="solo",
            instructions="Compare different types of securities for portfolio allocation.",
        )

    def compare(self, securities: dict) -> dict:
        """Compare different securities and provide insights.

        Args:
            securities: Dictionary with keys 'stock', 'crypto', 'etf', 'reit' and corresponding data

        Returns:
            Comparison analysis with insights and recommendations
        """
        # Use the ComparisonAgent to perform the comparison
        comparison_agent = self.members[0]
        comparison_result = comparison_agent.compare(securities)

        return comparison_result
