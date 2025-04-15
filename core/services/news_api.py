"""
News API service for the Financial Analysis Dashboard.
"""
from typing import List, Dict, Any
import random
from datetime import datetime, timedelta

class NewsAPI:
    """
    Service for fetching news articles related to financial assets.
    This is a placeholder implementation that returns mock data.
    """
    
    def __init__(self):
        """Initialize the NewsAPI service."""
        self.sources = [
            "Financial Times", "Bloomberg", "CNBC", "Reuters", 
            "Wall Street Journal", "MarketWatch", "Barron's", 
            "The Economist", "Forbes", "Business Insider"
        ]
    
    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get general market news.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        return self._generate_mock_articles("market", limit)
    
    def get_stock_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        return self._generate_mock_articles(f"stock:{ticker}", limit)
    
    def get_crypto_news(self, coin_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific cryptocurrency.
        
        Args:
            coin_id: Cryptocurrency ID or symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        return self._generate_mock_articles(f"crypto:{coin_id}", limit)
    
    def get_reit_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific REIT.
        
        Args:
            ticker: REIT ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        return self._generate_mock_articles(f"reit:{ticker}", limit)
    
    def get_etf_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news for a specific ETF.
        
        Args:
            ticker: ETF ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        return self._generate_mock_articles(f"etf:{ticker}", limit)
    
    def _generate_mock_articles(self, topic: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate mock news articles for a given topic.
        
        Args:
            topic: Topic to generate articles for
            limit: Maximum number of articles to return
            
        Returns:
            List of mock news articles
        """
        articles = []
        now = datetime.now()
        
        # Generate headlines based on topic
        if topic == "market":
            headlines = [
                "Markets React to Federal Reserve Decision",
                "Global Stocks Rise on Economic Recovery Hopes",
                "Inflation Concerns Weigh on Investor Sentiment",
                "Tech Stocks Lead Market Rally",
                "Oil Prices Surge on Supply Constraints",
                "Bond Yields Climb as Investors Reassess Risk",
                "Market Volatility Increases Amid Geopolitical Tensions",
                "Earnings Season Exceeds Analyst Expectations",
                "Retail Investors Drive Meme Stock Resurgence",
                "Economic Data Points to Continued Growth",
                "Central Banks Signal Policy Shift",
                "Market Breadth Improves as Small Caps Outperform",
                "Sector Rotation Accelerates as Economy Reopens",
                "Cryptocurrency Market Faces Regulatory Scrutiny",
                "IPO Market Remains Hot with New Listings"
            ]
        elif topic.startswith("stock:"):
            ticker = topic.split(":")[1].upper()
            headlines = [
                f"{ticker} Reports Strong Quarterly Earnings",
                f"{ticker} Announces New Product Line",
                f"{ticker} Shares Climb on Analyst Upgrade",
                f"{ticker} CEO Discusses Future Growth Strategy",
                f"{ticker} Expands into New Markets",
                f"{ticker} Faces Increased Competition",
                f"{ticker} Announces Stock Buyback Program",
                f"{ticker} Dividend Increase Pleases Investors",
                f"{ticker} Partners with Tech Giant on Innovation",
                f"Institutional Investors Increase Stakes in {ticker}",
                f"{ticker} Addresses Supply Chain Challenges",
                f"{ticker} Restructures Operations for Efficiency",
                f"{ticker} Beats Revenue Expectations",
                f"Analysts Divided on {ticker}'s Valuation",
                f"{ticker} Implements Sustainability Initiatives"
            ]
        elif topic.startswith("crypto:"):
            coin = topic.split(":")[1].upper()
            headlines = [
                f"{coin} Reaches New All-Time High",
                f"{coin} Adoption Increases Among Institutional Investors",
                f"{coin} Network Upgrade Improves Scalability",
                f"Regulatory Clarity Boosts {coin} Price",
                f"{coin} Mining Difficulty Adjusts After Hash Rate Changes",
                f"New {coin} DeFi Applications Gain Traction",
                f"{coin} Faces Volatility Amid Market Uncertainty",
                f"Major Exchange Lists {coin} for Trading",
                f"{coin} Foundation Announces Development Grants",
                f"Whale Movements Spotted in {coin} Blockchain",
                f"{coin} Correlation with Traditional Markets Shifts",
                f"Technical Analysis: {coin} Forms Bullish Pattern",
                f"{coin} Community Votes on Governance Proposal",
                f"Central Bank Digital Currencies Impact {coin} Outlook",
                f"{coin} Layer-2 Solutions Address Congestion Issues"
            ]
        elif topic.startswith("reit:"):
            ticker = topic.split(":")[1].upper()
            headlines = [
                f"{ticker} REIT Increases Dividend Payout",
                f"{ticker} Acquires Premium Commercial Properties",
                f"{ticker} REIT Benefits from Real Estate Market Recovery",
                f"{ticker} Occupancy Rates Exceed Industry Average",
                f"{ticker} REIT Refinances Debt at Lower Rates",
                f"{ticker} Expands Portfolio with Strategic Acquisitions",
                f"{ticker} REIT Focuses on Sustainable Building Practices",
                f"{ticker} Reports Strong Funds from Operations",
                f"{ticker} REIT Adapts to Changing Market Demands",
                f"{ticker} Management Discusses Long-term Growth Strategy",
                f"{ticker} REIT Navigates Interest Rate Environment",
                f"{ticker} Property Values Appreciate in Key Markets",
                f"{ticker} REIT Implements Technology Upgrades",
                f"{ticker} Leasing Activity Remains Robust",
                f"{ticker} REIT Explores International Expansion"
            ]
        elif topic.startswith("etf:"):
            ticker = topic.split(":")[1].upper()
            headlines = [
                f"{ticker} ETF Sees Record Inflows",
                f"{ticker} ETF Rebalances Portfolio",
                f"{ticker} Outperforms Benchmark Index",
                f"{ticker} ETF Adds New Holdings",
                f"{ticker} Expense Ratio Remains Competitive",
                f"{ticker} ETF Provides Sector Diversification",
                f"{ticker} Assets Under Management Grow Substantially",
                f"{ticker} ETF Adapts to Market Trends",
                f"{ticker} Dividend Yield Attracts Income Investors",
                f"{ticker} ETF Implements ESG Criteria",
                f"{ticker} Trading Volume Indicates Increased Interest",
                f"{ticker} ETF Strategy Explained by Fund Manager",
                f"{ticker} Performance Analysis: Risk vs. Reward",
                f"{ticker} ETF Celebrates Anniversary with Strong Returns",
                f"{ticker} Compared to Competing Funds"
            ]
        else:
            headlines = [
                "Financial Markets Update: Latest Trends and Analysis",
                "Economic Indicators Point to Recovery",
                "Investors React to Policy Announcements",
                "Market Sentiment Shifts on New Data",
                "Sector Analysis: Winners and Losers"
            ]
        
        # Generate random articles
        for i in range(min(limit, len(headlines))):
            # Random date within the last 7 days
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            article_date = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            articles.append({
                "id": f"article-{topic}-{i}",
                "title": headlines[i],
                "source": random.choice(self.sources),
                "url": f"https://example.com/news/{topic.replace(':', '-')}/{i}",
                "published_at": article_date.isoformat(),
                "summary": f"This is a mock summary for the article about {headlines[i].lower()}. "
                          f"The article discusses important developments and potential implications for investors.",
                "image_url": f"https://example.com/images/news/{i % 5 + 1}.jpg"
            })
        
        # Sort by date (newest first)
        articles.sort(key=lambda x: x["published_at"], reverse=True)
        
        return articles
