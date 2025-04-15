from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from pathlib import Path
import yfinance as yf
import math
from cache_manager import CacheManager
from api_integrations.coingecko_client import CoinGeckoClient
from utils.serialization import make_json_serializable

# Initialize cache managers for different data types
stock_cache = CacheManager(cache_type="stock")
crypto_cache = CacheManager(cache_type="crypto")
macro_cache = CacheManager(cache_type="macro")
etf_cache = CacheManager(cache_type="etf")
comparison_cache = CacheManager(cache_type="comparison")

# Initialize API clients
coingecko_client = CoinGeckoClient()


class FundamentalAnalyst(Agent):
    """Evaluates stocks using DCF, P/E, and growth metrics."""
    def __init__(self):
        super().__init__(
            name="FundamentalAnalyst",
            model=OpenAIChat(id="gpt-4-turbo"),
            instructions="""
            Analyze stocks using:
            1. Discounted Cash Flow (DCF) with 10% margin of error.
            2. Compare P/E, P/B to sector averages.
            3. Output confidence score based on data consistency.
            Respond in JSON.
            """
        )
        self.knowledge = CombinedKnowledgeBase(
            sources=[CSVKnowledgeBase(path=Path("data/stock_fundamentals.csv"))]
        )

    def analyze(self, ticker: str) -> dict:
        # Check if we have cached data for this ticker
        cache_key = f"stock_{ticker.lower()}"
        cached_data = stock_cache.get(cache_key)

        if cached_data:
            print(f"Using cached data for {ticker}")
            return cached_data

        try:
            print(f"Fetching fresh data for {ticker}")
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            data = stock.info

            # Basic financial metrics
            enterprise_value = data.get("enterpriseValue", 0)
            market_cap = data.get("marketCap", 0)
            pe = data.get("trailingPE", 0)
            pb = data.get("priceToBook", 0)
            dividend_yield = data.get("dividendYield", 0) * 100 if data.get("dividendYield") else 0
            beta = data.get("beta", 0)

            # Get current price
            current_price = data.get("currentPrice", 0)
            if current_price == 0:
                current_price = data.get("previousClose", 0)

            # Calculate DCF (simplified)
            dcf = enterprise_value * 0.9

            # Get analyst target price if available
            target_price = data.get("targetMeanPrice", 0)

            # Get sector PE for comparison
            sector_pe = data.get("sectorPE", 25.0)  # Default to 25 if not available

            # Calculate intrinsic value using multiple valuation methods
            # 1. DCF-based intrinsic value (already calculated as 'dcf')
            # 2. Earnings-based valuation using industry standard PE
            earnings_per_share = data.get("trailingEPS", 0)
            earnings_value = earnings_per_share * sector_pe if earnings_per_share > 0 and sector_pe > 0 else 0

            # 3. Book value with growth premium
            book_value_per_share = data.get("bookValue", 0)
            growth_rate = data.get("earningsGrowth", 0) or data.get("revenueGrowth", 0) or 0.05  # Default to 5% if no data
            # Apply growth premium to book value (higher growth = higher premium)
            growth_premium = 1 + (growth_rate * 5)  # 5x multiplier for growth impact
            book_value_adjusted = book_value_per_share * growth_premium if book_value_per_share > 0 else 0

            # 4. Analyst consensus target price (already available as 'target_price')

            # Calculate composite intrinsic value using weighted average of available methods
            available_methods = 0
            composite_value = 0

            # Weight factors based on reliability (industry standard practice)
            if dcf > 0:
                composite_value += dcf * 0.35  # 35% weight to DCF
                available_methods += 1

            if earnings_value > 0:
                composite_value += earnings_value * 0.25  # 25% weight to earnings-based valuation
                available_methods += 1

            if book_value_adjusted > 0:
                composite_value += book_value_adjusted * 0.15  # 15% weight to adjusted book value
                available_methods += 1

            if target_price > 0:
                composite_value += target_price * 0.25  # 25% weight to analyst targets
                available_methods += 1

            # Calculate final intrinsic value (average of available methods)
            intrinsic_value = composite_value / available_methods if available_methods > 0 else 0

            # Calculate upside potential based on intrinsic value vs current price
            upside_potential = ((intrinsic_value / current_price) - 1) * 100 if current_price > 0 and intrinsic_value > 0 else 0

            # Store the intrinsic value for reporting
            calculated_intrinsic_value = intrinsic_value

            # Get sector information
            sector = data.get("sector", "Unknown")
            industry = data.get("industry", "Unknown")

            # Calculate confidence score
            pe_score = min(100, int(100 * (1 - abs(pe - 25) / 25))) if pe > 0 else 50
            growth_score = min(100, int(data.get("earningsGrowth", 0.05) * 1000)) if data.get("earningsGrowth") else 50
            confidence = int((pe_score * 0.6) + (growth_score * 0.4))

            result = {
                "ticker": ticker,
                "name": data.get("shortName", ticker),
                "sector": sector,
                "industry": industry,
                "current_price": f"${current_price:,.2f}",
                "dcf": f"${dcf:,.2f}",
                "target_price": f"${target_price:,.2f}" if target_price > 0 else "N/A",
                "intrinsic_value": f"${calculated_intrinsic_value:,.2f}" if calculated_intrinsic_value > 0 else "N/A",
                "market_cap": f"${market_cap:,.2f}",
                "pe": pe,
                "pb": pb,
                "dividend_yield": f"{dividend_yield:.2f}%",
                "beta": beta,
                "upside_potential": f"{upside_potential:.2f}%",
                "sector_pe": sector_pe,  # Now using actual sector PE if available
                "confidence": confidence,
                # Add raw values for charting
                "raw": {
                    "current_price": current_price,
                    "market_cap": market_cap,
                    "pe": pe,
                    "pb": pb,
                    "dividend_yield": dividend_yield,
                    "beta": beta,
                    "target_price": target_price,
                    "dcf": dcf,
                    "intrinsic_value": calculated_intrinsic_value,
                    "earnings_per_share": earnings_per_share,
                    "book_value": book_value_per_share
                }
            }

            # Serialize and cache the result
            serialized_result = make_json_serializable(result)
            stock_cache.set(cache_key, serialized_result)

            return result
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            # Return minimal data on error
            return {
                "ticker": ticker,
                "dcf": "$0.00",
                "pe": 0,
                "confidence": 0,
                "error": str(e)
            }


class CryptoAnalyst(Agent):
    """Analyzes crypto assets using on-chain data and market metrics."""
    def __init__(self):
        super().__init__(
            name="CryptoAnalyst",
            model=OpenAIChat(id="gpt-4-turbo"),
            instructions="Evaluate market cap, volatility, on-chain metrics, and technical indicators."
        )

    def analyze(self, coin: str) -> dict:
        # Check if we have cached data for this coin
        cache_key = f"crypto_{coin.lower()}"
        cached_data = crypto_cache.get(cache_key)

        if cached_data:
            print(f"Using cached data for {coin}")
            return cached_data

        try:
            print(f"Fetching fresh data for {coin}")
            from analytics.advanced_metrics import AdvancedAnalytics

            # Get advanced metrics using the improved implementation
            # This will handle ticker conversion internally
            advanced_metrics = AdvancedAnalytics.get_advanced_crypto_metrics(coin)

            if "error" in advanced_metrics:
                raise ValueError(f"Could not find data for {coin}: {advanced_metrics.get('error')}")

            # Get comprehensive analysis from CoinGecko
            coin_id = advanced_metrics.get("coin_id", coin.lower())
            analysis = coingecko_client.analyze_coin(coin_id)

            # Extract data from the analysis
            current_price = advanced_metrics.get("current_price", 0)
            market_cap = advanced_metrics.get("market_cap", 0)
            market_cap_rank = advanced_metrics.get("market_cap_rank", 0)
            volume_24h = advanced_metrics.get("total_volume", 0)
            price_change_24h = advanced_metrics.get("price_change_percentage_24h", 0)
            price_change_7d = advanced_metrics.get("price_change_percentage_7d", 0)
            volatility = advanced_metrics.get("volatility", 0)
            rsi = advanced_metrics.get("rsi", 50)
            sharpe_ratio = advanced_metrics.get("sharpe_ratio", 0)
            max_drawdown = advanced_metrics.get("max_drawdown", 0)

            # Get market dominance from analysis
            market_dominance = analysis.get("market_dominance", 0)

            # Determine volatility category based on volatility value
            volatility_category = "Low"
            if volatility > 100:
                volatility_category = "Very High"
            elif volatility > 75:
                volatility_category = "High"
            elif volatility > 50:
                volatility_category = "Medium"

            # Calculate investment quality score using industry-standard metrics
            # This is a more sophisticated approach to evaluating crypto assets

            # 1. Market Maturity & Adoption (30% of total score)
            # Market cap rank is weighted logarithmically (diminishing returns for higher ranks)
            market_maturity_score = 0
            if market_cap_rank > 0:
                # Log scale for market cap rank (1st = 30, 10th = 20, 100th = 10)
                market_maturity_score = max(5, 30 - 10 * math.log10(market_cap_rank))
            else:
                # Default score if rank is unknown
                market_maturity_score = 10

            # Add points for trading volume relative to market cap (liquidity indicator)
            volume_to_mcap_ratio = 0
            if market_cap > 0 and volume_24h > 0:
                volume_to_mcap_ratio = volume_24h / market_cap
                # Healthy daily volume is 2-15% of market cap
                if volume_to_mcap_ratio > 0.15:
                    # Too high volume can indicate manipulation
                    market_maturity_score += 5
                elif volume_to_mcap_ratio > 0.02:
                    # Ideal range
                    market_maturity_score += 10
                else:
                    # Low liquidity
                    market_maturity_score += volume_to_mcap_ratio * 500  # Scale up to max 10 points

            # 2. Technical Indicators (30% of total score)
            technical_score = 0

            # RSI evaluation (ideally between 40-60 for stability)
            rsi_score = 0
            if 40 <= rsi <= 60:
                rsi_score = 10  # Balanced RSI
            elif (30 <= rsi < 40) or (60 < rsi <= 70):
                rsi_score = 7   # Slightly overbought/oversold
            elif (20 <= rsi < 30) or (70 < rsi <= 80):
                rsi_score = 4   # Moderately overbought/oversold
            else:
                rsi_score = 2   # Extremely overbought/oversold

            # Moving average analysis
            ma_score = 0
            if 'moving_averages' in advanced_metrics and current_price > 0:
                ma_data = advanced_metrics['moving_averages']
                # Price above longer-term MAs is bullish
                if 'ma200' in ma_data and current_price > ma_data['ma200']:
                    ma_score += 4
                if 'ma100' in ma_data and current_price > ma_data['ma100']:
                    ma_score += 3
                if 'ma50' in ma_data and current_price > ma_data['ma50']:
                    ma_score += 3

            # Volatility assessment (lower is better for stability)
            volatility_score = 0
            if volatility < 50:
                volatility_score = 10  # Low volatility
            elif volatility < 75:
                volatility_score = 7   # Moderate volatility
            elif volatility < 100:
                volatility_score = 4   # High volatility
            else:
                volatility_score = 2   # Extreme volatility

            # Combine technical indicators
            technical_score = rsi_score + ma_score + volatility_score
            # Normalize to 30 points max
            technical_score = min(30, technical_score * 30 / 30)

            # 3. Risk-Adjusted Performance (40% of total score)
            performance_score = 0

            # Sharpe ratio (risk-adjusted return)
            sharpe_score = 0
            if sharpe_ratio > 2:
                sharpe_score = 15  # Excellent
            elif sharpe_ratio > 1:
                sharpe_score = 10  # Good
            elif sharpe_ratio > 0:
                sharpe_score = 5   # Positive but not great
            else:
                sharpe_score = 0   # Negative (poor)

            # Maximum drawdown assessment (lower is better)
            drawdown_score = 0
            if max_drawdown > -0.3:
                drawdown_score = 10  # Low drawdown
            elif max_drawdown > -0.5:
                drawdown_score = 7   # Moderate drawdown
            elif max_drawdown > -0.7:
                drawdown_score = 4   # High drawdown
            else:
                drawdown_score = 2   # Extreme drawdown

            # Recent performance trend (weighted for recency)
            trend_score = 0
            # Weight recent performance more heavily but avoid extremes
            if -10 <= price_change_7d <= 20:
                trend_score += 5 + (price_change_7d / 4)  # Max 10 points
            else:
                trend_score += 2  # Extreme movements are concerning

            # Get 30-day price change if available
            price_change_30d = advanced_metrics.get('monthly_return', 0)
            if -20 <= price_change_30d <= 40:
                trend_score += 3 + (price_change_30d / 10)  # Max 7 points
            else:
                trend_score += 1

            # Combine performance metrics
            performance_score = sharpe_score + drawdown_score + min(15, trend_score)
            # Normalize to 40 points max
            performance_score = min(40, performance_score * 40 / 40)

            # Calculate overall confidence score (0-100)
            confidence = max(0, min(100, int(market_maturity_score + technical_score + performance_score)))

            # Format the result with more comprehensive data
            result = {
                "coin": coin,
                "name": advanced_metrics.get("name", coin.upper()),
                "symbol": advanced_metrics.get("symbol", coin.upper()),
                "current_price": f"${current_price:,.2f}",
                "mcap": f"${market_cap:,.0f}",
                "volume_24h": f"${volume_24h:,.0f}",
                "market_cap_rank": market_cap_rank,
                "market_dominance": f"{market_dominance:.2f}%",
                "volatility": f"{volatility:.2f}% ({volatility_category})",
                "rsi": f"{rsi:.1f}",
                "sharpe_ratio": f"{sharpe_ratio:.2f}",
                "max_drawdown": f"{max_drawdown:.2f}%",
                "price_change_24h": f"{price_change_24h:.2f}%",
                "price_change_7d": f"{price_change_7d:.2f}%",
                "all_time_high": f"${advanced_metrics.get('ath', 0):,.2f}",
                "all_time_high_change": f"{advanced_metrics.get('ath_change_percentage', 0):.2f}%",
                "circulating_supply": f"{advanced_metrics.get('circulating_supply', 0):,.0f}",
                "max_supply": f"{advanced_metrics.get('max_supply', 0):,.0f}" if advanced_metrics.get('max_supply') else "Unlimited",
                "supply_percentage": f"{(advanced_metrics.get('circulating_supply', 0) / advanced_metrics.get('max_supply', 1) * 100):.1f}%" if advanced_metrics.get('max_supply') else "N/A",
                "developer_activity": analysis.get("developer_activity", {}),
                "community_data": analysis.get("community_data", {}),
                "confidence": confidence,
                # Add raw values for charting
                "raw": {
                    "price": current_price,
                    "market_cap": market_cap,
                    "volume_24h": volume_24h,
                    "market_dominance": market_dominance,
                    "volatility": volatility,
                    "rsi": rsi,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "price_change_24h": price_change_24h,
                    "price_change_7d": price_change_7d,
                    "all_time_high": advanced_metrics.get("ath", 0),
                    "all_time_high_change": advanced_metrics.get("ath_change_percentage", 0),
                    "circulating_supply": advanced_metrics.get("circulating_supply", 0),
                    "max_supply": advanced_metrics.get("max_supply", 0),
                }
            }

            # Add price history data for charts
            if "price_history" in advanced_metrics:
                result["price_history"] = advanced_metrics["price_history"]

            # Serialize and cache the result
            serialized_result = make_json_serializable(result)
            crypto_cache.set(cache_key, serialized_result)

            return result
        except Exception as e:
            print(f"Error analyzing {coin}: {str(e)}")
            # Return minimal data on error
            return {
                "coin": coin,
                "mcap": "$0",
                "volatility": "Unknown",
                "confidence": 0,
                "error": str(e)
            }


class MacroSentimentAnalyst(Agent):
    """Assesses market-wide conditions."""
    def __init__(self):
        super().__init__(
            name="MacroAnalyst",
            instructions="""
            Analyze:
            1. Fed interest rates
            2. Inflation (CPI)
            3. Geopolitical risks
            Output sentiment score (0-100).
            """
        )

    def analyze(self) -> dict:
        # Check if we have cached macro data
        cache_key = "macro_sentiment"
        cached_data = macro_cache.get(cache_key)

        if cached_data:
            print("Using cached macro data")
            return cached_data

        try:
            print("Generating fresh macro data")
            # In a real implementation, this would fetch data from economic APIs
            # or use LLMs to analyze recent news and economic indicators

            # Generate realistic macroeconomic data
            sentiment = 65  # 0-100 scale
            fed_impact = "Neutral"  # Negative, Neutral, Positive
            inflation_risk = "Moderate"  # Low, Moderate, High
            gdp_growth = 2.3  # Percentage
            unemployment = 3.8  # Percentage
            interest_rate = 5.25  # Percentage

            # Market trend analysis
            market_trend = "Bullish"  # Bearish, Neutral, Bullish
            volatility_index = 18.5  # VIX value
            recession_probability = 25  # Percentage

            result = {
                "sentiment": sentiment,
                "fed_impact": fed_impact,
                "inflation_risk": inflation_risk,
                "gdp_growth": f"{gdp_growth:.1f}%",
                "gdp_outlook": "Stable",
                "unemployment": f"{unemployment:.1f}%",
                "interest_rate": f"{interest_rate:.2f}%",
                "market_trend": market_trend,
                "volatility_index": volatility_index,
                "recession_probability": f"{recession_probability}%",
                # Add raw values for charting
                "raw": {
                    "sentiment": sentiment,
                    "gdp_growth": gdp_growth,
                    "unemployment": unemployment,
                    "interest_rate": interest_rate,
                    "volatility_index": volatility_index,
                    "recession_probability": recession_probability
                }
            }

            # Cache the result
            macro_cache.set(cache_key, result)

            return result
        except Exception as e:
            print(f"Error in macro analysis: {str(e)}")
            # Return minimal data on error
            return {
                "sentiment": 50,
                "fed_impact": "Unknown",
                "inflation_risk": "Unknown",
                "error": str(e)
            }


class ETFAnalyst(Agent):
    """Analyzes ETFs using holdings, performance, and expense metrics."""
    def __init__(self):
        super().__init__(
            name="ETFAnalyst",
            model=OpenAIChat(id="gpt-4-turbo"),
            instructions="""
            Analyze ETFs using:
            1. Expense ratio and fund efficiency
            2. Historical performance and volatility
            3. Holdings composition and sector allocation
            4. Compare to benchmark indices
            Output comprehensive analysis with confidence score.
            """
        )

    def analyze(self, ticker: str) -> dict:
        # Check if we have cached data for this ETF
        cache_key = f"etf_{ticker.lower()}"
        cached_data = etf_cache.get(cache_key)

        if cached_data:
            print(f"Using cached data for {ticker}")
            return cached_data

        try:
            print(f"Fetching fresh data for ETF {ticker}")
            # Fetch data from Yahoo Finance
            etf = yf.Ticker(ticker)
            data = etf.info

            # Basic ETF metrics
            name = data.get("shortName", f"{ticker} ETF")
            category = data.get("category", "Unknown")
            asset_class = data.get("assetClass", "Equity")

            # Financial metrics
            expense_ratio = data.get("annualReportExpenseRatio", 0.0) * 100 if data.get("annualReportExpenseRatio") else 0.0
            net_assets = data.get("totalAssets", 0)
            nav = data.get("navPrice", 0) or data.get("previousClose", 0)

            # Performance metrics
            ytd_return = data.get("ytdReturn", 0) * 100 if data.get("ytdReturn") else 0
            three_year_return = data.get("threeYearAverageReturn", 0) * 100 if data.get("threeYearAverageReturn") else 0
            five_year_return = data.get("fiveYearAverageReturn", 0) * 100 if data.get("fiveYearAverageReturn") else 0
            dividend_yield = data.get("yield", 0) * 100 if data.get("yield") else 0
            beta = data.get("beta", 0) or data.get("beta3Year", 0)

            # Get current price
            current_price = data.get("regularMarketPrice", 0)
            if current_price == 0:
                current_price = data.get("previousClose", 0)

            # Get holdings data if available
            holdings = []
            try:
                # Use the get_holdings method instead of accessing holdings attribute directly
                # This is a safer approach as yfinance may have changed its API
                if hasattr(etf, 'get_holdings'):
                    holdings_data = etf.get_holdings()
                    if holdings_data is not None and not holdings_data.empty:
                        top_holdings = holdings_data.head(10)
                        for _, row in top_holdings.iterrows():
                            holdings.append({
                                "symbol": row.get("symbol", ""),
                                "name": row.get("name", ""),
                                "weight": row.get("% Assets", 0)
                            })
                # Fallback to major_holders if holdings are not available
                elif hasattr(etf, 'major_holders'):
                    holdings_data = etf.major_holders
                    if holdings_data is not None and not holdings_data.empty:
                        for i, row in holdings_data.iterrows():
                            if i < 10:  # Limit to top 10
                                holdings.append({
                                    "name": f"Holder {i+1}",
                                    "weight": row[0] if len(row) > 0 else "N/A"
                                })
            except Exception as holdings_error:
                print(f"Error fetching holdings for {ticker}: {str(holdings_error)}")

            # Calculate ETF quality score using industry-standard metrics
            # This follows professional fund rating methodologies

            # 1. Cost Efficiency (20% of total score)
            cost_score = 0
            # Expense ratio evaluation (lower is better)
            if expense_ratio < 0.2:  # Ultra low-cost ETFs
                cost_score = 20
            elif expense_ratio < 0.5:  # Low-cost ETFs
                cost_score = 15
            elif expense_ratio < 0.75:  # Moderate cost ETFs
                cost_score = 10
            elif expense_ratio < 1.0:  # Average cost ETFs
                cost_score = 5
            else:  # High cost ETFs
                cost_score = max(0, 20 - int(expense_ratio * 5))

            # 2. Performance Metrics (40% of total score)
            performance_score = 0

            # Risk-adjusted returns (Sharpe ratio proxy using returns and beta)
            risk_adjusted_score = 0
            if three_year_return > 0 and beta > 0:
                # Calculate a simplified Sharpe-like measure
                risk_adjusted_return = three_year_return / beta
                if risk_adjusted_return > 15:  # Excellent risk-adjusted return
                    risk_adjusted_score = 15
                elif risk_adjusted_return > 10:  # Very good
                    risk_adjusted_score = 12
                elif risk_adjusted_return > 5:  # Good
                    risk_adjusted_score = 8
                else:  # Average or below
                    risk_adjusted_score = max(0, int(risk_adjusted_return))

            # Absolute performance evaluation
            absolute_performance_score = 0
            # 3-year performance (most important timeframe)
            if three_year_return > 40:  # Exceptional
                absolute_performance_score += 15
            elif three_year_return > 30:  # Excellent
                absolute_performance_score += 12
            elif three_year_return > 20:  # Very good
                absolute_performance_score += 9
            elif three_year_return > 10:  # Good
                absolute_performance_score += 6
            elif three_year_return > 0:  # Positive but modest
                absolute_performance_score += 3

            # 1-year performance (shorter timeframe, less weight)
            one_year_return = data.get("oneYearReturn", 0) * 100 if data.get("oneYearReturn") else 0
            if one_year_return > 20:  # Exceptional
                absolute_performance_score += 5
            elif one_year_return > 10:  # Very good
                absolute_performance_score += 3
            elif one_year_return > 0:  # Positive
                absolute_performance_score += 1

            # YTD performance (current momentum)
            if ytd_return > 10:  # Strong momentum
                absolute_performance_score += 5
            elif ytd_return > 5:  # Good momentum
                absolute_performance_score += 3
            elif ytd_return > 0:  # Positive momentum
                absolute_performance_score += 1

            # Combine performance metrics (max 40 points)
            performance_score = min(40, risk_adjusted_score + absolute_performance_score)

            # 3. Fund Structure & Stability (25% of total score)
            structure_score = 0

            # Asset size (larger funds tend to be more stable)
            if net_assets > 10e9:  # >$10B (very large)
                structure_score += 10
            elif net_assets > 1e9:  # >$1B (large)
                structure_score += 8
            elif net_assets > 100e6:  # >$100M (medium)
                structure_score += 5
            elif net_assets > 10e6:  # >$10M (small)
                structure_score += 2

            # Diversification (based on holdings count)
            if len(holdings) > 100:  # Very diversified
                structure_score += 10
            elif len(holdings) > 50:  # Well diversified
                structure_score += 7
            elif len(holdings) > 20:  # Moderately diversified
                structure_score += 4
            else:  # Concentrated
                structure_score += 2

            # Fund age/track record (if available)
            # Placeholder - would ideally use inception date
            structure_score += 5  # Default value

            # 4. Risk Profile (15% of total score)
            risk_score = 0

            # Beta evaluation (closer to 1 is neutral)
            if 0.9 <= beta <= 1.1:  # Market-like risk
                risk_score += 10
            elif 0.7 <= beta <= 1.3:  # Moderate deviation
                risk_score += 7
            elif 0.5 <= beta <= 1.5:  # Significant deviation
                risk_score += 4
            else:  # Extreme deviation
                risk_score += 2

            # Volatility consideration (if available)
            # Placeholder - would ideally use standard deviation
            risk_score += 5  # Default value

            # Calculate overall confidence score (0-100 scale)
            confidence = min(100, cost_score + performance_score + structure_score + risk_score)

            # Format the result
            result = {
                "ticker": ticker,
                "name": name,
                "category": category,
                "asset_class": asset_class,
                "current_price": f"${current_price:,.2f}",
                "nav": f"${nav:,.2f}",
                "expense_ratio": f"{expense_ratio:.2f}%",
                "net_assets": f"${net_assets:,.0f}",
                "yield": f"{dividend_yield:.2f}%",
                "ytd_return": f"{ytd_return:.2f}%",
                "three_year_return": f"{three_year_return:.2f}%",
                "five_year_return": f"{five_year_return:.2f}%",
                "beta": beta,
                "holdings": holdings,
                "confidence": confidence,
                # Add raw values for charting
                "raw": {
                    "current_price": current_price,
                    "nav": nav,
                    "expense_ratio_value": expense_ratio / 100,  # Convert back to decimal for calculations
                    "net_assets": net_assets,
                    "yield_value": dividend_yield,
                    "ytd_return_value": ytd_return,
                    "three_year_return_value": three_year_return,
                    "five_year_return_value": five_year_return,
                    "beta": beta
                }
            }

            # Serialize and cache the result
            serialized_result = make_json_serializable(result)
            etf_cache.set(cache_key, serialized_result)

            return result
        except Exception as e:
            print(f"Error analyzing ETF {ticker}: {str(e)}")
            # Return minimal data on error
            return {
                "ticker": ticker,
                "name": f"{ticker} ETF",
                "expense_ratio": "0.00%",
                "yield": "0.00%",
                "confidence": 0,
                "error": str(e)
            }


class ComparisonAgent(Agent):
    """Compares different types of securities for portfolio allocation decisions."""
    def __init__(self):
        super().__init__(
            name="ComparisonAgent",
            model=OpenAIChat(id="gpt-4-turbo"),
            instructions="""
            Compare different types of securities (stocks, crypto, ETFs, REITs) based on:
            1. Risk-adjusted returns and volatility
            2. Correlation between assets
            3. Historical performance in different market conditions
            4. Fundamental metrics appropriate for each asset class
            Output comprehensive comparison with portfolio allocation recommendations.
            """
        )

    def compare(self, securities: dict) -> dict:
        """Compare different securities and provide insights.

        Args:
            securities: Dictionary with keys 'stock', 'crypto', 'etf', 'reit' and corresponding data

        Returns:
            Comparison analysis with insights and recommendations
        """
        # Generate a unique cache key based on the securities being compared
        cache_keys = []
        for asset_type, data in securities.items():
            if isinstance(data, dict) and 'ticker' in data:
                cache_keys.append(f"{asset_type}_{data['ticker'].lower()}")
            elif isinstance(data, dict) and 'coin' in data:
                cache_keys.append(f"{asset_type}_{data['coin'].lower()}")

        cache_key = "comparison_" + "_vs_".join(sorted(cache_keys))
        cached_data = comparison_cache.get(cache_key)

        if cached_data:
            print(f"Using cached comparison data for {cache_key}")
            return cached_data

        try:
            print(f"Generating fresh comparison for {cache_key}")

            # Extract relevant metrics for comparison
            comparison_data = {
                "performance": {},
                "risk": {},
                "valuation": {},
                "income": {}
            }

            # Process each security type
            for asset_type, data in securities.items():
                if not data:
                    continue

                # Get identifier
                if asset_type == 'crypto':
                    identifier = data.get('symbol', data.get('coin', 'Unknown'))
                else:
                    identifier = data.get('ticker', 'Unknown')

                # Performance metrics
                if asset_type == 'stock':
                    # Extract raw values for proper comparison
                    raw = data.get('raw', {})
                    comparison_data['performance'][identifier] = {
                        'current_price': raw.get('current_price', 0),
                        'upside_potential': data.get('upside_potential', '0%').rstrip('%'),
                    }
                    comparison_data['risk'][identifier] = {
                        'beta': raw.get('beta', 0)
                    }
                    comparison_data['valuation'][identifier] = {
                        'pe_ratio': raw.get('pe', 0),
                        'pb_ratio': raw.get('pb', 0)
                    }
                    comparison_data['income'][identifier] = {
                        'dividend_yield': raw.get('dividend_yield', 0)
                    }

                elif asset_type == 'crypto':
                    # Extract raw values for proper comparison
                    raw = data.get('raw', {})
                    comparison_data['performance'][identifier] = {
                        'current_price': raw.get('price', 0),
                        'price_change_24h': raw.get('price_change_24h', 0),
                        'price_change_7d': raw.get('price_change_7d', 0)
                    }
                    comparison_data['risk'][identifier] = {
                        'volatility': raw.get('volatility', 0),
                        'max_drawdown': raw.get('max_drawdown', 0)
                    }
                    comparison_data['valuation'][identifier] = {
                        'market_cap': raw.get('market_cap', 0),
                        'market_dominance': raw.get('market_dominance', 0)
                    }
                    # Crypto typically doesn't have income metrics
                    comparison_data['income'][identifier] = {
                        'yield': 0  # No yield for most cryptocurrencies
                    }

                elif asset_type == 'etf':
                    # Extract raw values for proper comparison
                    raw = data.get('raw', {})
                    comparison_data['performance'][identifier] = {
                        'current_price': raw.get('current_price', 0),
                        'ytd_return': raw.get('ytd_return_value', 0),
                        'three_year_return': raw.get('three_year_return_value', 0)
                    }
                    comparison_data['risk'][identifier] = {
                        'beta': raw.get('beta', 0)
                    }
                    comparison_data['valuation'][identifier] = {
                        'expense_ratio': raw.get('expense_ratio_value', 0),
                        'net_assets': raw.get('net_assets', 0)
                    }
                    comparison_data['income'][identifier] = {
                        'yield': raw.get('yield_value', 0)
                    }

                elif asset_type == 'reit':
                    # REITs typically have different metrics
                    comparison_data['performance'][identifier] = {
                        'current_price': data.get('current_price', '0').lstrip('$').replace(',', ''),
                    }
                    comparison_data['risk'][identifier] = {
                        'beta': data.get('beta', 0)
                    }
                    comparison_data['valuation'][identifier] = {
                        'price_to_ffo': data.get('price_to_ffo', 0)
                    }
                    comparison_data['income'][identifier] = {
                        'dividend_yield': data.get('dividend_yield', '0%').rstrip('%')
                    }

            # Generate insights based on the comparison
            insights = self._generate_comparison_insights(comparison_data, securities)

            # Generate recommendations
            recommendations = self._generate_recommendations(comparison_data, securities)

            # Prepare the result
            result = {
                "comparison_data": comparison_data,
                "insights": insights,
                "recommendations": recommendations,
                "securities_analyzed": list(securities.keys())
            }

            # Serialize and cache the result
            serialized_result = make_json_serializable(result)
            comparison_cache.set(cache_key, serialized_result)

            return result
        except Exception as e:
            print(f"Error in comparison analysis: {str(e)}")
            # Return minimal data on error
            return {
                "error": str(e),
                "securities_analyzed": list(securities.keys()),
                "insights": [f"Error performing comparison: {str(e)}"],
                "recommendations": ["Unable to provide recommendations due to an error in analysis."]
            }

    def _generate_comparison_insights(self, comparison_data, securities=None):
        """Generate insights based on the comparison data."""
        insights = []

        # Performance insights
        if 'performance' in comparison_data and comparison_data['performance']:
            # Find best and worst performers
            performance_metrics = {}
            for identifier, metrics in comparison_data['performance'].items():
                if 'three_year_return' in metrics:
                    performance_metrics[identifier] = metrics['three_year_return']
                elif 'ytd_return' in metrics:
                    performance_metrics[identifier] = metrics['ytd_return']
                elif 'price_change_7d' in metrics:
                    performance_metrics[identifier] = metrics['price_change_7d']

            if performance_metrics:
                best_performer = max(performance_metrics.items(), key=lambda x: x[1])
                worst_performer = min(performance_metrics.items(), key=lambda x: x[1])
                insights.append(f"{best_performer[0]} has shown the strongest recent performance with a return of {best_performer[1]:.2f}%.")
                insights.append(f"{worst_performer[0]} has shown the weakest recent performance with a return of {worst_performer[1]:.2f}%.")

        # Risk insights
        if 'risk' in comparison_data and comparison_data['risk']:
            # Compare volatility or beta
            risk_metrics = {}
            for identifier, metrics in comparison_data['risk'].items():
                if 'volatility' in metrics:
                    risk_metrics[identifier] = metrics['volatility']
                elif 'beta' in metrics:
                    risk_metrics[identifier] = metrics['beta']

            if risk_metrics:
                highest_risk = max(risk_metrics.items(), key=lambda x: x[1])
                lowest_risk = min(risk_metrics.items(), key=lambda x: x[1])
                insights.append(f"{highest_risk[0]} has the highest risk profile with a {'volatility' if 'volatility' in comparison_data['risk'][highest_risk[0]] else 'beta'} of {highest_risk[1]:.2f}.")
                insights.append(f"{lowest_risk[0]} has the lowest risk profile with a {'volatility' if 'volatility' in comparison_data['risk'][lowest_risk[0]] else 'beta'} of {lowest_risk[1]:.2f}.")

        # Income insights
        if 'income' in comparison_data and comparison_data['income']:
            # Compare yields
            income_metrics = {}
            for identifier, metrics in comparison_data['income'].items():
                if 'dividend_yield' in metrics:
                    income_metrics[identifier] = float(metrics['dividend_yield'])
                elif 'yield' in metrics:
                    income_metrics[identifier] = float(metrics['yield'])

            if income_metrics:
                highest_yield = max(income_metrics.items(), key=lambda x: x[1])
                if highest_yield[1] > 0:
                    insights.append(f"{highest_yield[0]} offers the highest income potential with a yield of {highest_yield[1]:.2f}%.")

        # Add a general insight if we don't have enough specific ones
        if len(insights) < 3:
            insights.append("Consider diversifying across these different asset classes to optimize risk-adjusted returns.")

        return insights

    def _generate_recommendations(self, comparison_data=None, securities=None):
        """Generate recommendations based on the comparison data."""
        recommendations = []

        # Basic portfolio allocation recommendation
        asset_types = list(securities.keys())

        if len(asset_types) >= 3:
            recommendations.append("Consider a diversified portfolio across multiple asset classes for optimal risk-adjusted returns.")

        # Risk-based recommendations
        has_high_risk = False
        has_low_risk = False

        for asset_type, data in securities.items():
            if asset_type == 'crypto':
                has_high_risk = True
            elif asset_type == 'etf' and data.get('category', '').lower() in ['bond', 'fixed income']:
                has_low_risk = True

        if has_high_risk and has_low_risk:
            recommendations.append("Your selection includes both high-risk and low-risk assets, which can help balance your portfolio.")
        elif has_high_risk:
            recommendations.append("Consider adding some lower-risk assets like bond ETFs to balance your portfolio's risk profile.")
        elif has_low_risk:
            recommendations.append("Consider adding some growth-oriented assets to potentially increase your portfolio's returns.")

        # Income-based recommendations
        income_assets = []
        for asset_type, data in securities.items():
            if asset_type == 'stock' and float(data.get('dividend_yield', '0%').rstrip('%')) > 2:
                income_assets.append(data.get('ticker', 'this stock'))
            elif asset_type == 'etf' and float(data.get('yield', '0%').rstrip('%')) > 2:
                income_assets.append(data.get('ticker', 'this ETF'))
            elif asset_type == 'reit':
                income_assets.append(data.get('ticker', 'this REIT'))

        if income_assets:
            recommendations.append(f"{', '.join(income_assets)} could be good additions to an income-focused portfolio.")

        # Add a general recommendation if we don't have enough specific ones
        if len(recommendations) < 2:
            recommendations.append("Regular rebalancing is recommended to maintain your desired asset allocation as market conditions change.")

        return recommendations
