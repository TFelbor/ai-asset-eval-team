"""
Cryptocurrency analysis team for analyzing cryptocurrencies.
This module provides a standardized way to analyze cryptocurrencies.
"""
from typing import Dict, Any
from .base_team import BaseAnalysisTeam

class CryptoAnalysisTeam(BaseAnalysisTeam):
    """Cryptocurrency analysis team."""

    def __init__(self):
        """Initialize the cryptocurrency analysis team."""
        super().__init__(asset_type="crypto")

    def _create_knowledge_base(self):
        """
        Create a knowledge base for the agent.

        Returns:
            Knowledge base
        """
        # Create a simple knowledge base
        kb = super()._create_knowledge_base()

        # Add crypto-specific knowledge as a source
        kb.add_source({
            "type": "text",
            "content": """
            Cryptocurrency Analysis Guidelines:

            1. Fundamental Analysis:
               - Evaluate the project's use case and problem it solves
               - Assess the technology and technical innovation
               - Review the team's experience and track record
               - Examine tokenomics (supply, distribution, inflation)
               - Analyze network activity and adoption metrics

            2. Technical Analysis:
               - Identify key support and resistance levels
               - Evaluate moving averages (50-day, 200-day) for trend direction
               - Assess relative strength index (RSI) for overbought/oversold conditions
               - Look for chart patterns (head and shoulders, double tops/bottoms)
               - Consider volume trends to confirm price movements

            3. Market Analysis:
               - Market capitalization and ranking
               - Trading volume and liquidity
               - Exchange listings and accessibility
               - Market sentiment and social metrics
               - Correlation with Bitcoin and other major cryptocurrencies

            4. Risk Assessment:
               - Regulatory risks
               - Security vulnerabilities
               - Competition from other projects
               - Centralization concerns
               - Volatility and market risks

            5. Investment Recommendations:
               - Strong Buy: High conviction with strong fundamentals and technical setup
               - Buy: Positive outlook with good risk/reward ratio
               - Hold: Neutral outlook or awaiting catalysts
               - Sell: Negative outlook or better opportunities elsewhere
               - Strong Sell: Serious concerns about fundamentals or technical breakdown
            """
        })

        return kb

    def _get_asset_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get data for the cryptocurrency.

        Args:
            ticker: Cryptocurrency ID or symbol

        Returns:
            Cryptocurrency data
        """
        return self.data_service.get_crypto_data(ticker)

    def _perform_analysis(
        self,
        ticker: str,
        asset_data: Dict[str, Any],
        macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform analysis on the cryptocurrency.

        Args:
            ticker: Cryptocurrency ID or symbol
            asset_data: Cryptocurrency data
            macro_data: Macroeconomic data

        Returns:
            Analysis results
        """
        try:
            # Debug info
            print(f"Analyzing crypto: {ticker}")
            print(f"Asset data keys: {asset_data.keys()}")

            # Extract key metrics
            market_data = asset_data.get("market_data", {})
            print(f"Market data keys: {market_data.keys() if isinstance(market_data, dict) else 'Not a dict'}")

            # Convert all values to appropriate types with better error handling
            def safe_float(value, default=0.0):
                try:
                    if isinstance(value, dict):
                        return default  # Return default if we got a dict instead of a value
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Get current price with better error handling
            current_price_data = market_data.get("current_price", {})
            if not isinstance(current_price_data, dict):
                current_price = safe_float(current_price_data)
            else:
                current_price = safe_float(current_price_data.get("usd", 0))

            # Get other metrics with better error handling
            market_cap_data = market_data.get("market_cap", {})
            market_cap = safe_float(market_cap_data.get("usd", 0) if isinstance(market_cap_data, dict) else market_cap_data)

            total_volume_data = market_data.get("total_volume", {})
            total_volume = safe_float(total_volume_data.get("usd", 0) if isinstance(total_volume_data, dict) else total_volume_data)

            price_change_24h = safe_float(market_data.get("price_change_percentage_24h", 0))
            price_change_7d = safe_float(market_data.get("price_change_percentage_7d", 0))
            price_change_30d = safe_float(market_data.get("price_change_percentage_30d", 0))

            ath_data = market_data.get("ath", {})
            ath = safe_float(ath_data.get("usd", 0) if isinstance(ath_data, dict) else ath_data)

            atl_data = market_data.get("atl", {})
            atl = safe_float(atl_data.get("usd", 0) if isinstance(atl_data, dict) else atl_data)

            print(f"Extracted values: price={current_price}, ath={ath}, atl={atl}")
        except Exception as e:
            print(f"Error extracting crypto data: {str(e)}")
            # Return empty data with error
            return {"error": f"Failed to extract crypto data: {str(e)}"}

        try:
            # Get price history with enhanced error handling
            try:
                # Get price history from data service
                price_history = self.data_service.get_price_history(ticker, "crypto", "1mo")
                print(f"Price history keys: {price_history.keys() if isinstance(price_history, dict) else 'Not a dict'}")

                # Check if there was an error getting price history
                if isinstance(price_history, dict) and "error" in price_history:
                    print(f"Error in price history: {price_history.get('error')}")
                    return {"error": f"Failed to get price history: {price_history.get('error')}"}

                # Process and validate price history data
                if isinstance(price_history, dict):
                    # Create new lists for processed data
                    processed_timestamps = []
                    processed_prices = []
                    processed_volumes = []

                    # Process timestamps and prices
                    raw_timestamps = price_history.get("timestamps", [])
                    raw_prices = price_history.get("prices", [])
                    raw_volumes = price_history.get("volumes", [])

                    # Ensure all values are properly converted to appropriate types
                    for i in range(len(raw_timestamps)):
                        try:
                            # Convert timestamp to float
                            timestamp = float(raw_timestamps[i])
                            processed_timestamps.append(timestamp)

                            # Convert price to float
                            if i < len(raw_prices):
                                price = float(raw_prices[i])
                                processed_prices.append(price)
                            else:
                                # If price is missing, use the last known price or 0
                                price = processed_prices[-1] if processed_prices else 0
                                processed_prices.append(price)

                            # Convert volume to float
                            if i < len(raw_volumes):
                                volume = float(raw_volumes[i])
                                processed_volumes.append(volume)
                            else:
                                # If volume is missing, use 0
                                processed_volumes.append(0)
                        except (ValueError, TypeError) as e:
                            print(f"Error processing data point {i}: {e}")
                            # Skip this data point
                            continue

                    # Create the processed price history dictionary
                    price_history = {
                        "timestamps": processed_timestamps,
                        "prices": processed_prices,
                        "volumes": processed_volumes
                    }

                    # Print sample data for debugging
                    if processed_timestamps and processed_prices:
                        print(f"Sample processed price history data:")
                        print(f"  First timestamp: {processed_timestamps[0]}")
                        print(f"  First price: {processed_prices[0]}")
                        print(f"  Data points: {len(processed_timestamps)}")
                else:
                    print("Warning: Price history is not a dictionary, creating empty structure")
                    price_history = {
                        "timestamps": [],
                        "prices": [],
                        "volumes": []
                    }
            except Exception as e:
                print(f"Error getting price history: {str(e)}")
                # Create an empty structure if there's an error
                price_history = {
                    "timestamps": [],
                    "prices": [],
                    "volumes": []
                }

            # Calculate volatility
            volatility = self._calculate_volatility(price_history)
            print(f"Calculated volatility: {volatility}")

            # Calculate potential
            potential = self._calculate_potential(current_price, ath)
            print(f"Calculated potential: {potential}")

            # Get news
            news = self.data_service.get_news("crypto", ticker, limit=5)

            # Perform AI analysis
            ai_analysis = self._perform_ai_analysis(ticker, asset_data, macro_data, volatility)

            # Ensure all values are the correct type before formatting
            def safe_format_price(value):
                try:
                    # Ensure value is a float
                    float_value = float(value) if value is not None else 0.0
                    return f"${float_value:,.2f}"
                except (ValueError, TypeError):
                    # If conversion fails, return a simple string
                    return f"${value}" if value is not None else "$0"

            def safe_format_percentage(value):
                try:
                    # Ensure value is a float
                    float_value = float(value) if value is not None else 0.0
                    return f"{float_value:.2f}%"
                except (ValueError, TypeError):
                    # If conversion fails, return a simple string
                    return f"{value}%" if value is not None else "0%"

            # Combine all analysis with safe string formatting
            try:
                analysis = {
                    "current_price": safe_format_price(current_price),
                    "market_cap": self._format_market_cap(market_cap),
                    "total_volume": self._format_market_cap(total_volume),
                    "price_change_24h": safe_format_percentage(price_change_24h),
                    "price_change_7d": safe_format_percentage(price_change_7d),
                    "price_change_30d": safe_format_percentage(price_change_30d),
                    "all_time_high": safe_format_price(ath),
                    "all_time_low": safe_format_price(atl),
                    "potential": potential,
                    "volatility": volatility,
                    "price_history": price_history,
                    "news": news,
                    "ai_analysis": ai_analysis
                }
            except Exception as e:
                print(f"Error formatting analysis values: {e}")
                # Fallback to very simple string formatting without any special formatting
                analysis = {
                    "current_price": f"${current_price}" if current_price is not None else "$0",
                    "market_cap": self._format_market_cap(market_cap),
                    "total_volume": self._format_market_cap(total_volume),
                    "price_change_24h": f"{price_change_24h}%" if price_change_24h is not None else "0%",
                    "price_change_7d": f"{price_change_7d}%" if price_change_7d is not None else "0%",
                    "price_change_30d": f"{price_change_30d}%" if price_change_30d is not None else "0%",
                    "all_time_high": f"${ath}" if ath is not None else "$0",
                    "all_time_low": f"${atl}" if atl is not None else "$0",
                    "potential": potential,
                    "volatility": volatility,
                    "price_history": price_history,
                    "news": news,
                    "ai_analysis": ai_analysis
                }

            return analysis
        except Exception as e:
            print(f"Error in crypto analysis: {str(e)}")
            return {"error": f"Failed to analyze crypto: {str(e)}"}

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
            ticker: Cryptocurrency ID or symbol
            asset_data: Cryptocurrency data
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
            "crypto": {
                "id": asset_data.get("id", ""),
                "symbol": asset_data.get("symbol", "").upper(),
                "name": asset_data.get("name", ""),
                "current_price": analysis["current_price"],
                "market_cap": analysis["market_cap"],
                "total_volume": analysis["total_volume"],
                "price_change_24h": analysis["price_change_24h"],
                "price_change_7d": analysis["price_change_7d"],
                "price_change_30d": analysis["price_change_30d"],
                "all_time_high": analysis["all_time_high"],
                "all_time_low": analysis["all_time_low"],
                "potential": analysis["potential"],
                "volatility": analysis["volatility"],
                "price_history": analysis["price_history"],
                "news": analysis["news"],
                "analysis": analysis["ai_analysis"],
                "confidence": confidence
            },
            # Also include under 'cryptocurrency' key for backward compatibility
            "cryptocurrency": {
                "id": asset_data.get("id", ""),
                "symbol": asset_data.get("symbol", "").upper(),
                "name": asset_data.get("name", ""),
                "current_price": analysis["current_price"],
                "market_cap": analysis["market_cap"],
                "total_volume": analysis["total_volume"],
                "price_change_24h": analysis["price_change_24h"],
                "price_change_7d": analysis["price_change_7d"],
                "price_change_30d": analysis["price_change_30d"],
                "all_time_high": analysis["all_time_high"],
                "all_time_low": analysis["all_time_low"],
                "potential": analysis["potential"],
                "volatility": analysis["volatility"],
                "price_history": analysis["price_history"],
                "news": analysis["news"],
                "analysis": analysis["ai_analysis"],
                "confidence": confidence
            },
            "recommendation": recommendation,
            "overall_score": confidence
        }

        return report

    def _calculate_volatility(self, price_history: Dict[str, Any]) -> str:
        """
        Calculate volatility based on price history.

        Args:
            price_history: Price history data

        Returns:
            Volatility category (Very Low, Low, Moderate, High, Very High)
        """
        prices = price_history.get("prices", [])

        if not prices or len(prices) < 2:
            return "Unknown"

        # Ensure all prices are floats
        try:
            prices = [float(price) for price in prices]
        except (ValueError, TypeError) as e:
            print(f"Error converting prices to float: {e}")
            print(f"Sample price values: {prices[:3] if len(prices) >= 3 else prices}")
            return "Unknown"

        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            try:
                daily_return = (prices[i] / prices[i-1]) - 1
                returns.append(daily_return)
            except (ZeroDivisionError, TypeError) as e:
                print(f"Error calculating return at index {i}: {e}")
                continue

        # Check if we have enough returns to calculate volatility
        if not returns:
            return "Unknown"

        # Calculate standard deviation of returns
        import numpy as np
        std_dev = np.std(returns) * 100  # Convert to percentage

        # Categorize volatility
        if std_dev < 1:
            return "Very Low"
        elif std_dev < 3:
            return "Low"
        elif std_dev < 5:
            return "Moderate"
        elif std_dev < 10:
            return "High"
        else:
            return "Very High"

    def _calculate_potential(self, current_price, ath) -> str:
        """
        Calculate potential based on all-time high.

        Args:
            current_price: Current price
            ath: All-time high price

        Returns:
            Potential as a percentage string
        """
        # Ensure values are floats with improved error handling
        try:
            # Handle None values
            if current_price is None or ath is None:
                return "0%"

            # Handle string values that might be formatted with currency symbols
            if isinstance(current_price, str):
                current_price = current_price.replace('$', '').replace(',', '')
            if isinstance(ath, str):
                ath = ath.replace('$', '').replace(',', '')

            # Convert to float
            current_price = float(current_price)
            ath = float(ath)

            # Validate values
            if current_price <= 0 or ath <= 0:
                return "0%"

            # Calculate potential
            potential = (ath / current_price - 1) * 100

            # Format as string with proper handling
            return f"{potential:.2f}%"
        except (ValueError, TypeError, ZeroDivisionError) as e:
            # Log the error for debugging
            print(f"Error calculating potential: {e}, current_price={current_price}, ath={ath}")
            # Fallback if any calculation fails
            return "0%"

    def _perform_ai_analysis(
        self,
        ticker: str,
        asset_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        volatility: str
    ) -> Dict[str, Any]:
        """
        Perform AI analysis on the cryptocurrency.

        Args:
            ticker: Cryptocurrency ID or symbol
            asset_data: Cryptocurrency data
            macro_data: Macroeconomic data
            volatility: Calculated volatility

        Returns:
            AI analysis results
        """
        # Prepare prompt for the agent
        market_data = asset_data.get("market_data", {})

        # Safely extract and convert numeric values
        def safe_get_float(data, key_path, default=0.0):
            try:
                value = data
                for key in key_path:
                    if isinstance(value, dict):
                        value = value.get(key, {})
                    else:
                        return default
                if value is None:
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default

        # Extract values with safe conversion
        current_price = safe_get_float(market_data, ["current_price", "usd"])
        market_cap = safe_get_float(market_data, ["market_cap", "usd"])
        total_volume = safe_get_float(market_data, ["total_volume", "usd"])
        price_change_24h = safe_get_float(market_data, ["price_change_percentage_24h"])
        price_change_7d = safe_get_float(market_data, ["price_change_percentage_7d"])
        price_change_30d = safe_get_float(market_data, ["price_change_percentage_30d"])
        ath = safe_get_float(market_data, ["ath", "usd"])

        # Create prompt with safe formatting
        prompt = f"""
        Analyze the cryptocurrency {asset_data.get('name', '')} ({asset_data.get('symbol', '').upper()}) based on the following data:

        Current Price: ${current_price:,.2f}
        Market Cap: ${market_cap:,.0f}
        24h Trading Volume: ${total_volume:,.0f}
        24h Price Change: {price_change_24h:.2f}%
        7d Price Change: {price_change_7d:.2f}%
        30d Price Change: {price_change_30d:.2f}%
        All-Time High: ${ath:,.2f}
        Volatility: {volatility}

        Macroeconomic Environment:
        GDP Outlook: {macro_data.get('gdp_outlook', 'Stable')}
        Inflation Risk: {macro_data.get('inflation_risk', 'Moderate')}

        Provide a comprehensive analysis including:
        1. Fundamental Analysis
        2. Technical Outlook
        3. Market Sentiment
        4. Risk Factors
        5. Investment Thesis

        Format your response as a JSON object with the following structure:
        {{"fundamental_analysis": "Your analysis here", "technical_outlook": "Your analysis here", "market_sentiment": "Your analysis here", "risk_factors": "Your analysis here", "investment_thesis": "Your analysis here", "strengths": ["Strength 1", "Strength 2"], "weaknesses": ["Weakness 1", "Weakness 2"], "opportunities": ["Opportunity 1", "Opportunity 2"], "threats": ["Threat 1", "Threat 2"]}}
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

        # Adjust based on price changes
        price_change_24h = float(analysis.get("price_change_24h", "0%").replace("%", ""))
        price_change_7d = float(analysis.get("price_change_7d", "0%").replace("%", ""))
        price_change_30d = float(analysis.get("price_change_30d", "0%").replace("%", ""))

        # Short-term momentum
        if price_change_24h > 5:
            confidence += 5
        elif price_change_24h < -5:
            confidence -= 5

        # Medium-term momentum
        if price_change_7d > 10:
            confidence += 10
        elif price_change_7d < -10:
            confidence -= 10

        # Long-term momentum
        if price_change_30d > 20:
            confidence += 15
        elif price_change_30d < -20:
            confidence -= 15

        # Adjust based on volatility
        volatility = analysis.get("volatility", "Moderate")
        if volatility == "Very High":
            confidence -= 10
        elif volatility == "High":
            confidence -= 5
        elif volatility == "Low":
            confidence += 5
        elif volatility == "Very Low":
            confidence += 10

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
        # Get price changes
        price_change_24h = float(analysis.get("price_change_24h", "0%").replace("%", ""))
        price_change_7d = float(analysis.get("price_change_7d", "0%").replace("%", ""))
        price_change_30d = float(analysis.get("price_change_30d", "0%").replace("%", ""))

        # Calculate weighted price change
        weighted_change = (
            0.2 * price_change_24h +
            0.3 * price_change_7d +
            0.5 * price_change_30d
        )

        # Determine recommendation based on weighted change and confidence
        if weighted_change > 30 and confidence >= 70:
            return "Strong Buy"
        elif weighted_change > 15 and confidence >= 60:
            return "Buy"
        elif weighted_change > -15 and weighted_change < 15:
            return "Hold"
        elif weighted_change < -30 and confidence <= 40:
            return "Strong Sell"
        elif weighted_change < -15 and confidence <= 50:
            return "Sell"
        else:
            return "Hold"

    def _format_market_cap(self, value: float) -> str:
        """
        Format market cap in a human-readable format.

        Args:
            value: Market cap value

        Returns:
            Formatted market cap string
        """
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"${value / 1e6:.2f}M"
        elif value > 0:
            return f"${value / 1e3:.2f}K"
        else:
            return "$0"
