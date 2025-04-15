"""
Comparison team for comparing different assets.
This module provides a standardized way to compare assets of the same type.
"""
from typing import Dict, Any, List, Optional, Tuple
from .base_team import BaseAnalysisTeam
import numpy as np
from datetime import datetime

class ComparisonTeam(BaseAnalysisTeam):
    """Comparison team for different assets."""

    def __init__(self):
        """Initialize the comparison team."""
        super().__init__(asset_type="comparison")

    def _create_knowledge_base(self):
        """
        Create a knowledge base for the agent.

        Returns:
            Knowledge base
        """
        # Create a simple knowledge base
        kb = super()._create_knowledge_base()

        # Add comparison-specific knowledge as a source
        kb.add_source({
            "type": "text",
            "content": """
        # Asset Comparison Guide

        ## Comparison Frameworks

        ### 1. Performance Comparison
        - Historical returns (1-month, 3-month, 6-month, 1-year, 3-year, 5-year)
        - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
        - Volatility comparison
        - Drawdown analysis
        - Performance in different market conditions

        ### 2. Valuation Comparison
        - Price multiples (P/E, P/B, P/S, etc.)
        - Dividend yield
        - Growth metrics
        - Intrinsic value estimates
        - Relative valuation

        ### 3. Financial Health Comparison
        - Balance sheet strength
        - Cash flow generation
        - Debt levels
        - Profitability metrics
        - Efficiency ratios

        ### 4. Technical Comparison
        - Price trends
        - Moving averages
        - Momentum indicators
        - Volume analysis
        - Support and resistance levels

        ### 5. Correlation Analysis
        - Correlation between assets
        - Diversification benefits
        - Portfolio impact

        ## Asset-Specific Comparison Metrics

        ### Stocks
        - P/E ratio
        - P/B ratio
        - Dividend yield
        - EPS growth
        - Revenue growth
        - Profit margins
        - Return on equity
        - Debt to equity
        - Beta
        - Market cap

        ### Cryptocurrencies
        - Market cap
        - Trading volume
        - Developer activity
        - Network metrics
        - Adoption metrics
        - Supply dynamics
        - Staking yield
        - Protocol revenue
        - Total value locked (TVL)
        - Regulatory status

        ### REITs
        - Dividend yield
        - Funds from operations (FFO)
        - Price to FFO
        - Occupancy rates
        - Property type
        - Geographic diversification
        - Lease terms
        - Development pipeline
        - Debt to equity
        - Interest coverage ratio

        ### ETFs
        - Expense ratio
        - Assets under management
        - Tracking error
        - Sector allocation
        - Geographic allocation
        - Yield
        - Liquidity
        - Fund age
        - Tax efficiency
        - Management style

        ## Comparison Visualization Methods

        ### 1. Side-by-Side Tables
        - Key metrics in tabular format
        - Color coding for better/worse metrics
        - Percentage differences

        ### 2. Radar/Spider Charts
        - Multiple metrics on a single chart
        - Normalized values for fair comparison
        - Visual representation of strengths and weaknesses

        ### 3. Bar Charts
        - Direct comparison of individual metrics
        - Grouped or stacked for multiple assets
        - Sorted by performance

        ### 4. Line Charts
        - Historical performance over time
        - Price trends
        - Growth metrics

        ### 5. Scatter Plots
        - Risk vs. return
        - Correlation analysis
        - Bubble charts for multi-dimensional comparison

        ## Comparison Analysis Framework

        ### 1. Identify Comparison Criteria
        - Determine relevant metrics for the asset class
        - Establish benchmark or reference points
        - Define time periods for analysis

        ### 2. Gather and Normalize Data
        - Collect data for all assets
        - Normalize data for fair comparison
        - Adjust for outliers

        ### 3. Perform Quantitative Analysis
        - Calculate performance metrics
        - Analyze risk measures
        - Evaluate valuation metrics

        ### 4. Conduct Qualitative Assessment
        - Management quality
        - Competitive positioning
        - Growth prospects
        - Regulatory environment

        ### 5. Synthesize Findings
        - Identify relative strengths and weaknesses
        - Determine overall ranking
        - Provide context for differences

        ### 6. Formulate Recommendations
        - Investment recommendations
        - Portfolio allocation suggestions
        - Risk management considerations
        """
        })

        return kb

    def compare_assets(
        self,
        tickers: List[str],
        asset_type: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple assets of the same type.

        Args:
            tickers: List of ticker symbols
            asset_type: Type of assets to compare (stock, crypto, reit, etf)
            metrics: Optional list of metrics to compare

        Returns:
            Comparison results
        """
        # Validate inputs
        if not tickers:
            return {"error": "No tickers provided"}

        if asset_type.lower() not in ["stock", "crypto", "reit", "etf"]:
            return {"error": f"Unsupported asset type: {asset_type}"}

        # Get data for each asset
        assets_data = {}
        price_histories = {}

        for ticker in tickers:
            # Get asset data based on type
            if asset_type.lower() == "stock":
                asset_data = self.data_service.get_stock_data(ticker)
            elif asset_type.lower() == "crypto":
                asset_data = self.data_service.get_crypto_data(ticker)
            elif asset_type.lower() == "reit":
                asset_data = self.data_service.get_reit_data(ticker)
            elif asset_type.lower() == "etf":
                asset_data = self.data_service.get_etf_data(ticker)

            # Check for errors
            if "error" in asset_data:
                return {"error": f"Error fetching data for {ticker}: {asset_data['error']}"}

            # Get price history
            price_history = self.data_service.get_price_history(ticker, asset_type, "1y")

            # Store data
            assets_data[ticker] = asset_data
            price_histories[ticker] = price_history

        # Prepare metrics for comparison
        if metrics is None:
            # Use default metrics based on asset type
            if asset_type.lower() == "stock":
                metrics = ["current_price", "market_cap", "pe", "pb", "dividend_yield", "beta"]
            elif asset_type.lower() == "crypto":
                metrics = ["current_price", "market_cap", "total_volume", "price_change_24h", "price_change_7d"]
            elif asset_type.lower() == "reit":
                metrics = ["current_price", "market_cap", "dividend_yield", "price_to_ffo", "debt_to_equity"]
            elif asset_type.lower() == "etf":
                metrics = ["current_price", "aum", "expense_ratio", "ytd_return", "one_year_return"]

        # Extract metrics for each asset
        comparison_data = {}
        for ticker, asset_data in assets_data.items():
            comparison_data[ticker] = {
                "name": asset_data.get("name", ticker),
                "metrics": {}
            }

            # Extract each metric
            for metric in metrics:
                if metric in asset_data:
                    comparison_data[ticker]["metrics"][metric] = asset_data[metric]
                elif asset_type.lower() in asset_data and metric in asset_data[asset_type.lower()]:
                    comparison_data[ticker]["metrics"][metric] = asset_data[asset_type.lower()][metric]
                else:
                    comparison_data[ticker]["metrics"][metric] = "N/A"

        # Calculate performance metrics
        performance_data = {}
        for ticker, price_history in price_histories.items():
            if "timestamps" in price_history and "prices" in price_history:
                timestamps = price_history["timestamps"]
                prices = price_history["prices"]

                if timestamps and prices:
                    # Calculate returns
                    current_price = prices[-1]
                    start_price = prices[0]
                    total_return = (current_price - start_price) / start_price * 100

                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = []
                    for i in range(1, len(prices)):
                        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                        daily_returns.append(daily_return)

                    import numpy as np
                    volatility = np.std(daily_returns) * 100 if daily_returns else 0

                    # Store performance data
                    performance_data[ticker] = {
                        "total_return": f"{total_return:.2f}%",
                        "volatility": f"{volatility:.2f}%",
                        "current_price": f"${current_price:.2f}",
                        "start_price": f"${start_price:.2f}"
                    }
                else:
                    performance_data[ticker] = {
                        "total_return": "N/A",
                        "volatility": "N/A",
                        "current_price": "N/A",
                        "start_price": "N/A"
                    }
            else:
                performance_data[ticker] = {
                    "total_return": "N/A",
                    "volatility": "N/A",
                    "current_price": "N/A",
                    "start_price": "N/A"
                }

        # Prepare context for the agent
        context = {
            "tickers": tickers,
            "asset_type": asset_type,
            "comparison_data": comparison_data,
            "performance_data": performance_data,
            "price_histories": price_histories
        }

        # Generate comparison prompt
        prompt = f"""
        You are a financial analyst tasked with comparing the following {asset_type}s: {', '.join(tickers)}.

        Here is the comparison data:

        {chr(10).join([f"### {ticker} ({comparison_data[ticker]['name']})" + chr(10) +
                      chr(10).join([f"- {metric}: {value}" for metric, value in comparison_data[ticker]['metrics'].items()]) +
                      chr(10) + f"- Total Return (1Y): {performance_data.get(ticker, {}).get('total_return', 'N/A')}" +
                      chr(10) + f"- Volatility: {performance_data.get(ticker, {}).get('volatility', 'N/A')}"
                      for ticker in tickers])}

        Based on this data and your knowledge of {asset_type}s, please provide:

        1. A comprehensive comparison of these assets, including:
           - Relative valuation
           - Performance comparison
           - Risk assessment
           - Investment outlook

        2. A ranking of the assets from most to least attractive investment

        3. Key strengths and weaknesses of each asset

        Format your response as a JSON object with the following structure:
        ```json
        {{
            "comparison": {{
                "valuation": "Your valuation comparison here",
                "performance": "Your performance comparison here",
                "risk": "Your risk assessment here",
                "outlook": "Your investment outlook here"
            }},
            "ranking": [
                {{
                    "ticker": "TICKER1",
                    "rank": 1,
                    "rationale": "Reason for this ranking"
                }},
                {{
                    "ticker": "TICKER2",
                    "rank": 2,
                    "rationale": "Reason for this ranking"
                }}
            ],
            "asset_analysis": {{
                "TICKER1": {{
                    "strengths": ["Strength 1", "Strength 2"],
                    "weaknesses": ["Weakness 1", "Weakness 2"]
                }},
                "TICKER2": {{
                    "strengths": ["Strength 1", "Strength 2"],
                    "weaknesses": ["Weakness 1", "Weakness 2"]
                }}
            }}
        }}
        ```

        Ensure your analysis is data-driven, balanced, and considers both the positives and negatives of each asset.
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
                        "comparison": {
                            "valuation": "Analysis not available in structured format.",
                            "performance": "Analysis not available in structured format.",
                            "risk": "Analysis not available in structured format.",
                            "outlook": "Analysis not available in structured format."
                        },
                        "ranking": [{"ticker": ticker, "rank": i+1, "rationale": "N/A"} for i, ticker in enumerate(tickers)],
                        "asset_analysis": {ticker: {"strengths": [], "weaknesses": []} for ticker in tickers}
                    }

            # Combine all data
            result = {
                "tickers": tickers,
                "asset_type": asset_type,
                "assets_data": assets_data,
                "price_histories": price_histories,
                "performance_data": performance_data,
                "comparison": analysis_result.get("comparison", {}),
                "ranking": analysis_result.get("ranking", []),
                "asset_analysis": analysis_result.get("asset_analysis", {})
            }

            return result
        except Exception as e:
            # Return error with raw response
            return {
                "tickers": tickers,
                "asset_type": asset_type,
                "error": f"Error parsing analysis: {str(e)}",
                "raw_response": response,
                "assets_data": assets_data,
                "price_histories": price_histories,
                "performance_data": performance_data
            }

    def compare_cross_asset(
        self,
        assets: Dict[str, Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Compare assets of different types.

        Args:
            assets: Dictionary mapping asset_id to (ticker, asset_type) tuples
                Example: {"asset1": ("AAPL", "stock"), "asset2": ("BTC", "crypto")}

        Returns:
            Cross-asset comparison results
        """
        # Validate inputs
        if not assets:
            return {"error": "No assets provided"}

        # Get data for each asset
        assets_data = {}
        price_histories = {}
        asset_info = {}

        for asset_id, (ticker, asset_type) in assets.items():
            # Validate asset type
            if asset_type.lower() not in ["stock", "crypto", "reit", "etf"]:
                return {"error": f"Unsupported asset type: {asset_type}"}

            # Get asset data based on type
            if asset_type.lower() == "stock":
                asset_data = self.data_service.get_stock_data(ticker)
            elif asset_type.lower() == "crypto":
                asset_data = self.data_service.get_crypto_data(ticker)
            elif asset_type.lower() == "reit":
                asset_data = self.data_service.get_reit_data(ticker)
            elif asset_type.lower() == "etf":
                asset_data = self.data_service.get_etf_data(ticker)

            # Check for errors
            if "error" in asset_data:
                return {"error": f"Error fetching data for {ticker} ({asset_type}): {asset_data['error']}"}

            # Get price history
            price_history = self.data_service.get_price_history(ticker, asset_type, "1y")

            # Store data
            assets_data[asset_id] = asset_data
            price_histories[asset_id] = price_history
            asset_info[asset_id] = {
                "ticker": ticker,
                "asset_type": asset_type,
                "name": asset_data.get("name", ticker)
            }

        # Extract common metrics for each asset
        comparison_data = {}
        for asset_id, asset_data in assets_data.items():
            ticker = asset_info[asset_id]["ticker"]
            asset_type = asset_info[asset_id]["asset_type"]

            # Extract basic metrics based on asset type
            metrics = {}

            # Common metrics across all asset types
            metrics["current_price"] = self._extract_metric(asset_data, "current_price", asset_type)
            metrics["market_cap"] = self._extract_metric(asset_data, "market_cap", asset_type)

            # Asset-specific metrics
            if asset_type.lower() == "stock":
                metrics["pe_ratio"] = self._extract_metric(asset_data, "pe", asset_type)
                metrics["pb_ratio"] = self._extract_metric(asset_data, "pb", asset_type)
                metrics["dividend_yield"] = self._extract_metric(asset_data, "dividend_yield", asset_type)
                metrics["beta"] = self._extract_metric(asset_data, "beta", asset_type)
            elif asset_type.lower() == "crypto":
                metrics["24h_volume"] = self._extract_metric(asset_data, "total_volume", asset_type)
                metrics["price_change_24h"] = self._extract_metric(asset_data, "price_change_24h", asset_type)
                metrics["price_change_7d"] = self._extract_metric(asset_data, "price_change_7d", asset_type)
            elif asset_type.lower() == "reit":
                metrics["dividend_yield"] = self._extract_metric(asset_data, "dividend_yield", asset_type)
                metrics["price_to_ffo"] = self._extract_metric(asset_data, "price_to_ffo", asset_type)
                metrics["debt_to_equity"] = self._extract_metric(asset_data, "debt_to_equity", asset_type)
            elif asset_type.lower() == "etf":
                metrics["expense_ratio"] = self._extract_metric(asset_data, "expense_ratio", asset_type)
                metrics["ytd_return"] = self._extract_metric(asset_data, "ytd_return", asset_type)
                metrics["one_year_return"] = self._extract_metric(asset_data, "one_year_return", asset_type)

            comparison_data[asset_id] = {
                "ticker": ticker,
                "name": asset_info[asset_id]["name"],
                "asset_type": asset_type,
                "metrics": metrics
            }

        # Calculate performance metrics
        performance_data = {}
        for asset_id, price_history in price_histories.items():
            ticker = asset_info[asset_id]["ticker"]

            if "timestamps" in price_history and "prices" in price_history:
                timestamps = price_history["timestamps"]
                prices = price_history["prices"]

                if timestamps and prices:
                    # Calculate returns
                    current_price = prices[-1]
                    start_price = prices[0]
                    total_return = (current_price - start_price) / start_price * 100

                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = []
                    for i in range(1, len(prices)):
                        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                        daily_returns.append(daily_return)

                    volatility = np.std(daily_returns) * 100 if daily_returns else 0

                    # Store performance data
                    performance_data[asset_id] = {
                        "total_return": f"{total_return:.2f}%",
                        "volatility": f"{volatility:.2f}%",
                        "current_price": f"${current_price:.2f}",
                        "start_price": f"${start_price:.2f}"
                    }
                else:
                    performance_data[asset_id] = {
                        "total_return": "N/A",
                        "volatility": "N/A",
                        "current_price": "N/A",
                        "start_price": "N/A"
                    }
            else:
                performance_data[asset_id] = {
                    "total_return": "N/A",
                    "volatility": "N/A",
                    "current_price": "N/A",
                    "start_price": "N/A"
                }

        # Calculate correlation matrix if there are at least 2 assets
        correlation_matrix = None
        if len(assets) >= 2:
            correlation_matrix = self._calculate_correlation_matrix(price_histories)

        # Prepare context for the agent
        context = {
            "assets": assets,
            "asset_info": asset_info,
            "comparison_data": comparison_data,
            "performance_data": performance_data,
            "price_histories": price_histories,
            "correlation_matrix": correlation_matrix
        }

        # Generate comparison prompt
        prompt = f"""
        You are a financial analyst tasked with comparing the following assets of different types:

        {chr(10).join([f"### {asset_id}: {comparison_data[asset_id]['ticker']} ({comparison_data[asset_id]['name']}) - {comparison_data[asset_id]['asset_type'].upper()}" + chr(10) +
                      chr(10).join([f"- {metric}: {value}" for metric, value in comparison_data[asset_id]['metrics'].items()]) +
                      chr(10) + f"- Total Return (1Y): {performance_data.get(asset_id, {}).get('total_return', 'N/A')}" +
                      chr(10) + f"- Volatility: {performance_data.get(asset_id, {}).get('volatility', 'N/A')}"
                      for asset_id in assets.keys()])}

        Based on this data, please provide:

        1. A comprehensive cross-asset comparison, including:
           - Relative performance
           - Risk assessment
           - Diversification benefits
           - Investment outlook

        2. A ranking of the assets from most to least attractive investment

        3. Key strengths and weaknesses of each asset

        4. Portfolio allocation recommendations

        Format your response as a JSON object with the following structure:
        ```json
        {{
            "comparison": {{
                "performance": "Your performance comparison here",
                "risk": "Your risk assessment here",
                "diversification": "Your diversification analysis here",
                "outlook": "Your investment outlook here"
            }},
            "ranking": [
                {{
                    "asset_id": "asset1",
                    "rank": 1,
                    "rationale": "Reason for this ranking"
                }},
                {{
                    "asset_id": "asset2",
                    "rank": 2,
                    "rationale": "Reason for this ranking"
                }}
            ],
            "asset_analysis": {{
                "asset1": {{
                    "strengths": ["Strength 1", "Strength 2"],
                    "weaknesses": ["Weakness 1", "Weakness 2"]
                }},
                "asset2": {{
                    "strengths": ["Strength 1", "Strength 2"],
                    "weaknesses": ["Weakness 1", "Weakness 2"]
                }}
            }},
            "portfolio_allocation": {{
                "recommendation": "Your overall allocation recommendation",
                "allocations": [
                    {{
                        "asset_id": "asset1",
                        "percentage": 60,
                        "rationale": "Reason for this allocation"
                    }},
                    {{
                        "asset_id": "asset2",
                        "percentage": 40,
                        "rationale": "Reason for this allocation"
                    }}
                ]
            }}
        }}
        ```

        Ensure your analysis is data-driven, balanced, and considers both the positives and negatives of each asset.
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
                        "comparison": {
                            "performance": "Analysis not available in structured format.",
                            "risk": "Analysis not available in structured format.",
                            "diversification": "Analysis not available in structured format.",
                            "outlook": "Analysis not available in structured format."
                        },
                        "ranking": [{
                            "asset_id": asset_id,
                            "rank": i+1,
                            "rationale": "N/A"
                        } for i, asset_id in enumerate(assets.keys())],
                        "asset_analysis": {
                            asset_id: {"strengths": [], "weaknesses": []}
                            for asset_id in assets.keys()
                        },
                        "portfolio_allocation": {
                            "recommendation": "Analysis not available in structured format.",
                            "allocations": [{
                                "asset_id": asset_id,
                                "percentage": 100 // len(assets),
                                "rationale": "N/A"
                            } for asset_id in assets.keys()]
                        }
                    }

            # Combine all data
            result = {
                "assets": assets,
                "asset_info": asset_info,
                "assets_data": assets_data,
                "price_histories": price_histories,
                "performance_data": performance_data,
                "correlation_matrix": correlation_matrix,
                "comparison": analysis_result.get("comparison", {}),
                "ranking": analysis_result.get("ranking", []),
                "asset_analysis": analysis_result.get("asset_analysis", {}),
                "portfolio_allocation": analysis_result.get("portfolio_allocation", {})
            }

            return result
        except Exception as e:
            # Return error with raw response
            return {
                "assets": assets,
                "asset_info": asset_info,
                "error": f"Error parsing analysis: {str(e)}",
                "raw_response": response,
                "assets_data": assets_data,
                "price_histories": price_histories,
                "performance_data": performance_data,
                "correlation_matrix": correlation_matrix
            }

    def _extract_metric(self, asset_data: Dict[str, Any], metric_name: str, asset_type: str) -> Any:
        """
        Extract a metric from asset data, handling different data structures.

        Args:
            asset_data: Asset data dictionary
            metric_name: Name of the metric to extract
            asset_type: Type of asset (stock, crypto, reit, etf)

        Returns:
            Extracted metric value or "N/A" if not found
        """
        # Check if metric is directly in asset_data
        if metric_name in asset_data:
            return asset_data[metric_name]

        # Check if metric is in asset_type-specific section
        if asset_type.lower() in asset_data and metric_name in asset_data[asset_type.lower()]:
            return asset_data[asset_type.lower()][metric_name]

        # For crypto, check in market_data
        if asset_type.lower() == "crypto" and "market_data" in asset_data:
            market_data = asset_data["market_data"]
            if metric_name in market_data:
                return market_data[metric_name]
            elif metric_name == "current_price" and "current_price" in market_data:
                # Handle nested current_price structure in CoinGecko data
                current_price = market_data["current_price"]
                if isinstance(current_price, dict) and "usd" in current_price:
                    return current_price["usd"]

        # Return N/A if not found
        return "N/A"

    def _calculate_correlation_matrix(self, price_histories: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between asset price histories.

        Args:
            price_histories: Dictionary mapping asset_ids to price history dictionaries

        Returns:
            Correlation matrix as a nested dictionary
        """
        # Extract price data for each asset
        price_data = {}
        for asset_id, history in price_histories.items():
            if "prices" in history and history["prices"]:
                price_data[asset_id] = history["prices"]

        # If we have less than 2 assets with price data, return None
        if len(price_data) < 2:
            return None

        # Find the minimum length of price data
        min_length = min(len(prices) for prices in price_data.values())

        # Truncate all price data to the same length
        for asset_id in price_data:
            price_data[asset_id] = price_data[asset_id][-min_length:]

        # Calculate correlation matrix
        correlation_matrix = {}
        for asset_id1 in price_data:
            correlation_matrix[asset_id1] = {}
            for asset_id2 in price_data:
                if asset_id1 == asset_id2:
                    correlation_matrix[asset_id1][asset_id2] = 1.0
                else:
                    # Calculate correlation coefficient
                    try:
                        corr = np.corrcoef(price_data[asset_id1], price_data[asset_id2])[0, 1]
                        correlation_matrix[asset_id1][asset_id2] = round(corr, 2)
                    except Exception:
                        correlation_matrix[asset_id1][asset_id2] = 0.0

        return correlation_matrix
