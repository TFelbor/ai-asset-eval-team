import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import yfinance as yf
import datetime
from config import settings as config

class ChartService:
    @staticmethod
    def create_stock_price_chart(ticker, stock_data):
        """Create a simple price vs DCF chart"""
        current_price = stock_data.get("raw", {}).get("current_price", 0)
        dcf_value = float(stock_data.get("dcf", "$0").replace("$", "").replace(",", ""))

        fig = go.Figure()

        # Add current price bar
        fig.add_trace(go.Bar(
            x=["Current Price"],
            y=[current_price],
            name="Current Price",
            marker_color="#2196F3"
        ))

        # Add DCF value bar
        fig.add_trace(go.Bar(
            x=["DCF Value"],
            y=[dcf_value],
            name="DCF Value",
            marker_color="#4CAF50"
        ))

        # Update layout
        fig.update_layout(
            title=f"{ticker} - Current Price vs DCF Value",
            xaxis_title="Metric",
            yaxis_title="Value ($)",
            barmode="group",
            height=500,
            width=800
        )

        chart_description = "Comparison of current price and discounted cash flow (DCF) value"
        insights = [
            f"Current price: {stock_data.get('current_price', '$0')}",
            f"DCF value: {stock_data.get('dcf', '$0')}",
            f"Upside potential: {stock_data.get('upside_potential', '0%')}"
        ]

        return {
            "fig": fig,
            "description": chart_description,
            "insights": insights
        }
    
    @staticmethod
    def create_stock_metrics_chart(ticker, stock_data, raw_data):
        """Create a radar chart of key metrics"""
        pe = raw_data.get("pe", 0)
        pb = raw_data.get("pb", 0)
        dividend_yield = raw_data.get("dividend_yield", 0)
        beta = raw_data.get("beta", 0)

        # Normalize values for radar chart
        pe_norm = min(1, 15 / max(1, pe)) if pe > 0 else 0.5  # Lower P/E is better
        pb_norm = min(1, 2 / max(0.1, pb)) if pb > 0 else 0.5  # Lower P/B is better
        div_norm = min(1, dividend_yield / 5)  # Higher dividend is better (up to 5%)
        beta_norm = 1 - min(1, abs(beta - 1) / 1)  # Beta closer to 1 is better

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[pe_norm, pb_norm, div_norm, beta_norm, stock_data.get("confidence", 0) / 100],
            theta=["P/E Ratio", "P/B Ratio", "Dividend Yield", "Beta", "Confidence"],
            fill="toself",
            name=ticker,
            line=dict(color=config.CHART_COLORS["primary"])
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"{ticker} - Key Metrics",
            height=600,
            width=800,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff")
        )

        chart_description = "Radar chart of key financial metrics (normalized values)"
        insights = [
            f"P/E ratio: {pe:.2f}",
            f"P/B ratio: {pb:.2f}",
            f"Dividend yield: {stock_data.get('dividend_yield', '0%')}",
            f"Beta: {beta:.2f}",
            f"Confidence score: {stock_data.get('confidence', 0)}/100"
        ]

        return {
            "fig": fig,
            "description": chart_description,
            "insights": insights
        }
        
    @staticmethod
    def get_chart_json(fig):
        """Convert a Plotly figure to JSON"""
        return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
