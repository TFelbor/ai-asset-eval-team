from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import json
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from api_integrations.coingecko import CoinGeckoAPI
from api_integrations.news_api import NewsAPI
from analytics.advanced_metrics import AdvancedAnalytics
from analytics.backtesting import run_backtest
from config import settings as config
import yfinance as yf
import numpy as np
import datetime
from utils.logger import app_logger
from app.middleware.error_handler import ErrorHandler
from app.api import stock_routes, crypto_routes, reit_routes, etf_routes, news_routes, backtest_routes

# Import all analysis teams
from teams import (
    StockAnalysisTeam,
    CryptoAnalysisTeam,
    REITAnalysisTeam,
    ETFAnalysisTeam,
    ComparisonTeam
)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analysis Dashboard",
    description="AI-powered analysis of stocks, cryptocurrencies, REITs, and ETFs",
    version="1.0.0"
)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(ErrorHandler.handle_exceptions)

app.include_router(stock_routes.router, prefix="/analyze")
app.include_router(crypto_routes.router, prefix="/analyze")
app.include_router(reit_routes.router, prefix="/analyze")
app.include_router(etf_routes.router, prefix="/analyze")
app.include_router(news_routes.router, prefix="/news")
app.include_router(backtest_routes.router, prefix="/backtest")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

# Add middleware for Content Security Policy
@app.middleware("http")
async def add_csp_header(request, call_next):
    response = await call_next(request)
    # Add CSP header to allow Plotly to work properly - more permissive for development
    response.headers["Content-Security-Policy"] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; script-src * 'unsafe-inline' 'unsafe-eval'; connect-src * 'unsafe-inline'; img-src * data: blob: 'unsafe-inline'; frame-src *; style-src * 'unsafe-inline';"
    return response

# Mount static files directory for frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import API routes
from app.api import stock_routes, crypto_routes, reit_routes, etf_routes

# Include routers
app.include_router(stock_routes.router, prefix="/analyze")
app.include_router(crypto_routes.router, prefix="/analyze")
app.include_router(reit_routes.router, prefix="/analyze")
app.include_router(etf_routes.router, prefix="/analyze")

# No WebSocket connection manager needed - using simulated data instead


@app.get("/")
async def root():
    """Return the dashboard homepage."""
    return FileResponse("static/index.html")


# WebSocket test endpoint removed to prevent connection attempts


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    return FileResponse("static/favicon.ico")


# WebSocket endpoint removed - using simulated data instead


@app.get("/plotly-test")
async def plotly_test():
    """Test page for Plotly."""
    return FileResponse("static/plotly_test.html")


@app.get("/simple-plotly-test")
async def simple_plotly_test():
    """Simple test page for Plotly without CSP restrictions."""
    response = FileResponse("static/simple_plotly_test.html")
    # Disable CSP for this page
    response.headers["Content-Security-Policy"] = ""
    return response


@app.get("/no-csp")
async def no_csp_dashboard():
    """Dashboard without CSP restrictions."""
    response = FileResponse("static/no_csp_index.html")
    # Disable CSP for this page
    response.headers["Content-Security-Policy"] = ""
    return response


@app.get("/analyze/stock/{ticker}")
async def analyze_stock(ticker: str):
    """Analyze a stock by ticker symbol."""
    try:
        # Validate ticker
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol")

        team = StockAnalysisTeam()
        report = team.analyze(ticker)

        # Generate insights based on analysis
        try:
            # Extract data from the report
            stock_data = report.get('stock', {})
            macro_data = report.get('macro', {})

            if not stock_data:
                raise ValueError("No stock data available")

            # Use the enhanced data from FundamentalAnalyst
            insights = [
                f"{stock_data.get('name', ticker)} ({ticker}) in {stock_data.get('sector', 'Unknown')} sector, {stock_data.get('industry', 'Unknown')} industry.",
                f"Current price: {stock_data.get('current_price', '$0.00')} with upside potential of {stock_data.get('upside_potential', '0.00%')}.",
                f"P/E ratio is {stock_data.get('pe', 0):.1f} compared to sector average of {stock_data.get('sector_pe', 0):.1f}. P/B ratio: {stock_data.get('pb', 0):.2f}.",
                f"Market cap: {stock_data.get('market_cap', '$0')} with dividend yield of {stock_data.get('dividend_yield', '0.00%')} and beta of {stock_data.get('beta', 0):.2f}.",
                f"Market sentiment is {macro_data.get('sentiment', 0)}/100 with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {report.get('overall_score', 0):.1f}/100 score."
            ]
        except Exception as insight_error:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed for {ticker}, but there was an error generating insights: {str(insight_error)}"]

        # Add chart links to the response
        chart_links = [
            {"type": "price", "url": f"/analyze/stock/chart/{ticker}?chart_type=price", "title": "Financial Health Dashboard"},
            {"type": "metrics", "url": f"/analyze/stock/chart/{ticker}?chart_type=metrics", "title": "Key Metrics Chart"},
            {"type": "history", "url": f"/analyze/stock/chart/{ticker}?chart_type=history", "title": "Price History"},
            {"type": "candlestick", "url": f"/analyze/stock/chart/{ticker}?chart_type=candlestick", "title": "Candlestick Chart"},
            {"type": "price_volume", "url": f"/analyze/stock/chart/{ticker}?chart_type=price_volume", "title": "Price & Volume Chart"}
        ]

        # Add news link
        news_link = {"url": f"/news/stock/{ticker}", "title": "Latest News"}

        # Add advanced analytics link
        advanced_link = {"url": f"/analyze/stock/advanced/{ticker}", "title": "Advanced Analytics"}

        return JSONResponse({
            "report": report,
            "insights": insights,
            "charts": chart_links,
            "news": news_link,
            "advanced": advanced_link
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except Exception as e:
        # Log the error for debugging (in a real app, use a proper logger)
        print(f"Error analyzing stock {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock {ticker}: {str(e)}")


@app.get("/analyze/crypto/{ticker}")
async def analyze_crypto(ticker: str):
    """Analyze a cryptocurrency by ticker symbol (e.g., BTC, ETH) or coin ID (e.g., bitcoin, ethereum)."""
    try:
        # Validate ticker/coin_id
        if not ticker or len(ticker) > 20:
            raise ValueError("Invalid cryptocurrency ticker or ID")

        team = CryptoAnalysisTeam()
        report = team.analyze(ticker)

        if "error" in report.get("crypto", {}):
            raise HTTPException(status_code=404, detail=f"Cryptocurrency not found: {ticker}")

        # Generate insights based on analysis
        try:
            crypto_data = report.get('crypto', {})
            macro_data = report.get('macro', {})

            if not crypto_data:
                raise ValueError("No cryptocurrency data available")

            # Get symbol and name for display
            symbol = crypto_data.get('symbol', ticker.upper())
            name = crypto_data.get('name', symbol)

            # Use the real data from CoinGecko with improved formatting
            insights = [
                f"{name} ({symbol}) - {crypto_data.get('current_price')} with market cap of {crypto_data.get('mcap', 'Unknown')}.",
                f"Market cap rank: #{crypto_data.get('market_cap_rank', 0)} with {crypto_data.get('market_dominance')} market dominance.",
                f"24h price change: {crypto_data.get('price_change_24h', '0%')} with 7d change of {crypto_data.get('price_change_7d', '0%')}.",
                f"Volatility: {crypto_data.get('volatility', 'Unknown')} with 24h volume of {crypto_data.get('volume_24h', '$0')}.",
                f"RSI: {crypto_data.get('rsi', 'N/A')} | Sharpe Ratio: {crypto_data.get('sharpe_ratio', 'N/A')} | Max Drawdown: {crypto_data.get('max_drawdown', 'N/A')}",
                f"All-time high: {crypto_data.get('all_time_high', '$0')} ({crypto_data.get('all_time_high_change', '0%')} from current price).",
                f"Supply: {crypto_data.get('circulating_supply', '0')} / {crypto_data.get('max_supply', 'Unlimited')} ({crypto_data.get('supply_percentage', 'N/A')} circulating).",
                f"Macroeconomic outlook: {macro_data.get('gdp_outlook', 'Stable')} with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {report.get('overall_score', 0):.1f}/100 score."
            ]
        except Exception as insight_error:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed for {ticker}, but there was an error generating insights: {str(insight_error)}"]

        # Add chart links to the response
        chart_links = [
            {"type": "price", "url": f"/analyze/crypto/chart/{ticker}?chart_type=price", "title": "Price Chart"},
            {"type": "price_volume", "url": f"/analyze/crypto/chart/{ticker}?chart_type=price_volume", "title": "Price & Volume Chart"},
            {"type": "performance", "url": f"/analyze/crypto/chart/{ticker}?chart_type=performance", "title": "Performance Chart"},
            {"type": "volume", "url": f"/analyze/crypto/chart/{ticker}?chart_type=volume", "title": "Volume Chart"}
        ]

        # Add news link
        news_link = {"url": f"/news/crypto/{ticker}", "title": "Latest News"}

        # Add advanced analytics link
        advanced_link = {"url": f"/analyze/crypto/advanced/{ticker}", "title": "Advanced Analytics"}

        return JSONResponse({
            "report": report,
            "insights": insights,
            "charts": chart_links,
            "news": news_link,
            "advanced": advanced_link
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging (in a real app, use a proper logger)
        print(f"Error analyzing cryptocurrency {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing cryptocurrency {ticker}: {str(e)}")


@app.get("/analyze/reit/{ticker}")
async def analyze_reit(ticker: str):
    """Analyze a REIT by ticker symbol."""
    try:
        # Validate ticker
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid REIT ticker symbol")

        team = REITAnalysisTeam()
        report = team.analyze(ticker)

        if "error" in report.get("reit", {}):
            raise HTTPException(status_code=404, detail=f"REIT not found: {ticker}")

        # Generate insights based on analysis
        try:
            reit_data = report.get('reit', {})
            macro_data = report.get('macro', {})

            if not reit_data:
                raise ValueError("No REIT data available")

            insights = [
                f"{reit_data.get('name', ticker)} is a {reit_data.get('property_type', 'Commercial')} REIT with market cap of ${reit_data.get('market_cap', 0):,.0f}.",
                f"Dividend yield: {reit_data.get('dividend_yield', '0.0%')} with price to FFO ratio of {reit_data.get('price_to_ffo', 0):.2f}.",
                f"Debt to equity ratio: {reit_data.get('debt_to_equity', 0):.2f} with beta of {reit_data.get('beta', 0):.2f}.",
                f"Macroeconomic outlook: Stable with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {report.get('overall_score', 0):.1f}/100 score."
            ]
        except Exception as insight_error:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed for {ticker}, but there was an error generating insights: {str(insight_error)}"]

        return JSONResponse({
            "report": report,
            "insights": insights
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging (in a real app, use a proper logger)
        print(f"Error analyzing REIT {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing REIT {ticker}: {str(e)}")


@app.get("/analyze/etf/{ticker}")
async def analyze_etf(ticker: str):
    """Analyze an ETF by ticker symbol."""
    try:
        # Validate ticker
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ETF ticker symbol")

        team = ETFAnalysisTeam()
        report = team.analyze(ticker)

        if "error" in report.get("etf", {}):
            raise HTTPException(status_code=404, detail=f"ETF not found: {ticker}")

        # Generate insights based on analysis
        try:
            etf_data = report.get('etf', {})
            macro_data = report.get('macro', {})

            if not etf_data:
                raise ValueError("No ETF data available")

            insights = [
                f"{etf_data.get('name', ticker)} is a {etf_data.get('category', 'Broad Market')} ETF in the {etf_data.get('asset_class', 'Equity')} asset class.",
                f"Expense ratio: {etf_data.get('expense_ratio', '0.0%')} with yield of {etf_data.get('yield', '0.0%')}.",
                f"YTD return: {etf_data.get('ytd_return', '0.0%')} with 3-year return of {etf_data.get('three_year_return', '0.0%')}.",
                f"Macroeconomic outlook: Stable with {macro_data.get('inflation_risk', 'Unknown')} inflation risk.",
                f"Overall recommendation: {report.get('recommendation', 'Hold')} with {report.get('overall_score', 0):.1f}/100 score."
            ]
        except Exception as insight_error:
            # Fallback insights if there's an error processing the data
            insights = [f"Analysis completed for {ticker}, but there was an error generating insights: {str(insight_error)}"]

        return JSONResponse({
            "report": report,
            "insights": insights
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging (in a real app, use a proper logger)
        print(f"Error analyzing ETF {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing ETF {ticker}: {str(e)}")


@app.get("/analyze/compare")
async def compare_securities(stock: Optional[str] = None, crypto: Optional[str] = None,
                           reit: Optional[str] = None, etf: Optional[str] = None):
    """Compare different types of securities."""
    try:
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

        if not results:
            raise HTTPException(status_code=400, detail="At least one security must be provided for comparison")

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

        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing securities: {str(e)}")


@app.get("/chart/stock/{ticker}")
async def stock_chart(request: Request, ticker: str, chart_type: str = "price"):
    """Generate a chart for a stock."""
    try:
        # Validate ticker
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol")

        # Get stock data
        team = StockAnalysisTeam()
        report = team.analyze(ticker)

        if "error" in report.get("stock", {}):
            raise HTTPException(status_code=404, detail=f"Stock not found: {ticker}")

        stock_data = report.get("stock", {})
        raw_data = stock_data.get("raw", {})

        # Create chart based on chart_type
        chart_title = f"{stock_data.get('name', ticker)} ({ticker})"
        chart_description = ""
        data_source = "Yahoo Finance"
        insights = []

        if chart_type == "price":
            # Create a simple price vs DCF chart
            current_price = raw_data.get("current_price", 0)
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

        elif chart_type == "metrics":
            # Create a radar chart of key metrics
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

        elif chart_type == "history":
            # Create a historical price chart
            # Get historical data from Yahoo Finance
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                raise ValueError("No historical data available")

            # Create the figure
            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=config.CHART_COLORS["primary"], width=2)
            ))

            # Add moving averages
            ma50 = hist['Close'].rolling(window=50).mean()
            ma200 = hist['Close'].rolling(window=200).mean()

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ma50,
                mode='lines',
                name='50-day MA',
                line=dict(color=config.CHART_COLORS["secondary"], width=1.5, dash='dot')
            ))

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ma200,
                mode='lines',
                name='200-day MA',
                line=dict(color=config.CHART_COLORS["accent"], width=1.5, dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title=f"{ticker} - 1 Year Price History",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                width=900,
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff")
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            # Calculate insights
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            price_change = ((current_price / start_price) - 1) * 100
            max_price = hist['Close'].max()
            min_price = hist['Close'].min()

            chart_description = "1-year price history with 50-day and 200-day moving averages"
            insights = [
                f"Current price: ${current_price:.2f}",
                f"1-year change: {price_change:.2f}%",
                f"1-year high: ${max_price:.2f}",
                f"1-year low: ${min_price:.2f}",
                f"50-day MA: ${ma50.iloc[-1]:.2f}",
                f"200-day MA: ${ma200.iloc[-1]:.2f}"
            ]

        elif chart_type == "candlestick":
            # Create a candlestick chart
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")

            if hist.empty:
                raise ValueError("No historical data available")

            # Create the figure
            fig = go.Figure()

            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=ticker,
                increasing_line_color=config.CHART_COLORS["success"],
                decreasing_line_color=config.CHART_COLORS["danger"]
            ))

            # Add volume bars
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color=config.CHART_COLORS["info"],
                opacity=0.3,
                yaxis="y2"
            ))

            # Update layout
            fig.update_layout(
                title=f"{ticker} - 3 Month Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                width=900,
                hovermode="x unified",
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff")
            )

            # Add range selector
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            # Calculate insights
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            price_change = ((current_price / start_price) - 1) * 100
            avg_volume = hist['Volume'].mean()

            chart_description = "3-month candlestick chart with volume"
            insights = [
                f"Current price: ${current_price:.2f}",
                f"3-month change: {price_change:.2f}%",
                f"Average daily volume: {avg_volume:,.0f}",
                f"Highest high: ${hist['High'].max():.2f}",
                f"Lowest low: ${hist['Low'].min():.2f}"
            ]

        # Convert the figure to JSON
        chart_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

        # Return the chart page
        return templates.TemplateResponse(
            "chart.html",
            {
                "request": request,
                "title": f"{ticker} Chart",
                "ticker": ticker,
                "asset_type": "stock",
                "chart_title": chart_title,
                "chart_description": chart_description,
                "data_source": data_source,
                "insights": insights,
                "chart_json": chart_json
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Error generating chart for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart for {ticker}: {str(e)}")


@app.get("/chart/crypto/{coin_id}")
async def crypto_chart(request: Request, coin_id: str, chart_type: str = "price"):
    """Generate a chart for a cryptocurrency."""
    try:
        # Validate coin_id
        if not coin_id:
            raise ValueError("Invalid cryptocurrency ID")

        # Get crypto data
        team = CryptoAnalysisTeam()
        report = team.analyze(coin_id)

        if "error" in report.get("crypto", {}):
            raise HTTPException(status_code=404, detail=f"Cryptocurrency not found: {coin_id}")

        crypto_data = report.get("crypto", {})
        raw_data = crypto_data.get("raw", {})

        # Get historical data for charts
        coin_gecko = CoinGeckoAPI()

        # Try to get the coin ID if a symbol was provided
        cg_coin_id = coin_id.lower()

        # Check if we need to convert symbol to ID
        if len(cg_coin_id) <= 5:  # Most symbols are short
            # Get coin list and find the ID for this symbol
            coin_list = coin_gecko.get_coin_list()
            for c in coin_list:
                if c.get('symbol', '').lower() == cg_coin_id:
                    cg_coin_id = c.get('id')
                    break

        # Get historical price data
        historical_data = coin_gecko.get_coin_price_history(cg_coin_id, days=30)

        # Create chart based on chart_type
        chart_title = f"{crypto_data.get('name', coin_id)} ({crypto_data.get('symbol', coin_id.upper())})"
        chart_description = ""
        data_source = "CoinGecko"
        insights = []

        if chart_type == "price":
            # Create a price chart
            prices = historical_data.get("prices", [])

            if not prices:
                raise ValueError("No historical price data available")

            # Convert timestamps to dates
            import datetime
            dates = [datetime.datetime.fromtimestamp(price[0]/1000).strftime('%Y-%m-%d') for price in prices]
            price_values = [price[1] for price in prices]

            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_values,
                mode='lines',
                name='Price (USD)',
                line=dict(color='#2196F3', width=2)
            ))

            # Update layout
            fig.update_layout(
                title=f"{chart_title} - 30 Day Price History",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                width=900,
                hovermode="x unified"
            )

            chart_description = "30-day price history in USD"
            insights = [
                f"Current price: ${crypto_data.get('current_price', 0):,.2f}",
                f"24h change: {crypto_data.get('price_change_24h', '0%')}",
                f"7d change: {crypto_data.get('price_change_7d', '0%')}",
                f"All-time high: ${raw_data.get('all_time_high', 0):,.2f}"
            ]

        elif chart_type == "performance":
            # Create a performance comparison chart
            prices = historical_data.get("prices", [])

            if not prices:
                raise ValueError("No historical price data available")

            # Convert timestamps to dates and calculate percentage change
            import datetime
            dates = [datetime.datetime.fromtimestamp(price[0]/1000).strftime('%Y-%m-%d') for price in prices]

            # Calculate percentage change from first day
            base_price = prices[0][1]
            price_percent_change = [(price[1] / base_price - 1) * 100 for price in prices]

            fig = go.Figure()

            # Add percentage change line
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_percent_change,
                mode='lines',
                name='% Change',
                line=dict(color='#4CAF50', width=2),
                fill='tozeroy'
            ))

            # Add zero line
            fig.add_shape(
                type="line",
                x0=dates[0],
                y0=0,
                x1=dates[-1],
                y1=0,
                line=dict(color="#888888", width=1, dash="dash")
            )

            # Update layout
            fig.update_layout(
                title=f"{chart_title} - 30 Day Performance",
                xaxis_title="Date",
                yaxis_title="% Change",
                height=500,
                width=900,
                hovermode="x unified"
            )

            chart_description = "30-day performance (percentage change)"

            # Calculate overall performance
            start_price = prices[0][1]
            end_price = prices[-1][1]
            overall_change = ((end_price / start_price) - 1) * 100

            insights = [
                f"30-day performance: {overall_change:.2f}%",
                f"Starting price (30 days ago): ${start_price:,.2f}",
                f"Current price: ${end_price:,.2f}",
                f"Volatility: {crypto_data.get('volatility', 'Unknown')}"
            ]

        elif chart_type == "volume":
            # Create a trading volume chart
            volumes = historical_data.get("total_volumes", [])

            if not volumes:
                raise ValueError("No historical volume data available")

            # Convert timestamps to dates
            import datetime
            dates = [datetime.datetime.fromtimestamp(volume[0]/1000).strftime('%Y-%m-%d') for volume in volumes]
            volume_values = [volume[1] for volume in volumes]

            fig = go.Figure()

            # Add volume bars
            fig.add_trace(go.Bar(
                x=dates,
                y=volume_values,
                name='Trading Volume (USD)',
                marker_color='#FF9800'
            ))

            # Update layout
            fig.update_layout(
                title=f"{chart_title} - 30 Day Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume (USD)",
                height=500,
                width=900,
                hovermode="x unified"
            )

            chart_description = "30-day trading volume in USD"

            # Calculate average daily volume
            avg_volume = sum(volume_values) / len(volume_values)
            max_volume = max(volume_values)
            min_volume = min(volume_values)
            current_volume = volume_values[-1]

            insights = [
                f"Current 24h volume: ${current_volume:,.0f}",
                f"Average 30-day volume: ${avg_volume:,.0f}",
                f"Highest volume: ${max_volume:,.0f}",
                f"Lowest volume: ${min_volume:,.0f}"
            ]

        # Convert the figure to JSON
        chart_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

        # Return the chart page
        return templates.TemplateResponse(
            "chart.html",
            {
                "request": request,
                "title": f"{crypto_data.get('symbol', coin_id.upper())} Chart",
                "ticker": coin_id,
                "asset_type": "crypto",
                "chart_title": chart_title,
                "chart_description": chart_description,
                "data_source": data_source,
                "insights": insights,
                "chart_json": chart_json
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Error generating chart for {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart for {coin_id}: {str(e)}")


@app.get("/news/market")
async def get_market_news():
    """Get general market news."""
    try:
        news_api = NewsAPI()
        articles = news_api.get_market_news()

        return JSONResponse({
            "articles": articles
        })
    except Exception as e:
        print(f"Error fetching market news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching market news: {str(e)}")


@app.get("/news/stock/{ticker}")
async def get_stock_news(ticker: str):
    """Get news for a specific stock."""
    try:
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol")

        news_api = NewsAPI()
        articles = news_api.get_stock_news(ticker)

        return JSONResponse({
            "ticker": ticker,
            "articles": articles
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching news for {ticker}: {str(e)}")


@app.get("/news/crypto/{coin_id}")
async def get_crypto_news(coin_id: str):
    """Get news for a specific cryptocurrency."""
    try:
        if not coin_id:
            raise ValueError("Invalid cryptocurrency ID")

        news_api = NewsAPI()
        articles = news_api.get_crypto_news(coin_id)

        return JSONResponse({
            "coin_id": coin_id,
            "articles": articles
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except Exception as e:
        print(f"Error fetching news for {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching news for {coin_id}: {str(e)}")


@app.get("/advanced/stock/{ticker}")
async def advanced_stock_analysis(ticker: str, period: str = "1y"):
    """Get advanced analytics for a stock."""
    try:
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol")

        # Get advanced metrics
        metrics = AdvancedAnalytics.get_advanced_stock_metrics(ticker, period)

        if "error" in metrics:
            raise HTTPException(status_code=404, detail=f"Error analyzing stock: {metrics['error']}")

        # Format metrics for display
        formatted_metrics = {
            "ticker": metrics["ticker"],
            "price": {
                "current": metrics["current_price"],
                "target": metrics["target_mean_price"],
                "upside_potential": ((metrics["target_mean_price"] / metrics["current_price"]) - 1) * 100 if metrics["current_price"] and metrics["target_mean_price"] else None
            },
            "valuation": {
                "market_cap": metrics["market_cap"],
                "enterprise_value": metrics["enterprise_value"],
                "price_to_book": metrics["price_to_book"],
                "forward_pe": metrics["forward_pe"],
                "peg_ratio": metrics["peg_ratio"],
                "enterprise_to_revenue": metrics["enterprise_to_revenue"],
                "enterprise_to_ebitda": metrics["enterprise_to_ebitda"]
            },
            "performance": {
                "volatility": metrics["volatility"],
                "beta": metrics["beta"],
                "alpha": metrics["alpha"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "current_drawdown": metrics["current_drawdown"],
                "rsi": metrics["rsi"]
            },
            "financials": {
                "profit_margins": metrics["profit_margins"],
                "debt_to_equity": metrics["debt_to_equity"],
                "return_on_equity": metrics["return_on_equity"],
                "return_on_assets": metrics["return_on_assets"],
                "free_cash_flow": metrics["free_cash_flow"],
                "operating_cash_flow": metrics["operating_cash_flow"],
                "revenue_growth": metrics["revenue_growth"],
                "earnings_growth": metrics["earnings_growth"]
            },
            "technical": {
                "moving_averages": metrics["moving_averages"],
                "ma_signals": get_ma_signals(metrics["current_price"], metrics["moving_averages"])
            },
            "dividend": {
                "yield": metrics["dividend_yield"]
            },
            "analyst": {
                "rating": metrics["analyst_rating"]
            }
        }

        # Generate insights
        insights = generate_stock_insights(formatted_metrics)

        return JSONResponse({
            "metrics": formatted_metrics,
            "insights": insights
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in advanced stock analysis for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock {ticker}: {str(e)}")


@app.get("/advanced/crypto/{coin_id}")
async def advanced_crypto_analysis(coin_id: str, days: int = 365):
    """Get advanced analytics for a cryptocurrency."""
    try:
        if not coin_id:
            raise ValueError("Invalid cryptocurrency ID")

        # Get advanced metrics
        metrics = AdvancedAnalytics.get_advanced_crypto_metrics(coin_id, days=days)

        if "error" in metrics:
            raise HTTPException(status_code=404, detail=f"Error analyzing cryptocurrency: {metrics['error']}")

        # Format metrics for display
        formatted_metrics = {
            "coin_id": metrics["coin_id"],
            "price": {
                "current": metrics["current_price"],
                "ath": metrics["ath"],
                "ath_change": metrics["ath_change_percentage"],
                "atl": metrics["atl"],
                "atl_change": metrics["atl_change_percentage"],
                "high_24h": metrics["high_24h"],
                "low_24h": metrics["low_24h"]
            },
            "market": {
                "market_cap": metrics["market_cap"],
                "market_cap_rank": metrics["market_cap_rank"],
                "fully_diluted_valuation": metrics["fully_diluted_valuation"],
                "total_volume": metrics["total_volume"],
                "circulating_supply": metrics["circulating_supply"],
                "total_supply": metrics["total_supply"],
                "max_supply": metrics["max_supply"]
            },
            "performance": {
                "daily_return": metrics["daily_return"],
                "weekly_return": metrics["weekly_return"],
                "monthly_return": metrics["monthly_return"],
                "yearly_return": metrics["yearly_return"],
                "volatility": metrics["volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "current_drawdown": metrics["current_drawdown"]
            },
            "technical": {
                "rsi": metrics["rsi"],
                "moving_averages": metrics["moving_averages"],
                "ma_signals": get_ma_signals(metrics["current_price"], metrics["moving_averages"])
            },
            "changes": {
                "price_change_24h": metrics["price_change_24h"],
                "price_change_percentage_24h": metrics["price_change_percentage_24h"],
                "market_cap_change_24h": metrics["market_cap_change_24h"],
                "market_cap_change_percentage_24h": metrics["market_cap_change_percentage_24h"]
            }
        }

        # Generate insights
        insights = generate_crypto_insights(formatted_metrics)

        return JSONResponse({
            "metrics": formatted_metrics,
            "insights": insights
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in advanced crypto analysis for {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing cryptocurrency {coin_id}: {str(e)}")


def get_ma_signals(current_price: float, moving_averages: Dict[str, float]) -> Dict[str, str]:
    """Generate signals based on moving averages."""
    signals = {}

    for ma_key, ma_value in moving_averages.items():
        if current_price > ma_value:
            signals[ma_key] = "bullish"
        elif current_price < ma_value:
            signals[ma_key] = "bearish"
        else:
            signals[ma_key] = "neutral"

    return signals


def generate_stock_insights(metrics: Dict[str, Any]) -> List[str]:
    """Generate insights from stock metrics."""
    insights = []

    # Price insights
    if metrics["price"]["upside_potential"] is not None:
        if metrics["price"]["upside_potential"] > 20:
            insights.append(f"Analysts suggest significant upside potential of {metrics['price']['upside_potential']:.1f}%")
        elif metrics["price"]["upside_potential"] < -20:
            insights.append(f"Analysts suggest significant downside risk of {abs(metrics['price']['upside_potential']):.1f}%")

    # Valuation insights
    if metrics["valuation"]["forward_pe"] is not None:
        if metrics["valuation"]["forward_pe"] < 15:
            insights.append(f"Forward P/E ratio of {metrics['valuation']['forward_pe']:.2f} suggests potential undervaluation")
        elif metrics["valuation"]["forward_pe"] > 30:
            insights.append(f"Forward P/E ratio of {metrics['valuation']['forward_pe']:.2f} suggests potential overvaluation")

    # Performance insights
    if metrics["performance"]["volatility"] > 40:
        insights.append(f"High volatility of {metrics['performance']['volatility']:.1f}% indicates significant price swings")

    if metrics["performance"]["beta"] is not None:
        if metrics["performance"]["beta"] > 1.5:
            insights.append(f"Beta of {metrics['performance']['beta']:.2f} indicates higher volatility than the market")
        elif metrics["performance"]["beta"] < 0.5:
            insights.append(f"Beta of {metrics['performance']['beta']:.2f} indicates lower volatility than the market")

    if metrics["performance"]["alpha"] is not None:
        if metrics["performance"]["alpha"] > 5:
            insights.append(f"Alpha of {metrics['performance']['alpha']:.2f}% indicates outperformance relative to risk")
        elif metrics["performance"]["alpha"] < -5:
            insights.append(f"Alpha of {metrics['performance']['alpha']:.2f}% indicates underperformance relative to risk")

    # Financial insights
    if metrics["financials"]["profit_margins"] is not None:
        if metrics["financials"]["profit_margins"] > 0.2:
            insights.append(f"High profit margin of {metrics['financials']['profit_margins']*100:.1f}% indicates strong profitability")
        elif metrics["financials"]["profit_margins"] < 0.05:
            insights.append(f"Low profit margin of {metrics['financials']['profit_margins']*100:.1f}% may indicate competitive pressures")

    if metrics["financials"]["debt_to_equity"] is not None:
        if metrics["financials"]["debt_to_equity"] > 2:
            insights.append(f"High debt-to-equity ratio of {metrics['financials']['debt_to_equity']:.2f} indicates significant leverage")

    # Technical insights
    if metrics["technical"]["rsi"] > 70:
        insights.append(f"RSI of {metrics['technical']['rsi']:.1f} suggests the stock may be overbought")
    elif metrics["technical"]["rsi"] < 30:
        insights.append(f"RSI of {metrics['technical']['rsi']:.1f} suggests the stock may be oversold")

    # Dividend insights
    if metrics["dividend"]["yield"] > 4:
        insights.append(f"High dividend yield of {metrics['dividend']['yield']:.2f}% may be attractive for income investors")

    # Analyst insights
    if metrics["analyst"]["rating"] in ["buy", "strongBuy"]:
        insights.append(f"Analysts generally recommend {metrics['analyst']['rating']} for this stock")
    elif metrics["analyst"]["rating"] in ["sell", "strongSell"]:
        insights.append(f"Analysts generally recommend {metrics['analyst']['rating']} for this stock")

    # If no insights were generated, add a default one
    if not insights:
        insights.append("No significant insights found based on the current metrics.")

    return insights


@app.get("/compare/stocks")
async def compare_stocks(tickers: str, period: str = "1y"):
    """Compare multiple stocks."""
    try:
        # Parse tickers (comma-separated)
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

        if not ticker_list:
            raise ValueError("No valid tickers provided")

        if len(ticker_list) > 5:
            raise ValueError("Maximum 5 tickers can be compared at once")

        # Get data for each ticker
        results = {}
        for ticker in ticker_list:
            try:
                metrics = AdvancedAnalytics.get_advanced_stock_metrics(ticker, period)
                if "error" not in metrics:
                    results[ticker] = metrics
            except Exception as e:
                print(f"Error getting data for {ticker}: {str(e)}")

        if not results:
            raise HTTPException(status_code=404, detail="Could not retrieve data for any of the provided tickers")

        # Generate comparison data
        comparison = {
            "tickers": list(results.keys()),
            "price": {},
            "valuation": {},
            "performance": {},
            "financials": {},
            "technical": {},
            "dividend": {}
        }

        # Price comparison
        comparison["price"]["current"] = {ticker: metrics["current_price"] for ticker, metrics in results.items()}

        # Valuation comparison
        comparison["valuation"]["market_cap"] = {ticker: metrics["market_cap"] for ticker, metrics in results.items()}
        comparison["valuation"]["pe"] = {ticker: metrics.get("trailingPE", None) for ticker, metrics in results.items()}
        comparison["valuation"]["forward_pe"] = {ticker: metrics.get("forward_pe", None) for ticker, metrics in results.items()}
        comparison["valuation"]["price_to_book"] = {ticker: metrics.get("price_to_book", None) for ticker, metrics in results.items()}
        comparison["valuation"]["peg_ratio"] = {ticker: metrics.get("peg_ratio", None) for ticker, metrics in results.items()}

        # Performance comparison
        comparison["performance"]["beta"] = {ticker: metrics.get("beta", None) for ticker, metrics in results.items()}
        comparison["performance"]["alpha"] = {ticker: metrics.get("alpha", None) for ticker, metrics in results.items()}
        comparison["performance"]["volatility"] = {ticker: metrics.get("volatility", None) for ticker, metrics in results.items()}
        comparison["performance"]["sharpe_ratio"] = {ticker: metrics.get("sharpe_ratio", None) for ticker, metrics in results.items()}

        # Financial comparison
        comparison["financials"]["profit_margins"] = {ticker: metrics.get("profit_margins", None) for ticker, metrics in results.items()}
        comparison["financials"]["debt_to_equity"] = {ticker: metrics.get("debt_to_equity", None) for ticker, metrics in results.items()}
        comparison["financials"]["return_on_equity"] = {ticker: metrics.get("return_on_equity", None) for ticker, metrics in results.items()}
        comparison["financials"]["return_on_assets"] = {ticker: metrics.get("return_on_assets", None) for ticker, metrics in results.items()}

        # Technical comparison
        comparison["technical"]["rsi"] = {ticker: metrics.get("rsi", None) for ticker, metrics in results.items()}

        # Dividend comparison
        comparison["dividend"]["yield"] = {ticker: metrics.get("dividend_yield", None) for ticker, metrics in results.items()}

        # Generate comparison insights
        insights = generate_stock_comparison_insights(comparison)

        # Generate comparison chart data
        chart_data = generate_stock_comparison_chart_data(ticker_list, period)

        return JSONResponse({
            "comparison": comparison,
            "insights": insights,
            "chart_data": chart_data
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in stock comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing stocks: {str(e)}")


@app.get("/compare/cryptos")
async def compare_cryptos(coins: str, days: int = 30):
    """Compare multiple cryptocurrencies."""
    try:
        # Parse coin IDs (comma-separated)
        coin_list = [c.strip() for c in coins.split(",") if c.strip()]

        if not coin_list:
            raise ValueError("No valid coin IDs provided")

        if len(coin_list) > 5:
            raise ValueError("Maximum 5 cryptocurrencies can be compared at once")

        # Get data for each coin
        results = {}
        for coin_id in coin_list:
            try:
                metrics = AdvancedAnalytics.get_advanced_crypto_metrics(coin_id, days=days)
                if "error" not in metrics:
                    results[coin_id] = metrics
            except Exception as e:
                print(f"Error getting data for {coin_id}: {str(e)}")

        if not results:
            raise HTTPException(status_code=404, detail="Could not retrieve data for any of the provided coins")

        # Generate comparison data
        comparison = {
            "coins": list(results.keys()),
            "price": {},
            "market": {},
            "performance": {},
            "technical": {},
            "changes": {}
        }

        # Price comparison
        comparison["price"]["current"] = {coin: metrics["current_price"] for coin, metrics in results.items()}
        comparison["price"]["ath"] = {coin: metrics["ath"] for coin, metrics in results.items()}
        comparison["price"]["ath_change"] = {coin: metrics["ath_change_percentage"] for coin, metrics in results.items()}

        # Market comparison
        comparison["market"]["market_cap"] = {coin: metrics["market_cap"] for coin, metrics in results.items()}
        comparison["market"]["market_cap_rank"] = {coin: metrics["market_cap_rank"] for coin, metrics in results.items()}
        comparison["market"]["total_volume"] = {coin: metrics["total_volume"] for coin, metrics in results.items()}
        comparison["market"]["circulating_supply"] = {coin: metrics["circulating_supply"] for coin, metrics in results.items()}

        # Performance comparison
        comparison["performance"]["volatility"] = {coin: metrics["volatility"] for coin, metrics in results.items()}
        comparison["performance"]["sharpe_ratio"] = {coin: metrics["sharpe_ratio"] for coin, metrics in results.items()}
        comparison["performance"]["max_drawdown"] = {coin: metrics["max_drawdown"] for coin, metrics in results.items()}
        comparison["performance"]["daily_return"] = {coin: metrics["daily_return"] for coin, metrics in results.items()}
        comparison["performance"]["weekly_return"] = {coin: metrics["weekly_return"] for coin, metrics in results.items()}
        comparison["performance"]["monthly_return"] = {coin: metrics["monthly_return"] for coin, metrics in results.items()}

        # Technical comparison
        comparison["technical"]["rsi"] = {coin: metrics["rsi"] for coin, metrics in results.items()}

        # Changes comparison
        comparison["changes"]["price_change_24h"] = {coin: metrics["price_change_percentage_24h"] for coin, metrics in results.items()}

        # Generate comparison insights
        insights = generate_crypto_comparison_insights(comparison)

        # Generate comparison chart data
        chart_data = generate_crypto_comparison_chart_data(coin_list, days)

        return JSONResponse({
            "comparison": comparison,
            "insights": insights,
            "chart_data": chart_data
        })
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in crypto comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing cryptocurrencies: {str(e)}")


def generate_stock_comparison_insights(comparison: Dict[str, Any]) -> List[str]:
    """Generate insights from stock comparison data."""
    insights = []
    tickers = comparison["tickers"]

    if len(tickers) < 2:
        return [f"Comparison insights require at least 2 stocks, but only {len(tickers)} provided."]

    # Find best and worst performers for various metrics
    try:
        # Valuation comparison
        if all(pe is not None for pe in comparison["valuation"]["forward_pe"].values()):
            lowest_pe = min(tickers, key=lambda t: comparison["valuation"]["forward_pe"][t])
            highest_pe = max(tickers, key=lambda t: comparison["valuation"]["forward_pe"][t])
            insights.append(f"{lowest_pe} has the lowest forward P/E ratio at {comparison['valuation']['forward_pe'][lowest_pe]:.2f}, while {highest_pe} has the highest at {comparison['valuation']['forward_pe'][highest_pe]:.2f}.")

        # Performance comparison
        if all(vol is not None for vol in comparison["performance"]["volatility"].values()):
            lowest_vol = min(tickers, key=lambda t: comparison["performance"]["volatility"][t])
            highest_vol = max(tickers, key=lambda t: comparison["performance"]["volatility"][t])
            insights.append(f"{lowest_vol} has the lowest volatility at {comparison['performance']['volatility'][lowest_vol]:.1f}%, while {highest_vol} has the highest at {comparison['performance']['volatility'][highest_vol]:.1f}%.")

        if all(beta is not None for beta in comparison["performance"]["beta"].values()):
            lowest_beta = min(tickers, key=lambda t: abs(comparison["performance"]["beta"][t] - 1))
            highest_beta = max(tickers, key=lambda t: comparison["performance"]["beta"][t])
            insights.append(f"{lowest_beta} has a beta of {comparison['performance']['beta'][lowest_beta]:.2f} (closest to market), while {highest_beta} has the highest beta at {comparison['performance']['beta'][highest_beta]:.2f}.")

        # Financial comparison
        if all(margin is not None for margin in comparison["financials"]["profit_margins"].values()):
            best_margin = max(tickers, key=lambda t: comparison["financials"]["profit_margins"][t])
            worst_margin = min(tickers, key=lambda t: comparison["financials"]["profit_margins"][t])
            insights.append(f"{best_margin} has the highest profit margin at {comparison['financials']['profit_margins'][best_margin]*100:.1f}%, while {worst_margin} has the lowest at {comparison['financials']['profit_margins'][worst_margin]*100:.1f}%.")

        # Dividend comparison
        if all(div is not None for div in comparison["dividend"]["yield"].values()):
            highest_div = max(tickers, key=lambda t: comparison["dividend"]["yield"][t])
            lowest_div = min(tickers, key=lambda t: comparison["dividend"]["yield"][t])
            insights.append(f"{highest_div} offers the highest dividend yield at {comparison['dividend']['yield'][highest_div]:.2f}%, while {lowest_div} offers the lowest at {comparison['dividend']['yield'][lowest_div]:.2f}%.")
    except Exception as e:
        print(f"Error generating comparison insights: {str(e)}")

    # If no insights were generated, add a default one
    if not insights:
        insights.append("No significant comparison insights found based on the current metrics.")

    return insights


def generate_crypto_comparison_insights(comparison: Dict[str, Any]) -> List[str]:
    """Generate insights from cryptocurrency comparison data."""
    insights = []
    coins = comparison["coins"]

    if len(coins) < 2:
        return [f"Comparison insights require at least 2 cryptocurrencies, but only {len(coins)} provided."]

    # Find best and worst performers for various metrics
    try:
        # Price comparison
        if all(ath_change is not None for ath_change in comparison["price"]["ath_change"].values()):
            closest_to_ath = max(coins, key=lambda c: comparison["price"]["ath_change"][c])
            furthest_from_ath = min(coins, key=lambda c: comparison["price"]["ath_change"][c])
            insights.append(f"{closest_to_ath} is closest to its all-time high, down only {abs(comparison['price']['ath_change'][closest_to_ath]):.1f}%, while {furthest_from_ath} is down {abs(comparison['price']['ath_change'][furthest_from_ath]):.1f}% from its ATH.")

        # Market comparison
        if all(rank is not None for rank in comparison["market"]["market_cap_rank"].values()):
            highest_rank = min(coins, key=lambda c: comparison["market"]["market_cap_rank"][c])
            lowest_rank = max(coins, key=lambda c: comparison["market"]["market_cap_rank"][c])
            insights.append(f"{highest_rank} has the highest market cap rank at #{comparison['market']['market_cap_rank'][highest_rank]}, while {lowest_rank} ranks #{comparison['market']['market_cap_rank'][lowest_rank]}.")

        # Performance comparison
        if all(vol is not None for vol in comparison["performance"]["volatility"].values()):
            lowest_vol = min(coins, key=lambda c: comparison["performance"]["volatility"][c])
            highest_vol = max(coins, key=lambda c: comparison["performance"]["volatility"][c])
            insights.append(f"{lowest_vol} has the lowest volatility at {comparison['performance']['volatility'][lowest_vol]:.1f}%, while {highest_vol} has the highest at {comparison['performance']['volatility'][highest_vol]:.1f}%.")

        if all(sharpe is not None for sharpe in comparison["performance"]["sharpe_ratio"].values()):
            best_sharpe = max(coins, key=lambda c: comparison["performance"]["sharpe_ratio"][c])
            worst_sharpe = min(coins, key=lambda c: comparison["performance"]["sharpe_ratio"][c])
            insights.append(f"{best_sharpe} has the best risk-adjusted returns with a Sharpe ratio of {comparison['performance']['sharpe_ratio'][best_sharpe]:.2f}, while {worst_sharpe} has the worst at {comparison['performance']['sharpe_ratio'][worst_sharpe]:.2f}.")

        # Recent performance
        if all(change is not None for change in comparison["changes"]["price_change_24h"].values()):
            best_24h = max(coins, key=lambda c: comparison["changes"]["price_change_24h"][c])
            worst_24h = min(coins, key=lambda c: comparison["changes"]["price_change_24h"][c])
            insights.append(f"{best_24h} has the best 24-hour performance at {comparison['changes']['price_change_24h'][best_24h]:.1f}%, while {worst_24h} has the worst at {comparison['changes']['price_change_24h'][worst_24h]:.1f}%.")
    except Exception as e:
        print(f"Error generating comparison insights: {str(e)}")

    # If no insights were generated, add a default one
    if not insights:
        insights.append("No significant comparison insights found based on the current metrics.")

    return insights


def generate_stock_comparison_chart_data(tickers: List[str], period: str = "1y") -> Dict[str, Any]:
    """Generate chart data for stock comparison."""
    try:
        # Get historical data for each ticker
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    # Convert to list of [timestamp, price] pairs
                    prices = [[int(date.timestamp() * 1000), price] for date, price in zip(hist.index, hist["Close"])]
                    data[ticker] = prices
            except Exception as e:
                print(f"Error getting historical data for {ticker}: {str(e)}")

        return data
    except Exception as e:
        print(f"Error generating stock comparison chart data: {str(e)}")
        return {}


def generate_crypto_comparison_chart_data(coins: List[str], days: int = 30) -> Dict[str, Any]:
    """Generate chart data for cryptocurrency comparison."""
    try:
        # This is a placeholder - in a real implementation, you would use the CoinGecko API
        # For now, we'll return mock data

        data = {}
        for coin in coins:
            try:
                # Generate mock price data (random walk)
                np.random.seed(int(hash(coin) % 2**32))
                start_price = 1000 + np.random.rand() * 10000
                daily_returns = np.random.normal(0.001, 0.03, days)
                prices = start_price * np.cumprod(1 + daily_returns)

                # Generate timestamps (one per day, going back from today)
                import datetime
                today = datetime.datetime.now()
                timestamps = [(today - datetime.timedelta(days=i)).timestamp() * 1000 for i in range(days, 0, -1)]

                # Combine timestamps and prices
                data[coin] = [[int(ts), price] for ts, price in zip(timestamps, prices)]
            except Exception as e:
                print(f"Error generating mock data for {coin}: {str(e)}")

        return data
    except Exception as e:
        print(f"Error generating crypto comparison chart data: {str(e)}")
        return {}


@app.get("/compare/stocks/view")
async def view_stock_comparison(request: Request, tickers: str, period: str = "1y"):
    """View stock comparison page."""
    try:
        # Parse tickers (comma-separated)
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

        if not ticker_list:
            raise ValueError("No valid tickers provided")

        if len(ticker_list) > 5:
            raise ValueError("Maximum 5 tickers can be compared at once")

        # Get comparison data
        comparison_data = await compare_stocks(tickers, period)
        comparison = comparison_data.body
        comparison_dict = json.loads(comparison)

        # Create chart data
        chart_data = comparison_dict["chart_data"]

        # Create chart figure
        fig = go.Figure()

        # Add a line for each ticker
        for ticker, prices in chart_data.items():
            if prices:
                # Extract timestamps and prices
                timestamps = [p[0] for p in prices]
                values = [p[1] for p in prices]

                # Add line to chart
                fig.add_trace(go.Scatter(
                    x=[datetime.datetime.fromtimestamp(ts/1000) for ts in timestamps],
                    y=values,
                    mode='lines',
                    name=ticker
                ))

        # Update layout
        fig.update_layout(
            title=f"Price Comparison - {', '.join(ticker_list)}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # Convert the figure to JSON
        chart_json = json.dumps({
            "data": fig.data,
            "layout": fig.layout
        }, cls=PlotlyJSONEncoder)

        # Format comparison data for template
        formatted_comparison = {
            "price": comparison_dict["comparison"]["price"],
            "valuation": comparison_dict["comparison"]["valuation"],
            "performance": comparison_dict["comparison"]["performance"],
            "financials": comparison_dict["comparison"]["financials"],
            "technical": comparison_dict["comparison"]["technical"],
            "dividend": comparison_dict["comparison"]["dividend"]
        }

        # Helper function for template
        def get_value_class(metric_name, ticker, values):
            if ticker not in values or values[ticker] is None:
                return ""

            # Determine if this is the best or worst value
            tickers_with_values = [t for t in ticker_list if t in values and values[t] is not None]
            if not tickers_with_values:
                return ""

            # Metrics where higher is better
            higher_better = ["market_cap", "profit_margins", "return_on_equity", "return_on_assets", "yield"]

            # Metrics where lower is better
            lower_better = ["pe", "forward_pe", "peg_ratio", "debt_to_equity"]

            # For other metrics, just show the value without highlighting
            if metric_name in higher_better:
                if ticker == max(tickers_with_values, key=lambda t: values[t]):
                    return "best-value"
                if ticker == min(tickers_with_values, key=lambda t: values[t]):
                    return "worst-value"
            elif metric_name in lower_better:
                if ticker == min(tickers_with_values, key=lambda t: values[t]):
                    return "best-value"
                if ticker == max(tickers_with_values, key=lambda t: values[t]):
                    return "worst-value"

            return ""

        # Helper function to format values
        def format_value(value):
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                if value > 1000000000:  # Billions
                    return f"${value/1000000000:.2f}B"
                elif value > 1000000:  # Millions
                    return f"${value/1000000:.2f}M"
                elif value > 1000:  # Thousands
                    return f"${value/1000:.2f}K"
                else:
                    return f"{value:.2f}"
            return str(value)

        return templates.TemplateResponse(
            "comparison.html",
            {
                "request": request,
                "title": f"Stock Comparison: {', '.join(ticker_list)}",
                "chart_title": f"Price Comparison - {', '.join(ticker_list)}",
                "chart_description": f"Historical price comparison for {period} period",
                "data_source": "Yahoo Finance",
                "insights": comparison_dict["insights"],
                "comparison_data": formatted_comparison,
                "items": ticker_list,
                "chart_json": chart_json,
                "get_value_class": get_value_class,
                "format_value": format_value
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in stock comparison view: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing stock comparison: {str(e)}")


@app.get("/compare/cryptos/view")
async def view_crypto_comparison(request: Request, coins: str, days: int = 30):
    """View cryptocurrency comparison page."""
    try:
        # Parse coin IDs (comma-separated)
        coin_list = [c.strip() for c in coins.split(",") if c.strip()]

        if not coin_list:
            raise ValueError("No valid coin IDs provided")

        if len(coin_list) > 5:
            raise ValueError("Maximum 5 cryptocurrencies can be compared at once")

        # Get comparison data
        comparison_data = await compare_cryptos(coins, days)
        comparison = comparison_data.body
        comparison_dict = json.loads(comparison)

        # Create chart data
        chart_data = comparison_dict["chart_data"]

        # Create chart figure
        fig = go.Figure()

        # Add a line for each coin
        for coin, prices in chart_data.items():
            if prices:
                # Extract timestamps and prices
                timestamps = [p[0] for p in prices]
                values = [p[1] for p in prices]

                # Add line to chart
                fig.add_trace(go.Scatter(
                    x=[datetime.datetime.fromtimestamp(ts/1000) for ts in timestamps],
                    y=values,
                    mode='lines',
                    name=coin
                ))

        # Update layout
        fig.update_layout(
            title=f"Price Comparison - {', '.join(coin_list)}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # Convert the figure to JSON
        chart_json = json.dumps({
            "data": fig.data,
            "layout": fig.layout
        }, cls=PlotlyJSONEncoder)

        # Format comparison data for template
        formatted_comparison = {
            "price": comparison_dict["comparison"]["price"],
            "market": comparison_dict["comparison"]["market"],
            "performance": comparison_dict["comparison"]["performance"],
            "technical": comparison_dict["comparison"]["technical"],
            "changes": comparison_dict["comparison"]["changes"]
        }

        # Helper function for template
        def get_value_class(metric_name, coin, values):
            if coin not in values or values[coin] is None:
                return ""

            # Determine if this is the best or worst value
            coins_with_values = [c for c in coin_list if c in values and values[c] is not None]
            if not coins_with_values:
                return ""

            # Metrics where higher is better
            higher_better = ["market_cap", "total_volume", "sharpe_ratio", "daily_return", "weekly_return", "monthly_return", "price_change_24h"]

            # Metrics where lower is better
            lower_better = ["market_cap_rank", "volatility", "max_drawdown"]

            # For other metrics, just show the value without highlighting
            if metric_name in higher_better:
                if coin == max(coins_with_values, key=lambda c: values[c]):
                    return "best-value"
                if coin == min(coins_with_values, key=lambda c: values[c]):
                    return "worst-value"
            elif metric_name in lower_better:
                if coin == min(coins_with_values, key=lambda c: values[c]):
                    return "best-value"
                if coin == max(coins_with_values, key=lambda c: values[c]):
                    return "worst-value"

            return ""

        # Helper function to format values
        def format_value(value):
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                if value > 1000000000:  # Billions
                    return f"${value/1000000000:.2f}B"
                elif value > 1000000:  # Millions
                    return f"${value/1000000:.2f}M"
                elif value > 1000:  # Thousands
                    return f"${value/1000:.2f}K"
                else:
                    return f"{value:.2f}"
            return str(value)

        return templates.TemplateResponse(
            "comparison.html",
            {
                "request": request,
                "title": f"Cryptocurrency Comparison: {', '.join(coin_list)}",
                "chart_title": f"Price Comparison - {', '.join(coin_list)}",
                "chart_description": f"Historical price comparison for {days} days",
                "data_source": "CoinGecko",
                "insights": comparison_dict["insights"],
                "comparison_data": formatted_comparison,
                "items": coin_list,
                "chart_json": chart_json,
                "get_value_class": get_value_class,
                "format_value": format_value
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in crypto comparison view: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing crypto comparison: {str(e)}")


@app.get("/compare/view")
async def view_asset_comparison(request: Request, stock: Optional[str] = None, crypto: Optional[str] = None,
                             reit: Optional[str] = None, etf: Optional[str] = None):
    """View comparison page for different asset types."""
    try:
        # Validate that at least one asset is provided
        if not any([stock, crypto, reit, etf]):
            raise ValueError("At least one asset must be provided for comparison")

        # Get comparison data
        comparison_data = await compare_securities(stock, crypto, reit, etf)
        comparison_dict = json.loads(comparison_data.body)

        # Create asset list for display
        asset_list = []
        if stock:
            asset_list.append(f"Stock: {stock}")
        if crypto:
            asset_list.append(f"Crypto: {crypto}")
        if reit:
            asset_list.append(f"REIT: {reit}")
        if etf:
            asset_list.append(f"ETF: {etf}")

        # Format comparison data for display
        formatted_comparison = {}
        if "comparison_data" in comparison_dict:
            for category, metrics in comparison_dict["comparison_data"].items():
                formatted_comparison[category] = metrics

        # Create chart data for performance comparison
        fig = go.Figure()

        # Add a bar for each asset's performance metric if available
        if "performance" in formatted_comparison:
            performance_data = formatted_comparison["performance"]

            # Extract performance metrics for each asset
            labels = []
            values = []
            colors = []

            for asset_id, metrics in performance_data.items():
                # Use the most relevant performance metric available
                perf_value = None
                if "three_year_return" in metrics:
                    perf_value = metrics["three_year_return"]
                    metric_name = "3-Year Return"
                elif "ytd_return" in metrics:
                    perf_value = metrics["ytd_return"]
                    metric_name = "YTD Return"
                elif "price_change_7d" in metrics:
                    perf_value = metrics["price_change_7d"]
                    metric_name = "7-Day Change"
                elif "upside_potential" in metrics:
                    # Convert from string to float if needed
                    if isinstance(metrics["upside_potential"], str):
                        perf_value = float(metrics["upside_potential"].rstrip("%"))
                    else:
                        perf_value = metrics["upside_potential"]
                    metric_name = "Upside Potential"

                if perf_value is not None:
                    labels.append(asset_id)
                    values.append(perf_value)
                    # Set color based on value
                    colors.append("green" if perf_value >= 0 else "red")

            if labels and values:
                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.2f}%" for v in values],
                    textposition="auto"
                ))

                # Update layout
                fig.update_layout(
                    title=f"Performance Comparison - {metric_name}",
                    xaxis_title="Asset",
                    yaxis_title="Performance (%)",
                    template="plotly_dark",
                    autosize=True,
                    margin=dict(l=50, r=50, t=80, b=50)
                )

        # Convert the figure to JSON
        chart_json = json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)

        # Helper functions for template
        def get_value_class(metric, asset_id, values):
            if asset_id not in values or values[asset_id] is None:
                return "neutral"

            # Determine if higher or lower values are better for this metric
            higher_is_better = True
            if metric in ["beta", "volatility", "expense_ratio", "pe_ratio", "pb_ratio"]:
                higher_is_better = False

            # Find min and max values
            valid_values = [v for k, v in values.items() if v is not None]
            if not valid_values:
                return "neutral"

            min_val = min(valid_values)
            max_val = max(valid_values)

            # If all values are the same, return neutral
            if min_val == max_val:
                return "neutral"

            # Get the value for this asset
            value = values[asset_id]

            # Determine if this is the best, worst, or in between
            if higher_is_better:
                if value == max_val:
                    return "positive"
                elif value == min_val:
                    return "negative"
                else:
                    return "neutral"
            else:
                if value == min_val:
                    return "positive"
                elif value == max_val:
                    return "negative"
                else:
                    return "neutral"

        def format_value(value):
            if value is None:
                return "N/A"
            elif isinstance(value, (int, float)):
                if abs(value) >= 1000000000:
                    return f"${value/1000000000:.2f}B"
                elif abs(value) >= 1000000:
                    return f"${value/1000000:.2f}M"
                elif abs(value) >= 1000:
                    return f"${value/1000:.2f}K"
                elif isinstance(value, float):
                    return f"{value:.2f}"
                else:
                    return str(value)
            else:
                return str(value)

        return templates.TemplateResponse(
            "comparison.html",
            {
                "request": request,
                "title": f"Asset Comparison: {', '.join(asset_list)}",
                "chart_title": "Performance Comparison",
                "chart_description": "Comparing performance metrics across different asset types",
                "data_source": "Multiple Sources",
                "insights": comparison_dict.get("insights", []),
                "recommendations": comparison_dict.get("recommendations", []),
                "comparison_data": formatted_comparison,
                "items": [stock, crypto, reit, etf] if all([stock, crypto, reit, etf]) else \
                         [a for a in [stock, crypto, reit, etf] if a],
                "chart_json": chart_json,
                "get_value_class": get_value_class,
                "format_value": format_value
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in asset comparison view: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing asset comparison: {str(e)}")


@app.get("/backtest/stock/{ticker}")
async def backtest_stock(ticker: str, strategy: str, period: str = "5y", short_window: int = 50, long_window: int = 200,
                        rsi_window: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                        macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
    """Run a backtest for a stock."""
    try:
        if not ticker or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol")

        # Validate strategy
        valid_strategies = ["ma_cross", "rsi", "macd"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}")

        # Set strategy parameters
        params = {}
        if strategy == "ma_cross":
            params = {
                "short_window": short_window,
                "long_window": long_window
            }
        elif strategy == "rsi":
            params = {
                "window": rsi_window,
                "overbought": rsi_overbought,
                "oversold": rsi_oversold
            }
        elif strategy == "macd":
            params = {
                "fast_period": macd_fast,
                "slow_period": macd_slow,
                "signal_period": macd_signal
            }

        # Run backtest
        results = run_backtest(ticker, strategy, params, period)

        if "error" in results:
            raise HTTPException(status_code=404, detail=f"Error running backtest: {results['error']}")

        # Format results for response
        formatted_results = {
            "ticker": results["ticker"],
            "strategy": {
                "name": results["strategy"],
                "params": results["params"],
                "description": get_strategy_description(results["strategy"])
            },
            "performance": {
                "initial_capital": results["initial_capital"],
                "final_value": results["final_value"],
                "total_return": results["total_return"],
                "annual_return": results["annual_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "alpha": results["alpha"],
                "beta": results["beta"]
            },
            "benchmark": {
                "total_return": results["benchmark_return"],
                "annual_return": results["benchmark_annual_return"]
            },
            "trades": {
                "count": results["num_trades"],
                "details": results["trades"][:10]  # Limit to first 10 trades
            },
            "chart": results["chart"]
        }

        return JSONResponse(formatted_results)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in backtest for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running backtest for {ticker}: {str(e)}")


@app.get("/backtest/view/{ticker}")
async def view_backtest(request: Request, ticker: str, strategy: str, period: str = "5y", short_window: int = 50, long_window: int = 200,
                      rsi_window: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                      macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
    """View backtest results page."""
    try:
        # Get backtest results
        backtest_data = await backtest_stock(ticker, strategy, period, short_window, long_window,
                                           rsi_window, rsi_overbought, rsi_oversold,
                                           macd_fast, macd_slow, macd_signal)
        backtest_dict = json.loads(backtest_data.body)

        # Get strategy description
        strategy_description = get_strategy_description(strategy)

        # Format parameters for display
        strategy_params = []
        for param, value in backtest_dict["strategy"]["params"].items():
            strategy_params.append(f"{param.replace('_', ' ').title()}: {value}")

        # Format performance metrics
        performance_metrics = [
            f"Total Return: {backtest_dict['performance']['total_return']*100:.2f}%",
            f"Annual Return: {backtest_dict['performance']['annual_return']*100:.2f}%",
            f"Sharpe Ratio: {backtest_dict['performance']['sharpe_ratio']:.2f}",
            f"Max Drawdown: {backtest_dict['performance']['max_drawdown']*100:.2f}%",
            f"Alpha: {backtest_dict['performance']['alpha']*100:.2f}%",
            f"Beta: {backtest_dict['performance']['beta']:.2f}"
        ]

        # Format benchmark metrics
        benchmark_metrics = [
            f"Total Return: {backtest_dict['benchmark']['total_return']*100:.2f}%",
            f"Annual Return: {backtest_dict['benchmark']['annual_return']*100:.2f}%"
        ]

        # Format trade details
        trade_details = []
        for trade in backtest_dict["trades"]["details"]:
            trade_date = datetime.datetime.fromisoformat(trade["date"].replace("Z", "+00:00")).strftime("%Y-%m-%d")
            trade_details.append(f"{trade_date}: {trade['type'].title()} {trade['shares']} shares at ${trade['price']:.2f} (${trade['value']:.2f})")

        return templates.TemplateResponse(
            "backtest.html",
            {
                "request": request,
                "title": f"Backtest Results: {ticker} - {strategy.upper()}",
                "ticker": ticker,
                "strategy": strategy.upper(),
                "strategy_description": strategy_description,
                "strategy_params": strategy_params,
                "period": period,
                "performance_metrics": performance_metrics,
                "benchmark_metrics": benchmark_metrics,
                "trade_count": backtest_dict["trades"]["count"],
                "trade_details": trade_details,
                "chart_image": backtest_dict["chart"]
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error viewing backtest for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing backtest for {ticker}: {str(e)}")


def get_strategy_description(strategy: str) -> str:
    """Get a description of a trading strategy."""
    if strategy == "ma_cross":
        return "Moving Average Crossover strategy generates buy signals when the short-term moving average crosses above the long-term moving average, and sell signals when it crosses below."
    elif strategy == "rsi":
        return "Relative Strength Index (RSI) strategy generates buy signals when the RSI falls below the oversold threshold, and sell signals when it rises above the overbought threshold."
    elif strategy == "macd":
        return "Moving Average Convergence Divergence (MACD) strategy generates buy signals when the MACD line crosses above the signal line, and sell signals when it crosses below."
    else:
        return "Unknown strategy"


def generate_crypto_insights(metrics: Dict[str, Any]) -> List[str]:
    """Generate insights from cryptocurrency metrics."""
    insights = []

    # Price insights
    if metrics["price"]["ath_change"] < -50:
        insights.append(f"Currently {abs(metrics['price']['ath_change']):.1f}% below all-time high, suggesting potential recovery room")
    elif metrics["price"]["ath_change"] > -10:
        insights.append(f"Near all-time high prices, only {abs(metrics['price']['ath_change']):.1f}% below ATH")

    # Market insights
    if metrics["market"]["market_cap_rank"] <= 10:
        insights.append(f"Top {metrics['market']['market_cap_rank']} cryptocurrency by market capitalization")

    if metrics["market"]["circulating_supply"] and metrics["market"]["max_supply"]:
        supply_ratio = metrics["market"]["circulating_supply"] / metrics["market"]["max_supply"] * 100
        if supply_ratio > 90:
            insights.append(f"Limited supply growth potential with {supply_ratio:.1f}% of maximum supply already in circulation")
        elif supply_ratio < 50:
            insights.append(f"Significant supply growth expected with only {supply_ratio:.1f}% of maximum supply in circulation")

    # Performance insights
    if metrics["performance"]["volatility"] > 100:
        insights.append(f"Extremely high volatility of {metrics['performance']['volatility']:.1f}% indicates significant price risk")
    elif metrics["performance"]["volatility"] < 50:
        insights.append(f"Relatively low volatility of {metrics['performance']['volatility']:.1f}% for a cryptocurrency")

    if metrics["performance"]["sharpe_ratio"] > 1:
        insights.append(f"Favorable risk-adjusted returns with Sharpe ratio of {metrics['performance']['sharpe_ratio']:.2f}")
    elif metrics["performance"]["sharpe_ratio"] < 0:
        insights.append(f"Poor risk-adjusted returns with negative Sharpe ratio of {metrics['performance']['sharpe_ratio']:.2f}")

    if metrics["performance"]["max_drawdown"] < -70:
        insights.append(f"Extreme historical drawdown of {abs(metrics['performance']['max_drawdown']):.1f}% indicates high historical volatility")

    # Technical insights
    if metrics["technical"]["rsi"] > 70:
        insights.append(f"RSI of {metrics['technical']['rsi']:.1f} suggests the cryptocurrency may be overbought")
    elif metrics["technical"]["rsi"] < 30:
        insights.append(f"RSI of {metrics['technical']['rsi']:.1f} suggests the cryptocurrency may be oversold")

    # Recent performance
    if metrics["changes"]["price_change_percentage_24h"] > 10:
        insights.append(f"Strong 24-hour performance with {metrics['changes']['price_change_percentage_24h']:.1f}% price increase")
    elif metrics["changes"]["price_change_percentage_24h"] < -10:
        insights.append(f"Weak 24-hour performance with {metrics['changes']['price_change_percentage_24h']:.1f}% price decrease")

    # If no insights were generated, add a default one
    if not insights:
        insights.append("No significant insights found based on the current metrics.")

    return insights


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
