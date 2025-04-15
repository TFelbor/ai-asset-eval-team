# AI Finance Dashboard

An AI-powered dashboard for comprehensive analysis of different financial securities including stocks, cryptocurrencies, REITs, and ETFs.

## Overview

The AI Finance Dashboard is a Streamlit-based application that provides in-depth analysis of various financial assets. It uses specialized AI agents to analyze different aspects of financial securities and combines them into a cohesive analysis. The dashboard features a modern dark-themed UI with vibrant accent colors, interactive charts, and educational content to help users make informed investment decisions.

## Live Demos

### Dashboard Home 

![](https://github.com/TFelbor/ai-asset-eval-team/blob/main/demos/dashboard_home.gif)


### Stock Analysis

![](https://github.com/TFelbor/ai-asset-eval-team/blob/main/demos/stock_analysis.gif)

## Project Structure

The project has been reorganized and cleaned up for better maintainability and clarity:

```
ai_asset_eval_team/
├── core/                   # Core functionality
│   ├── analytics/          # Chart generation and data analysis
│   ├── api/                # API integrations
│   ├── config/             # Configuration files
│   ├── data/               # Data management and caching
│   │   ├── cache/          # Cached API responses
│   │   └── temp/           # Temporary files
│   ├── logs/               # Log files
│   ├── models/             # Data models
│   ├── services/           # Service modules
│   ├── static/             # Static assets (CSS, JS, images)
│   ├── teams/              # Analysis teams
│   ├── templates/          # HTML templates
│   ├── tests/              # Unit tests
│   ├── ui/                 # UI components
│   ├── utils/              # Utility functions
│   ├── dashboard.py        # Main dashboard implementation
│   └── main.py             # Core entry point
├── deprecated/             # Deprecated code (no longer used)
├── .streamlit/             # Streamlit configuration
├── run.py                  # Unified Python runner
├── run.sh                  # Unified shell script runner
└── README.md               # This file
```

## Features

- **Multi-agent AI System**: Specialized AI agents analyze different aspects of financial securities
- **Comprehensive Analysis**: Combines fundamental, technical, and macroeconomic factors
- **Multiple Security Types**: Support for stocks, cryptocurrencies, REITs, and ETFs
- **Interactive Dashboard**: Visual representation of analysis with charts and insights
- **In-Page Chart Tabs**: Charts open in tabs that stay on the same page with ability to be minimized
- **Advanced Analytics**: In-depth metrics including volatility, alpha, beta, Sharpe ratio, and more
- **Machine Learning Analysis**: ML-based price predictions and trend analysis
- **Comparison Tool**: Compare multiple securities with side-by-side metrics and charts
  - Same asset type comparison (e.g., comparing multiple stocks)
  - Cross-asset type comparison (e.g., comparing stocks, crypto, REITs, ETFs)
- **Backtesting**: Test trading strategies against historical data with performance metrics
  - Multiple strategy options (Moving Average Crossover, RSI, MACD)
  - Customizable parameters for each strategy
  - Detailed performance metrics and visualizations
- **Advanced Chart Types**:
  - Candlestick charts with volume
  - Technical analysis charts with indicators (RSI, MACD, Bollinger Bands)
  - Price & volume charts
  - Correlation matrices for comparing multiple assets
- **Financial News Integration**:
  - Market news
  - Asset-specific news
  - Trending topics analysis
- **Mobile Optimized**: Responsive design that works on desktop, tablet, and mobile devices
- **Dark Mode**: Modern dark-themed UI with vibrant accent colors for better readability
- **Enhanced Caching**: Two-level caching system (memory + file) for faster access and reduced API calls
- **Advanced Logging**: Comprehensive logging system with structured logging, color-coded output, and performance tracking
  - Loguru integration for enhanced logging features
  - Fallback to standard logging when Loguru is not available
  - Performance tracking for critical operations
  - Structured logging with detailed context information
- **Educational Content**: Detailed financial term definitions and explanations for educational purposes
- **Debug Mode**: Toggle to display detailed request/response information for developers
- **Server Management**: Built-in server shutdown functionality for clean application termination

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_asset_eval_team.git
   cd ai_asset_eval_team
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install additional required packages for enhanced features:
   ```bash
   ./install_requirements.sh
   ```
   This will install Loguru for enhanced logging features.

5. Set up API keys:
   - Copy the example config file and edit it:
     ```bash
     cp config/local.py.example config/local.py
     # Edit config/local.py with your API keys
     ```
   - Alternatively, set environment variables:
     ```bash
     export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key_here"
     export COINGECKO_API_KEY="your_coingecko_key_here"
     export NEWS_API_KEY="your_news_api_key_here"
     ```

### Running the Dashboard

The simplest way to run the dashboard is using the provided unified shell script:

```bash
./run.sh
```

You can also use the Python script directly:

```bash
./run.py
```

Both scripts accept the following options:

```bash
# Open browser automatically
./run.sh --browser

# Enable debug mode
./run.sh --debug

# Run on a specific port
./run.sh --port=8502

# Use a specific theme (dark or light)
./run.sh --theme=light

# Run in production mode
./run.sh --production
```

Or use the Makefile:

```bash
make streamlit
```

This will automatically open your browser to the dashboard at http://localhost:8501.

## Usage

### Main Dashboard
1. Select an asset type from the sidebar (Stock, Cryptocurrency, ETF, or REIT)
2. Enter a ticker symbol or ID in the input field
3. Click the "Analyze" button
4. View the analysis results in the tabs:
   - **Overview**: Key metrics and summary
   - **Charts**: Interactive charts for different aspects of the asset
   - **Insights**: AI-generated insights about the asset
   - **ML Analysis**: Machine learning-based predictions and trend analysis
   - **News**: Latest news related to the asset
   - **Raw Data**: Detailed data for advanced users

### Comparison Tool
1. Click the "Asset Comparison Tool" button in the sidebar
2. Choose between "Same Asset Type" or "Cross-Asset Type" comparison
3. For same asset type comparison:
   - Select the asset type (Stock, Cryptocurrency, ETF, or REIT)
   - Enter ticker symbols and click "Add to Comparison"
   - Select chart type and time period
   - Click "Generate Comparison"
4. For cross-asset type comparison:
   - Enter ticker symbols for different asset types
   - Click "Generate Comparison"

### Backtesting Tool
1. Click the "Backtesting Tool" button in the sidebar
2. Enter a stock ticker symbol
3. Select a trading strategy (Moving Average Crossover, RSI, MACD)
4. Configure strategy parameters
5. Set initial capital amount
6. Click "Run Backtest"
7. View results in the Performance, Trades, Chart, and Raw Data tabs

### Financial News
1. Click the "Financial News" button in the sidebar
2. Browse news in different tabs:
   - Market News: General financial market news
   - Asset-Specific News: News related to a specific asset
   - Trending Topics: Analysis of trending financial topics

## Asset Types

### Stocks
- Enter standard ticker symbols (e.g., AAPL, MSFT, GOOGL)
- View fundamental metrics, technical indicators, and price history
- Get AI-generated insights about the company's financial health

### Cryptocurrencies
- Enter cryptocurrency tickers (e.g., BTC, ETH) or IDs (e.g., bitcoin, ethereum)
- View market metrics, price history, and volatility
- Get AI-generated insights about the cryptocurrency's market position
- Access real-time data from CoinGecko API

### REITs (Real Estate Investment Trusts)
- Enter REIT ticker symbols (e.g., VNQ, O, AMT)
- View property type, dividend yield, and other REIT-specific metrics
- Get AI-generated insights about the REIT's performance

### ETFs (Exchange-Traded Funds)
- Enter ETF ticker symbols (e.g., SPY, QQQ, VTI)
- View fund composition, expense ratio, and performance metrics
- Get AI-generated insights about the ETF's strategy and performance

## Data Sources

The dashboard uses data from the following sources:

- **Yahoo Finance**: Stocks, ETFs, REITs
- **CoinGecko**: Cryptocurrencies
- **Alpha Vantage**: Additional financial data
- **News API**: Financial news

## Recent Updates

### UI and Navigation Improvements

1. **Enhanced UI**: Improved visual design with better organization and styling
2. **Fixed Navigation**: All "Return to Home" buttons now correctly return to the landing page
3. **Sidebar Organization**: Reorganized sidebar with clear sections and improved button styling
4. **Welcome Screen**: Redesigned welcome screen with feature highlights and getting started guide

### Technical Improvements

1. **Enhanced Logging System**: Implemented advanced logging with Loguru integration
   - Added structured logging with detailed context information
   - Added performance tracking for critical operations
   - Added fallback to standard logging when Loguru is not available
   - Added installation script for required packages
2. **Crypto ML Analysis**: Fixed array length mismatch errors in cryptocurrency ML analysis
3. **Enhanced Data Fetching**: Improved data fetching with better error handling and fallbacks
4. **Server Management**: Added server shutdown functionality for clean application termination
5. **Documentation**: Updated documentation to reflect current features and implementation

### Project Cleanup and Organization

The project has been cleaned up and organized for better maintainability:

1. **Core Directory Structure**: Essential files have been organized into a core directory with clear separation of concerns
2. **Deprecated Code Removal**: Redundant and deprecated files have been moved to the deprecated directory
3. **Code Cleanup**: Unnecessary code sections have been removed
4. **Documentation Updates**: README and documentation have been updated to reflect the current state of the project
5. **Unified Run Scripts**: All run scripts have been consolidated into a single unified script (run.py/run.sh)
6. **Enhanced Setup**: Improved setup process with better dependency management
7. **Git Configuration**: Added proper .gitignore and .gitattributes files for better version control
8. **Removed Duplicate Repository**: Cleaned up duplicate repository files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
