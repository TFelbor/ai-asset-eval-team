.PHONY: setup run test clean lint format backtest optimize

# Default target
all: setup

# Setup the project with enhanced dependencies
setup:
	python -m pip install -e .
	python -m pip install -r requirements.txt
	./install_requirements.sh

# Run the dashboard (default)
run:
	./run.sh

# Run the dashboard with browser auto-open
run-browser:
	./run.sh --browser

# Run the dashboard with debug mode
run-debug:
	./run.sh --debug

# Run the dashboard in production mode
run-production:
	./run.sh --production

# Run the dashboard with browser and debug mode
run-dev:
	./run.sh --browser --debug

# Legacy commands for backward compatibility
run-dashboard: run
run-dashboard-browser: run-browser
run-streamlit: run
run-streamlit-browser: run-browser
streamlit: run

# Run tests
test:
	pytest tests/

# Clean up temporary files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.c" -delete
	find . -type f -name "*.h" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf core/data/cache/*
	rm -rf core/data/temp/*
	rm -rf core/logs/*

# Run linting
lint:
	flake8 .

# Format code
format:
	black .

# Run backtesting analysis
backtest:
    python -m core.analytics.enhanced_analytics

# Run portfolio optimization
optimize:
    python -m core.analytics.portfolio_optimization

# Run machine learning analysis
ml-analysis:
    python -m core.analytics.ml_analysis

# Run all analyses
analyze-all: backtest optimize ml-analysis
