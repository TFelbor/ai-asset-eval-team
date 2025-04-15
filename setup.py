from setuptools import setup, find_packages

setup(
    name="ai_finance_dashboard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Main dependencies
        "streamlit>=1.29.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.1",
        "numpy>=1.26.0",
        "plotly>=5.17.0",
        "yfinance>=0.2.31",
        "pycoingecko>=3.3.0",
        "requests>=2.31.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.3.1",

        # Enhanced logging
        "loguru",

        # Optional dependencies for advanced features
        # Uncomment as needed
        # "backtrader>=1.9.78",
        # "PyPortfolioOpt>=1.5.5",
        # "empyrical>=0.5.5",
        # "quantstats>=0.0.59",
        # "pandas-ta>=0.3.14b0",
        # "mplfinance>=0.12.10b0",
        # "pandas-datareader>=0.10.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
