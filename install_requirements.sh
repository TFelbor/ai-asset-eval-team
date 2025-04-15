#!/bin/bash
# Install required packages for the AI Finance Dashboard

echo "Installing required packages..."

# Install enhanced logging
echo "Installing Loguru for enhanced logging..."
pip install loguru

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Loguru installed successfully!"
else
    echo "Warning: Failed to install Loguru. Enhanced logging features will not be available."
    echo "You can manually install it later with: pip install loguru"
fi

# Install optional dependencies
echo "Do you want to install optional dependencies for advanced features? (y/n)"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Installing optional dependencies..."
    pip install backtrader PyPortfolioOpt empyrical quantstats pandas-ta mplfinance pandas-datareader
    echo "Optional dependencies installed!"
fi

echo "Installation complete!"
