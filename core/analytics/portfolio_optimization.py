"""
Portfolio optimization module for financial analysis.
This module provides portfolio optimization tools using PyPortfolioOpt and riskfolio-lib.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import riskfolio as rp
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions, DiscreteAllocation
from typing import Dict, List, Union, Optional, Tuple, Any

class PortfolioOptimizer:
    """
    A class that provides portfolio optimization tools.
    Combines functionality from PyPortfolioOpt and riskfolio-lib.
    """
    
    @staticmethod
    def optimize_portfolio(prices_df: pd.DataFrame, 
                          method: str = 'sharpe', 
                          risk_free_rate: float = 0.02,
                          weight_bounds: Tuple[float, float] = (0, 1)) -> Dict[str, Any]:
        """
        Optimize a portfolio using PyPortfolioOpt.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            method: Optimization method ('sharpe', 'min_volatility', 'max_return', 'efficient_risk', 'efficient_return')
            risk_free_rate: Risk-free rate
            weight_bounds: Tuple of (min_weight, max_weight) for each asset
            
        Returns:
            Dict containing weights and performance metrics
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Create the Efficient Frontier object
            ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            
            # Add objective for diversification
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            # Optimize based on the selected method
            if method == 'sharpe':
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif method == 'min_volatility':
                weights = ef.min_volatility()
            elif method == 'max_return':
                weights = ef.max_return()
            elif method == 'efficient_risk':
                target_volatility = 0.15  # Example target volatility
                weights = ef.efficient_risk(target_volatility)
            elif method == 'efficient_return':
                target_return = 0.20  # Example target return
                weights = ef.efficient_return(target_return)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Clean weights (round to 4 decimal places)
            cleaned_weights = ef.clean_weights()
            
            # Get portfolio performance metrics
            expected_return, volatility, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            # Calculate additional metrics
            portfolio_return = sum(mu * cleaned_weights)
            portfolio_variance = np.dot(np.dot(cleaned_weights, S), cleaned_weights)
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate discrete allocation for a given portfolio value
            portfolio_value = 10000  # Example portfolio value
            latest_prices = prices_df.iloc[-1]
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            
            return {
                'weights': cleaned_weights,
                'performance': {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_volatility
                },
                'allocation': {
                    'discrete_allocation': allocation,
                    'leftover': leftover
                }
            }
        except Exception as e:
            raise ValueError(f"Portfolio optimization failed: {str(e)}")
    
    @staticmethod
    def optimize_portfolio_riskfolio(prices_df: pd.DataFrame,
                                    method: str = 'hrp',
                                    risk_measure: str = 'MV',
                                    risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Optimize a portfolio using riskfolio-lib.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            method: Optimization method ('hrp', 'herc', 'nco')
            risk_measure: Risk measure ('MV', 'MAD', 'MSV', 'CVaR', 'EVaR', 'WR', 'FLPM', 'SLPM', 'CDaR', 'UCI', 'EDaR')
            risk_free_rate: Risk-free rate
            
        Returns:
            Dict containing weights and performance metrics
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Create portfolio object
            port = rp.Portfolio(returns=returns_df)
            
            # Calculate covariance matrix
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Optimize based on the selected method
            if method == 'hrp':
                # Hierarchical Risk Parity
                weights = port.optimization(model='HRP', rm=risk_measure, rf=risk_free_rate)
            elif method == 'herc':
                # Hierarchical Equal Risk Contribution
                weights = port.optimization(model='HERC', rm=risk_measure, rf=risk_free_rate)
            elif method == 'nco':
                # Nested Clustered Optimization
                weights = port.optimization(model='NCO', rm=risk_measure, rf=risk_free_rate, obj='Sharpe')
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio performance
            port.set_weights(weights.iloc[0].values)
            ret, vol, sharpe = port.portfolio_performance(rm=risk_measure, rf=risk_free_rate)
            
            # Convert weights to dictionary
            weights_dict = weights.iloc[0].to_dict()
            
            # Calculate discrete allocation for a given portfolio value
            portfolio_value = 10000  # Example portfolio value
            latest_prices = prices_df.iloc[-1]
            da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            
            return {
                'weights': weights_dict,
                'performance': {
                    'expected_return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                },
                'allocation': {
                    'discrete_allocation': allocation,
                    'leftover': leftover
                }
            }
        except Exception as e:
            raise ValueError(f"Portfolio optimization failed: {str(e)}")
    
    @staticmethod
    def generate_efficient_frontier(prices_df: pd.DataFrame, 
                                   risk_free_rate: float = 0.02,
                                   points: int = 50) -> Dict[str, Any]:
        """
        Generate the efficient frontier for a portfolio.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            risk_free_rate: Risk-free rate
            points: Number of points on the efficient frontier
            
        Returns:
            Dict containing efficient frontier data
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Create the Efficient Frontier object
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            # Generate efficient frontier
            returns = []
            volatilities = []
            sharpe_ratios = []
            weights_list = []
            
            # Find the range of returns
            ef_min_vol = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef_min_vol.min_volatility()
            min_ret, min_vol, _ = ef_min_vol.portfolio_performance()
            
            ef_max_ret = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef_max_ret.max_return()
            max_ret, max_vol, _ = ef_max_ret.portfolio_performance()
            
            # Generate points along the efficient frontier
            target_returns = np.linspace(min_ret, max_ret, points)
            
            for target_return in target_returns:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                try:
                    ef.efficient_return(target_return)
                    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                    returns.append(ret)
                    volatilities.append(vol)
                    sharpe_ratios.append(sharpe)
                    weights_list.append(ef.clean_weights())
                except:
                    continue
            
            # Find the optimal portfolio (maximum Sharpe ratio)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            opt_ret, opt_vol, opt_sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            opt_weights = ef.clean_weights()
            
            # Find the minimum volatility portfolio
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.min_volatility()
            min_vol_ret, min_vol_vol, min_vol_sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            min_vol_weights = ef.clean_weights()
            
            # Calculate the capital market line
            cml_returns = np.linspace(risk_free_rate, max(returns) * 1.5, 100)
            cml_volatilities = [(r - risk_free_rate) / opt_sharpe for r in cml_returns]
            
            return {
                'efficient_frontier': {
                    'returns': returns,
                    'volatilities': volatilities,
                    'sharpe_ratios': sharpe_ratios,
                    'weights': weights_list
                },
                'optimal_portfolio': {
                    'return': opt_ret,
                    'volatility': opt_vol,
                    'sharpe_ratio': opt_sharpe,
                    'weights': opt_weights
                },
                'min_volatility_portfolio': {
                    'return': min_vol_ret,
                    'volatility': min_vol_vol,
                    'sharpe_ratio': min_vol_sharpe,
                    'weights': min_vol_weights
                },
                'capital_market_line': {
                    'returns': cml_returns.tolist(),
                    'volatilities': cml_volatilities
                },
                'risk_free_rate': risk_free_rate
            }
        except Exception as e:
            raise ValueError(f"Efficient frontier generation failed: {str(e)}")
    
    @staticmethod
    def plot_efficient_frontier(prices_df: pd.DataFrame, 
                               risk_free_rate: float = 0.02,
                               points: int = 50,
                               show_assets: bool = True,
                               show_cml: bool = True) -> plt.Figure:
        """
        Plot the efficient frontier for a portfolio.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            risk_free_rate: Risk-free rate
            points: Number of points on the efficient frontier
            show_assets: Whether to show individual assets
            show_cml: Whether to show the capital market line
            
        Returns:
            Matplotlib figure
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Get efficient frontier data
            ef_data = PortfolioOptimizer.generate_efficient_frontier(prices_df, risk_free_rate, points)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            ax.plot(ef_data['efficient_frontier']['volatilities'], 
                    ef_data['efficient_frontier']['returns'], 
                    'b-', linewidth=2, label='Efficient Frontier')
            
            # Plot optimal portfolio (maximum Sharpe ratio)
            ax.scatter(ef_data['optimal_portfolio']['volatility'], 
                      ef_data['optimal_portfolio']['return'], 
                      marker='*', s=200, color='r', label='Maximum Sharpe Ratio')
            
            # Plot minimum volatility portfolio
            ax.scatter(ef_data['min_volatility_portfolio']['volatility'], 
                      ef_data['min_volatility_portfolio']['return'], 
                      marker='o', s=150, color='g', label='Minimum Volatility')
            
            # Plot individual assets if requested
            if show_assets:
                returns = expected_returns.mean_historical_return(prices_df)
                cov = risk_models.sample_cov(prices_df)
                for i, asset in enumerate(prices_df.columns):
                    asset_return = returns[i]
                    asset_volatility = np.sqrt(cov.iloc[i, i])
                    ax.scatter(asset_volatility, asset_return, marker='o', s=50, 
                              label=f'{asset}', alpha=0.7)
            
            # Plot capital market line if requested
            if show_cml:
                ax.plot(ef_data['capital_market_line']['volatilities'], 
                        ef_data['capital_market_line']['returns'], 
                        'r--', linewidth=1.5, label='Capital Market Line')
                
                # Plot risk-free rate
                ax.scatter(0, risk_free_rate, marker='o', s=80, color='k', label='Risk-Free Rate')
            
            # Set labels and title
            ax.set_xlabel('Volatility (Standard Deviation)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            return fig
        except Exception as e:
            raise ValueError(f"Efficient frontier plotting failed: {str(e)}")
    
    @staticmethod
    def plot_hierarchical_clusters(prices_df: pd.DataFrame, method: str = 'hrp') -> plt.Figure:
        """
        Plot hierarchical clusters for portfolio optimization.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            method: Clustering method ('hrp', 'herc')
            
        Returns:
            Matplotlib figure
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Create portfolio object
            port = rp.Portfolio(returns=returns_df)
            
            # Calculate covariance matrix
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Plot hierarchical clusters
            if method == 'hrp':
                # Hierarchical Risk Parity
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                port.plot_clusters(model='HRP', linkage='ward', k=None, max_k=10, leaf_order=True, ax=ax)
                ax.set_title('Hierarchical Risk Parity Clusters')
            elif method == 'herc':
                # Hierarchical Equal Risk Contribution
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                port.plot_clusters(model='HERC', linkage='ward', k=None, max_k=10, leaf_order=True, ax=ax)
                ax.set_title('Hierarchical Equal Risk Contribution Clusters')
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            return fig
        except Exception as e:
            raise ValueError(f"Hierarchical clustering failed: {str(e)}")
    
    @staticmethod
    def plot_risk_contribution(prices_df: pd.DataFrame, method: str = 'hrp', risk_measure: str = 'MV') -> plt.Figure:
        """
        Plot risk contribution for portfolio optimization.
        
        Args:
            prices_df: DataFrame with asset prices (columns are assets, index is dates)
            method: Optimization method ('hrp', 'herc', 'nco')
            risk_measure: Risk measure ('MV', 'MAD', 'MSV', 'CVaR', 'EVaR', 'WR', 'FLPM', 'SLPM', 'CDaR', 'UCI', 'EDaR')
            
        Returns:
            Matplotlib figure
        """
        if prices_df.empty:
            raise ValueError("Empty price data provided")
            
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Create portfolio object
            port = rp.Portfolio(returns=returns_df)
            
            # Calculate covariance matrix
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Optimize based on the selected method
            if method == 'hrp':
                # Hierarchical Risk Parity
                weights = port.optimization(model='HRP', rm=risk_measure)
            elif method == 'herc':
                # Hierarchical Equal Risk Contribution
                weights = port.optimization(model='HERC', rm=risk_measure)
            elif method == 'nco':
                # Nested Clustered Optimization
                weights = port.optimization(model='NCO', rm=risk_measure, obj='Sharpe')
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Set weights
            port.set_weights(weights.iloc[0].values)
            
            # Plot risk contribution
            fig, ax = plt.subplots(figsize=(10, 6))
            port.plot_risk_con(rm=risk_measure, ax=ax)
            ax.set_title(f'Risk Contribution - {method.upper()} ({risk_measure})')
            
            return fig
        except Exception as e:
            raise ValueError(f"Risk contribution plotting failed: {str(e)}")
