"""
Risk analysis module for financial analysis.
This module provides risk analysis tools using empyrical and quantstats.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import empyrical as ep
from typing import Dict, List, Union, Optional, Tuple, Any

class RiskAnalyzer:
    """
    A class that provides risk analysis tools.
    Combines functionality from empyrical and quantstats.
    """
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series, 
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a returns series.
        
        Args:
            returns: Series of asset returns
            benchmark_returns: Series of benchmark returns (optional)
            risk_free: Risk-free rate
            
        Returns:
            Dict of risk metrics
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = ep.cum_returns_final(returns)
            metrics['annual_return'] = ep.annual_return(returns)
            metrics['annual_volatility'] = ep.annual_volatility(returns)
            metrics['sharpe_ratio'] = ep.sharpe_ratio(returns, risk_free=risk_free)
            metrics['calmar_ratio'] = ep.calmar_ratio(returns)
            metrics['stability'] = ep.stability_of_timeseries(returns)
            metrics['max_drawdown'] = ep.max_drawdown(returns)
            metrics['omega_ratio'] = ep.omega_ratio(returns, risk_free=risk_free)
            metrics['sortino_ratio'] = ep.sortino_ratio(returns, required_return=risk_free)
            
            # Value at Risk (VaR)
            metrics['var_95'] = qs.stats.var(returns, sigma=2.0)
            metrics['var_99'] = qs.stats.var(returns, sigma=2.5)
            metrics['cvar_95'] = qs.stats.cvar(returns, sigma=2.0)
            
            # Additional metrics from quantstats
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            metrics['expected_return'] = qs.stats.expected_return(returns)
            metrics['expected_shortfall'] = qs.stats.expected_shortfall(returns)
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
            metrics['value_at_risk'] = qs.stats.value_at_risk(returns)
            
            # Benchmark-related metrics (if benchmark is provided)
            if benchmark_returns is not None:
                metrics['alpha'] = ep.alpha(returns, benchmark_returns, risk_free=risk_free)
                metrics['beta'] = ep.beta(returns, benchmark_returns)
                metrics['information_ratio'] = ep.excess_sharpe(returns, benchmark_returns)
                metrics['capture_ratio'] = qs.stats.capture(returns, benchmark_returns)
                metrics['up_capture'] = qs.stats.up_capture(returns, benchmark_returns)
                metrics['down_capture'] = qs.stats.down_capture(returns, benchmark_returns)
                metrics['up_down_ratio'] = qs.stats.up_down_capture(returns, benchmark_returns)
                metrics['correlation'] = qs.stats.correlation(returns, benchmark_returns)
                metrics['r_squared'] = qs.stats.r_squared(returns, benchmark_returns)
                metrics['treynor_ratio'] = qs.stats.treynor_ratio(returns, benchmark_returns, risk_free=risk_free)
            
            # Validate metrics
            return {k: v for k, v in metrics.items() if not (np.isnan(v) or np.isinf(v))}
        except Exception as e:
            raise ValueError(f"Risk metrics calculation failed: {str(e)}")
    
    @staticmethod
    def generate_risk_report(returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free: float = 0.0,
                            title: str = 'Risk Analysis Report') -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.
        
        Args:
            returns: Series of asset returns
            benchmark_returns: Series of benchmark returns (optional)
            risk_free: Risk-free rate
            title: Report title
            
        Returns:
            Dict containing risk report data
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Calculate risk metrics
            metrics = RiskAnalyzer.calculate_risk_metrics(returns, benchmark_returns, risk_free)
            
            # Calculate rolling metrics
            rolling_metrics = {}
            rolling_metrics['rolling_sharpe'] = qs.stats.rolling_sharpe(returns, window=252)
            rolling_metrics['rolling_sortino'] = qs.stats.rolling_sortino(returns, window=252)
            rolling_metrics['rolling_volatility'] = qs.stats.rolling_volatility(returns, window=252)
            
            if benchmark_returns is not None:
                rolling_metrics['rolling_beta'] = qs.stats.rolling_beta(returns, benchmark_returns, window=252)
                rolling_metrics['rolling_alpha'] = qs.stats.rolling_alpha(returns, benchmark_returns, window=252)
            
            # Calculate drawdowns
            drawdowns = qs.stats.to_drawdown_series(returns)
            top_drawdowns = qs.stats.drawdown_details(returns)
            
            # Calculate monthly and yearly returns
            monthly_returns = qs.stats.monthly_returns(returns)
            yearly_returns = qs.stats.yearly_returns(returns)
            
            # Calculate distribution statistics
            best_day = qs.stats.best(returns)
            worst_day = qs.stats.worst(returns)
            win_rate = qs.stats.win_rate(returns)
            win_loss_ratio = qs.stats.win_loss_ratio(returns)
            
            # Compile report
            report = {
                'title': title,
                'metrics': metrics,
                'rolling_metrics': {k: v.to_dict() for k, v in rolling_metrics.items()},
                'drawdowns': {
                    'drawdown_series': drawdowns.to_dict(),
                    'top_drawdowns': top_drawdowns.to_dict('records') if isinstance(top_drawdowns, pd.DataFrame) else []
                },
                'returns': {
                    'monthly': monthly_returns.to_dict(),
                    'yearly': yearly_returns.to_dict()
                },
                'distribution': {
                    'best_day': best_day,
                    'worst_day': worst_day,
                    'win_rate': win_rate,
                    'win_loss_ratio': win_loss_ratio
                }
            }
            
            return report
        except Exception as e:
            raise ValueError(f"Risk report generation failed: {str(e)}")
    
    @staticmethod
    def plot_risk_metrics(returns: pd.Series, 
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free: float = 0.0,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot key risk metrics.
        
        Args:
            returns: Series of asset returns
            benchmark_returns: Series of benchmark returns (optional)
            risk_free: Risk-free rate
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot cumulative returns
            qs.plots.returns(returns, benchmark_returns, ax=axes[0, 0])
            axes[0, 0].set_title('Cumulative Returns')
            
            # Plot drawdowns
            qs.plots.drawdown(returns, ax=axes[0, 1])
            axes[0, 1].set_title('Drawdowns')
            
            # Plot monthly returns heatmap
            qs.plots.monthly_heatmap(returns, ax=axes[1, 0])
            axes[1, 0].set_title('Monthly Returns')
            
            # Plot rolling volatility
            qs.plots.rolling_volatility(returns, benchmark_returns, ax=axes[1, 1])
            axes[1, 1].set_title('Rolling Volatility')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        except Exception as e:
            raise ValueError(f"Risk metrics plotting failed: {str(e)}")
    
    @staticmethod
    def plot_rolling_metrics(returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            window: int = 252,
                            figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot rolling risk metrics.
        
        Args:
            returns: Series of asset returns
            benchmark_returns: Series of benchmark returns (optional)
            window: Rolling window size
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=figsize)
            
            # Plot rolling Sharpe ratio
            qs.plots.rolling_sharpe(returns, benchmark_returns, window=window, ax=axes[0])
            axes[0].set_title(f'Rolling Sharpe Ratio ({window} days)')
            
            # Plot rolling Sortino ratio
            qs.plots.rolling_sortino(returns, window=window, ax=axes[1])
            axes[1].set_title(f'Rolling Sortino Ratio ({window} days)')
            
            # Plot rolling beta (if benchmark is provided)
            if benchmark_returns is not None:
                qs.plots.rolling_beta(returns, benchmark_returns, window=window, ax=axes[2])
                axes[2].set_title(f'Rolling Beta ({window} days)')
            else:
                # If no benchmark, plot rolling volatility instead
                qs.plots.rolling_volatility(returns, window=window, ax=axes[2])
                axes[2].set_title(f'Rolling Volatility ({window} days)')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        except Exception as e:
            raise ValueError(f"Rolling metrics plotting failed: {str(e)}")
    
    @staticmethod
    def plot_drawdown_periods(returns: pd.Series, top_n: int = 5, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot top drawdown periods.
        
        Args:
            returns: Series of asset returns
            top_n: Number of top drawdowns to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            
            # Plot underwater chart
            qs.plots.drawdowns_periods(returns, ax=axes[0])
            axes[0].set_title('Underwater Chart')
            
            # Plot drawdown details
            qs.plots.drawdown_details(returns, top_n=top_n, ax=axes[1])
            axes[1].set_title(f'Top {top_n} Drawdown Periods')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        except Exception as e:
            raise ValueError(f"Drawdown periods plotting failed: {str(e)}")
    
    @staticmethod
    def plot_return_distribution(returns: pd.Series, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot return distribution.
        
        Args:
            returns: Series of asset returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            
            # Plot histogram
            qs.plots.histogram(returns, ax=axes[0])
            axes[0].set_title('Return Distribution')
            
            # Plot daily returns
            qs.plots.daily_returns(returns, ax=axes[1])
            axes[1].set_title('Daily Returns')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        except Exception as e:
            raise ValueError(f"Return distribution plotting failed: {str(e)}")
    
    @staticmethod
    def generate_full_report(returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free: float = 0.0,
                            title: str = 'Performance Report',
                            output_file: Optional[str] = None) -> None:
        """
        Generate a full HTML performance report using quantstats.
        
        Args:
            returns: Series of asset returns
            benchmark_returns: Series of benchmark returns (optional)
            risk_free: Risk-free rate
            title: Report title
            output_file: Output file path (if None, report is displayed in browser)
            
        Returns:
            None
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Generate report
            if benchmark_returns is not None:
                qs.reports.html(returns, benchmark_returns, risk_free=risk_free, 
                               title=title, output=output_file)
            else:
                qs.reports.html(returns, risk_free=risk_free, 
                               title=title, output=output_file)
        except Exception as e:
            raise ValueError(f"Full report generation failed: {str(e)}")
    
    @staticmethod
    def calculate_stress_test(returns: pd.Series, 
                             scenarios: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, float]:
        """
        Perform stress testing on returns series.
        
        Args:
            returns: Series of asset returns
            scenarios: Dict of scenario names and date ranges (if None, default scenarios are used)
            
        Returns:
            Dict of scenario results
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Default scenarios if none provided
            if scenarios is None:
                scenarios = {
                    'Financial Crisis': ('2008-09-01', '2009-03-31'),
                    'COVID-19': ('2020-02-19', '2020-03-23'),
                    'Dot-com Bubble': ('2000-03-10', '2002-10-09'),
                    'Black Monday': ('1987-10-19', '1987-10-19'),
                    'Flash Crash': ('2010-05-06', '2010-05-06'),
                    'Brexit Vote': ('2016-06-23', '2016-06-24'),
                    '2018 Q4 Selloff': ('2018-10-01', '2018-12-24')
                }
            
            results = {}
            
            # Calculate returns during each scenario
            for scenario_name, (start_date, end_date) in scenarios.items():
                try:
                    # Check if dates are within the returns index
                    if pd.Timestamp(start_date) >= returns.index.min() and pd.Timestamp(end_date) <= returns.index.max():
                        scenario_returns = returns.loc[start_date:end_date]
                        results[scenario_name] = ep.cum_returns_final(scenario_returns)
                except:
                    # Skip scenarios that don't apply to the data
                    continue
            
            return results
        except Exception as e:
            raise ValueError(f"Stress test calculation failed: {str(e)}")
    
    @staticmethod
    def calculate_var_cvar(returns: pd.Series, 
                          confidence_level: float = 0.95,
                          method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of asset returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Method for calculation ('historical', 'gaussian', 'cornish_fisher')
            
        Returns:
            Dict containing VaR and CVaR values
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            results = {}
            
            if method == 'historical':
                # Historical method
                var = np.percentile(returns, 100 * (1 - confidence_level))
                cvar = returns[returns <= var].mean()
                
                results['var'] = var
                results['cvar'] = cvar
                
            elif method == 'gaussian':
                # Gaussian method
                mean = returns.mean()
                std = returns.std()
                z_score = -np.sqrt(2) * np.erfinv(2 * confidence_level - 1)
                var = mean - z_score * std
                
                # Calculate CVaR for Gaussian distribution
                pdf_at_var = np.exp(-0.5 * z_score**2) / np.sqrt(2 * np.pi)
                cvar = mean - std * pdf_at_var / (1 - confidence_level)
                
                results['var'] = var
                results['cvar'] = cvar
                
            elif method == 'cornish_fisher':
                # Cornish-Fisher expansion
                mean = returns.mean()
                std = returns.std()
                skew = returns.skew()
                kurt = returns.kurtosis()
                
                z_score = -np.sqrt(2) * np.erfinv(2 * confidence_level - 1)
                cf_z = z_score + (z_score**2 - 1) * skew / 6 + (z_score**3 - 3 * z_score) * kurt / 24 - (2 * z_score**3 - 5 * z_score) * skew**2 / 36
                var = mean + cf_z * std
                
                # For CVaR, use historical method on the tail
                cvar = returns[returns <= var].mean()
                
                results['var'] = var
                results['cvar'] = cvar
                
            else:
                raise ValueError(f"Unknown VaR/CVaR method: {method}")
            
            return results
        except Exception as e:
            raise ValueError(f"VaR/CVaR calculation failed: {str(e)}")
    
    @staticmethod
    def calculate_risk_contribution(returns: pd.Series, 
                                   weights: pd.Series,
                                   cov_matrix: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate risk contribution of each asset in a portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Series of portfolio weights
            cov_matrix: Covariance matrix (if None, calculated from returns)
            
        Returns:
            Series of risk contributions
        """
        if returns.empty:
            raise ValueError("Empty returns data provided")
            
        try:
            # Calculate covariance matrix if not provided
            if cov_matrix is None:
                cov_matrix = returns.cov()
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(weights.dot(cov_matrix).dot(weights))
            
            # Calculate marginal contribution to risk
            mcr = cov_matrix.dot(weights) / portfolio_vol
            
            # Calculate risk contribution
            rc = weights * mcr
            
            # Normalize to sum to 1
            rc = rc / rc.sum()
            
            return rc
        except Exception as e:
            raise ValueError(f"Risk contribution calculation failed: {str(e)}")
