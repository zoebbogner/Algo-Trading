"""
Portfolio Optimization Module

This module provides advanced portfolio optimization techniques:
- Modern Portfolio Theory (Efficient Frontier)
- Risk Parity optimization
- Black-Litterman model
- Dynamic rebalancing
- Multi-objective optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timezone
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings

from ..data_models.trading import Portfolio, Position


class PortfolioOptimizer:
    """
    Advanced portfolio optimization system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Optimization parameters
        self.optimization_method = config.get('optimization_method', 'efficient_frontier')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        self.max_iterations = config.get('max_iterations', 1000)
        self.constraint_tolerance = config.get('constraint_tolerance', 1e-6)
        
        # Portfolio constraints
        self.min_weight = config.get('min_weight', 0.01)  # 1% minimum position
        self.max_weight = config.get('max_weight', 0.40)  # 40% maximum position
        self.target_volatility = config.get('target_volatility', None)
        self.target_return = config.get('target_return', None)
        
        # Rebalancing parameters
        self.rebalancing_threshold = config.get('rebalancing_threshold', 0.05)  # 5% threshold
        self.rebalancing_frequency = config.get('rebalancing_frequency', 'monthly')
        
        print(f"🎯 Portfolio Optimizer initialized")
        print(f"   Method: {self.optimization_method}")
        print(f"   Min Weight: {self.min_weight:.1%}")
        print(f"   Max Weight: {self.max_weight:.1%}")
    
    def optimize_portfolio(self, returns_data: Dict[str, np.ndarray], 
                         method: str = None, constraints: Dict = None) -> Dict:
        """
        Optimize portfolio allocation using specified method
        """
        if method is None:
            method = self.optimization_method
        
        # Prepare data
        returns_matrix, symbols = self._prepare_returns_matrix(returns_data)
        
        if returns_matrix.size == 0:
            return {'error': 'Insufficient data for optimization'}
        
        # Calculate expected returns and covariance matrix
        expected_returns = self._calculate_expected_returns(returns_matrix)
        covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
        
        # Apply constraints
        if constraints is None:
            constraints = self._get_default_constraints(len(symbols))
        
        # Optimize based on method
        if method == 'efficient_frontier':
            result = self._optimize_efficient_frontier(expected_returns, covariance_matrix, constraints)
        elif method == 'risk_parity':
            result = self._optimize_risk_parity(covariance_matrix, constraints)
        elif method == 'maximum_sharpe':
            result = self._optimize_maximum_sharpe(expected_returns, covariance_matrix, constraints)
        elif method == 'minimum_variance':
            result = self._optimize_minimum_variance(covariance_matrix, constraints)
        elif method == 'black_litterman':
            result = self._optimize_black_litterman(expected_returns, covariance_matrix, constraints)
        else:
            return {'error': f'Unknown optimization method: {method}'}
        
        # Add metadata
        result['method'] = method
        result['symbols'] = symbols
        result['constraints'] = constraints
        result['optimization_timestamp'] = datetime.now(timezone.utc)
        
        return result
    
    def _prepare_returns_matrix(self, returns_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Prepare returns data for optimization"""
        if not returns_data:
            return np.array([]), []
        
        # Find common length
        lengths = [len(returns) for returns in returns_data.values()]
        if not lengths:
            return np.array([]), []
        
        min_length = min(lengths)
        if min_length < 30:  # Need at least 30 observations
            return np.array([]), []
        
        # Create returns matrix
        symbols = list(returns_data.keys())
        returns_matrix = np.zeros((min_length, len(symbols)))
        
        for i, symbol in enumerate(symbols):
            returns_matrix[:, i] = returns_data[symbol][:min_length]
        
        return returns_matrix, symbols
    
    def _calculate_expected_returns(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate expected returns using multiple methods"""
        # Method 1: Simple mean
        simple_mean = np.mean(returns_matrix, axis=0)
        
        # Method 2: Exponentially weighted mean (more recent data has higher weight)
        weights = np.exp(np.linspace(-1, 0, returns_matrix.shape[0]))
        weights = weights / np.sum(weights)
        exp_weighted_mean = np.average(returns_matrix, axis=0, weights=weights)
        
        # Method 3: Shrinkage towards global mean
        global_mean = np.mean(simple_mean)
        shrinkage_factor = 0.3
        shrunk_mean = shrinkage_factor * global_mean + (1 - shrinkage_factor) * simple_mean
        
        # Use shrunk mean as default
        return shrunk_mean
    
    def _calculate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix with shrinkage"""
        # Sample covariance
        sample_cov = np.cov(returns_matrix, rowvar=False)
        
        # Target matrix (diagonal with average variance)
        avg_var = np.mean(np.diag(sample_cov))
        target_matrix = np.eye(sample_cov.shape[0]) * avg_var
        
        # Shrinkage towards target
        shrinkage_factor = 0.3
        shrunk_cov = shrinkage_factor * target_matrix + (1 - shrinkage_factor) * sample_cov
        
        # Ensure positive definiteness
        shrunk_cov = self._ensure_positive_definite(shrunk_cov)
        
        return shrunk_cov
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        # Check if matrix is positive definite
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # Matrix is not positive definite, fix it
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            fixed_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Add small diagonal term for numerical stability
            fixed_matrix += np.eye(matrix.shape[0]) * 1e-8
            
            return fixed_matrix
    
    def _get_default_constraints(self, n_assets: int) -> Dict:
        """Get default optimization constraints"""
        return {
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'target_volatility': self.target_volatility,
            'target_return': self.target_return,
            'risk_free_rate': self.risk_free_rate
        }
    
    def _optimize_efficient_frontier(self, expected_returns: np.ndarray, 
                                   covariance_matrix: np.ndarray, constraints: Dict) -> Dict:
        """Optimize portfolio for efficient frontier"""
        n_assets = len(expected_returns)
        
        # Generate efficient frontier points
        target_returns = np.linspace(
            np.min(expected_returns), 
            np.max(expected_returns), 
            50
        )
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                # Minimize variance for given target return
                result = minimize(
                    fun=lambda w: self._portfolio_variance(w, covariance_matrix),
                    x0=np.ones(n_assets) / n_assets,  # Equal weight initial guess
                    constraints=[
                        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                        {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}  # Target return
                    ],
                    bounds=[(constraints['min_weight'], constraints['max_weight'])] * n_assets,
                    method='SLSQP',
                    options={'maxiter': self.max_iterations}
                )
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_volatility = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
                    sharpe_ratio = (portfolio_return - constraints['risk_free_rate']/252) / portfolio_volatility
                    
                    efficient_portfolios.append({
                        'weights': weights,
                        'expected_return': portfolio_return,
                        'volatility': portfolio_volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'target_return': target_return
                    })
            except Exception as e:
                continue
        
        if not efficient_portfolios:
            return {'error': 'Failed to generate efficient frontier'}
        
        # Find optimal portfolio (maximum Sharpe ratio)
        optimal_portfolio = max(efficient_portfolios, key=lambda x: x['sharpe_ratio'])
        
        return {
            'efficient_frontier': efficient_portfolios,
            'optimal_portfolio': optimal_portfolio,
            'all_portfolios': efficient_portfolios
        }
    
    def _optimize_risk_parity(self, covariance_matrix: np.ndarray, constraints: Dict) -> Dict:
        """Optimize portfolio for risk parity (equal risk contribution)"""
        n_assets = len(covariance_matrix)
        
        def risk_parity_objective(weights):
            """Objective function: minimize difference in risk contributions"""
            portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
            
            # Calculate risk contributions
            risk_contributions = []
            for i in range(n_assets):
                # Marginal risk contribution
                marginal_risk = np.dot(covariance_matrix[i, :], weights) / portfolio_vol
                risk_contribution = weights[i] * marginal_risk
                risk_contributions.append(risk_contribution)
            
            # Penalty for unequal risk contributions
            target_risk = portfolio_vol / n_assets
            risk_parity_penalty = sum((rc - target_risk)**2 for rc in risk_contributions)
            
            return risk_parity_penalty
        
        try:
            result = minimize(
                fun=risk_parity_objective,
                x0=np.ones(n_assets) / n_assets,
                constraints=[
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                ],
                bounds=[(constraints['min_weight'], constraints['max_weight'])] * n_assets,
                method='SLSQP',
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
                
                # Calculate risk contributions
                risk_contributions = []
                for i in range(n_assets):
                    marginal_risk = np.dot(covariance_matrix[i, :], weights) / portfolio_vol
                    risk_contribution = weights[i] * marginal_risk
                    risk_contributions.append(risk_contribution)
                
                return {
                    'weights': weights,
                    'volatility': portfolio_vol,
                    'risk_contributions': risk_contributions,
                    'risk_parity_score': result.fun,
                    'optimization_success': True
                }
            else:
                return {'error': 'Risk parity optimization failed', 'optimization_success': False}
                
        except Exception as e:
            return {'error': f'Risk parity optimization error: {e}', 'optimization_success': False}
    
    def _optimize_maximum_sharpe(self, expected_returns: np.ndarray, 
                                covariance_matrix: np.ndarray, constraints: Dict) -> Dict:
        """Optimize portfolio for maximum Sharpe ratio"""
        n_assets = len(expected_returns)
        
        def negative_sharpe_ratio(weights):
            """Negative Sharpe ratio (minimization)"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - constraints['risk_free_rate']/252) / portfolio_vol
            return -sharpe_ratio  # Negative for minimization
        
        try:
            result = minimize(
                fun=negative_sharpe_ratio,
                x0=np.ones(n_assets) / n_assets,
                constraints=[
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                ],
                bounds=[(constraints['min_weight'], constraints['max_weight'])] * n_assets,
                method='SLSQP',
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
                sharpe_ratio = (portfolio_return - constraints['risk_free_rate']/252) / portfolio_vol
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_success': True
                }
            else:
                return {'error': 'Maximum Sharpe optimization failed', 'optimization_success': False}
                
        except Exception as e:
            return {'error': f'Maximum Sharpe optimization error: {e}', 'optimization_success': False}
    
    def _optimize_minimum_variance(self, covariance_matrix: np.ndarray, constraints: Dict) -> Dict:
        """Optimize portfolio for minimum variance"""
        n_assets = len(covariance_matrix)
        
        try:
            result = minimize(
                fun=lambda w: self._portfolio_variance(w, covariance_matrix),
                x0=np.ones(n_assets) / n_assets,
                constraints=[
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                ],
                bounds=[(constraints['min_weight'], constraints['max_weight'])] * n_assets,
                method='SLSQP',
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                weights = result.x
                portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
                
                return {
                    'weights': weights,
                    'volatility': portfolio_vol,
                    'optimization_success': True
                }
            else:
                return {'error': 'Minimum variance optimization failed', 'optimization_success': False}
                
        except Exception as e:
            return {'error': f'Minimum variance optimization error: {e}', 'optimization_success': False}
    
    def _optimize_black_litterman(self, expected_returns: np.ndarray, 
                                 covariance_matrix: np.ndarray, constraints: Dict) -> Dict:
        """Optimize portfolio using Black-Litterman model"""
        # This is a simplified implementation
        # In practice, Black-Litterman requires market equilibrium returns and investor views
        
        n_assets = len(expected_returns)
        
        # Use market equilibrium returns (simplified)
        market_cap_weights = np.ones(n_assets) / n_assets  # Equal weight assumption
        risk_aversion = 3.0  # Typical risk aversion parameter
        
        # Market equilibrium returns
        market_returns = risk_aversion * np.dot(covariance_matrix, market_cap_weights)
        
        # Investor views (simplified - no specific views)
        tau = 0.05  # Confidence in market equilibrium
        
        # Black-Litterman returns
        bl_returns = market_returns
        
        # Optimize using Black-Litterman returns
        return self._optimize_maximum_sharpe(bl_returns, covariance_matrix, constraints)
    
    def _portfolio_variance(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """Calculate portfolio variance"""
        return np.dot(weights.T, np.dot(covariance_matrix, weights))
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, expected_returns: np.ndarray, 
                                  covariance_matrix: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        if weights.size == 0:
            return {}
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
        
        # Risk metrics
        var_95 = self._calculate_portfolio_var(weights, covariance_matrix, 0.95)
        var_99 = self._calculate_portfolio_var(weights, covariance_matrix, 0.99)
        
        # Diversification metrics
        concentration_index = np.sum(weights**2)  # Herfindahl index
        effective_n = 1 / concentration_index  # Effective number of assets
        
        # Risk decomposition
        risk_contributions = []
        for i in range(len(weights)):
            marginal_risk = np.dot(covariance_matrix[i, :], weights) / portfolio_volatility
            risk_contribution = weights[i] * marginal_risk
            risk_contributions.append(risk_contribution)
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': (portfolio_return - self.risk_free_rate/252) / portfolio_volatility if portfolio_volatility > 0 else 0,
            'var_95': var_95,
            'var_99': var_99,
            'concentration_index': concentration_index,
            'effective_n': effective_n,
            'risk_contributions': risk_contributions,
            'weights': weights
        }
    
    def _calculate_portfolio_var(self, weights: np.ndarray, covariance_matrix: np.ndarray, 
                               confidence_level: float) -> float:
        """Calculate portfolio Value at Risk"""
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights, covariance_matrix))
        z_score = stats.norm.ppf(confidence_level)
        return z_score * portfolio_volatility
    
    def rebalance_portfolio(self, current_portfolio: Portfolio, target_weights: Dict[str, float], 
                           current_prices: Dict[str, float]) -> Dict:
        """Calculate rebalancing trades"""
        if not current_portfolio.positions:
            return {'rebalancing_trades': [], 'total_cost': 0}
        
        current_weights = {}
        total_value = float(current_portfolio.equity)
        
        # Calculate current weights
        for position in current_portfolio.positions:
            current_weights[position.symbol] = float(position.market_value) / total_value
        
        # Calculate target values
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = weight * total_value
        
        # Calculate rebalancing trades
        rebalancing_trades = []
        total_cost = 0
        
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_value = current_weights.get(symbol, 0) * total_value
            target_value = target_values.get(symbol, 0)
            
            # Check if rebalancing is needed
            if abs(target_value - current_value) / total_value > self.rebalancing_threshold:
                trade_value = target_value - current_value
                trade_quantity = trade_value / current_prices.get(symbol, 1)
                
                if abs(trade_quantity) > 0:
                    trade = {
                        'symbol': symbol,
                        'action': 'buy' if trade_quantity > 0 else 'sell',
                        'quantity': abs(trade_quantity),
                        'value': abs(trade_value),
                        'current_weight': current_weights.get(symbol, 0),
                        'target_weight': target_weights.get(symbol, 0)
                    }
                    rebalancing_trades.append(trade)
                    
                    # Estimate transaction cost (simplified)
                    transaction_cost = abs(trade_value) * 0.001  # 0.1% transaction cost
                    total_cost += transaction_cost
        
        return {
            'rebalancing_trades': rebalancing_trades,
            'total_cost': total_cost,
            'rebalancing_threshold': self.rebalancing_threshold,
            'current_weights': current_weights,
            'target_weights': target_weights
        }
