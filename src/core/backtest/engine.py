"""Enhanced Backtesting Engine."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Enhanced backtesting engine."""
    
    def __init__(self, config: Dict):
        """Initialize backtesting engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Basic configuration
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.commission_rate = config.get('costs', {}).get('commission_rate', 0.001)
        self.slippage_bps = config.get('costs', {}).get('slippage_bps', 5)
        self.fee_rate = self.commission_rate
        self.slippage = self.slippage_bps / 10000
        
        # Risk management
        risk_config = config.get('risk', {})
        self.max_position_size = risk_config.get('max_position_size', 0.05)
        self.max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.2)
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = risk_config.get('take_profit_pct', 0.10)
        
        # Initialize state
        self.reset()
        
        # Generate run ID
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info("BacktestEngine initialized")
    
    def reset(self):
        """Reset engine state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.position_prices = {}
        self.position_stops = {}
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []
        self.equity_curve = []
        
        self.logger.info("Engine state reset")
    
    def run_backtest(self, data_path: str, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Run backtest."""
        try:
            self.logger.info(f"Starting backtest for {len(symbols)} symbols")
            
            # Load and prepare data
            data = self._load_data(data_path, symbols, start_date, end_date)
            if data is None:
                return False
            
            # Run simulation
            self._run_simulation(data)
            
            # Calculate final metrics
            self._calculate_metrics()
            
            # Save results
            self._save_results()
            
            self.logger.info("Backtest completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            return False
    
    def _load_data(self, data_path: str, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare data for backtesting."""
        try:
            self.logger.info(f"Loading data from {data_path}")
            
            # Load data
            data = pd.read_parquet(data_path)
            
            # Filter by symbols
            if symbols:
                data = data[data['symbol'].isin(symbols)]
            
            # Filter by date range
            if start_date:
                data = data[data['date'] >= start_date]
            if end_date:
                data = data[data['date'] <= end_date]
            
            # Sort by timestamp
            data = data.sort_values(['symbol', 'ts'])
            
            self.logger.info(f"Loaded {len(data):,} data points for {data['symbol'].nunique()} symbols")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def _run_simulation(self, data: pd.DataFrame) -> None:
        """Run the backtest simulation."""
        self.logger.info("Starting simulation...")
        
        # Group by timestamp for portfolio-level decisions
        for timestamp, current_data in data.groupby('ts'):
            # Check risk management (stop losses, take profits)
            self._check_risk_management(current_data)
            
            # Generate trading signals for each symbol
            for _, row in current_data.iterrows():
                symbol = row['symbol']
                
                # Generate trading signal
                signal = self._generate_trading_signal(row)
                
                # Execute trades based on signals
                if signal == 'buy' and symbol not in self.positions:
                    self._execute_trade(symbol, 'buy', row['close'], timestamp, row)
                elif signal == 'sell' and symbol in self.positions and self.positions[symbol] > 0:
                    self._execute_trade(symbol, 'sell', row['close'], timestamp, row)
            
            # Update equity curve
            self._update_equity_curve(timestamp, current_data)
    
    def _generate_trading_signal(self, row: pd.Series) -> str:
        """Generate trading signal using traditional indicators."""
        signal = 'hold'
        
        # Check if we have basic indicators
        if not all(indicator in row and pd.notna(row[indicator]) for indicator in ['ma_20', 'ma_50', 'rsi_14']):
            return signal
        
        close_price = row['close']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        rsi = row['rsi_14']
        
        # Traditional signal generation
        signal = self._generate_traditional_signal(row)
        
        return signal
    
    def _generate_traditional_signal(self, row: pd.Series) -> str:
        """Generate traditional technical analysis signal."""
        signal = 'hold'
        
        close_price = row['close']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        rsi = row['rsi_14']
        
        # AGGRESSIVE STRATEGY 1: Quick momentum trades with tight thresholds
        if close_price > ma_20 * 1.005:  # Very small move above MA
            if rsi < 70:  # Not too overbought
                signal = 'buy'
        elif close_price < ma_20 * 0.995:  # Very small move below MA
            if rsi > 30:  # Not too oversold
                signal = 'sell'
        
        # AGGRESSIVE STRATEGY 2: RSI extremes for mean reversion
        if rsi < 30:  # Oversold
            signal = 'buy'
        elif rsi > 75:  # Overbought
            signal = 'sell'
        
        # AGGRESSIVE STRATEGY 3: Trend following with minimal requirements
        if ma_20 > ma_50:  # Any uptrend
            if close_price > ma_20 * 1.003:  # Small buffer
                signal = 'buy'
        elif ma_20 < ma_50:  # Any downtrend
            if close_price < ma_20 * 0.997:  # Small buffer
                signal = 'sell'
        
        # AGGRESSIVE STRATEGY 4: Breakout plays
        if close_price > ma_20 * 1.01:  # Strong move up
            signal = 'buy'
        elif close_price < ma_20 * 0.99:  # Strong move down
            signal = 'sell'
        
        # AGGRESSIVE STRATEGY 5: RSI momentum shifts
        if 35 <= rsi <= 45 and close_price > ma_20 * 0.999:  # RSI recovering
            signal = 'buy'
        elif 55 <= rsi <= 65 and close_price < ma_20 * 1.001:  # RSI weakening
            signal = 'sell'
        
        return signal

    def _check_risk_management(self, current_data: pd.DataFrame) -> None:
        """Check and execute stop losses and take profits."""
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            close_price = row['close']
            
            if symbol in self.positions and self.positions[symbol] > 0:
                entry_price = self.position_prices[symbol]
                
                # Check stop loss
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                if close_price <= stop_loss_price:
                    self.logger.info(f"Stop loss triggered for {symbol} at ${close_price:.2f} (entry: ${entry_price:.2f})")
                    self._execute_trade(symbol, 'sell', close_price, row['ts'], row)
                
                # Check take profit
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                if close_price >= take_profit_price:
                    self.logger.info(f"Take profit triggered for {symbol} at ${close_price:.2f} (entry: ${entry_price:.2f})")
                    self._execute_trade(symbol, 'sell', close_price, row['ts'], row)
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size using Kelly Criterion."""
        # Simple Kelly Criterion implementation
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply constraints
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_position_size))
        
        # Calculate position size
        position_value = self.cash * kelly_fraction
        shares = position_value / price
        
        return shares
    
    def _execute_trade(self, symbol: str, action: str, price: float, timestamp: datetime, row: pd.Series) -> None:
        """Execute a trade."""
        if action == 'buy':
            # Calculate position size
            shares = self._calculate_position_size(symbol, price)
            
            if shares > 0:
                # Calculate costs
                trade_value = shares * price
                commission = trade_value * self.fee_rate
                slippage_cost = trade_value * self.slippage
                total_cost = commission + slippage_cost
                
                # Execute trade
                if self.cash >= (trade_value + total_cost):
                    self.cash -= (trade_value + total_cost)
                    self.positions[symbol] = shares
                    self.position_prices[symbol] = price
                    
                    # Record trade
                    self._record_trade(symbol, 'buy', shares, price, timestamp, total_cost)
                    
                    self.logger.info(f"Bought {shares:.2f} shares of {symbol} at ${price:.2f}")
                else:
                    self.logger.warning(f"Insufficient cash for {symbol} trade")
        
        elif action == 'sell':
            if symbol in self.positions and self.positions[symbol] > 0:
                shares = self.positions[symbol]
                trade_value = shares * price
                
                # Calculate costs
                commission = trade_value * self.fee_rate
                slippage_cost = trade_value * self.slippage
                total_cost = commission + slippage_cost
                
                # Execute trade
                self.cash += (trade_value - total_cost)
                self.positions[symbol] = 0
                entry_price = self.position_prices[symbol]
                
                # Calculate P&L
                pnl = (price - entry_price) * shares - total_cost
                self.total_pnl += pnl
                
                # Update trade statistics
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Record trade
                self._record_trade(symbol, 'sell', shares, price, timestamp, total_cost, pnl)
                
                self.logger.info(f"Sold {shares:.2f} shares of {symbol} at ${price:.2f}, P&L: ${pnl:.2f}")
    
    def _record_trade(self, symbol: str, action: str, shares: float, price: float, timestamp: datetime, costs: float, pnl: float = 0.0) -> None:
        """Record trade details."""
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'costs': costs,
            'pnl': pnl,
            'cash': self.cash,
            'total_pnl': self.total_pnl
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
    
    def _update_equity_curve(self, timestamp: datetime, current_data: pd.DataFrame) -> None:
        """Update equity curve with current portfolio value."""
        portfolio_value = self.cash
        
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            if symbol in self.positions and self.positions[symbol] > 0:
                portfolio_value += self.positions[symbol] * row['close']
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash
        })
    
    def _calculate_metrics(self) -> None:
        """Calculate final performance metrics."""
        if not self.equity_curve:
            return
        
        # Calculate returns
        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1]['portfolio_value']
            curr_value = self.equity_curve[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate drawdown
        peak_value = initial_value
        max_drawdown = 0
        
        for point in self.equity_curve:
            if point['portfolio_value'] > peak_value:
                peak_value = point['portfolio_value']
            else:
                drawdown = (peak_value - point['portfolio_value']) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
        
        # Store metrics
        self.performance_metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_pnl': self.total_pnl,
            'final_portfolio_value': final_value
        }
        
        self.logger.info(f"Performance metrics calculated: Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
    
    def _save_results(self) -> None:
        """Save backtest results."""
        try:
            # Create reports directory
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            # Save trade history
            if self.trade_history:
                trades_df = pd.DataFrame(self.trade_history)
                trades_file = reports_dir / f'trades_{self.run_id}.parquet'
                trades_df.to_parquet(trades_file)
                self.logger.info(f"Trade history saved to {trades_file}")
            
            # Save equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_file = reports_dir / f'equity_curve_{self.run_id}.parquet'
                equity_df.to_parquet(equity_file)
                self.logger.info(f"Equity curve saved to {equity_file}")
            
            # Save performance metrics
            if hasattr(self, 'performance_metrics'):
                import json
                metrics_file = reports_dir / f'performance_metrics_{self.run_id}.json'
                with open(metrics_file, 'w') as f:
                    json.dump(self.performance_metrics, f, indent=2, default=str)
                self.logger.info(f"Performance metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_results(self) -> Dict:
        """Get backtest results."""
        return {
            'performance_metrics': getattr(self, 'performance_metrics', {}),
            'trade_history': self.trade_history,
            'equity_curve': self.equity_curve,
            'run_id': self.run_id
        }
