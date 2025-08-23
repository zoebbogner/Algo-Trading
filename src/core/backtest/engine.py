"""Minimal backtesting engine for crypto algorithmic trading."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.utils.logging import get_logger, get_run_id
from src.utils.config import load_and_log_config


class BacktestEngine:
    """Minimal backtesting engine with trade-next-bar execution."""
    
    def __init__(self, config: dict):
        """Initialize the backtesting engine."""
        self.config = config
        self.logger = get_logger(__name__)
        self.run_id = get_run_id()
        
        # Trading state
        self.positions = {}  # symbol -> quantity
        self.cash = 100000.0  # Starting cash
        self.equity_history = []
        self.trade_history = []
        
        # Performance metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Load trading costs
        self.fee_rate = config.get('costs', {}).get('fee_rate', 0.001)  # 0.1%
        self.slippage_bps = config.get('costs', {}).get('slippage_bps', 5)  # 5 bps
        
        # Ensure output directories exist
        self.reports_dir = Path(f"reports/runs/{self.run_id}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Backtest engine initialized with run_id: {self.run_id}")
        self.logger.info(f"Starting cash: ${self.cash:,.2f}")
        self.logger.info(f"Fee rate: {self.fee_rate:.3f}, Slippage: {self.slippage_bps} bps")
    
    def run_backtest(self, data_path: str, symbols: List[str], start_date: str = None, end_date: str = None) -> bool:
        """Run backtest on historical data."""
        try:
            self.logger.info(f"Starting backtest for symbols: {symbols}")
            
            # Load data
            data = self._load_data(data_path, symbols, start_date, end_date)
            if data is None or data.empty:
                self.logger.error("No data loaded for backtest")
                return False
            
            self.logger.info(f"Loaded {len(data)} data points")
            
            # Run simulation
            self._run_simulation(data)
            
            # Generate reports
            self._generate_reports()
            
            self.logger.info("Backtest completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return False
    
    def _load_data(self, data_path: str, symbols: List[str], start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load historical data for backtesting."""
        try:
            # Try to load from partitioned dataset first
            partition_path = Path(data_path)
            if partition_path.exists():
                # Load from partitioned dataset
                data = pd.read_parquet(partition_path)
                
                # Filter by symbols
                if symbols:
                    data = data[data['symbol'].isin(symbols)]
                
                # Filter by date range
                if start_date:
                    data = data[data['ts'] >= start_date]
                if end_date:
                    data = data[data['ts'] <= end_date]
                
                # Sort by timestamp
                data = data.sort_values(['symbol', 'ts'])
                
                return data
            
            else:
                # Fallback to individual files
                all_data = []
                for symbol in symbols:
                    symbol_file = Path(data_path) / f"{symbol.lower()}_bars_1m.parquet"
                    if symbol_file.exists():
                        symbol_data = pd.read_parquet(symbol_file)
                        symbol_data['symbol'] = symbol
                        all_data.append(symbol_data)
                
                if not all_data:
                    return None
                
                data = pd.concat(all_data, ignore_index=True)
                data = data.sort_values(['symbol', 'ts'])
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def _run_simulation(self, data: pd.DataFrame) -> None:
        """Run the trading simulation."""
        self.logger.info("Starting trading simulation")
        
        # Group by timestamp to process all symbols at once
        timestamps = sorted(data['ts'].unique())
        
        for i, ts in enumerate(timestamps):
            current_data = data[data['ts'] == ts]
            
            # Make trading decisions for each symbol
            for _, row in current_data.iterrows():
                symbol = row['symbol']
                close_price = row['close']
                
                # Simple strategy: buy if price > 20-period MA, sell if < 20-period MA
                if 'ma_20' in row and pd.notna(row['ma_20']):
                    ma_20 = row['ma_20']
                    
                    if close_price > ma_20 * 1.01:  # Buy signal with 1% buffer
                        self._execute_trade(symbol, 'buy', 1.0, close_price, ts, row)
                    elif close_price < ma_20 * 0.99:  # Sell signal with 1% buffer
                        if symbol in self.positions and self.positions[symbol] > 0:
                            self._execute_trade(symbol, 'sell', self.positions[symbol], close_price, ts, row)
            
            # Record equity at each timestamp
            self._record_equity(ts)
            
            # Progress logging
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(timestamps)} timestamps")
    
    def _execute_trade(self, symbol: str, action: str, quantity: float, price: float, timestamp: datetime, features: pd.Series) -> None:
        """Execute a trade with costs and slippage."""
        # Calculate execution price with slippage
        slippage_multiplier = 1 + (self.slippage_bps / 10000)
        if action == 'buy':
            execution_price = price * slippage_multiplier
        else:
            execution_price = price / slippage_multiplier
        
        # Calculate costs
        trade_value = quantity * execution_price
        fees = trade_value * self.fee_rate
        
        # Update positions and cash
        if action == 'buy':
            if symbol not in self.positions:
                self.positions[symbol] = 0
            self.positions[symbol] += quantity
            self.cash -= (trade_value + fees)
        else:  # sell
            self.positions[symbol] -= quantity
            self.cash += (trade_value - fees)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'fees': fees,
            'cash_after': self.cash,
            'run_id': self.run_id
        }
        self.trade_history.append(trade_record)
        
        self.total_trades += 1
        self.logger.debug(f"Executed {action} {quantity} {symbol} at ${execution_price:.4f}")
    
    def _record_equity(self, timestamp: datetime) -> None:
        """Record current equity value."""
        # Calculate current portfolio value
        portfolio_value = self.cash
        
        # Add value of all positions (simplified - would need current prices in real implementation)
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # For simplicity, use last known price or assume no change
                portfolio_value += quantity * 0  # This would need to be current price
        
        equity_record = {
            'timestamp': timestamp,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_pnl': portfolio_value - 100000.0,  # Starting cash
            'run_id': self.run_id
        }
        self.equity_history.append(equity_record)
    
    def _generate_reports(self) -> None:
        """Generate backtest reports."""
        self.logger.info("Generating backtest reports")
        
        # Equity report
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_df.to_csv(self.reports_dir / "equity.csv", index=False)
            self.logger.info(f"Equity report saved to {self.reports_dir / 'equity.csv'}")
        
        # Trade log
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(self.reports_dir / "trades.csv", index=False)
            self.logger.info(f"Trade log saved to {self.reports_dir / 'trades.csv'}")
        
        # Performance metrics
        metrics = self._calculate_metrics()
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.reports_dir / "metrics.csv", index=False)
        
        # Save metrics as JSON for easy consumption
        import json
        with open(self.reports_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Performance metrics saved to {self.reports_dir / 'metrics.json'}")
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.equity_history:
            return {}
        
        equity_df = pd.DataFrame(self.equity_history)
        
        # Basic metrics
        final_equity = equity_df['portfolio_value'].iloc[-1]
        initial_equity = 100000.0
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate returns for Sharpe ratio
        equity_df['returns'] = equity_df['portfolio_value'].pct_change().fillna(0)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if equity_df['returns'].std() > 0:
            sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(1440)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Hit rate
        if self.total_trades > 0:
            hit_rate = self.winning_trades / self.total_trades
        else:
            hit_rate = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        metrics = {
            'run_id': self.run_id,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'hit_rate': hit_rate,
            'final_cash': self.cash,
            'final_positions': dict(self.positions)
        }
        
        self.logger.info(f"Final equity: ${final_equity:,.2f}")
        self.logger.info(f"Total return: {total_return*100:.2f}%")
        self.logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Max drawdown: {max_drawdown*100:.2f}%")
        self.logger.info(f"Hit rate: {hit_rate*100:.1f}%")
        
        return metrics


def main():
    """Main function to run backtest."""
    # Load configuration
    config = load_and_log_config()
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest on BTCUSDT and ETHUSDT for 7 days
    symbols = ['BTCUSDT', 'ETHUSDT']
    data_path = "data/features/features_1m.parquet"  # Use features dataset
    
    success = engine.run_backtest(data_path, symbols)
    
    if success:
        print("✅ Backtest completed successfully")
        print(f"📊 Reports saved to: {engine.reports_dir}")
    else:
        print("❌ Backtest failed")


if __name__ == "__main__":
    main()
