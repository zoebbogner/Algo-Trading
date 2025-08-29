"""Minimal backtesting engine for crypto algorithmic trading."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_and_log_config
from src.utils.logging import get_logger, get_run_id


class BacktestEngine:
    """Minimal backtesting engine with trade-next-bar execution."""

    def __init__(self, config: dict):
        """Initialize the backtesting engine."""
        self.config = config
        self.logger = get_logger(__name__)
        self.run_id = get_run_id()

        # Trading state
        self.positions = {}  # symbol -> quantity
        self.position_prices = {}  # symbol -> average entry price
        self.position_stops = {}  # symbol -> stop loss price
        self.cash = 100000.0  # Starting cash
        self.equity_history = []
        self.trade_history = []

        # Performance metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk management
        self.max_position_size = config.get('risk', {}).get('max_position_size', 0.05)  # 5% max per position
        self.max_portfolio_risk = config.get('risk', {}).get('max_portfolio_risk', 0.20)  # 20% max portfolio risk
        self.stop_loss_pct = config.get('risk', {}).get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = config.get('risk', {}).get('take_profit_pct', 0.10)  # 10% take profit

        # Load trading costs
        self.fee_rate = config.get('costs', {}).get('fee_rate', 0.001)  # 0.1%
        self.slippage_bps = config.get('costs', {}).get('slippage_bps', 5)  # 5 bps

        # Ensure output directories exist
        self.reports_dir = Path(f"reports/runs/{self.run_id}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Backtest engine initialized with run_id: {self.run_id}")
        self.logger.info(f"Starting cash: ${self.cash:,.2f}")
        self.logger.info(f"Fee rate: {self.fee_rate:.3f}, Slippage: {self.slippage_bps} bps")
        self.logger.info(f"Risk settings: Max position {self.max_position_size*100:.1f}%, Stop loss {self.stop_loss_pct*100:.1f}%, Take profit {self.take_profit_pct*100:.1f}%")

    def run_backtest(self, data_path: str, symbols: list[str], start_date: str = None, end_date: str = None) -> bool:
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

    def _load_data(self, data_path: str, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame | None:
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

            # Check stop losses and take profits first
            self._check_risk_management(current_data)

            # Make trading decisions for each symbol
            for _, row in current_data.iterrows():
                symbol = row['symbol']
                close_price = row['close']

                # Skip if we already have a position and it's not time to sell
                if symbol in self.positions and self.positions[symbol] > 0:
                    continue

                # Enhanced strategy with multiple indicators
                signal = self._generate_trading_signal(row)
                
                if signal == 'buy':
                    self._execute_trade(symbol, 'buy', close_price, ts, row)
                elif signal == 'sell' and symbol in self.positions and self.positions[symbol] > 0:
                    self._execute_trade(symbol, 'sell', close_price, ts, row)

            # Record equity at each timestamp
            self._record_equity(ts, current_data)

            # Progress logging
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(timestamps)} timestamps")

    def _generate_trading_signal(self, row: pd.Series) -> str:
        """Generate trading signal using multiple technical indicators."""
        signal = 'hold'
        
        # Check if we have basic indicators
        if not all(indicator in row and pd.notna(row[indicator]) for indicator in ['ma_20', 'ma_50', 'rsi_14']):
            return signal
        
        close_price = row['close']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        rsi = row['rsi_14']
        
        # Buy conditions (2 out of 3 must be true)
        buy_conditions = [
            close_price > ma_20 * 1.01,  # Price above 20-period MA with buffer
            ma_20 > ma_50,  # Short-term trend above long-term trend
            rsi < 75  # Not severely overbought
        ]
        
        # Sell conditions (2 out of 3 must be true)
        sell_conditions = [
            close_price < ma_20 * 0.99,  # Price below 20-period MA
            ma_20 < ma_50,  # Short-term trend below long-term trend
            rsi > 75  # Overbought
        ]
        
        # Generate signal (more flexible)
        if sum(buy_conditions) >= 2:
            signal = 'buy'
        elif sum(sell_conditions) >= 2:
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
                    continue
                
                # Check take profit
                take_profit_price = entry_price * (1 + self.take_profit_pct)
                if close_price >= take_profit_price:
                    self.logger.info(f"Take profit triggered for {symbol} at ${close_price:.2f} (entry: ${entry_price:.2f})")
                    self._execute_trade(symbol, 'sell', close_price, row['ts'], row)
                    continue

    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size using Kelly Criterion and risk management."""
        # Get current portfolio value
        portfolio_value = self.cash
        for sym, qty in self.positions.items():
            if qty > 0:
                portfolio_value += qty * self.position_prices[sym]
        
        # Calculate Kelly position size (simplified)
        # In a real implementation, this would use win rate and average win/loss
        win_rate = 0.4  # Conservative estimate
        avg_win = 0.05  # 5% average win
        avg_loss = 0.03  # 3% average loss
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))  # Cap at max position size
        else:
            kelly_fraction = self.max_position_size * 0.5  # Conservative fallback
        
        # Calculate position size in currency
        position_value = portfolio_value * kelly_fraction
        
        # Convert to quantity
        quantity = position_value / price
        
        # Ensure we don't exceed available cash
        max_quantity = self.cash / (price * (1 + self.fee_rate))
        quantity = min(quantity, max_quantity)
        
        return max(0, quantity)

    def _execute_trade(self, symbol: str, action: str, price: float, timestamp: datetime, features: pd.Series) -> None:
        """Execute a trade with costs and slippage."""
        # Calculate execution price with slippage
        slippage_multiplier = 1 + (self.slippage_bps / 10000)
        if action == 'buy':
            execution_price = price * slippage_multiplier
            # Calculate position size using Kelly Criterion
            quantity = self._calculate_position_size(symbol, execution_price)
            if quantity <= 0:
                return  # Skip trade if no position size
        else:
            execution_price = price / slippage_multiplier
            quantity = self.positions[symbol]  # Sell entire position

        # Calculate costs
        trade_value = quantity * execution_price
        fees = trade_value * self.fee_rate

        # Calculate trade PnL for sell orders
        trade_pnl = 0.0
        if action == 'sell' and symbol in self.position_prices:
            avg_entry_price = self.position_prices[symbol]
            trade_pnl = (execution_price - avg_entry_price) * quantity - fees
            
            # Track winning/losing trades
            if trade_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

        # Update positions and cash
        if action == 'buy':
            if symbol not in self.positions:
                self.positions[symbol] = 0
                self.position_prices[symbol] = execution_price
                self.position_stops[symbol] = execution_price * (1 - self.stop_loss_pct)
            else:
                # Update average entry price
                total_cost = (self.positions[symbol] * self.position_prices[symbol]) + (quantity * execution_price)
                total_quantity = self.positions[symbol] + quantity
                self.position_prices[symbol] = total_cost / total_quantity
                # Update stop loss to new average entry
                self.position_stops[symbol] = self.position_prices[symbol] * (1 - self.stop_loss_pct)
            
            self.positions[symbol] += quantity
            self.cash -= (trade_value + fees)
        else:  # sell
            self.positions[symbol] -= quantity
            self.cash += (trade_value - fees)
            
            # Remove position if fully sold
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
                del self.position_prices[symbol]
                del self.position_stops[symbol]

        # Update total PnL
        self.total_pnl += trade_pnl

        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'fees': fees,
            'trade_pnl': trade_pnl,
            'cash_after': self.cash,
            'run_id': self.run_id
        }
        self.trade_history.append(trade_record)

        self.total_trades += 1
        self.logger.debug(f"Executed {action} {quantity:.4f} {symbol} at ${execution_price:.4f}, PnL: ${trade_pnl:.2f}")

    def _record_equity(self, timestamp: datetime, current_data: pd.DataFrame) -> None:
        """Record current equity value."""
        # Calculate current portfolio value
        portfolio_value = self.cash

        # Add value of all positions using current market prices
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # Find current price for this symbol
                symbol_data = current_data[current_data['symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['close']
                    position_value = quantity * current_price
                    portfolio_value += position_value

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

    def _calculate_metrics(self) -> dict[str, Any]:
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

        # Profit factor
        if self.losing_trades > 0:
            avg_win = self.total_pnl / self.winning_trades if self.winning_trades > 0 else 0
            avg_loss = abs(self.total_pnl / self.losing_trades) if self.losing_trades > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_factor = 0

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
            'losing_trades': self.losing_trades,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.total_pnl,
            'final_cash': self.cash,
            'final_positions': dict(self.positions)
        }

        self.logger.info(f"Final equity: ${final_equity:,.2f}")
        self.logger.info(f"Total return: {total_return*100:.2f}%")
        self.logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Max drawdown: {max_drawdown*100:.2f}%")
        self.logger.info(f"Hit rate: {hit_rate*100:.1f}%")
        self.logger.info(f"Profit factor: {profit_factor:.2f}")
        self.logger.info(f"Total PnL: ${self.total_pnl:,.2f}")

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
