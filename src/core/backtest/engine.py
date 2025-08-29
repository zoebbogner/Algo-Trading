"""Enhanced Backtesting Engine with ML Integration."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Import ML components
try:
    from ..ml import MLSignalGenerator, MLFeatureEngineer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML components not available. Using traditional signals only.")

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Enhanced backtesting engine with ML integration."""
    
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
        
        # ML configuration
        self.use_ml = config.get('use_ml', False) and ML_AVAILABLE
        if self.use_ml:
            self.ml_config = config.get('ml', {})
            self.ml_feature_engineer = MLFeatureEngineer(self.ml_config)
            self.ml_signal_generator = MLSignalGenerator(self.ml_config)
            self.ml_models_trained = False
            self.ml_weight = self.ml_config.get('weight', 0.7)
            self.technical_weight = 1.0 - self.ml_weight
        
        # Initialize state
        self.reset()
        
        # Generate run ID
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info(f"BacktestEngine initialized with ML: {self.use_ml}")
    
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
        """Run backtest with optional ML integration."""
        try:
            self.logger.info(f"Starting backtest for {len(symbols)} symbols")
            
            # Load and prepare data
            data = self._load_data(data_path, symbols, start_date, end_date)
            if data is None:
                return False
            
            # Prepare ML features if enabled
            if self.use_ml:
                data = self._prepare_ml_features(data)
                if not self.ml_models_trained:
                    self.logger.warning("ML models not trained. Training on available data...")
                    self._train_ml_models(data)
            
            # Run simulation
            self._run_simulation(data)
            
            # Calculate final metrics
            self._calculate_metrics()
            
            # Save results
            self._save_results()
            
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
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare ML features for the dataset."""
        self.logger.info("Preparing ML features...")
        
        try:
            # Create ML features
            data_with_features = self.ml_feature_engineer.create_ml_features(data)
            
            # Fit scaler on training data
            self.ml_feature_engineer.fit_scaler(data_with_features)
            
            self.logger.info(f"ML features prepared: {len(self.ml_feature_engineer.feature_columns)} features")
            return data_with_features
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            return data
    
    def _train_ml_models(self, data: pd.DataFrame) -> None:
        """Train ML models on available data."""
        try:
            self.logger.info("Training ML models...")
            
            # Prepare target variable (future returns)
            data_with_target = self._prepare_target_variable(data)
            
            # Train models
            self.ml_signal_generator.train_models(data_with_target, 'target')
            
            self.ml_models_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.use_ml = False
    
    def _prepare_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variable for ML training."""
        data = data.copy()
        
        # Create target: 1 for buy, 0 for sell, 2 for hold
        # Based on future returns
        future_returns = data['close'].pct_change(5).shift(-5)  # 5-period ahead return
        threshold = future_returns.std() * 0.5  # Dynamic threshold
        
        data['target'] = np.where(future_returns > threshold, 1, 0)
        data['target'] = np.where(future_returns < -threshold, 2, data['target'])
        
        # Remove rows where we can't calculate future returns
        data = data.dropna(subset=['target'])
        
        self.logger.info(f"Target variable prepared. Distribution: {data['target'].value_counts().to_dict()}")
        return data
    
    def _generate_ml_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML trading signals."""
        if not self.ml_models_trained:
            self.logger.warning("ML models not trained. Using traditional signals only.")
            return data
        
        try:
            # Generate ML signals
            data_with_ml = self.ml_signal_generator.generate_signals(data)
            
            self.logger.info("ML signals generated successfully")
            return data_with_ml
            
        except Exception as e:
            self.logger.error(f"Error generating ML signals: {e}")
            return data
    
    def _generate_trading_signal(self, row: pd.Series) -> str:
        """Generate trading signal using traditional indicators and optional ML."""
        signal = 'hold'
        
        # Check if we have basic indicators
        if not all(indicator in row and pd.notna(row[indicator]) for indicator in ['ma_20', 'ma_50', 'rsi_14']):
            return signal
        
        close_price = row['close']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        rsi = row['rsi_14']
        
        # Traditional signal generation
        traditional_signal = self._generate_traditional_signal(row)
        
        # ML signal if available
        ml_signal = None
        if self.use_ml and 'ml_signal' in row and pd.notna(row['ml_signal']):
            ml_signal = row['ml_signal']
            ml_confidence = row.get('ml_confidence', 0.5)
        else:
            ml_confidence = 0.0
        
        # Combine signals
        if ml_signal and ml_confidence > 0.6:
            # High confidence ML signal
            if ml_confidence > 0.8:
                signal = ml_signal  # Trust ML completely
            else:
                # Combine ML and traditional
                if ml_signal == traditional_signal:
                    signal = ml_signal  # Agreement
                else:
                    signal = traditional_signal  # Fallback to traditional
        else:
            # Use traditional signal
            signal = traditional_signal
        
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
        win_rate = 0.45  # Increased from 0.4 to 0.45 for more aggressive sizing
        avg_win = 0.06  # Increased from 0.05 to 0.06 for better win potential
        avg_loss = 0.04  # Increased from 0.03 to 0.04 for larger positions
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            # Apply aggressive multiplier to increase position sizes
            kelly_fraction = kelly_fraction * 1.8  # Increased from 1.0 to 1.8
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))  # Cap at max position size
        else:
            kelly_fraction = self.max_position_size * 0.8  # Increased from 0.5 to 0.8
        
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
        self.equity_curve.append(equity_record)

    def _generate_reports(self) -> None:
        """Generate backtest reports."""
        self.logger.info("Generating backtest reports")

        # Equity report
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
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
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)

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
