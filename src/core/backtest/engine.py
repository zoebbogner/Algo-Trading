"""Enhanced Backtesting Engine for Algorithmic Trading."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Enhanced backtesting engine with risk management and performance tracking."""
    
    def __init__(self, config: Dict):
        """Initialize backtesting engine with configuration."""
        self._setup_configuration(config)
        self._setup_risk_management(config)
        self._initialize_state()
        self._generate_run_id()
        
        logger.info("BacktestEngine initialized successfully")
    
    def _setup_configuration(self, config: Dict) -> None:
        """Setup basic configuration parameters."""
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000.0)
        
        # Cost configuration
        costs = config.get('costs', {})
        self.commission_rate = costs.get('commission_rate', 0.001)
        self.slippage_bps = costs.get('slippage_bps', 5)
        self.fee_rate = self.commission_rate
        self.slippage = self.slippage_bps / 10000
    
    def _setup_risk_management(self, config: Dict) -> None:
        """Setup risk management parameters."""
        risk_config = config.get('risk', {})
        self.max_position_size = risk_config.get('max_position_size', 0.05)
        self.max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.2)
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = risk_config.get('take_profit_pct', 0.10)
    
    def _initialize_state(self) -> None:
        """Initialize engine state variables."""
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
    
    def _generate_run_id(self) -> None:
        """Generate unique run identifier."""
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def reset(self) -> None:
        """Reset engine state to initial conditions."""
        self._initialize_state()
        logger.info("Engine state reset to initial conditions")
    
    def run_backtest(self, data_path: str, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Run complete backtest simulation."""
        try:
            logger.info(f"Starting backtest for {len(symbols)} symbols")
            
            data = self._load_and_prepare_data(data_path, symbols, start_date, end_date)
            if data is None:
                return False
            
            self._execute_simulation(data)
            self._calculate_performance_metrics()
            self._save_backtest_results()
            
            logger.info("Backtest completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return False
    
    def _load_and_prepare_data(self, data_path: str, symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare data for backtesting."""
        try:
            logger.info(f"Loading data from {data_path}")
            
            data = self._load_raw_data(data_path)
            data = self._filter_data_by_symbols(data, symbols)
            data = self._filter_data_by_dates(data, start_date, end_date)
            data = self._sort_data_by_timestamp(data)
            
            logger.info(f"Loaded {len(data):,} data points for {data['symbol'].nunique()} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _load_raw_data(self, data_path: str) -> pd.DataFrame:
        """Load raw data from file."""
        return pd.read_parquet(data_path)
    
    def _filter_data_by_symbols(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Filter data by specified symbols."""
        if symbols:
            return data[data['symbol'].isin(symbols)]
        return data
    
    def _filter_data_by_dates(self, data: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Filter data by date range."""
        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]
        return data
    
    def _sort_data_by_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by symbol and timestamp."""
        return data.sort_values(['symbol', 'ts'])
    
    def _execute_simulation(self, data: pd.DataFrame) -> None:
        """Execute the backtest simulation."""
        logger.info("Starting simulation execution...")
        
        for timestamp, current_data in data.groupby('ts'):
            self._process_timestamp_data(timestamp, current_data)
    
    def _process_timestamp_data(self, timestamp: datetime, current_data: pd.DataFrame) -> None:
        """Process data for a specific timestamp."""
        self._execute_risk_management(current_data)
        self._execute_trading_signals(current_data, timestamp)
        self._update_portfolio_equity(timestamp, current_data)
    
    def _execute_risk_management(self, current_data: pd.DataFrame) -> None:
        """Execute risk management checks (stop losses, take profits)."""
        for _, row in current_data.iterrows():
            self._check_position_risk(row)
    
    def _check_position_risk(self, row: pd.Series) -> None:
        """Check risk for a specific position."""
        symbol = row['symbol']
        close_price = row['close']
        
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return
        
        entry_price = self.position_prices[symbol]
        
        if self._should_trigger_stop_loss(close_price, entry_price):
            self._execute_risk_exit(symbol, close_price, row, 'stop_loss')
        elif self._should_trigger_take_profit(close_price, entry_price):
            self._execute_risk_exit(symbol, close_price, row, 'take_profit')
    
    def _should_trigger_stop_loss(self, current_price: float, entry_price: float) -> bool:
        """Check if stop loss should be triggered."""
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        return current_price <= stop_loss_price
    
    def _should_trigger_take_profit(self, current_price: float, entry_price: float) -> bool:
        """Check if take profit should be triggered."""
        take_profit_price = entry_price * (1 + self.take_profit_pct)
        return current_price >= take_profit_price
    
    def _execute_risk_exit(self, symbol: str, price: float, row: pd.Series, exit_type: str) -> None:
        """Execute risk-based position exit."""
        logger.info(f"{exit_type.title()} triggered for {symbol} at ${price:.2f}")
        self._execute_trade(symbol, 'sell', price, row['ts'], row)
    
    def _execute_trading_signals(self, current_data: pd.DataFrame, timestamp: datetime) -> None:
        """Execute trading signals for current data."""
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            signal = self._generate_trading_signal(row)
            self._execute_signal(symbol, signal, row, timestamp)
    
    def _execute_signal(self, symbol: str, signal: str, row: pd.Series, timestamp: datetime) -> None:
        """Execute a trading signal."""
        if signal == 'buy' and symbol not in self.positions:
            self._execute_trade(symbol, 'buy', row['close'], timestamp, row)
        elif signal == 'sell' and symbol in self.positions and self.positions[symbol] > 0:
            self._execute_trade(symbol, 'sell', row['close'], timestamp, row)
    
    def _generate_trading_signal(self, row: pd.Series) -> str:
        """Generate trading signal using technical indicators."""
        if not self._has_required_indicators(row):
            return 'hold'
        
        return self._generate_traditional_signal(row)
    
    def _has_required_indicators(self, row: pd.Series) -> bool:
        """Check if required indicators are available."""
        required_indicators = ['ma_20', 'ma_50', 'rsi_14']
        return all(indicator in row and pd.notna(row[indicator]) for indicator in required_indicators)
    
    def _generate_traditional_signal(self, row: pd.Series) -> str:
        """Generate traditional technical analysis signal."""
        close_price = row['close']
        ma_20 = row['ma_20']
        ma_50 = row['ma_50']
        rsi = row['rsi_14']
        
        # Strategy 1: Quick momentum trades
        if self._check_momentum_buy_signal(close_price, ma_20, rsi):
            return 'buy'
        elif self._check_momentum_sell_signal(close_price, ma_20, rsi):
            return 'sell'
        
        # Strategy 2: RSI extremes for mean reversion
        if self._check_rsi_buy_signal(rsi):
            return 'buy'
        elif self._check_rsi_sell_signal(rsi):
            return 'sell'
        
        # Strategy 3: Trend following
        if self._check_trend_buy_signal(close_price, ma_20, ma_50):
            return 'buy'
        elif self._check_trend_sell_signal(close_price, ma_20, ma_50):
            return 'sell'
        
        # Strategy 4: Breakout plays
        if self._check_breakout_buy_signal(close_price, ma_20):
            return 'buy'
        elif self._check_breakout_sell_signal(close_price, ma_20):
            return 'sell'
        
        # Strategy 5: RSI momentum shifts
        if self._check_rsi_momentum_buy_signal(rsi, close_price, ma_20):
            return 'buy'
        elif self._check_rsi_momentum_sell_signal(rsi, close_price, ma_20):
            return 'sell'
        
        return 'hold'
    
    def _check_momentum_buy_signal(self, close_price: float, ma_20: float, rsi: float) -> bool:
        """Check momentum buy signal conditions."""
        return close_price > ma_20 * 1.005 and rsi < 70
    
    def _check_momentum_sell_signal(self, close_price: float, ma_20: float, rsi: float) -> bool:
        """Check momentum sell signal conditions."""
        return close_price < ma_20 * 0.995 and rsi > 30
    
    def _check_rsi_buy_signal(self, rsi: float) -> bool:
        """Check RSI buy signal conditions."""
        return rsi < 30
    
    def _check_rsi_sell_signal(self, rsi: float) -> bool:
        """Check RSI sell signal conditions."""
        return rsi > 75
    
    def _check_trend_buy_signal(self, close_price: float, ma_20: float, ma_50: float) -> bool:
        """Check trend buy signal conditions."""
        return ma_20 > ma_50 and close_price > ma_20 * 1.003
    
    def _check_trend_sell_signal(self, close_price: float, ma_20: float, ma_50: float) -> bool:
        """Check trend sell signal conditions."""
        return ma_20 < ma_50 and close_price < ma_20 * 0.997
    
    def _check_breakout_buy_signal(self, close_price: float, ma_20: float) -> bool:
        """Check breakout buy signal conditions."""
        return close_price > ma_20 * 1.01
    
    def _check_breakout_sell_signal(self, close_price: float, ma_20: float) -> bool:
        """Check breakout sell signal conditions."""
        return close_price < ma_20 * 0.99
    
    def _check_rsi_momentum_buy_signal(self, rsi: float, close_price: float, ma_20: float) -> bool:
        """Check RSI momentum buy signal conditions."""
        return 35 <= rsi <= 45 and close_price > ma_20 * 0.999
    
    def _check_rsi_momentum_sell_signal(self, rsi: float, close_price: float, ma_20: float) -> bool:
        """Check RSI momentum sell signal conditions."""
        return 55 <= rsi <= 65 and close_price < ma_20 * 1.001
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size using Kelly Criterion."""
        kelly_fraction = self._calculate_kelly_fraction()
        kelly_fraction = self._apply_position_constraints(kelly_fraction)
        
        position_value = self.cash * kelly_fraction
        return position_value / price
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction."""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        if avg_loss == 0:
            return 0.0
        
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    def _apply_position_constraints(self, kelly_fraction: float) -> float:
        """Apply position size constraints."""
        return max(0.0, min(kelly_fraction, self.max_position_size))
    
    def _execute_trade(self, symbol: str, action: str, price: float, timestamp: datetime, row: pd.Series) -> None:
        """Execute a trade with proper cost calculation."""
        if action == 'buy':
            self._execute_buy_trade(symbol, price, timestamp, row)
        elif action == 'sell':
            self._execute_sell_trade(symbol, price, timestamp, row)
    
    def _execute_buy_trade(self, symbol: str, price: float, timestamp: datetime, row: pd.Series) -> None:
        """Execute a buy trade."""
        shares = self._calculate_position_size(symbol, price)
        
        if shares <= 0:
            return
        
        trade_value = shares * price
        total_cost = self._calculate_trade_costs(trade_value)
        
        if self._can_afford_trade(trade_value, total_cost):
            self._complete_buy_trade(symbol, shares, price, timestamp, total_cost)
        else:
            logger.warning(f"Insufficient cash for {symbol} trade")
    
    def _execute_sell_trade(self, symbol: str, price: float, timestamp: datetime, row: pd.Series) -> None:
        """Execute a sell trade."""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return
        
        shares = self.positions[symbol]
        trade_value = shares * price
        total_cost = self._calculate_trade_costs(trade_value)
        
        self._complete_sell_trade(symbol, shares, price, timestamp, total_cost)
    
    def _calculate_trade_costs(self, trade_value: float) -> float:
        """Calculate total trade costs (commission + slippage)."""
        commission = trade_value * self.fee_rate
        slippage_cost = trade_value * self.slippage
        return commission + slippage_cost
    
    def _can_afford_trade(self, trade_value: float, total_cost: float) -> bool:
        """Check if we can afford the trade."""
        return self.cash >= (trade_value + total_cost)
    
    def _complete_buy_trade(self, symbol: str, shares: float, price: float, timestamp: datetime, total_cost: float) -> None:
        """Complete a buy trade."""
        trade_value = shares * price
        self.cash -= (trade_value + total_cost)
        self.positions[symbol] = shares
        self.position_prices[symbol] = price
        
        self._record_trade(symbol, 'buy', shares, price, timestamp, total_cost)
        logger.info(f"Bought {shares:.2f} shares of {symbol} at ${price:.2f}")
    
    def _complete_sell_trade(self, symbol: str, shares: float, price: float, timestamp: datetime, total_cost: float) -> None:
        """Complete a sell trade."""
        trade_value = shares * price
        self.cash += (trade_value - total_cost)
        
        entry_price = self.position_prices[symbol]
        pnl = (price - entry_price) * shares - total_cost
        
        self._update_trade_statistics(pnl)
        self._record_trade(symbol, 'sell', shares, price, timestamp, total_cost, pnl)
        
        # Clear position
        self.positions[symbol] = 0
        
        logger.info(f"Sold {shares:.2f} shares of {symbol} at ${price:.2f}, P&L: ${pnl:.2f}")
    
    def _update_trade_statistics(self, pnl: float) -> None:
        """Update trade statistics."""
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def _record_trade(self, symbol: str, action: str, shares: float, price: float, timestamp: datetime, costs: float, pnl: float = 0.0) -> None:
        """Record trade details in history."""
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
    
    def _update_portfolio_equity(self, timestamp: datetime, current_data: pd.DataFrame) -> None:
        """Update portfolio equity curve."""
        portfolio_value = self._calculate_portfolio_value(current_data)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash
        })
    
    def _calculate_portfolio_value(self, current_data: pd.DataFrame) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash
        
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            if symbol in self.positions and self.positions[symbol] > 0:
                portfolio_value += self.positions[symbol] * row['close']
        
        return portfolio_value
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return
        
        self.performance_metrics = {
            'total_return': self._calculate_total_return(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self._calculate_win_rate(),
            'total_pnl': self.total_pnl,
            'final_portfolio_value': self.equity_curve[-1]['portfolio_value']
        }
        
        self._log_performance_summary()
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage."""
        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]['portfolio_value']
        return (final_value - initial_value) / initial_value
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        returns = self._calculate_daily_returns()
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_daily_returns(self) -> List[float]:
        """Calculate daily returns from equity curve."""
        returns = []
        
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1]['portfolio_value']
            curr_value = self.equity_curve[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        return returns
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        peak_value = self.initial_capital
        max_drawdown = 0.0
        
        for point in self.equity_curve:
            if point['portfolio_value'] > peak_value:
                peak_value = point['portfolio_value']
            else:
                drawdown = (peak_value - point['portfolio_value']) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        return self.winning_trades / max(self.total_trades, 1)
    
    def _log_performance_summary(self) -> None:
        """Log performance summary."""
        metrics = self.performance_metrics
        logger.info(
            f"Performance metrics calculated: "
            f"Return: {metrics['total_return']:.2%}, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
            f"Max DD: {metrics['max_drawdown']:.2%}"
        )
    
    def _save_backtest_results(self) -> None:
        """Save all backtest results to files."""
        try:
            reports_dir = self._create_reports_directory()
            self._save_trade_history(reports_dir)
            self._save_equity_curve(reports_dir)
            self._save_performance_metrics(reports_dir)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _create_reports_directory(self) -> Path:
        """Create reports directory."""
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        return reports_dir
    
    def _save_trade_history(self, reports_dir: Path) -> None:
        """Save trade history to file."""
        if not self.trade_history:
            return
        
        trades_df = pd.DataFrame(self.trade_history)
        trades_file = reports_dir / f'trades_{self.run_id}.parquet'
        trades_df.to_parquet(trades_file)
        logger.info(f"Trade history saved to {trades_file}")
    
    def _save_equity_curve(self, reports_dir: Path) -> None:
        """Save equity curve to file."""
        if not self.equity_curve:
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_file = reports_dir / f'equity_curve_{self.run_id}.parquet'
        equity_df.to_parquet(equity_file)
        logger.info(f"Equity curve saved to {equity_file}")
    
    def _save_performance_metrics(self, reports_dir: Path) -> None:
        """Save performance metrics to file."""
        if not hasattr(self, 'performance_metrics'):
            return
        
        import json
        metrics_file = reports_dir / f'performance_metrics_{self.run_id}.json'
        
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        logger.info(f"Performance metrics saved to {metrics_file}")
    
    def get_results(self) -> Dict:
        """Get comprehensive backtest results."""
        return {
            'performance_metrics': getattr(self, 'performance_metrics', {}),
            'trade_history': self.trade_history,
            'equity_curve': self.equity_curve,
            'run_id': self.run_id
        }
