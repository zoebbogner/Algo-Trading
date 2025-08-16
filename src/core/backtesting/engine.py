"""
Backtesting Engine

Core engine for running backtests with comprehensive performance tracking
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import logging

from ..data_models.market import MarketData
from ..data_models.trading import Portfolio, Position, Order, Fill
from ..strategy.base import Strategy
from .analyzer import BacktestResult
from ...adapters.data.manager import DataManager
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine that simulates trading strategies on historical data
    """
    
    def __init__(self, strategy: Strategy, data_manager: DataManager, 
                 risk_manager: RiskManager, initial_capital: float = 10000.0):
        self.strategy = strategy
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.initial_capital = Decimal(str(initial_capital))
        
        # Portfolio state
        self.portfolio = Portfolio(
            cash=self.initial_capital,
            equity=self.initial_capital,
            exposure_gross=Decimal("0"),
            exposure_net=Decimal("0"),
            drawdown=Decimal("0"),
            peak_equity=self.initial_capital,
            pnl_total=Decimal("0"),
            pnl_daily=Decimal("0"),
            positions=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        # Backtest state
        self.equity_curve = []
        self.trades = []
        self.signals = []
        self.hourly_pnl = []  # New: Track hourly P&L
        self.portfolio_snapshots = []  # New: Track portfolio state each hour
        
        # Performance tracking
        self.start_time = None
        self.end_date = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Backtest engine initialized with {initial_capital} capital")
    
    async def run_backtest(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, timeframe: str = "1h") -> BacktestResult:
        """
        Run a complete backtest
        """
        self.start_time = start_date
        self.end_date = end_date
        
        logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Load historical data
            await self._load_historical_data(symbols, start_date, end_date, timeframe)
            
            # Run simulation
            await self._run_simulation()
            
            # Calculate results
            results = self._calculate_results()
            
            logger.info(f"Backtest completed. Final portfolio value: {self.portfolio.equity}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _load_historical_data(self, symbols: List[str], start_date: datetime, 
                                   end_date: datetime, timeframe: str):
        """Load historical data for all symbols"""
        logger.info("Loading historical data...")
        
        # Ensure data manager is connected
        if not any(adapter.is_connected() for adapter in self.data_manager.adapters.values()):
            await self.data_manager.connect_all()
        
        self.historical_data = {}
        
        for symbol in symbols:
            try:
                bars = await self.data_manager.get_historical_data_range(
                    symbol, start_date, end_date, timeframe
                )
                if bars:
                    self.historical_data[symbol] = bars
                    logger.info(f"Loaded {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
    
    async def _run_simulation(self):
        """Run the trading simulation tick by tick"""
        logger.info("Running trading simulation...")
        
        # Get all unique timestamps
        all_timestamps = set()
        for bars in self.historical_data.values():
            all_timestamps.update(bar.timestamp for bar in bars)
        
        sorted_timestamps = sorted(all_timestamps)
        
        if not sorted_timestamps:
            # Record initial state if no data
            self._record_portfolio_state(datetime.now(timezone.utc))
            return
        
        # Record initial portfolio state
        self._record_portfolio_state(sorted_timestamps[0])
        
        # Process each timestamp
        for timestamp in sorted_timestamps:
            # Create market data snapshot for this timestamp
            market_data = self._create_market_data_snapshot(timestamp)
            
            if not market_data.bars:
                continue
            
            # Generate trading signals
            signals = self._generate_signals(market_data)
            
            # Execute trades
            if signals:
                await self._execute_trades(signals, timestamp)
            
            # Update portfolio state
            self._update_portfolio(market_data, timestamp)
            
            # Risk management checks
            self._risk_management_checks(timestamp)
            
            # Record portfolio state for this hour
            self._record_portfolio_state(timestamp)
            
            # Record hourly P&L
            self._record_hourly_pnl(timestamp)
    
    def _create_market_data_snapshot(self, timestamp: datetime) -> MarketData:
        """Create market data snapshot for a specific timestamp"""
        bars = {}
        features = {}
        
        for symbol, symbol_bars in self.historical_data.items():
            # Find the bar closest to this timestamp
            closest_bar = None
            min_diff = timedelta(hours=1)  # Within 1 hour
            
            for bar in symbol_bars:
                diff = abs(bar.timestamp - timestamp)
                if diff <= min_diff:
                    min_diff = diff
                    closest_bar = bar
            
            if closest_bar:
                bars[symbol] = [closest_bar]
                
                # Calculate features for this bar
                symbol_features = self._calculate_features(symbol, symbol_bars, closest_bar)
                if symbol_features:
                    features[symbol] = symbol_features
        
        return MarketData(
            bars=bars,
            features=features,
            timestamp=timestamp,
            timestamp_received=timestamp
        )
    
    def _calculate_features(self, symbol: str, all_bars: List, current_bar) -> List:
        """Calculate technical features for a symbol"""
        features = []
        
        if len(all_bars) < 20:
            return features
        
        # Get recent prices
        recent_bars = all_bars[-20:]
        prices = [float(bar.close) for bar in recent_bars]
        volumes = [float(bar.volume) for bar in recent_bars]
        
        # Simple moving averages
        if len(prices) >= 10:
            sma_10 = sum(prices[-10:]) / 10
            features.append({
                'timestamp': current_bar.timestamp,
                'symbol': symbol,
                'feature_name': 'SMA_10',
                'feature_value': sma_10,
                'lookback_period': 10,
                'source': 'price'
            })
        
        if len(prices) >= 20:
            sma_20 = sum(prices[-20:]) / 20
            features.append({
                'timestamp': current_bar.timestamp,
                'symbol': symbol,
                'feature_name': 'SMA_20',
                'feature_value': sma_20,
                'lookback_period': 20,
                'source': 'price'
            })
        
        # RSI
        if len(prices) >= 14:
            rsi = self._calculate_rsi(prices)
            if rsi is not None:
                features.append({
                    'timestamp': current_bar.timestamp,
                    'symbol': symbol,
                    'feature_name': 'RSI',
                    'feature_value': rsi,
                    'lookback_period': 14,
                    'source': 'price'
                })
        
        # Volume ratio
        if len(volumes) >= 20:
            current_volume = volumes[-1]
            avg_volume = sum(volumes[-20:]) / 20
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            features.append({
                'timestamp': current_bar.timestamp,
                'symbol': symbol,
                'feature_name': 'Volume_Ratio',
                'feature_value': volume_ratio,
                'lookback_period': 20,
                'source': 'volume'
            })
        
        return features
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gains = sum(gains[-period:]) / period
        avg_losses = sum(losses[-period:]) / period
        
        if avg_losses == 0:
            return 100
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_signals(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Generate trading signals from strategy"""
        try:
            strategy_signals = self.strategy.generate_signals(market_data, self.portfolio)
            
            signals = []
            
            # Process entry signals
            for symbol, signal_data in strategy_signals.get('entry_signals', {}).items():
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'entry',
                    'side': signal_data.get('side'),
                    'price': signal_data.get('price'),
                    'quantity': signal_data.get('quantity'),
                    'reason': signal_data.get('reason'),
                    'timestamp': signal_data.get('timestamp'),
                    'details': signal_data
                })
            
            # Process exit signals
            for symbol, signal_data in strategy_signals.get('exit_signals', {}).items():
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'exit',
                    'side': signal_data.get('side'),
                    'price': signal_data.get('price'),
                    'quantity': signal_data.get('quantity'),
                    'reason': signal_data.get('reason'),
                    'timestamp': signal_data.get('timestamp'),
                    'details': signal_data
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _execute_trades(self, signals: List[Dict[str, Any]], timestamp: datetime):
        """Execute trading signals"""
        for signal in signals:
            try:
                if signal['signal_type'] == 'entry':
                    if signal['side'] == 'buy':
                        await self._execute_buy_order(signal, timestamp)
                    elif signal['side'] == 'sell':
                        await self._execute_sell_order(signal, timestamp)
                elif signal['signal_type'] == 'exit':
                    if signal['side'] == 'sell':
                        await self._execute_sell_order(signal, timestamp)
                    elif signal['side'] == 'buy':
                        await self._execute_buy_order(signal, timestamp)
                        
            except Exception as e:
                logger.error(f"Error executing trade: {e}")
    
    async def _execute_buy_order(self, signal: Dict[str, Any], timestamp: datetime):
        """Execute a buy order"""
        symbol = signal['symbol']
        price = Decimal(str(signal['price']))
        quantity = signal['quantity']
        
        # Risk management check (temporarily disabled)
        # if not self.risk_manager.check_trade_allowed(signal):
        #     return
        
        # Calculate position size
        position_size = self._calculate_position_size(price, quantity)
        
        # Check if we have enough cash
        if position_size > self.portfolio.cash:
            logger.warning(f"Insufficient cash for {symbol} buy order")
            return
        
        # Create fill
        fill = Fill(
            price=price,
            quantity=Decimal(str(quantity)),
            fee=Decimal("0"),  # Simplified for backtesting
            slippage=Decimal("0"),
            value=position_size,
            timestamp=timestamp
        )
        
        # Update portfolio
        self.portfolio.cash -= position_size
        
        # Find or create position
        existing_position = None
        for pos in self.portfolio.positions:
            if pos.symbol == symbol:
                existing_position = pos
                break
        
        if existing_position:
            # Update existing position
            total_quantity = existing_position.quantity + Decimal(str(quantity))
            total_cost = existing_position.average_cost * existing_position.quantity + position_size
            existing_position.average_cost = total_cost / total_quantity
            existing_position.quantity = total_quantity
            existing_position.last_update_timestamp = timestamp
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                quantity=Decimal(str(quantity)),
                average_cost=price,
                market_value=position_size,
                entry_timestamp=timestamp,
                last_update_timestamp=timestamp
            )
            self.portfolio.positions.append(position)
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'buy',
            'quantity': quantity,
            'price': float(price),
            'value': float(position_size),
            'reason': signal.get('reason', ''),
            'fill': fill
        })
        
        self.total_trades += 1
        logger.info(f"Executed BUY order: {quantity} {symbol} @ {price}")
    
    async def _execute_sell_order(self, signal: Dict[str, Any], timestamp: datetime):
        """Execute a sell order"""
        symbol = signal['symbol']
        price = Decimal(str(signal['price']))
        quantity = signal['quantity']
        
        # Find position
        position_index = None
        for i, pos in enumerate(self.portfolio.positions):
            if pos.symbol == symbol:
                position_index = i
                break
        
        if position_index is None:
            logger.warning(f"No position found for {symbol} sell order")
            return
        
        position = self.portfolio.positions[position_index]
        
        # Check if we have enough quantity
        if abs(position.quantity) < quantity:
            logger.warning(f"Insufficient quantity for {symbol} sell order")
            return
        
        # Calculate P&L
        cost_basis = position.average_cost * Decimal(str(quantity))
        sale_value = price * Decimal(str(quantity))
        pnl = sale_value - cost_basis
        
        # Create fill
        fill = Fill(
            price=price,
            quantity=Decimal(str(quantity)),
            fee=Decimal("0"),
            slippage=Decimal("0"),
            value=float(sale_value),
            timestamp=timestamp
        )
        
        # Update portfolio
        self.portfolio.cash += sale_value
        self.portfolio.pnl_total += pnl
        
        # Update position
        if position.quantity > 0:  # Long position
            position.quantity -= Decimal(str(quantity))
        else:  # Short position
            position.quantity += Decimal(str(quantity))
        
        # Remove position if fully closed
        if position.quantity == 0:
            self.portfolio.positions.pop(position_index)
        else:
            position.last_update_timestamp = timestamp
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'sell',
            'quantity': quantity,
            'price': float(price),
            'value': float(sale_value),
            'pnl': float(pnl),
            'reason': signal.get('reason', ''),
            'fill': fill
        })
        
        self.total_trades += 1
        
        # Update win/loss counts
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"Executed SELL order: {quantity} {symbol} @ {price}, P&L: {pnl}")
    
    def _calculate_position_size(self, price: Decimal, quantity: float) -> Decimal:
        """Calculate the dollar value of a position"""
        return price * Decimal(str(quantity))
    
    def _update_portfolio(self, market_data: MarketData, timestamp: datetime):
        """Update portfolio with current market data"""
        total_market_value = Decimal("0")
        
        for symbol, bars in market_data.bars.items():
            if not bars:
                continue
            
            current_price = bars[0].close
            
            # Find position for this symbol
            for position in self.portfolio.positions:
                if position.symbol == symbol:
                    # Update market value
                    position.market_value = current_price * abs(position.quantity)
                    position.last_update_timestamp = timestamp
                    total_market_value += position.market_value
        
        # Update portfolio equity
        self.portfolio.equity = self.portfolio.cash + total_market_value
        
        # Update drawdown
        if self.portfolio.equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.equity
        
        current_drawdown = (self.portfolio.peak_equity - self.portfolio.equity) / self.portfolio.peak_equity
        if current_drawdown > self.portfolio.drawdown:
            self.portfolio.drawdown = current_drawdown
    
    def _risk_management_checks(self, timestamp: datetime):
        """Perform risk management checks"""
        try:
            # Temporarily disabled - methods not yet implemented
            # self.risk_manager.check_circuit_breaker(self.portfolio)
            # self.risk_manager.check_position_limits(self.portfolio)
            pass
        except Exception as e:
            logger.warning(f"Risk management check failed: {e}")
    
    def _record_portfolio_state(self, timestamp: datetime):
        """Record portfolio state for analysis"""
        # Calculate exposure
        exposure_gross = sum(abs(pos.market_value) for pos in self.portfolio.positions)
        exposure_net = sum(pos.market_value for pos in self.portfolio.positions)
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': float(self.portfolio.equity),
            'cash': float(self.portfolio.cash),
            'exposure_gross': float(exposure_gross),
            'exposure_net': float(exposure_net),
            'drawdown': float(self.portfolio.drawdown),
            'positions_count': len(self.portfolio.positions)
        })
        
        # Record portfolio snapshot
        self.portfolio_snapshots.append({
            'timestamp': timestamp,
            'portfolio': self.portfolio.copy() if hasattr(self.portfolio, 'copy') else self.portfolio
        })
    
    def _record_hourly_pnl(self, timestamp: datetime):
        """Record hourly P&L for detailed tracking"""
        if len(self.equity_curve) < 2:
            return
        
        current_equity = self.equity_curve[-1]['equity']
        previous_equity = self.equity_curve[-2]['equity']
        hourly_pnl = current_equity - previous_equity
        hourly_pnl_pct = (hourly_pnl / previous_equity * 100) if previous_equity > 0 else 0
        
        # Get current positions for this hour
        current_positions = []
        for pos in self.portfolio.positions:
            current_positions.append({
                'symbol': pos.symbol,
                'quantity': float(pos.quantity),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.market_value - (pos.average_cost * abs(pos.quantity)))
            })
        
        # Get trades for this hour
        hourly_trades = []
        for trade in self.trades:
            if trade['timestamp'].hour == timestamp.hour and trade['timestamp'].date() == timestamp.date():
                hourly_trades.append({
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'value': trade['value'],
                    'pnl': trade.get('pnl', 0),
                    'reason': trade.get('reason', '')
                })
        
        self.hourly_pnl.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'hourly_pnl': hourly_pnl,
            'hourly_pnl_pct': hourly_pnl_pct,
            'positions': current_positions,
            'trades': hourly_trades,
            'cash': float(self.portfolio.cash),
            'exposure': float(sum(abs(pos.market_value) for pos in self.portfolio.positions))
        })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not self.equity_curve:
            return BacktestResult(
                strategy_name=self.strategy.name,
                start_date=self.start_time,
                end_date=self.end_date,
                initial_capital=float(self.initial_capital),
                final_capital=float(self.portfolio.equity),
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=self.total_trades,
                win_rate=0.0,
                profit_factor=0.0,
                equity_curve=[],
                trades=self.trades,
                hourly_pnl=self.hourly_pnl,  # New: Include hourly P&L
                portfolio_snapshots=self.portfolio_snapshots,  # New: Include portfolio snapshots
                parameters=self.strategy.config
            )
        
        # Calculate returns
        initial_equity = self.equity_curve[0]['equity']
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate annualized return
        if self.start_time and self.end_date:
            duration_days = (self.end_date - self.start_time).days
            if duration_days > 0:
                annualized_return = ((1 + total_return) ** (365 / duration_days)) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]['equity']
            curr_equity = self.equity_curve[i]['equity']
            if prev_equity > 0:
                returns.append((curr_equity - prev_equity) / prev_equity)
        
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Calculate profit factor
        winning_pnl = sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) > 0)
        losing_pnl = abs(sum(trade.get('pnl', 0) for trade in self.trades if trade.get('pnl', 0) < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0.0
        
        # Get max drawdown
        max_drawdown = max(equity['drawdown'] for equity in self.equity_curve)
        
        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=self.start_time,
            end_date=self.end_date,
            initial_capital=float(self.initial_capital),
            final_capital=float(self.portfolio.equity),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=self.total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=self.equity_curve,
            trades=self.trades,
            hourly_pnl=self.hourly_pnl,  # New: Include hourly P&L
            portfolio_snapshots=self.portfolio_snapshots,  # New: Include portfolio snapshots
            parameters=self.strategy.config
        )
