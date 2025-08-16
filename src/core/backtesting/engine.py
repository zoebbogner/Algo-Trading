"""Core backtesting engine for strategy validation."""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..data_models.market import Bar, MarketData
from ..data_models.trading import Order, Position, Portfolio, OrderSide, OrderType
from ..strategies.base import Strategy
from ..risk.manager import RiskManager
from ...adapters.data.manager import DataManager
from ...utils.logging import logger


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    
    # Performance metrics
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    
    # Portfolio metrics
    final_portfolio_value: Decimal
    peak_portfolio_value: Decimal
    final_cash: Decimal
    final_exposure: Decimal
    
    # Time metrics
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Strategy details
    strategy_name: str
    parameters: Dict[str, Any]
    
    # Detailed data
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]


class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(
        self, 
        strategy: Strategy,
        data_manager: DataManager,
        risk_manager: RiskManager,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),  # 0.1%
        slippage_rate: Decimal = Decimal("0.0005"),   # 0.05%
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Initialize backtesting engine.
        
        Args:
            strategy: Trading strategy to test
            data_manager: Data manager for market data
            risk_manager: Risk management system
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            start_date: Backtest start date
            end_date: Backtest end date
        """
        self.strategy = strategy
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.start_date = start_date or (datetime.now(timezone.utc) - timedelta(days=30))
        self.end_date = end_date or datetime.now(timezone.utc)
        
        # Portfolio state
        self.portfolio = Portfolio(
            cash=initial_capital,
            equity=initial_capital,
            exposure=Decimal("0"),
            drawdown=Decimal("0"),
            peak_equity=initial_capital,
            total_pnl=Decimal("0"),
            positions={}
        )
        
        # Backtest state
        self.current_date: Optional[datetime] = None
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.positions: List[Dict[str, Any]] = []
        self.market_data_cache: Dict[str, List[Bar]] = {}
        
        logger.logger.info(f"Backtest engine initialized with {initial_capital} capital")
    
    async def run_backtest(
        self, 
        symbols: List[str], 
        timeframe: str,
        include_features: bool = True
    ) -> BacktestResult:
        """Run the backtest.
        
        Args:
            symbols: List of symbols to trade
            timeframe: Data timeframe
            include_features: Whether to include technical features
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.logger.info(f"Starting backtest for {len(symbols)} symbols from {self.start_date} to {self.end_date}")
        
        # Initialize strategy
        self.strategy.initialize()
        
        # Get historical data for all symbols
        await self._load_historical_data(symbols, timeframe)
        
        # Run simulation
        await self._run_simulation(symbols, timeframe, include_features)
        
        # Calculate results
        result = self._calculate_results()
        
        logger.logger.info(f"Backtest completed. Final portfolio value: {result.final_portfolio_value}")
        return result
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> None:
        """Load historical data for all symbols."""
        logger.logger.info("Loading historical data...")
        
        for symbol in symbols:
            try:
                # Get data from start_date to end_date
                bars = await self.data_manager.get_historical_data(
                    symbol, 
                    timeframe, 
                    limit=1000,  # Large enough for most timeframes
                    since=self.start_date
                )
                
                if bars:
                    # Filter by date range
                    filtered_bars = [
                        bar for bar in bars 
                        if self.start_date <= bar.timestamp <= self.end_date
                    ]
                    
                    if filtered_bars:
                        self.market_data_cache[symbol] = sorted(filtered_bars, key=lambda x: x.timestamp)
                        logger.logger.info(f"Loaded {len(filtered_bars)} bars for {symbol}")
                    else:
                        logger.logger.warning(f"No data in date range for {symbol}")
                else:
                    logger.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.logger.error(f"Error loading data for {symbol}: {e}")
    
    async def _run_simulation(
        self, 
        symbols: List[str], 
        timeframe: str, 
        include_features: bool
    ) -> None:
        """Run the trading simulation."""
        logger.logger.info("Running trading simulation...")
        
        # Get all unique timestamps across symbols
        all_timestamps = set()
        for bars in self.market_data_cache.values():
            all_timestamps.update(bar.timestamp for bar in bars)
        
        # Sort timestamps chronologically
        sorted_timestamps = sorted(all_timestamps)
        
        # Run simulation tick by tick
        for timestamp in sorted_timestamps:
            self.current_date = timestamp
            
            # Update portfolio with current market data
            await self._update_portfolio(timestamp, symbols, timeframe, include_features)
            
            # Generate strategy signals
            signals = await self._generate_signals(timestamp, symbols, timeframe, include_features)
            
            # Execute trades based on signals
            await self._execute_trades(signals, timestamp)
            
            # Record portfolio state
            self._record_portfolio_state(timestamp)
            
            # Risk management checks
            await self._risk_management_checks(timestamp)
    
    async def _update_portfolio(
        self, 
        timestamp: datetime, 
        symbols: List[str], 
        timeframe: str, 
        include_features: bool
    ) -> None:
        """Update portfolio with current market data."""
        for symbol in symbols:
            if symbol in self.market_data_cache:
                # Find the bar for this timestamp
                bars = self.market_data_cache[symbol]
                current_bar = None
                
                for bar in bars:
                    if bar.timestamp == timestamp:
                        current_bar = bar
                        break
                
                if current_bar:
                    # Update position values
                    if symbol in self.portfolio.positions:
                        position = self.portfolio.positions[symbol]
                        position.market_value = position.quantity * current_bar.close
                        position.last_update_timestamp = timestamp
                        
                        # Calculate unrealized P&L
                        position.unrealized_pnl = position.market_value - position.cost_basis
    
    async def _generate_signals(
        self, 
        timestamp: datetime, 
        symbols: List[str], 
        timeframe: str, 
        include_features: bool
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from strategy."""
        signals = []
        
        for symbol in symbols:
            if symbol in self.market_data_cache:
                # Get market data for signal generation
                bars = self.market_data_cache[symbol]
                
                # Find bars up to current timestamp
                current_bars = [bar for bar in bars if bar.timestamp <= timestamp]
                
                if len(current_bars) >= 50:  # Need enough data for features
                    try:
                        # Create market data object
                        latest_bar = current_bars[-1]
                        
                        # Calculate features if requested
                        features = []
                        if include_features:
                            from ...core.features.technical import FeatureEngine
                            feature_engine = FeatureEngine()
                            features_dict = feature_engine.calculate_features(current_bars)
                            
                            for feature_name, feature_value in features_dict.items():
                                if isinstance(feature_value, (int, float)) and feature_value == feature_value:  # Not NaN
                                    from ...core.data_models.market import Feature
                                    feature = Feature(
                                        timestamp=latest_bar.timestamp,
                                        symbol=symbol,
                                        feature_name=feature_name,
                                        feature_value=float(feature_value),
                                        source="technical"
                                    )
                                    features.append(feature)
                        
                        market_data = MarketData(
                            timestamp=latest_bar.timestamp,
                            symbol=symbol,
                            bar=latest_bar,
                            features=features,
                            timestamp_received=timestamp
                        )
                        
                        # Generate signal
                        signal = self.strategy.generate_signal(market_data)
                        if signal:
                            signals.append({
                                'symbol': symbol,
                                'signal': signal,
                                'timestamp': timestamp,
                                'market_data': market_data
                            })
                            
                    except Exception as e:
                        logger.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _execute_trades(self, signals: List[Dict[str, Any]], timestamp: datetime) -> None:
        """Execute trades based on signals."""
        for signal_data in signals:
            signal = signal_data['signal']
            symbol = signal_data['symbol']
            
            try:
                # Check risk management
                if not await self.risk_manager.check_trade_allowed(signal, self.portfolio):
                    logger.logger.debug(f"Trade blocked by risk manager for {symbol}")
                    continue
                
                # Execute the trade
                if signal.side == OrderSide.BUY:
                    await self._execute_buy_order(signal, symbol, timestamp)
                elif signal.side == OrderSide.SELL:
                    await self._execute_sell_order(signal, symbol, timestamp)
                    
            except Exception as e:
                logger.logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def _execute_buy_order(self, signal: Any, symbol: str, timestamp: datetime) -> None:
        """Execute a buy order."""
        # Get current price
        current_price = self._get_current_price(symbol, timestamp)
        if not current_price:
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, current_price)
        if position_size <= 0:
            return
        
        # Check if we have enough cash
        total_cost = position_size * current_price
        if total_cost > self.portfolio.cash:
            # Adjust position size to available cash
            position_size = self.portfolio.cash / current_price
            total_cost = self.portfolio.cash
        
        # Calculate fees and slippage
        commission = total_cost * self.commission_rate
        slippage = total_cost * self.slippage_rate
        total_cost_with_fees = total_cost + commission + slippage
        
        # Update portfolio
        self.portfolio.cash -= total_cost_with_fees
        
        # Create or update position
        if symbol in self.portfolio.positions:
            # Add to existing position
            position = self.portfolio.positions[symbol]
            old_quantity = position.quantity
            old_cost = position.cost_basis
            
            new_quantity = old_quantity + position_size
            new_cost = old_cost + total_cost
            
            position.quantity = new_quantity
            position.cost_basis = new_cost
            position.avg_entry_price = new_cost / new_quantity
            position.last_update_timestamp = timestamp
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                quantity=position_size,
                cost_basis=total_cost,
                avg_entry_price=current_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                market_value=position_size * current_price,
                entry_timestamp=timestamp,
                last_update_timestamp=timestamp
            )
            self.portfolio.positions[symbol] = position
        
        # Record trade
        self._record_trade(symbol, OrderSide.BUY, position_size, current_price, timestamp)
        
        logger.logger.debug(f"Executed BUY order: {position_size} {symbol} @ {current_price}")
    
    async def _execute_sell_order(self, signal: Any, symbol: str, timestamp: datetime) -> None:
        """Execute a sell order."""
        if symbol not in self.portfolio.positions:
            return
        
        position = self.portfolio.positions[symbol]
        if position.quantity <= 0:
            return
        
        # Get current price
        current_price = self._get_current_price(symbol, timestamp)
        if not current_price:
            return
        
        # Calculate sell quantity
        sell_quantity = min(position.quantity, signal.quantity) if hasattr(signal, 'quantity') else position.quantity
        
        # Calculate proceeds
        gross_proceeds = sell_quantity * current_price
        commission = gross_proceeds * self.commission_rate
        slippage = gross_proceeds * self.slippage_rate
        net_proceeds = gross_proceeds - commission - slippage
        
        # Calculate P&L
        cost_basis = (sell_quantity / position.quantity) * position.cost_basis
        realized_pnl = gross_proceeds - cost_basis - commission - slippage
        
        # Update portfolio
        self.portfolio.cash += net_proceeds
        
        # Update position
        position.quantity -= sell_quantity
        position.cost_basis -= cost_basis
        position.realized_pnl += realized_pnl
        
        if position.quantity <= 0:
            # Close position
            del self.portfolio.positions[symbol]
        else:
            # Update average entry price
            position.avg_entry_price = position.cost_basis / position.quantity
            position.last_update_timestamp = timestamp
        
        # Record trade
        self._record_trade(symbol, OrderSide.SELL, sell_quantity, current_price, timestamp)
        
        logger.logger.debug(f"Executed SELL order: {sell_quantity} {symbol} @ {current_price}")
    
    def _get_current_price(self, symbol: str, timestamp: datetime) -> Optional[Decimal]:
        """Get current price for a symbol at a specific timestamp."""
        if symbol in self.market_data_cache:
            bars = self.market_data_cache[symbol]
            for bar in bars:
                if bar.timestamp == timestamp:
                    return bar.close
        return None
    
    def _calculate_position_size(self, signal: Any, current_price: Decimal) -> Decimal:
        """Calculate position size based on signal and risk management."""
        # Simple position sizing - can be enhanced
        if hasattr(signal, 'quantity') and signal.quantity:
            return signal.quantity
        
        # Default to 10% of available cash
        return (self.portfolio.cash * Decimal("0.1")) / current_price
    
    def _record_trade(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal, timestamp: datetime) -> None:
        """Record a trade for analysis."""
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side.value,
            'quantity': float(quantity),
            'price': float(price),
            'value': float(quantity * price),
            'portfolio_value': float(self.portfolio.equity)
        }
        self.trades.append(trade)
    
    def _record_portfolio_state(self, timestamp: datetime) -> None:
        """Record portfolio state for equity curve."""
        # Calculate current portfolio value
        total_position_value = sum(
            pos.market_value for pos in self.portfolio.positions.values()
        )
        
        self.portfolio.equity = self.portfolio.cash + total_position_value
        self.portfolio.exposure = total_position_value
        
        # Update peak equity and drawdown
        if self.portfolio.equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.equity
        
        if self.portfolio.peak_equity > 0:
            self.portfolio.drawdown = (self.portfolio.peak_equity - self.portfolio.equity) / self.portfolio.peak_equity
        
        # Record state
        state = {
            'timestamp': timestamp,
            'equity': float(self.portfolio.equity),
            'cash': float(self.portfolio.cash),
            'exposure': float(self.portfolio.exposure),
            'drawdown': float(self.portfolio.drawdown),
            'peak_equity': float(self.portfolio.peak_equity)
        }
        self.equity_curve.append(state)
    
    async def _risk_management_checks(self, timestamp: datetime) -> None:
        """Perform risk management checks."""
        try:
            # Check circuit breakers
            if await self.risk_manager.check_circuit_breaker(self.portfolio):
                logger.logger.warning("Circuit breaker triggered - stopping trading")
                # Could implement emergency stop here
            
            # Check position limits
            await self.risk_manager.check_position_limits(self.portfolio)
            
        except Exception as e:
            logger.logger.error(f"Error in risk management checks: {e}")
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results."""
        if not self.equity_curve:
            raise ValueError("No equity curve data available")
        
        # Basic metrics
        start_equity = self.equity_curve[0]['equity']
        end_equity = self.equity_curve[-1]['equity']
        total_return = end_equity - start_equity
        total_return_pct = (total_return / start_equity) * 100
        
        # Duration
        start_date = self.equity_curve[0]['timestamp']
        end_date = self.equity_curve[-1]['timestamp']
        duration_days = (end_date - start_date).days
        
        # Annualized return
        if duration_days > 0:
            annualized_return = ((end_equity / start_equity) ** (365 / duration_days) - 1) * 100
        else:
            annualized_return = Decimal("0")
        
        # Drawdown
        max_drawdown = max(state['drawdown'] for state in self.equity_curve)
        max_drawdown_pct = max_drawdown * 100
        
        # Trade analysis
        winning_trades = [t for t in self.trades if t['side'] == 'SELL' and t['value'] > 0]
        losing_trades = [t for t in self.trades if t['side'] == 'SELL' and t['value'] <= 0]
        
        total_trades = len([t for t in self.trades if t['side'] == 'SELL'])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate returns for each trade
        trade_returns = []
        for i, trade in enumerate(self.trades):
            if trade['side'] == 'SELL':
                # Find corresponding buy trade
                buy_trades = [t for t in self.trades[:i] if t['symbol'] == trade['symbol'] and t['side'] == 'BUY']
                if buy_trades:
                    buy_trade = buy_trades[-1]
                    buy_value = buy_trade['value']
                    sell_value = trade['value']
                    trade_return = (sell_value - buy_value) / buy_value
                    trade_returns.append(trade_return)
        
        # Performance metrics
        if trade_returns:
            avg_win = sum(r for r in trade_returns if r > 0) / len([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
            avg_loss = sum(r for r in trade_returns if r < 0) / len([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
            largest_win = max(trade_returns) if trade_returns else 0
            largest_loss = min(trade_returns) if trade_returns else 0
            
            # Sharpe ratio (simplified)
            if len(trade_returns) > 1:
                mean_return = sum(trade_returns) / len(trade_returns)
                std_return = (sum((r - mean_return) ** 2 for r in trade_returns) / (len(trade_returns) - 1)) ** 0.5
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Profit factor
            gross_profit = sum(r for r in trade_returns if r > 0)
            gross_loss = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            avg_win = avg_loss = largest_win = largest_loss = sharpe_ratio = profit_factor = 0
        
        return BacktestResult(
            total_return=Decimal(str(total_return)),
            total_return_pct=Decimal(str(total_return_pct)),
            annualized_return=Decimal(str(annualized_return)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            max_drawdown_pct=Decimal(str(max_drawdown_pct)),
            win_rate=Decimal(str(win_rate)),
            profit_factor=Decimal(str(profit_factor)),
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=Decimal(str(avg_win)),
            avg_loss=Decimal(str(avg_loss)),
            largest_win=Decimal(str(largest_win)),
            largest_loss=Decimal(str(largest_loss)),
            final_portfolio_value=Decimal(str(end_equity)),
            peak_portfolio_value=Decimal(str(max(state['equity'] for state in self.equity_curve))),
            final_cash=Decimal(str(self.portfolio.cash)),
            final_exposure=Decimal(str(self.portfolio.exposure)),
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            strategy_name=self.strategy.__class__.__name__,
            parameters=self.strategy.get_parameters(),
            equity_curve=self.equity_curve,
            trades=self.trades,
            positions=[{
                'symbol': pos.symbol,
                'quantity': float(pos.quantity),
                'cost_basis': float(pos.cost_basis),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pnl)
            } for pos in self.portfolio.positions.values()]
        )
