#!/usr/bin/env python3
"""
Backtesting CLI for the Algo-Trading system.

Provides command-line interface for:
- Strategy backtesting execution
- Performance analysis and reporting
- Strategy comparison and optimization
- Risk management and position sizing
"""

import sys
from pathlib import Path

import click
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.backtest.engine import BacktestEngine
from src.utils.config import load_merged_config
from src.utils.logging import get_logger, setup_logging


@click.group()
@click.option('--config', '-c', default='configs/base.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def backtesting(ctx, config: str, verbose: bool):
    """Backtesting CLI for Algo-Trading system."""
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level=log_level)

    # Load configuration
    base_config = load_merged_config()
    ctx.obj = base_config

    # Ensure reports directory exists
    Path(base_config['reports_root']).mkdir(parents=True, exist_ok=True)


@backtesting.command()
@click.option('--data-path', '-d', required=True, help='Path to features dataset')
@click.option('--symbols', '-s', multiple=True,
              default=['BTCUSDT', 'ETHUSDT'], help='Trading symbols to backtest')
@click.option('--start-date', help='Start date for backtest (YYYY-MM-DD)')
@click.option('--end-date', help='End date for backtest (YYYY-MM-DD)')
@click.option('--initial-cash', '-c', default=100000, help='Initial cash amount')
@click.option('--fee-rate', '-f', default=0.001, help='Trading fee rate (0.001 = 0.1%)')
@click.option('--slippage-bps', default=5, help='Slippage in basis points (5 = 0.05%)')
@click.pass_context
def run_backtest(ctx, data_path: str, symbols: list[str], start_date: str, end_date: str,
                 initial_cash: float, fee_rate: float, slippage_bps: int):
    """Run a backtest on historical data."""
    logger = get_logger(__name__)
    config = ctx.obj

    logger.info("Starting backtest execution")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Initial cash: ${initial_cash:,.2f}")
    logger.info(f"Fee rate: {fee_rate:.3f}")
    logger.info(f"Slippage: {slippage_bps} bps")

    try:
        # Validate data path
        data_file = Path(data_path)
        if not data_file.exists():
            logger.error(f"Data file does not exist: {data_path}")
            sys.exit(1)

        # Update config with backtest parameters
        config['costs'] = {
            'fee_rate': fee_rate,
            'slippage_bps': slippage_bps
        }

        # Initialize backtest engine
        engine = BacktestEngine(config)

        # Override initial cash
        engine.cash = initial_cash

        # Run backtest
        success = engine.run_backtest(
            data_path=data_path,
            symbols=list(symbols),
            start_date=start_date,
            end_date=end_date
        )

        if success:
            logger.info("✅ Backtest completed successfully")
            logger.info(f"📊 Reports saved to: {engine.reports_dir}")

            # Display summary metrics
            if engine.equity_history:
                final_equity = engine.equity_history[-1]['portfolio_value']
                total_return = (final_equity - initial_cash) / initial_cash
                logger.info(f"💰 Final equity: ${final_equity:,.2f}")
                logger.info(f"📈 Total return: {total_return*100:.2f}%")
                logger.info(f"🔄 Total trades: {engine.total_trades}")
                if engine.total_trades > 0:
                    hit_rate = engine.winning_trades / engine.total_trades
                    logger.info(f"🎯 Hit rate: {hit_rate*100:.1f}%")
        else:
            logger.error("❌ Backtest failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during backtest execution: {e}")
        sys.exit(1)


@backtesting.command()
@click.option('--run-id', '-r', help='Specific run ID to analyze (latest if not specified)')
@click.option('--output-dir', '-o', default='reports/backtest_analysis', help='Output directory for analysis')
@click.pass_context
def analyze_backtest(ctx, run_id: str, output_dir: str):
    """Analyze backtest results and generate detailed reports."""
    logger = get_logger(__name__)
    config = ctx.obj

    logger.info("Starting backtest analysis")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        # Find backtest run
        reports_dir = Path(config['reports_root']) / 'runs'
        if not reports_dir.exists():
            logger.error("No backtest runs found")
            sys.exit(1)

        if run_id:
            run_dir = reports_dir / run_id
            if not run_dir.exists():
                logger.error(f"Backtest run {run_id} not found")
                sys.exit(1)
            run_dirs = [run_dir]
        else:
            # Get latest run
            run_dirs = sorted(reports_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if not run_dirs:
                logger.error("No backtest runs found")
                sys.exit(1)
            run_dirs = [run_dirs[0]]

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for run_dir in run_dirs:
            run_id = run_dir.name
            logger.info(f"Analyzing backtest run: {run_id}")

            # Load data
            equity_file = run_dir / 'equity.csv'
            trades_file = run_dir / 'trades.csv'
            metrics_file = run_dir / 'metrics.json'

            if not equity_file.exists():
                logger.warning(f"Equity file not found for run {run_id}")
                continue

            # Load equity data
            equity_df = pd.read_csv(equity_file)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

            # Load trades if available
            trades_df = None
            if trades_file.exists():
                trades_df = pd.read_csv(trades_file)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

            # Load metrics if available
            metrics = {}
            if metrics_file.exists():
                import json
                with open(metrics_file) as f:
                    metrics = json.load(f)

            # Basic analysis
            logger.info(f"📊 Equity data: {len(equity_df)} rows")
            if trades_df is not None:
                logger.info(f"🔄 Trades data: {len(trades_df)} rows")

            # Calculate additional metrics
            equity_df['returns'] = equity_df['portfolio_value'].pct_change().fillna(0)
            equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()

            # Rolling metrics
            equity_df['rolling_volatility'] = equity_df['returns'].rolling(20).std() * (252 ** 0.5)  # Annualized
            equity_df['rolling_sharpe'] = equity_df['returns'].rolling(20).mean() / equity_df['returns'].rolling(20).std() * (252 ** 0.5)

            # Drawdown analysis
            equity_df['peak'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']

            # Save enhanced data
            enhanced_equity_file = output_path / f'{run_id}_enhanced_equity.csv'
            equity_df.to_csv(enhanced_equity_file, index=False)
            logger.info(f"✅ Enhanced equity data saved to {enhanced_equity_file}")

            # Generate summary statistics
            summary_stats = {
                'run_id': run_id,
                'total_return': metrics.get('total_return_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'total_trades': metrics.get('total_trades', 0),
                'hit_rate': metrics.get('hit_rate', 0),
                'volatility': equity_df['returns'].std() * (252 ** 0.5),
                'avg_daily_return': equity_df['returns'].mean() * 252,
                'winning_trades': metrics.get('winning_trades', 0),
                'losing_trades': metrics.get('total_trades', 0) - metrics.get('winning_trades', 0)
            }

            # Save summary
            summary_file = output_path / f'{run_id}_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            logger.info(f"✅ Summary statistics saved to {summary_file}")

            # Generate plots if matplotlib is available
            try:
                # Equity curve
                plt.figure(figsize=(12, 8))

                plt.subplot(2, 2, 1)
                plt.plot(equity_df['timestamp'], equity_df['portfolio_value'])
                plt.title('Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.xticks(rotation=45)

                plt.subplot(2, 2, 2)
                plt.plot(equity_df['timestamp'], equity_df['cumulative_returns'])
                plt.title('Cumulative Returns')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Returns')
                plt.xticks(rotation=45)

                plt.subplot(2, 2, 3)
                plt.plot(equity_df['timestamp'], equity_df['drawdown'] * 100)
                plt.title('Drawdown (%)')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.xticks(rotation=45)

                plt.subplot(2, 2, 4)
                plt.plot(equity_df['timestamp'], equity_df['rolling_sharpe'])
                plt.title('Rolling Sharpe Ratio (20-period)')
                plt.xlabel('Date')
                plt.ylabel('Sharpe Ratio')
                plt.xticks(rotation=45)

                plt.tight_layout()

                # Save plot
                plot_file = output_path / f'{run_id}_analysis_plots.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"✅ Analysis plots saved to {plot_file}")

            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")

            # Display key metrics
            logger.info(f"📈 Key Metrics for Run {run_id}:")
            logger.info(f"  Total Return: {summary_stats['total_return']:.2f}%")
            logger.info(f"  Sharpe Ratio: {summary_stats['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {summary_stats['max_drawdown']*100:.2f}%")
            logger.info(f"  Volatility: {summary_stats['volatility']*100:.2f}%")
            logger.info(f"  Total Trades: {summary_stats['total_trades']}")
            logger.info(f"  Hit Rate: {summary_stats['hit_rate']*100:.1f}%")

        logger.info("✅ Backtest analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during backtest analysis: {e}")
        sys.exit(1)


@backtesting.command()
@click.option('--runs-dir', '-d', default='reports/runs', help='Directory containing backtest runs')
@click.pass_context
def list_runs(ctx, runs_dir: str):
    """List all available backtest runs."""
    logger = get_logger(__name__)

    logger.info("Listing available backtest runs")

    try:
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            logger.error(f"Runs directory does not exist: {runs_dir}")
            sys.exit(1)

        # Find all run directories
        run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]

        if not run_dirs:
            logger.info("No backtest runs found")
            return

        # Sort by creation time (newest first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        logger.info(f"Found {len(run_dirs)} backtest runs:")
        logger.info("-" * 80)

        for run_dir in run_dirs:
            run_id = run_dir.name
            creation_time = pd.to_datetime(run_dir.stat().st_mtime, unit='s')

            # Try to load metrics if available
            metrics_file = run_dir / 'metrics.json'
            metrics_info = ""

            if metrics_file.exists():
                try:
                    import json
                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    total_return = metrics.get('total_return_pct', 'N/A')
                    sharpe = metrics.get('sharpe_ratio', 'N/A')
                    trades = metrics.get('total_trades', 'N/A')

                    metrics_info = f" | Return: {total_return}% | Sharpe: {sharpe:.2f} | Trades: {trades}"

                except Exception:
                    metrics_info = " | Metrics unavailable"

            logger.info(f"📊 {run_id} | Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}{metrics_info}")

        logger.info("-" * 80)

    except Exception as e:
        logger.error(f"Error listing backtest runs: {e}")
        sys.exit(1)


if __name__ == '__main__':
    backtesting()

