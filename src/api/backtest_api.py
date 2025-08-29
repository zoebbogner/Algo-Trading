#!/usr/bin/env python3
"""
Backtesting API endpoints for the Algo-Trading system.

Provides REST API for:
- Backtesting execution and monitoring
- Performance analysis and reporting
- Strategy management and optimization
"""

import sys
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.core.backtest.engine import BacktestEngine
from src.utils.logging import get_logger

# Create blueprint
backtest_router = Blueprint('backtest', __name__, url_prefix='/api/backtest')
logger = get_logger(__name__)

# Load configuration
config = load_base_config()


@backtest_router.route('/health', methods=['GET'])
def health_check():
    """Check backtesting API health."""
    return jsonify({
        'status': 'healthy',
        'service': 'backtest-api',
        'config_loaded': bool(config)
    })


@backtest_router.route('/run', methods=['POST'])
def run_backtest():
    """Run a backtest with specified parameters."""
    try:
        data = request.get_json()
        
        # Required parameters
        features_file = data.get('features_file')
        if not features_file:
            return jsonify({
                'success': False,
                'error': 'features_file is required'
            }), 400

        # Optional parameters with defaults
        symbols = data.get('symbols', ['BTCUSDT'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = data.get('initial_capital', 100000)
        commission_rate = data.get('commission_rate', 0.001)
        slippage = data.get('slippage', 0.0005)
        
        # Risk management parameters
        max_position_size = data.get('max_position_size', 0.05)  # 5% max per position
        max_portfolio_risk = data.get('max_portfolio_risk', 0.20)  # 20% max portfolio risk
        stop_loss_pct = data.get('stop_loss_pct', 0.05)  # 5% stop loss
        take_profit_pct = data.get('take_profit_pct', 0.10)  # 10% take profit

        logger.info("Starting backtest execution")

        # Load features data
        df = pd.read_parquet(features_file)
        
        # Filter by date range
        if start_date:
            df = df[df['ts'] >= start_date]
        if end_date:
            df = df[df['ts'] <= end_date]

        # Filter by symbols
        if symbols:
            df = df[df['symbol'].isin(symbols)]

        if len(df) == 0:
            return jsonify({
                'success': False,
                'error': 'No data found for specified parameters'
            }), 400

        logger.info(f"Loaded {len(df)} rows for backtesting")

        # Initialize backtest engine with config
        backtest_config = {
            'costs': {
                'fee_rate': commission_rate,
                'slippage_bps': int(slippage * 10000)  # Convert to basis points
            },
            'risk': {
                'max_position_size': max_position_size,
                'max_portfolio_risk': max_portfolio_risk,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            },
            'initial_capital': initial_capital
        }
        engine = BacktestEngine(config=backtest_config)

        # Generate unique run ID
        import uuid
        run_id = str(uuid.uuid4())

        # Run backtest
        logger.info("Executing backtest...")

        # Save the filtered data temporarily for backtesting
        temp_data_path = f"data/temp/backtest_data_{run_id}.parquet"
        Path(temp_data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(temp_data_path)

        # Run backtest with the correct parameters
        success = engine.run_backtest(
            data_path=temp_data_path,
            symbols=symbols if symbols else df['symbol'].unique().tolist(),
            start_date=start_date,
            end_date=end_date
        )

        if not success:
            return jsonify({
                'success': False,
                'error': 'Backtest execution failed'
            }), 500

        # Get results from the engine
        results = {
            'equity_curve': pd.DataFrame(engine.equity_history),
            'trade_history': pd.DataFrame(engine.trade_history),
            'performance_metrics': {
                'total_pnl': engine.total_pnl,
                'total_trades': engine.total_trades,
                'winning_trades': engine.winning_trades,
                'losing_trades': engine.losing_trades,
                'win_rate': engine.winning_trades / engine.total_trades if engine.total_trades > 0 else 0,
                'profit_factor': engine.total_pnl / abs(engine.total_pnl) if engine.total_pnl != 0 else 0
            }
        }

        # Save results
        output_dir = Path('reports/backtest')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        equity_file = output_dir / f'equity_{run_id}.csv'
        results['equity_curve'].to_csv(equity_file, index=False)

        # Save trade history
        trades_file = output_dir / f'trades_{run_id}.csv'
        results['trade_history'].to_csv(trades_file, index=False)

        # Save performance summary
        summary_file = output_dir / f'summary_{run_id}.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(results['performance_metrics'], f, indent=2, default=str)

        # Prepare response
        response_data = {
            'run_id': run_id,
            'success': True,
            'message': 'Backtest completed successfully',
            'performance_metrics': results['performance_metrics'],
            'output_files': {
                'equity_curve': str(equity_file),
                'trade_history': str(trades_file),
                'summary': str(summary_file)
            },
            'backtest_info': {
                'start_date': str(df['ts'].min()),
                'end_date': str(df['ts'].max()),
                'symbols': df['symbol'].unique().tolist(),
                'total_rows': len(df),
                'initial_capital': initial_capital,
                'commission_rate': commission_rate,
                'slippage': slippage,
                'risk_management': {
                    'max_position_size': max_position_size,
                    'max_portfolio_risk': max_portfolio_risk,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct
                }
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@backtest_router.route('/analyze', methods=['POST'])
def analyze_backtest():
    """Analyze backtest results and generate detailed reports."""
    try:
        data = request.get_json()
        run_id = data.get('run_id')
        output_dir = data.get('output_dir', 'reports/backtest')

        if not run_id:
            return jsonify({
                'success': False,
                'error': 'run_id is required'
            }), 400

        logger.info(f"Starting analysis for backtest run: {run_id}")

        # Load backtest results
        base_dir = Path(output_dir)
        equity_file = base_dir / f'equity_{run_id}.csv'
        trades_file = base_dir / f'trades_{run_id}.csv'
        summary_file = base_dir / f'summary_{run_id}.json'

        if not all([equity_file.exists(), trades_file.exists(), summary_file.exists()]):
            return jsonify({
                'success': False,
                'error': f'Backtest results not found for run_id: {run_id}'
            }), 404

        # Load data
        import json

        import pandas as pd

        equity_df = pd.read_csv(equity_file)
        trades_df = pd.read_csv(trades_file)

        with open(summary_file) as f:
            performance_metrics = json.load(f)

        # Create analysis directory
        analysis_dir = base_dir / f'analysis_{run_id}'
        analysis_dir.mkdir(exist_ok=True)

        # Additional analysis
        logger.info("Performing detailed analysis...")

        # Drawdown analysis
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()
        max_drawdown = equity_df['drawdown'].min()
        drawdown_duration = (equity_df['drawdown'] < 0).sum()

        # Volatility analysis
        daily_returns = equity_df['equity'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized

        # Trade analysis
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            trade_analysis = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')
            }
        else:
            trade_analysis = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        # Enhanced performance metrics
        enhanced_metrics = {
            **performance_metrics,
            'max_drawdown': float(max_drawdown),
            'drawdown_duration': int(drawdown_duration),
            'annualized_volatility': float(volatility),
            'trade_analysis': trade_analysis
        }

        # Save enhanced analysis
        enhanced_file = analysis_dir / 'enhanced_analysis.json'
        with open(enhanced_file, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2, default=str)

        # Generate charts data
        chart_data = {
            'equity_curve': {
                'timestamps': equity_df['timestamp'].tolist(),
                'equity': equity_df['equity'].tolist(),
                'drawdown': equity_df['drawdown'].tolist()
            },
            'trade_distribution': {
                'winning_pnl': winning_trades['pnl'].tolist() if len(trades_df) > 0 else [],
                'losing_pnl': losing_trades['pnl'].tolist() if len(trades_df) > 0 else []
            }
        }

        charts_file = analysis_dir / 'charts_data.json'
        with open(charts_file, 'w') as f:
            json.dump(chart_data, f, indent=2, default=str)

        return jsonify({
            'success': True,
            'message': f'Backtest analysis completed for run: {run_id}',
            'run_id': run_id,
            'enhanced_metrics': enhanced_metrics,
            'output_files': {
                'enhanced_analysis': str(enhanced_file),
                'charts_data': str(charts_file)
            }
        })

    except Exception as e:
        logger.error(f"Error during backtest analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@backtest_router.route('/runs', methods=['GET'])
def list_backtest_runs():
    """List all available backtest runs."""
    try:
        output_dir = Path('reports/backtest')

        if not output_dir.exists():
            return jsonify({
                'success': True,
                'runs': [],
                'total_runs': 0
            })

        # Find all backtest runs
        equity_files = list(output_dir.glob('equity_*.csv'))
        run_ids = [f.stem.replace('equity_', '') for f in equity_files]

        runs_info = []
        for run_id in run_ids:
            try:
                # Load summary for each run
                summary_file = output_dir / f'summary_{run_id}.json'
                if summary_file.exists():
                    import json
                    with open(summary_file) as f:
                        summary = json.load(f)

                    runs_info.append({
                        'run_id': run_id,
                        'total_return': summary.get('total_return', 0),
                        'sharpe_ratio': summary.get('sharpe_ratio', 0),
                        'max_drawdown': summary.get('max_drawdown', 0),
                        'win_rate': summary.get('win_rate', 0),
                        'total_trades': summary.get('total_trades', 0),
                        'summary_file': str(summary_file)
                    })
            except Exception as e:
                logger.warning(f"Could not load summary for run {run_id}: {e}")
                continue

        # Sort by total return
        runs_info.sort(key=lambda x: x['total_return'], reverse=True)

        return jsonify({
            'success': True,
            'runs': runs_info,
            'total_runs': len(runs_info)
        })

    except Exception as e:
        logger.error(f"Error listing backtest runs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@backtest_router.route('/runs/<run_id>', methods=['GET'])
def get_backtest_run(run_id):
    """Get detailed information about a specific backtest run."""
    try:
        output_dir = Path('reports/backtest')

        # Check if run exists
        equity_file = output_dir / f'equity_{run_id}.csv'
        trades_file = output_dir / f'trades_{run_id}.csv'
        summary_file = output_dir / f'summary_{run_id}.json'

        if not all([equity_file.exists(), trades_file.exists(), summary_file.exists()]):
            return jsonify({
                'success': False,
                'error': f'Backtest run not found: {run_id}'
            }), 404

        # Load run data
        import json

        import pandas as pd

        equity_df = pd.read_csv(equity_file)
        trades_df = pd.read_csv(trades_file)

        with open(summary_file) as f:
            summary = json.load(f)

        # Prepare response
        run_data = {
            'run_id': run_id,
            'performance_metrics': summary,
            'equity_curve': {
                'total_rows': len(equity_df),
                'date_range': {
                    'start': equity_df['timestamp'].min(),
                    'end': equity_df['timestamp'].max()
                }
            },
            'trade_history': {
                'total_trades': len(trades_df),
                'symbols': trades_df['symbol'].unique().tolist() if len(trades_df) > 0 else []
            },
            'files': {
                'equity_curve': str(equity_file),
                'trade_history': str(trades_file),
                'summary': str(summary_file)
            }
        }

        return jsonify({
            'success': True,
            'data': run_data
        })

    except Exception as e:
        logger.error(f"Error getting backtest run {run_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@backtest_router.route('/status', methods=['GET'])
def get_backtest_status():
    """Get current backtesting system status."""
    try:
        output_dir = Path('reports/backtest')

        status = {
            'backtest_directory': str(output_dir),
            'exists': output_dir.exists(),
            'total_runs': 0,
            'recent_runs': [],
            'system_status': 'ready'
        }

        if output_dir.exists():
            # Count total runs
            equity_files = list(output_dir.glob('equity_*.csv'))
            status['total_runs'] = len(equity_files)

            # Get recent runs (last 5)
            if equity_files:
                # Sort by modification time
                equity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                recent_files = equity_files[:5]

                recent_runs = []
                for file_path in recent_files:
                    run_id = file_path.stem.replace('equity_', '')
                    summary_file = output_dir / f'summary_{run_id}.json'

                    if summary_file.exists():
                        try:
                            import json
                            with open(summary_file) as f:
                                summary = json.load(f)

                            recent_runs.append({
                                'run_id': run_id,
                                'total_return': summary.get('total_return', 0),
                                'sharpe_ratio': summary.get('sharpe_ratio', 0),
                                'max_drawdown': summary.get('max_drawdown', 0),
                                'modified': file_path.stat().st_mtime
                            })
                        except Exception:
                            continue

                status['recent_runs'] = recent_runs

        return jsonify({
            'success': True,
            'data': status
        })

    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

