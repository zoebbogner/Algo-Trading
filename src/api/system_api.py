#!/usr/bin/env python3
"""
System API endpoints for the Algo-Trading system.

Provides REST API for:
- System health and monitoring
- Test execution and reporting
- Dashboard generation
- System configuration
"""

import subprocess
import sys
from pathlib import Path

import psutil
from flask import Blueprint, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.utils.logging import get_logger

# Create blueprint
system_router = Blueprint('system', __name__, url_prefix='/api/system')
logger = get_logger(__name__)

# Load configuration
config = load_base_config()


@system_router.route('/health', methods=['GET'])
def health_check():
    """Check overall system health."""
    try:
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        disk_free_gb = disk_usage.free / (1024**3)

        # Check memory usage
        memory = psutil.virtual_memory()
        memory_used_percent = memory.percent

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Check key directories
        key_dirs = {
            'data': Path('data').exists(),
            'logs': Path('logs').exists(),
            'reports': Path('reports').exists(),
            'configs': Path('configs').exists()
        }

        # Overall health status
        health_status = 'healthy'
        if disk_free_gb < 1:  # Less than 1GB free
            health_status = 'warning'
        if memory_used_percent > 90:  # More than 90% memory used
            health_status = 'warning'
        if not all(key_dirs.values()):
            health_status = 'warning'

        return jsonify({
            'status': 'healthy',
            'service': 'system-api',
            'system_health': health_status,
            'resources': {
                'disk_free_gb': round(disk_free_gb, 2),
                'memory_used_percent': round(memory_used_percent, 2),
                'cpu_percent': round(cpu_percent, 2)
            },
            'directories': key_dirs,
            'config_loaded': bool(config)
        })

    except Exception as e:
        logger.error(f"Error in system health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'system-api',
            'error': str(e)
        }), 500


@system_router.route('/tests/run', methods=['POST'])
def run_tests():
    """Run the test suite and return results."""
    try:
        data = request.get_json() or {}
        test_pattern = data.get('pattern', '')
        coverage = data.get('coverage', True)
        verbose = data.get('verbose', True)

        logger.info("Starting test execution")

        # Build pytest command
        cmd = ['python3', '-m', 'pytest']

        if verbose:
            cmd.append('-v')

        if coverage:
            cmd.append('--cov=src')
            cmd.append('--cov-report=html:reports/coverage')
            cmd.append('--cov-report=term-missing')

        if test_pattern:
            cmd.append(f'-k={test_pattern}')

        # Add test directory
        cmd.append('tests/')

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run tests
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Parse results
        test_results = {
            'command': ' '.join(cmd),
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }

        # Extract test summary if available
        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    test_results['summary'] = line.strip()
                    break

        # Save test results
        output_dir = Path('reports/tests')
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        from datetime import datetime

        test_file = output_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)

        return jsonify({
            'success': True,
            'message': 'Tests completed',
            'test_results': test_results,
            'output_file': str(test_file)
        })

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_router.route('/tests/status', methods=['GET'])
def get_test_status():
    """Get current test status and recent results."""
    try:
        tests_dir = Path('reports/tests')
        coverage_dir = Path('reports/coverage')

        status = {
            'tests_directory': str(tests_dir),
            'coverage_directory': str(coverage_dir),
            'recent_results': [],
            'coverage_available': coverage_dir.exists()
        }

        if tests_dir.exists():
            # Find recent test results
            test_files = list(tests_dir.glob('test_results_*.json'))
            test_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            recent_results = []
            for test_file in test_files[:5]:  # Last 5 results
                try:
                    import json
                    with open(test_file) as f:
                        result = json.load(f)

                    recent_results.append({
                        'file': test_file.name,
                        'timestamp': test_file.stat().st_mtime,
                        'success': result.get('success', False),
                        'return_code': result.get('return_code', -1),
                        'summary': result.get('summary', 'No summary available')
                    })
                except Exception:
                    continue

            status['recent_results'] = recent_results
            status['total_test_files'] = len(test_files)

        if coverage_dir.exists():
            # Check for coverage report
            index_file = coverage_dir / 'index.html'
            status['coverage_report'] = str(index_file) if index_file.exists() else None

        return jsonify({
            'success': True,
            'data': status
        })

    except Exception as e:
        logger.error(f"Error getting test status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_router.route('/dashboard/generate', methods=['POST'])
def generate_dashboard():
    """Generate a static HTML dashboard."""
    try:
        data = request.get_json() or {}
        output_file = data.get('output_file', 'reports/dashboard.html')
        include_tests = data.get('include_tests', True)
        include_backtests = data.get('include_backtests', True)

        logger.info("Generating static HTML dashboard")

        # Create output directory
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate dashboard HTML
        dashboard_html = generate_dashboard_html(include_tests, include_backtests)

        # Save dashboard
        with open(output_path, 'w') as f:
            f.write(dashboard_html)

        logger.info(f"Dashboard generated: {output_path}")

        return jsonify({
            'success': True,
            'message': 'Dashboard generated successfully',
            'output_file': str(output_path)
        })

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_router.route('/dashboard/status', methods=['GET'])
def get_dashboard_status():
    """Get current dashboard status."""
    try:
        dashboard_file = Path('reports/dashboard.html')

        status = {
            'dashboard_file': str(dashboard_file),
            'exists': dashboard_file.exists(),
            'last_modified': None,
            'size_mb': 0
        }

        if dashboard_file.exists():
            status.update({
                'last_modified': dashboard_file.stat().st_mtime,
                'size_mb': round(dashboard_file.stat().st_size / (1024 * 1024), 2)
            })

        return jsonify({
            'success': True,
            'data': status
        })

    except Exception as e:
        logger.error(f"Error getting dashboard status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_router.route('/logs', methods=['GET'])
def get_logs():
    """Get recent log entries."""
    try:
        data = request.get_json() or {}
        log_file = data.get('log_file', 'logs/algo_trading.log')
        lines = data.get('lines', 100)

        log_path = Path(log_file)
        if not log_path.exists():
            return jsonify({
                'success': False,
                'error': f'Log file not found: {log_file}'
            }), 404

        # Read last N lines
        with open(log_path) as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        # Parse log entries (basic parsing)
        log_entries = []
        for line in recent_lines:
            try:
                # Basic log parsing - adjust based on your log format
                if ' - ' in line:
                    parts = line.split(' - ', 2)
                    if len(parts) >= 3:
                        timestamp, level, message = parts[0], parts[1], parts[2]
                        log_entries.append({
                            'timestamp': timestamp.strip(),
                            'level': level.strip(),
                            'message': message.strip()
                        })
                    else:
                        log_entries.append({
                            'timestamp': '',
                            'level': 'INFO',
                            'message': line.strip()
                        })
                else:
                    log_entries.append({
                        'timestamp': '',
                        'level': 'INFO',
                        'message': line.strip()
                    })
            except Exception:
                log_entries.append({
                    'timestamp': '',
                    'level': 'INFO',
                    'message': line.strip()
                })

        return jsonify({
            'success': True,
            'log_file': str(log_file),
            'total_lines': len(all_lines),
            'recent_lines': len(recent_lines),
            'entries': log_entries
        })

    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_router.route('/config', methods=['GET'])
def get_config():
    """Get current system configuration."""
    try:
        config_files = {
            'base': 'configs/base.yaml',
            'data_history': 'configs/data.history.yaml',
            'features': 'configs/features.yaml',
            'costs_sim': 'configs/costs.sim.yaml'
        }

        config_status = {}
        for name, path in config_files.items():
            config_path = Path(path)
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)

                    config_status[name] = {
                        'exists': True,
                        'path': str(config_path),
                        'size_mb': round(config_path.stat().st_size / (1024 * 1024), 2),
                        'keys': list(config_data.keys()) if config_data else []
                    }
                except Exception as e:
                    config_status[name] = {
                        'exists': True,
                        'path': str(config_path),
                        'error': str(e)
                    }
            else:
                config_status[name] = {
                    'exists': False,
                    'path': str(config_path)
                }

        return jsonify({
            'success': True,
            'config_files': config_status
        })

    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_dashboard_html(include_tests=True, include_backtests=True):
    """Generate static HTML dashboard content."""

    # Collect data for dashboard
    dashboard_data = {}

    # System info
    try:
        disk_usage = psutil.disk_usage('.')
        memory = psutil.virtual_memory()
        dashboard_data['system'] = {
            'disk_free_gb': round(disk_usage.free / (1024**3), 2),
            'memory_used_percent': round(memory.percent, 2),
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
    except Exception:
        dashboard_data['system'] = {'error': 'Could not collect system info'}

    # Test results
    if include_tests:
        try:
            tests_dir = Path('reports/tests')
            if tests_dir.exists():
                test_files = list(tests_dir.glob('test_results_*.json'))
                if test_files:
                    test_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    latest_test = test_files[0]

                    import json
                    with open(latest_test) as f:
                        test_result = json.load(f)

                    dashboard_data['tests'] = {
                        'latest_result': test_result.get('summary', 'No summary'),
                        'success': test_result.get('success', False),
                        'timestamp': latest_test.stat().st_mtime
                    }
        except Exception:
            dashboard_data['tests'] = {'error': 'Could not collect test info'}

    # Backtest results
    if include_backtests:
        try:
            backtest_dir = Path('reports/backtest')
            if backtest_dir.exists():
                equity_files = list(backtest_dir.glob('equity_*.csv'))
                if equity_files:
                    dashboard_data['backtests'] = {
                        'total_runs': len(equity_files),
                        'recent_runs': []
                    }

                    # Get recent runs
                    equity_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    for file_path in equity_files[:3]:
                        run_id = file_path.stem.replace('equity_', '')
                        summary_file = backtest_dir / f'summary_{run_id}.json'

                        if summary_file.exists():
                            try:
                                with open(summary_file) as f:
                                    summary = json.load(f)

                                dashboard_data['backtests']['recent_runs'].append({
                                    'run_id': run_id[:8],  # Short ID
                                    'total_return': summary.get('total_return', 0),
                                    'sharpe_ratio': summary.get('sharpe_ratio', 0)
                                })
                            except Exception:
                                continue
        except Exception:
            dashboard_data['backtests'] = {'error': 'Could not collect backtest info'}

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Algo-Trading System Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
                margin: 10px 0;
            }}
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .card h3 {{
                color: #333;
                margin-top: 0;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 15px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .metric-label {{
                font-weight: 600;
                color: #555;
            }}
            .metric-value {{
                font-weight: bold;
                color: #667eea;
            }}
            .status-indicator {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }}
            .status-healthy {{ background: #28a745; }}
            .status-warning {{ background: #ffc107; }}
            .status-error {{ background: #dc3545; }}
            .refresh-button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                margin: 20px auto;
                display: block;
                transition: background 0.3s ease;
            }}
            .refresh-button:hover {{
                background: #5a6fd8;
            }}
            .timestamp {{
                text-align: center;
                color: #666;
                font-size: 0.9em;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="dashboard-header">
                <h1>🚀 Algo-Trading System Dashboard</h1>
                <p>Real-time System Dashboard & Monitoring</p>
            </div>

            <div class="dashboard-grid">
                <!-- System Status -->
                <div class="card">
                    <h3>System Status</h3>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-label">Status</span>
                            <span class="metric-value status-{dashboard_data.get('status', 'unknown')}">
                                {dashboard_data.get('status', 'Unknown').upper()}
                            </span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Uptime</span>
                            <span class="metric-value">{dashboard_data.get('uptime', 'Unknown')}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Last Check</span>
                            <span class="metric-value">{dashboard_data.get('last_check', 'Unknown')}</span>
                        </div>
                    </div>
                </div>

                <!-- Test Results -->
                <div class="card">
                    <h3>Test Results</h3>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-label">Total Tests</span>
                            <span class="metric-value">{dashboard_data.get('tests', {}).get('total', 0)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Passed</span>
                            <span class="metric-value status-passed">{dashboard_data.get('tests', {}).get('passed', 0)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Failed</span>
                            <span class="metric-value status-failed">{dashboard_data.get('tests', {}).get('failed', 0)}</span>
                        </div>
                    </div>
                </div>

                <!-- Backtest Results -->
                <div class="card">
                    <h3>Backtest Results</h3>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-label">Total Runs</span>
                            <span class="metric-value">{dashboard_data.get('backtests', {}).get('total_runs', 0)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Recent Runs</span>
                            <span class="metric-value">{dashboard_data.get('backtests', {}).get('recent_runs', 0)}</span>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="card">
                    <h3>Quick Actions</h3>
                    <div class="actions">
                        <button onclick="runTests()" class="action-btn">🧪 Run Tests</button>
                        <button onclick="runBacktest()" class="action-btn">📊 Run Backtest</button>
                        <button onclick="collectData()" class="action-btn">📥 Collect Data</button>
                        <button onclick="generateFeatures()" class="action-btn">⚡ Generate Features</button>
                    </div>
                </div>
            </div>

            <button class="refresh-button" onclick="location.reload()">🔄 Refresh Dashboard</button>

            <div class="timestamp">
                Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    return html





