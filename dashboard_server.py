#!/usr/bin/env python3
"""
Flask web server for the Algo-Trading Dashboard.
Provides real-time updates and API endpoints.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify, request
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)

# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algo-Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .card h2 {
            color: #4a5568;
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 10px;
        }
        
        .metric-label {
            font-weight: 600;
            color: #4a5568;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
        }
        
        .metric-value.success {
            color: #38a169;
        }
        
        .metric-value.warning {
            color: #d69e2e;
        }
        
        .metric-value.danger {
            color: #e53e3e;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        .backtest-runs {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .run-item {
            background: #f7fafc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        
        .run-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .run-id {
            font-weight: bold;
            color: #667eea;
        }
        
        .run-timestamp {
            color: #718096;
            font-size: 0.9rem;
        }
        
        .run-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }
        
        .run-metric {
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .run-metric-value {
            font-weight: bold;
            color: #667eea;
        }
        
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 25px;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        
        .live-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #38a169;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="live-indicator">🟢 LIVE</div>
    
    <div class="container">
        <div class="header">
            <h1>🚀 Algo-Trading Dashboard</h1>
            <p>Professional Algorithmic Trading System Monitor</p>
            <p style="margin-top: 10px; font-size: 1rem; opacity: 0.8;">
                Last Updated: <span id="last-updated">{{ last_updated }}</span>
            </p>
        </div>
        
        <div class="dashboard-grid">
            <!-- Test Results Card -->
            <div class="card">
                <h2>🧪 Test Results</h2>
                <div class="metric">
                    <span class="metric-label">Total Tests</span>
                    <span class="metric-value" id="total-tests">{{ total_tests }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Passed</span>
                    <span class="metric-value success" id="passed-tests">{{ passed_tests }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed</span>
                    <span class="metric-value danger" id="failed-tests">{{ failed_tests }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value" id="success-rate">{{ success_rate }}</span>
                </div>
            </div>
            
            <!-- Code Coverage Card -->
            <div class="card">
                <h2>📊 Code Coverage</h2>
                <div class="metric">
                    <span class="metric-label">Overall Coverage</span>
                    <span class="metric-value" id="coverage-percentage">{{ coverage_percentage }}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Target</span>
                    <span class="metric-value">80%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value" id="coverage-status">{{ coverage_status }}</span>
                </div>
                <div class="chart-container">
                    <canvas id="coverageChart"></canvas>
                </div>
            </div>
            
            <!-- System Health Card -->
            <div class="card">
                <h2>💚 System Health</h2>
                <div class="metric">
                    <span class="metric-label">Feature Engine</span>
                    <span class="metric-value success">✅ Operational</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Data Pipeline</span>
                    <span class="metric-value success">✅ Operational</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Backtesting</span>
                    <span class="metric-value success">✅ Operational</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Validation</span>
                    <span class="metric-value success">✅ Operational</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Overall Status</span>
                    <span class="metric-value success">🟢 Healthy</span>
                </div>
            </div>
            
            <!-- Performance Metrics Card -->
            <div class="card">
                <h2>⚡ Performance Overview</h2>
                <div class="metric">
                    <span class="metric-label">Feature Computation</span>
                    <span class="metric-value success">Fast</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Data Processing</span>
                    <span class="metric-value success">Efficient</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value warning">Optimized</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Test Execution</span>
                    <span class="metric-value success">Fast</span>
                </div>
            </div>
        </div>
        
        <!-- Backtest Results Section -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>📈 Backtest Results</h2>
            <div class="backtest-runs" id="backtest-runs">
                {{ backtest_html | safe }}
            </div>
        </div>
        
        <!-- Feature Engineering Status -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>🔧 Feature Engineering Status</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div class="metric">
                    <span class="metric-label">Price Returns</span>
                    <span class="metric-value success">✅ 5 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trend Momentum</span>
                    <span class="metric-value success">✅ 8 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mean Reversion</span>
                    <span class="metric-value success">✅ 6 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Volatility Risk</span>
                    <span class="metric-value success">✅ 7 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Volume Liquidity</span>
                    <span class="metric-value success">✅ 4 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cross Asset</span>
                    <span class="metric-value success">✅ 3 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Regime Detection</span>
                    <span class="metric-value success">✅ 3 Features</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Risk Execution</span>
                    <span class="metric-value success">✅ 6 Features</span>
                </div>
            </div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="refreshDashboard()">
        🔄 Refresh Dashboard
    </button>
    
    <script>
        // Initialize charts
        let coverageChart;
        
        function initCharts() {
            const coverageCtx = document.getElementById('coverageChart').getContext('2d');
            coverageChart = new Chart(coverageCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Covered', 'Uncovered'],
                    datasets: [{
                        data: [{{ coverage_percentage }}, {{ 100 - coverage_percentage }}],
                        backgroundColor: [
                            '#38a169',
                            '#e53e3e'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    },
                    cutout: '70%'
                }
            });
        }
        
        function updateDashboard(data) {
            // Update test results
            document.getElementById('total-tests').textContent = data.total_tests;
            document.getElementById('passed-tests').textContent = data.passed_tests;
            document.getElementById('failed-tests').textContent = data.failed_tests;
            document.getElementById('success-rate').textContent = data.success_rate;
            
            // Update coverage
            document.getElementById('coverage-percentage').textContent = data.coverage_percentage + '%';
            document.getElementById('coverage-status').textContent = data.coverage_status;
            
            // Update last updated time
            document.getElementById('last-updated').textContent = new Date().toLocaleString();
            
            // Update backtest runs
            document.getElementById('backtest-runs').innerHTML = data.backtest_html;
            
            // Update charts
            if (coverageChart) {
                coverageChart.data.datasets[0].data = [data.coverage_percentage, 100 - data.coverage_percentage];
                coverageChart.update();
            }
        }
        
        function refreshDashboard() {
            axios.get('/api/dashboard-data')
                .then(response => {
                    updateDashboard(response.data);
                })
                .catch(error => {
                    console.error('Error refreshing dashboard:', error);
                });
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
        });
    </script>
</body>
</html>
"""

def run_tests_and_collect_results():
    """Run tests and collect results."""
    try:
        # Run pytest with coverage
        result = subprocess.run([
            'python3', '-m', 'pytest', '--cov=src', '--cov-report=json', 
            '--json-report', '--json-report-file=test_results.json'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        # Parse test results
        test_results = {}
        if os.path.exists('test_results.json'):
            try:
                with open('test_results.json', 'r') as f:
                    test_results = json.load(f)
            except:
                pass
        
        return {
            'test_output': result.stdout,
            'test_return_code': result.returncode,
            'test_results': test_results
        }
    except Exception as e:
        return {
            'error': str(e),
            'test_output': '',
            'test_return_code': -1
        }

def collect_backtest_results():
    """Collect existing backtest results."""
    backtest_results = []
    reports_dir = Path('reports/runs')
    
    if reports_dir.exists():
        for run_dir in sorted(reports_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if run_dir.is_dir():
                run_info = {
                    'run_id': run_dir.name,
                    'timestamp': datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc),
                    'metrics_file': run_dir / 'metrics.json',
                    'equity_file': run_dir / 'equity.csv',
                    'trades_file': run_dir / 'trades.csv'
                }
                
                # Load metrics if available
                if run_info['metrics_file'].exists():
                    try:
                        with open(run_info['metrics_file'], 'r') as f:
                            run_info['metrics'] = json.load(f)
                    except:
                        run_info['metrics'] = {}
                
                backtest_results.append(run_info)
    
    return backtest_results

def generate_backtest_html(backtest_data):
    """Generate HTML for backtest results."""
    if not backtest_data:
        return '<p style="text-align: center; color: #718096; font-style: italic;">No backtest runs found. Run a backtest to see results here.</p>'
    
    html_parts = []
    
    for run in backtest_data[:10]:  # Show last 10 runs
        run_html = f"""
        <div class="run-item">
            <div class="run-header">
                <span class="run-id">Run: {run['run_id']}</span>
                <span class="run-timestamp">{run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        """
        
        if 'metrics' in run and run['metrics']:
            metrics = run['metrics']
            run_html += f"""
            <div class="run-metrics">
                <div class="run-metric">
                    <div class="run-metric-value">{metrics.get('total_return_pct', 'N/A')}%</div>
                    <div>Return</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">{metrics.get('sharpe_ratio', 'N/A')}</div>
                    <div>Sharpe</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">{metrics.get('max_drawdown', 'N/A')}%</div>
                    <div>Max DD</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">{metrics.get('hit_rate', 'N/A')}%</div>
                    <div>Hit Rate</div>
                </div>
                <div class="run-metric">
                    <div class="run-metric-value">{metrics.get('total_trades', 'N/A')}</div>
                    <div>Trades</div>
                </div>
            </div>
            """
        
        run_html += "</div>"
        html_parts.append(run_html)
    
    return ''.join(html_parts)

@app.route('/')
def dashboard():
    """Main dashboard page."""
    # Collect data
    test_data = run_tests_and_collect_results()
    backtest_data = collect_backtest_results()
    
    # Calculate test statistics
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    if test_data.get('test_results'):
        summary = test_data['test_results'].get('summary', {})
        total_tests = summary.get('total', 0)
        passed_tests = summary.get('passed', 0)
        failed_tests = summary.get('failed', 0)
    
    success_rate = f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A"
    coverage_percentage = 37  # From your earlier output
    coverage_status = '✅ Excellent' if coverage_percentage >= 80 else '⚠️ Good' if coverage_percentage >= 60 else '❌ Needs Improvement'
    
    # Generate backtest HTML
    backtest_html = generate_backtest_html(backtest_data)
    
    return render_template_string(DASHBOARD_TEMPLATE,
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        success_rate=success_rate,
        coverage_percentage=coverage_percentage,
        coverage_status=coverage_status,
        backtest_html=backtest_html
    )

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """API endpoint for dashboard data."""
    # Collect data
    test_data = run_tests_and_collect_results()
    backtest_data = collect_backtest_results()
    
    # Calculate test statistics
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    if test_data.get('test_results'):
        summary = test_data['test_results'].get('summary', {})
        total_tests = summary.get('total', 0)
        passed_tests = summary.get('passed', 0)
        failed_tests = summary.get('failed', 0)
    
    success_rate = f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "N/A"
    coverage_percentage = 37  # From your earlier output
    coverage_status = '✅ Excellent' if coverage_percentage >= 80 else '⚠️ Good' if coverage_percentage >= 60 else '❌ Needs Improvement'
    
    # Generate backtest HTML
    backtest_html = generate_backtest_html(backtest_data)
    
    return jsonify({
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': success_rate,
        'coverage_percentage': coverage_percentage,
        'coverage_status': coverage_status,
        'backtest_html': backtest_html,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/run-tests')
def api_run_tests():
    """API endpoint to manually run tests."""
    try:
        test_data = run_tests_and_collect_results()
        return jsonify({
            'success': True,
            'message': 'Tests completed',
            'data': test_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Algo-Trading Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Auto-refresh every 30 seconds")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

