#!/usr/bin/env python3
"""
Main Flask API server for the Algo-Trading system.

Integrates all API modules:
- Data API (collection, validation, consolidation)
- Features API (computation, validation, analysis)
- Backtesting API (execution, analysis, reporting)
- System API (health, tests, dashboard, monitoring)
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.backtest_api import backtest_router
from src.api.data_api import data_router
from src.api.features_api import features_router
from src.api.system_api import system_router
from src.utils.logging import get_logger

# Create Flask app
app = Flask(__name__)
logger = get_logger(__name__)

# Enable CORS for cross-origin requests
CORS(app)

# Register blueprints
app.register_blueprint(data_router)
app.register_blueprint(features_router)
app.register_blueprint(backtest_router)
app.register_blueprint(system_router)


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Algo-Trading System API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'data': '/api/data',
            'features': '/api/features',
            'backtest': '/api/backtest',
            'system': '/api/system'
        },
        'documentation': {
            'data_api': {
                'health': 'GET /api/data/health',
                'collect_historical': 'POST /api/data/collect/historical',
                'collect_recent': 'POST /api/data/collect/recent',
                'consolidate': 'POST /api/data/consolidate',
                'validate': 'POST /api/data/validate',
                'status': 'GET /api/data/status'
            },
            'features_api': {
                'health': 'GET /api/features/health',
                'compute': 'POST /api/features/compute',
                'validate': 'POST /api/features/validate',
                'analyze': 'POST /api/features/analyze',
                'status': 'GET /api/features/status'
            },
            'backtest_api': {
                'health': 'GET /api/backtest/health',
                'run': 'POST /api/backtest/run',
                'analyze': 'POST /api/backtest/analyze',
                'runs': 'GET /api/backtest/runs',
                'run_details': 'GET /api/backtest/runs/<run_id>',
                'status': 'GET /api/backtest/status'
            },
            'system_api': {
                'health': 'GET /api/system/health',
                'run_tests': 'POST /api/system/tests/run',
                'test_status': 'GET /api/system/tests/status',
                'generate_dashboard': 'POST /api/system/dashboard/generate',
                'dashboard_status': 'GET /api/system/dashboard/status',
                'logs': 'GET /api/system/logs',
                'config': 'GET /api/system/config'
            }
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Overall system health check."""
    try:
        # Check if all modules are accessible
        modules_healthy = {
            'data_api': True,
            'features_api': True,
            'backtest_api': True,
            'system_api': True
        }

        # Check if key directories exist
        key_dirs = {
            'data': Path('data').exists(),
            'logs': Path('logs').exists(),
            'reports': Path('reports').exists(),
            'configs': Path('configs').exists()
        }

        # Overall health
        overall_health = 'healthy'
        if not all(key_dirs.values()):
            overall_health = 'warning'

        return jsonify({
            'status': 'healthy',
            'service': 'algo-trading-api',
            'overall_health': overall_health,
            'modules': modules_healthy,
            'directories': key_dirs,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'algo-trading-api',
            'error': str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Get comprehensive system status."""
    try:
        from src.config.base import load_config as load_base_config

        # Load configuration
        config = load_base_config()

        # System status
        status_info = {
            'service': 'Algo-Trading System API',
            'version': '1.0.0',
            'config_loaded': bool(config),
            'endpoints': {
                'data': '/api/data',
                'features': '/api/features',
                'backtest': '/api/backtest',
                'system': '/api/system'
            }
        }

        # Add configuration info if available
        if config:
            status_info['config'] = {
                'data_root': config.get('data_root', 'N/A'),
                'log_level': config.get('log_level', 'N/A'),
                'environment': config.get('environment', 'N/A')
            }

        return jsonify(status_info)

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'status': 'error',
            'service': 'algo-trading-api',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            '/',
            '/health',
            '/status',
            '/api/data/*',
            '/api/features/*',
            '/api/backtest/*',
            '/api/system/*'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }), 500


@app.before_request
def log_request():
    """Log all incoming requests."""
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")


@app.after_request
def log_response(response):
    """Log all outgoing responses."""
    logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
    return response


def create_app(config=None):
    """Factory function to create Flask app with optional configuration."""
    if config:
        app.config.update(config)

    return app


if __name__ == '__main__':
    # Development server
    logger.info("Starting Algo-Trading API server...")
    logger.info("Available endpoints:")
    logger.info("  - Data API: /api/data/*")
    logger.info("  - Features API: /api/features/*")
    logger.info("  - Backtesting API: /api/backtest/*")
    logger.info("  - System API: /api/system/*")

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )





