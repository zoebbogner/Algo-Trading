#!/usr/bin/env python3
"""Refactored System API endpoints for the Algo-Trading system."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

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


class SystemHealthChecker:
    """Handles system health monitoring."""
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
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
            if disk_free_gb < 1:
                health_status = 'warning'
            if memory_used_percent > 90:
                health_status = 'warning'
            if not all(key_dirs.values()):
                health_status = 'warning'

            return {
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
            }

        except Exception as e:
            logger.error(f"Error in system health check: {e}")
            return {
                'status': 'unhealthy',
                'service': 'system-api',
                'error': str(e)
            }


class TestRunner:
    """Handles test execution."""
    
    @staticmethod
    def run_tests(test_pattern: str = '', coverage: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """Run tests and return results."""
        try:
            # Build pytest command
            cmd = ['python3', '-m', 'pytest']
            
            if test_pattern:
                cmd.append(test_pattern)
            
            if coverage:
                cmd.extend(['--cov=src', '--cov-report=html'])
            
            if verbose:
                cmd.append('-v')

            logger.info(f"Executing test command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }

        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return {
                'success': False,
                'error': 'Test execution timed out after 5 minutes'
            }
        except Exception as e:
            logger.error(f"Error executing tests: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# API Endpoints
@system_router.route('/health', methods=['GET'])
def health_check():
    """Check overall system health."""
    try:
        health_data = SystemHealthChecker.check_system_health()
        return jsonify(health_data)
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

        result = TestRunner.run_tests(test_pattern, coverage, verbose)

        if result['success']:
            logger.info("Test execution completed successfully")
            return jsonify({
                'status': 'success',
                'message': 'Tests completed successfully',
                'result': result
            })
        else:
            logger.error("Test execution failed")
            return jsonify({
                'status': 'error',
                'message': 'Test execution failed',
                'result': result
            }), 500

    except Exception as e:
        logger.error(f"Error in test execution: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Test execution failed',
            'error': str(e)
        }), 500


@system_router.route('/config', methods=['GET'])
def get_config():
    """Get system configuration information."""
    try:
        if not config:
            return jsonify({
                'status': 'error',
                'message': 'Configuration not loaded'
            }), 500
        
        # Return safe configuration (no sensitive data)
        safe_config = {
            'data_root': config.get('data_root'),
            'reports_root': config.get('reports_root'),
            'logs_root': config.get('logs_root'),
            'config_version': config.get('version', 'unknown')
        }
        
        return jsonify({
            'status': 'success',
            'config': safe_config
        })
        
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Configuration retrieval failed',
            'error': str(e)
        }), 500
