"""LLM backend loader and dependency injection."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..backends.base import LLMClient, LLMConfigError
from ..backends.local_llama import LocalLlamaClient

logger = logging.getLogger(__name__)


def get_llm_client(
    run_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> LLMClient:
    """
    Get LLM client based on configuration and environment.
    
    Args:
        run_id: Optional run identifier for logging
        config: Optional configuration override
        
    Returns:
        Configured LLM client instance
        
    Raises:
        LLMConfigError: If configuration is invalid or backend not supported
    """
    
    # Load configuration
    llm_config = _load_llm_config(config)
    
    # Get backend selection (env overrides config)
    backend = os.getenv('LLM_BACKEND') or llm_config.get('backend', 'local_llama')
    
    # Validate backend is supported
    if backend not in SUPPORTED_BACKENDS:
        raise LLMConfigError(f"Unsupported backend: {backend}. Supported: {list(SUPPORTED_BACKENDS.keys())}")
    
    # Create client instance
    try:
        client = SUPPORTED_BACKENDS[backend](llm_config)
        
        # Set run_id if provided
        if run_id:
            client.run_id = run_id
        
        logger.info(f"LLM client initialized: {backend} with model {client.model}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM client for backend {backend}: {e}")
        raise LLMConfigError(f"Backend initialization failed: {e}")


def _load_llm_config(config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load LLM configuration from config files and environment.
    
    Args:
        config_override: Optional configuration override
        
    Returns:
        Merged configuration dictionary
    """
    
    # Start with default configuration
    config = {
        'backend': 'local_llama',
        'model': 'Meta-Llama-3-8B-Instruct.Q4_0.gguf',
        'temperature': 0.2,
        'max_tokens': 512,
        'stop': [],
        'json_mode': True,
        'timeout_s': 30,
        'retry': {
            'max_attempts': 3,
            'backoff_initial_ms': 250,
            'backoff_max_ms': 2000
        }
    }
    
    # Load from config file if it exists
    config_file = Path('configs/llm.yaml')
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
                logger.info(f"Loaded LLM config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load LLM config from {config_file}: {e}")
    
    # Apply environment overrides
    config = _apply_env_overrides(config)
    
    # Apply config override if provided
    if config_override:
        config.update(config_override)
        logger.info("Applied configuration override")
    
    # Validate required fields
    _validate_llm_config(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    
    # Backend selection
    if os.getenv('LLM_BACKEND'):
        config['backend'] = os.getenv('LLM_BACKEND')
    
    # Model path for local models
    if os.getenv('LLM_INSTRUCT_MODEL_PATH'):
        config['model_path'] = os.getenv('LLM_INSTRUCT_MODEL_PATH')
    
    # Temperature
    if os.getenv('LLM_TEMPERATURE'):
        try:
            config['temperature'] = float(os.getenv('LLM_TEMPERATURE'))
        except ValueError:
            logger.warning(f"Invalid LLM_TEMPERATURE: {os.getenv('LLM_TEMPERATURE')}")
    
    # Max tokens
    if os.getenv('LLM_MAX_TOKENS'):
        try:
            config['max_tokens'] = int(os.getenv('LLM_MAX_TOKENS'))
        except ValueError:
            logger.warning(f"Invalid LLM_MAX_TOKENS: {os.getenv('LLM_MAX_TOKENS')}")
    
    # Timeout
    if os.getenv('LLM_TIMEOUT_S'):
        try:
            config['timeout_s'] = int(os.getenv('LLM_TIMEOUT_S'))
        except ValueError:
            logger.warning(f"Invalid LLM_TIMEOUT_S: {os.getenv('LLM_TIMEOUT_S')}")
    
    return config


def _validate_llm_config(config: Dict[str, Any]) -> None:
    """Validate LLM configuration."""
    
    required_fields = ['backend', 'temperature', 'max_tokens', 'timeout_s']
    for field in required_fields:
        if field not in config:
            raise LLMConfigError(f"Missing required field: {field}")
    
    # Validate temperature
    if not (0.0 <= config['temperature'] <= 2.0):
        raise LLMConfigError(f"Temperature must be between 0.0 and 2.0, got {config['temperature']}")
    
    # Validate max_tokens
    if config['max_tokens'] <= 0:
        raise LLMConfigError(f"max_tokens must be positive, got {config['max_tokens']}")
    
    # Validate timeout
    if config['timeout_s'] <= 0:
        raise LLMConfigError(f"timeout_s must be positive, got {config['timeout_s']}")
    
    # Validate retry configuration
    retry_config = config.get('retry', {})
    if 'max_attempts' in retry_config and retry_config['max_attempts'] <= 0:
        raise LLMConfigError(f"retry.max_attempts must be positive, got {retry_config['max_attempts']}")
    
    logger.info("LLM configuration validation passed")


# Supported backends mapping
SUPPORTED_BACKENDS = {
    'local_llama': LocalLlamaClient,
    # Add more backends here as they're implemented
    # 'openai_like': OpenAILikeClient,
    # 'together': TogetherClient,
    # 'bedrock': BedrockClient,
}


def list_available_backends() -> list[str]:
    """List all available LLM backends."""
    return list(SUPPORTED_BACKENDS.keys())


def get_backend_info(backend: str) -> Dict[str, Any]:
    """Get information about a specific backend."""
    if backend not in SUPPORTED_BACKENDS:
        raise LLMConfigError(f"Unknown backend: {backend}")
    
    backend_class = SUPPORTED_BACKENDS[backend]
    
    return {
        'name': backend,
        'class': backend_class.__name__,
        'module': backend_class.__module__,
        'description': backend_class.__doc__ or 'No description available'
    }


def healthcheck_all_backends() -> Dict[str, Any]:
    """Health check all available backends."""
    results = {}
    
    for backend_name in SUPPORTED_BACKENDS:
        try:
            # Create a minimal config for health check
            config = {
                'backend': backend_name,
                'model': 'test',
                'temperature': 0.1,
                'max_tokens': 10,
                'timeout_s': 5
            }
            
            # Try to create client (this will test basic initialization)
            client = SUPPORTED_BACKENDS[backend_name](config)
            
            # Try health check if available
            if hasattr(client, 'healthcheck'):
                health = client.healthcheck()
                results[backend_name] = {
                    'status': 'healthy' if health.healthy else 'unhealthy',
                    'error': health.error_message,
                    'latency_ms': health.latency_ms
                }
            else:
                results[backend_name] = {
                    'status': 'unknown',
                    'error': 'No healthcheck method available',
                    'latency_ms': None
                }
            
            # Clean up
            if hasattr(client, 'close'):
                client.close()
                
        except Exception as e:
            results[backend_name] = {
                'status': 'error',
                'error': str(e),
                'latency_ms': None
            }
    
    return results
