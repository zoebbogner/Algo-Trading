"""LLM Runtime Loader - Backend selection and client instantiation."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from src.llm.backends.base import LLMClient, LLMConfigError
from src.llm.backends.local_llama import LocalLlamaClient

logger = logging.getLogger(__name__)


def get_llm_client(run_id: str, config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """Get configured LLM client instance."""
    try:
        llm_config = _load_llm_config()
        _apply_env_overrides(llm_config)
        _validate_config(llm_config)
        
        backend = _get_backend_selection(llm_config)
        client = _create_client_instance(backend, llm_config, run_id)
        
        _check_backend_health(client)
        return client
        
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        raise


def _load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from file."""
    config_path = Path("configs/llm.yaml")
    
    if not config_path.exists():
        logger.warning("No LLM config found, using defaults")
        return _get_default_config()
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded LLM config from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load LLM config: {e}")
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Get default LLM configuration."""
    return {
        'backend': 'local_llama',
        'model': 'Meta-Llama-3-8B-Instruct.Q4_0.gguf',
        'temperature': 0.2,
        'max_tokens': 512,
        'timeout_s': 30,
        'retry': {
            'max_attempts': 3,
            'backoff_initial_ms': 250,
            'backoff_max_ms': 2000
        }
    }


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Apply environment variable overrides to config."""
    _override_backend_selection(config)
    _override_model_path(config)
    _override_generation_params(config)


def _override_backend_selection(config: Dict[str, Any]) -> None:
    """Override backend selection from environment."""
    env_backend = os.getenv('LLM_BACKEND')
    if env_backend:
        config['backend'] = env_backend
        logger.info(f"Backend overridden from env: {env_backend}")


def _override_model_path(config: Dict[str, Any]) -> None:
    """Override model path from environment."""
    env_model_path = os.getenv('LLM_INSTRUCT_MODEL_PATH')
    if env_model_path:
        config['model'] = env_model_path
        logger.info(f"Model path overridden from env: {Path(env_model_path).name}")


def _override_generation_params(config: Dict[str, Any]) -> None:
    """Override generation parameters from environment."""
    _override_numeric_param(config, 'LLM_TEMPERATURE', 'temperature')
    _override_numeric_param(config, 'LLM_MAX_TOKENS', 'max_tokens')
    _override_numeric_param(config, 'LLM_TIMEOUT_S', 'timeout_s')


def _override_numeric_param(config: Dict[str, Any], env_key: str, config_key: str) -> None:
    """Override a numeric parameter from environment."""
    env_value = os.getenv(env_key)
    if env_value:
        try:
            config[config_key] = float(env_value)
            logger.info(f"{config_key} overridden from env: {env_value}")
        except ValueError:
            logger.warning(f"Invalid {env_key} value: {env_value}")


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    _validate_required_fields(config)
    _validate_parameter_ranges(config)


def _validate_required_fields(config: Dict[str, Any]) -> None:
    """Validate that required configuration fields are present."""
    required_fields = ['backend', 'model', 'temperature', 'max_tokens', 'timeout_s']
    
    for field in required_fields:
        if field not in config:
            raise LLMConfigError(f"Missing required field: {field}")


def _validate_parameter_ranges(config: Dict[str, Any]) -> None:
    """Validate that configuration parameters are within valid ranges."""
    if not (0.0 <= config['temperature'] <= 2.0):
        raise LLMConfigError(f"Temperature must be between 0.0 and 2.0, got {config['temperature']}")
    
    if config['max_tokens'] <= 0:
        raise LLMConfigError("max_tokens must be positive")
    
    if config['timeout_s'] <= 0:
        raise LLMConfigError("timeout_s must be positive")
    
    retry_config = config.get('retry', {})
    if retry_config.get('max_attempts', 1) <= 0:
        raise LLMConfigError("max_attempts must be positive")


def _get_backend_selection(config: Dict[str, Any]) -> str:
    """Get the selected backend from configuration."""
    backend = config['backend']
    _validate_backend_support(backend)
    return backend


def _validate_backend_support(backend: str) -> None:
    """Validate that the backend is supported."""
    supported_backends = ['local_llama', 'openai_like']
    
    if backend not in supported_backends:
        raise LLMConfigError(f"Unsupported backend: {backend}. Supported: {supported_backends}")


def _create_client_instance(backend: str, config: Dict[str, Any], run_id: str) -> LLMClient:
    """Create the appropriate LLM client instance."""
    if backend == 'local_llama':
        return LocalLlamaClient(config)
    elif backend == 'openai_like':
        # TODO: Implement OpenAI-like backend
        raise NotImplementedError("OpenAI-like backend not yet implemented")
    else:
        raise LLMConfigError(f"Unknown backend: {backend}")


def _check_backend_health(client: LLMClient) -> None:
    """Check backend health and log status."""
    try:
        health = client.healthcheck()
        if health['status'] == 'healthy':
            logger.info(f"Backend {client.backend} is healthy")
        else:
            logger.warning(f"Backend {client.backend} health check failed: {health['error']}")
    except Exception as e:
        logger.warning(f"Health check failed for {client.backend}: {e}")


def healthcheck_all_backends() -> Dict[str, Dict[str, Any]]:
    """Health check all available backends."""
    results = {}
    
    try:
        # Check local llama backend
        local_config = _get_default_config()
        local_config['backend'] = 'local_llama'
        
        local_client = LocalLlamaClient(local_config)
        results['local_llama'] = local_client.healthcheck()
        local_client.close()
        
    except Exception as e:
        results['local_llama'] = {
            'status': 'unhealthy',
            'error': str(e),
            'latency_ms': None
        }
    
    return results
