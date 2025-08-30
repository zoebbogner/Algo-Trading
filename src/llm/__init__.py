"""LLM Module for Algo-Trading System.

This module provides a pluggable interface for Large Language Models
to act as trading orchestrators and advisors.
"""

from .backends.base import (
    LLMClient, LLMResult, LLMChunk, LLMHealth,
    LLMError, LLMConfigError, LLMConnectionError,
    LLMRateLimitError, LLMTimeoutError, LLMParsingError, LLMUnknownError
)

from .runtime.loader import (
    get_llm_client,
    list_available_backends,
    get_backend_info,
    healthcheck_all_backends
)

from .backends.local_llama import LocalLlamaClient

__version__ = "1.0.0"

__all__ = [
    # Core interfaces
    "LLMClient",
    "LLMResult", 
    "LLMChunk",
    "LLMHealth",
    
    # Error types
    "LLMError",
    "LLMConfigError",
    "LLMConnectionError", 
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMParsingError",
    "LLMUnknownError",
    
    # Runtime utilities
    "get_llm_client",
    "list_available_backends",
    "get_backend_info", 
    "healthcheck_all_backends",
    
    # Backend implementations
    "LocalLlamaClient",
]
