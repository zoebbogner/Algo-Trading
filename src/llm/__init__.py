"""LLM Module - Pluggable Large Language Model Interface."""

# Core classes and interfaces
from .backends.base import (
    LLMClient,
    LLMResult,
    LLMChunk,
    LLMHealth,
    LLMConfigError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMParsingError,
    LLMUnknownError
)

# Backend implementations
from .backends.local_llama import LocalLlamaClient

# Runtime utilities
from .runtime.loader import (
    get_llm_client,
    healthcheck_all_backends
)

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Core classes
    "LLMClient",
    "LLMResult", 
    "LLMChunk",
    "LLMHealth",
    
    # Error types
    "LLMConfigError",
    "LLMConnectionError", 
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMParsingError",
    "LLMUnknownError",
    
    # Backends
    "LocalLlamaClient",
    
    # Runtime
    "get_llm_client",
    "healthcheck_all_backends",
]
