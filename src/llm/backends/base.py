"""Abstract base interface for LLM backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class LLMResult:
    """Standardized result from any LLM backend."""
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    finish_reason: str
    raw: Any = None
    model: str = ""
    backend: str = ""


@dataclass
class LLMChunk:
    """Streaming chunk from LLM."""
    delta: str
    done: bool
    usage_partial: Optional[Dict[str, Any]] = None


@dataclass
class LLMHealth:
    """Health status of LLM backend."""
    healthy: bool
    backend: str
    model: str
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    last_check: datetime = None


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMConfigError(LLMError):
    """Configuration error (bad config/unsupported param)."""
    pass


class LLMConnectionError(LLMError):
    """Network/connection error or model unavailable."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded (429-style)."""
    pass


class LLMTimeoutError(LLMError):
    """Request timeout."""
    pass


class LLMParsingError(LLMError):
    """Invalid JSON in json_mode."""
    pass


class LLMUnknownError(LLMError):
    """Unknown/unexpected error."""
    pass


class LLMClient(ABC):
    """Abstract interface for LLM clients."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM client with configuration."""
        self.config = config
        self.model = config.get('model', '')
        self.backend = config.get('backend', '')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout_s = config.get('timeout_s', 30)
        
        # Retry configuration
        retry_config = config.get('retry', {})
        self.max_attempts = retry_config.get('max_attempts', 3)
        self.backoff_initial_ms = retry_config.get('backoff_initial_ms', 250)
        self.backoff_max_ms = retry_config.get('backoff_max_ms', 2000)
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        *, 
        system: Optional[str] = None,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        seed: Optional[int] = None
    ) -> LLMResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        *, 
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False,
        stream: bool = False
    ) -> Union[LLMResult, List[LLMChunk]]:
        """Chat with the model using message history."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts (optional)."""
        pass
    
    @abstractmethod
    def healthcheck(self) -> LLMHealth:
        """Check health status of the backend."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the client and free resources."""
        pass
    
    def _measure_latency(self, func):
        """Decorator to measure function latency."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = int((time.time() - start_time) * 1000)
                if hasattr(result, 'latency_ms'):
                    result.latency_ms = latency_ms
                return result
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                raise e
        return wrapper
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.model:
            raise LLMConfigError("Model name/path is required")
        
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise LLMConfigError("Temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise LLMConfigError("max_tokens must be positive")
        
        if self.timeout_s <= 0:
            raise LLMConfigError("timeout_s must be positive")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except (LLMConnectionError, LLMRateLimitError, LLMTimeoutError) as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    # Calculate backoff delay
                    delay_ms = min(
                        self.backoff_initial_ms * (2 ** attempt),
                        self.backoff_max_ms
                    )
                    time.sleep(delay_ms / 1000.0)
                continue
            except Exception as e:
                # Don't retry on non-retryable errors
                raise e
        
        # All retries exhausted
        raise last_exception
