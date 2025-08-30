"""Abstract base interface for LLM backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


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
        self._setup_config(config)
        self._setup_retry_config(config)
        self._validate_config()
    
    def _setup_config(self, config: Dict[str, Any]) -> None:
        """Setup basic configuration parameters."""
        self.config = config
        self.model = config.get('model', '')
        self.backend = config.get('backend', '')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout_s = config.get('timeout_s', 30)
    
    def _setup_retry_config(self, config: Dict[str, Any]) -> None:
        """Setup retry configuration parameters."""
        retry_config = config.get('retry', {})
        self.max_attempts = retry_config.get('max_attempts', 3)
        self.backoff_initial_ms = retry_config.get('backoff_initial_ms', 250)
        self.backoff_max_ms = retry_config.get('backoff_max_ms', 2000)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        self._validate_required_fields()
        self._validate_parameter_ranges()
    
    def _validate_required_fields(self) -> None:
        """Validate that required configuration fields are present."""
        required_fields = ['model', 'temperature', 'max_tokens', 'timeout_s']
        for field in required_fields:
            if field not in self.config:
                raise LLMConfigError(f"Missing required field: {field}")
    
    def _validate_parameter_ranges(self) -> None:
        """Validate that configuration parameters are within valid ranges."""
        if not (0.0 <= self.temperature <= 2.0):
            raise LLMConfigError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise LLMConfigError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if self.timeout_s <= 0:
            raise LLMConfigError(f"timeout_s must be positive, got {self.timeout_s}")
        
        if self.max_attempts <= 0:
            raise LLMConfigError(f"max_attempts must be positive, got {self.max_attempts}")
    
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
                self._update_result_latency(result, start_time)
                return result
            except Exception as e:
                self._log_latency_on_error(start_time)
                raise e
        return wrapper
    
    def _update_result_latency(self, result: Any, start_time: float) -> None:
        """Update result with latency information if possible."""
        latency_ms = int((time.time() - start_time) * 1000)
        if hasattr(result, 'latency_ms'):
            result.latency_ms = latency_ms
    
    def _log_latency_on_error(self, start_time: float) -> None:
        """Log latency information when an error occurs."""
        latency_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Operation failed after {latency_ms}ms")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except (LLMConnectionError, LLMRateLimitError, LLMTimeoutError) as e:
                last_exception = e
                if self._should_retry(attempt):
                    self._wait_before_retry(attempt)
                    continue
                break
            except Exception as e:
                # Don't retry on non-retryable errors
                raise e
        
        # All retries exhausted
        raise last_exception
    
    def _should_retry(self, attempt: int) -> bool:
        """Determine if we should retry based on attempt number."""
        return attempt < self.max_attempts - 1
    
    def _wait_before_retry(self, attempt: int) -> None:
        """Wait before retrying with exponential backoff."""
        delay_ms = min(
            self.backoff_initial_ms * (2 ** attempt),
            self.backoff_max_ms
        )
        time.sleep(delay_ms / 1000.0)
