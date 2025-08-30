#!/usr/bin/env python3
"""OpenAI API backend for LLM integration."""

import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base import (
    LLMClient, LLMResult, LLMChunk, LLMHealth,
    LLMConfigError, LLMConnectionError, LLMTimeoutError, LLMUnknownError
)

logger = logging.getLogger(__name__)


class OpenAILikeClient(LLMClient):
    """OpenAI API client for LLM operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI client with configuration."""
        super().__init__(config)
        
        # Load environment variables
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Configuration
        self.model = config.get('model', 'gpt-4o-mini')  # Default to gpt-4o-mini
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout = config.get('timeout_s', 30)
        
        # Audit logging
        self.audit_log_path = "logs/llm_audit.jsonl"
        self._ensure_audit_log()
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
        logger.info(f"Base URL: {self.base_url}")
    
    def _ensure_audit_log(self):
        """Ensure audit log directory exists."""
        import os
        from pathlib import Path
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create audit log file if it doesn't exist
        if not os.path.exists(self.audit_log_path):
            with open(self.audit_log_path, 'w') as f:
                f.write("")  # Create empty file
    
    def _log_audit(self, prompt: str, response: str, latency_ms: int, 
                   token_usage: Optional[Dict[str, int]] = None, error: Optional[str] = None):
        """Log audit information to JSONL file."""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "prompt": prompt,
                "response": response,
                "latency_ms": latency_ms,
                "token_usage": token_usage or {},
                "error": error,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")
    
    def generate(self, prompt: str, *, system: Optional[str] = None, 
                stop: Optional[List[str]] = None, max_tokens: Optional[int] = None, 
                json_mode: bool = False, seed: Optional[int] = None) -> LLMResult:
        """Generate text using OpenAI API."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            
            messages.append({"role": "user", "content": prompt})
            
            # Prepare generation parameters
            gen_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "timeout": self.timeout
            }
            
            # Add JSON mode if requested
            if json_mode:
                gen_kwargs["response_format"] = {"type": "json_object"}
            
            # Add stop sequences if provided
            if stop:
                gen_kwargs["stop"] = stop
            
            # Add seed if provided (for reproducibility)
            if seed is not None:
                gen_kwargs["seed"] = seed
            
            logger.info(f"Calling OpenAI API with model: {self.model}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(**gen_kwargs)
            
            # Extract response content
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage
            token_usage = {}
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log audit information
            self._log_audit(prompt, content, latency_ms, token_usage)
            
            logger.info(f"OpenAI API call completed in {latency_ms}ms")
            logger.info(f"Token usage: {token_usage}")
            
            # Create LLMResult
            result = LLMResult(
                text=content,
                tokens_in=token_usage.get("prompt_tokens", 0),
                tokens_out=token_usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason or "stop",
                raw=response,
                model=self.model,
                backend="openai_like"
            )
            
            return result
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            # Log audit information with error
            self._log_audit(prompt, "", latency_ms, error=error_msg)
            
            logger.error(f"OpenAI API call failed: {error_msg}")
            raise
    
    def close(self):
        """Close the OpenAI client."""
        # OpenAI client doesn't need explicit closing, but we can log it
        logger.info("OpenAI client closed")
    
    def chat(self, messages: List[Dict[str, str]], *, tools: Optional[List[Dict[str, Any]]] = None,
             json_mode: bool = False, stream: bool = False) -> Union[LLMResult, List[LLMChunk]]:
        """Chat with the model using message history."""
        if stream:
            raise NotImplementedError("Streaming not yet implemented for OpenAI backend")
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if 'role' in msg and 'content' in msg:
                openai_messages.append(msg)
            else:
                # Assume user message if format unclear
                openai_messages.append({"role": "user", "content": str(msg)})
        
        # Use the generate method with the last message as prompt
        if openai_messages:
            last_message = openai_messages[-1]["content"]
            return self.generate(last_message, json_mode=json_mode)
        else:
            raise ValueError("No valid messages provided")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts (optional)."""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",  # OpenAI's embedding model
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def healthcheck(self) -> LLMHealth:
        """Check health status of the backend."""
        start_time = time.time()
        
        try:
            # Make a simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMHealth(
                healthy=True,
                backend="openai_like",
                model=self.model,
                latency_ms=latency_ms,
                error_message=None,
                last_check=datetime.now()
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMHealth(
                healthy=False,
                backend="openai_like",
                model=self.model,
                latency_ms=latency_ms,
                error_message=str(e),
                last_check=datetime.now()
            )
    
    def is_healthy(self) -> bool:
        """Check if the OpenAI API is healthy."""
        try:
            # Make a simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=5
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
