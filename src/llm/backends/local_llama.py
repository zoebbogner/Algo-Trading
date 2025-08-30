"""Local Llama GGUF backend using GPT4All."""

import os
import time
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .base import (
    LLMClient, LLMResult, LLMChunk, LLMHealth,
    LLMConfigError, LLMConnectionError, LLMTimeoutError, LLMUnknownError
)

logger = logging.getLogger(__name__)


class LocalLlamaClient(LLMClient):
    """Local Llama GGUF client using GPT4All."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize local Llama client."""
        super().__init__(config)
        
        # Get model path from environment or config
        self.model_path = os.getenv('LLM_INSTRUCT_MODEL_PATH') or config.get('model_path')
        if not self.model_path:
            raise LLMConfigError("LLM_INSTRUCT_MODEL_PATH environment variable or model_path config required")
        
        # Validate model file exists
        if not Path(self.model_path).exists():
            raise LLMConfigError(f"Model file not found: {self.model_path}")
        
        # Initialize GPT4All
        self._init_gpt4all()
        
        # Set backend identifier
        self.backend = "local_llama"
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Local Llama client initialized with model: {Path(self.model_path).name}")
    
    def _init_gpt4all(self) -> None:
        """Initialize GPT4All model."""
        try:
            from gpt4all import GPT4All
            
            # Load the model - GPT4All API has changed
            self.model = GPT4All(
                model_name=self.model_path,  # Use model_path as model_name
                model_path=self.model_path,  # Also set model_path for compatibility
                allow_download=False,
                device="cpu"  # Can be enhanced to detect GPU
            )
            
            logger.info(f"GPT4All model loaded: {Path(self.model_path).name}")
            
        except ImportError:
            raise LLMConfigError(
                "GPT4All not installed. Install with: pip install gpt4all"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to load GPT4All model: {e}")
    
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
        start_time = time.time()
        
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, system, json_mode)
            
            # Set generation parameters using correct GPT4All API
            gen_kwargs = {
                'prompt': full_prompt,
                'max_tokens': max_tokens or self.max_tokens,
                'temp': self.temperature,  # GPT4All uses 'temp' not 'temperature'
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
                'streaming': False  # GPT4All uses 'streaming' not 'stream'
            }
            
            # Remove None values to avoid API errors
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            
            # Generate response
            response = self.model.generate(**gen_kwargs)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract text and handle JSON mode
            if json_mode:
                response_text = self._extract_json(response)
            else:
                response_text = response.strip()
            
            # Estimate token counts (rough approximation)
            tokens_in = len(full_prompt.split()) * 1.3  # Rough token estimation
            tokens_out = len(response_text.split()) * 1.3
            
            return LLMResult(
                text=response_text,
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                latency_ms=latency_ms,
                finish_reason="stop",
                model=Path(self.model_path).name,
                backend=self.backend
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Generation failed: {e}")
            raise LLMUnknownError(f"Generation failed: {e}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        *, 
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False,
        stream: bool = False
    ) -> Union[LLMResult, List[LLMChunk]]:
        """Chat with the model using message history."""
        if stream:
            return self._chat_stream(messages, tools, json_mode)
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Add tools if provided
        if tools:
            prompt += "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
        
        # Generate response
        return self.generate(prompt, json_mode=json_mode)
    
    def _chat_stream(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False
    ) -> List[LLMChunk]:
        """Stream chat response (not fully implemented for local models)."""
        # For local models, streaming is limited - return chunks after completion
        result = self.chat(messages, tools=tools, json_mode=json_mode, stream=False)
        
        # Split response into chunks (simulated streaming)
        chunks = []
        words = result.text.split()
        chunk_size = max(1, len(words) // 5)  # 5 chunks
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(LLMChunk(
                delta=chunk_text + (" " if i + chunk_size < len(words) else ""),
                done=i + chunk_size >= len(words)
            ))
        
        return chunks
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            # GPT4All doesn't have native embeddings, so we'll use a simple approach
            # In production, you might want to use a dedicated embedding model
            embeddings = []
            for text in texts:
                # Simple hash-based embedding (not recommended for production)
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                
                # Convert to 128-dimensional vector
                embedding = [float(b) / 255.0 for b in hash_bytes] * 8  # Repeat to get 128 dims
                embeddings.append(embedding[:128])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise LLMUnknownError(f"Embedding generation failed: {e}")
    
    def healthcheck(self) -> LLMHealth:
        """Check health status of the local model."""
        try:
            start_time = time.time()
            
            # Try a simple generation
            test_result = self.generate("Hello", max_tokens=5)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMHealth(
                healthy=True,
                backend=self.backend,
                model=Path(self.model_path).name,
                latency_ms=latency_ms,
                last_check=time.time()
            )
            
        except Exception as e:
            return LLMHealth(
                healthy=False,
                backend=self.backend,
                model=Path(self.model_path).name,
                error_message=str(e),
                last_check=time.time()
            )
    
    def close(self) -> None:
        """Close the client and free resources."""
        try:
            if hasattr(self, 'model') and self.model:
                # GPT4All doesn't have explicit cleanup, but we can clear references
                del self.model
                self.model = None
            
            logger.info("Local Llama client closed")
            
        except Exception as e:
            logger.error(f"Error closing client: {e}")
    
    def _prepare_prompt(self, prompt: str, system: Optional[str], json_mode: bool) -> str:
        """Prepare the full prompt with system message and JSON formatting."""
        full_prompt = ""
        
        # Add system message if provided
        if system:
            full_prompt += f"System: {system}\n\n"
        
        # Add JSON mode instructions if needed
        if json_mode:
            full_prompt += "IMPORTANT: Respond with valid JSON only. No additional text or explanations.\n\n"
        
        # Add user prompt
        full_prompt += f"User: {prompt}\n\n"
        
        # Add assistant prefix
        full_prompt += "Assistant: "
        
        return full_prompt
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        # Try to find JSON in the response
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON code block
            r'```\s*(\{.*?\})\s*```',      # Code block
            r'(\{.*\})',                   # Raw JSON
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    # Validate JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, try to repair common issues
        return self._repair_json(text)
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON."""
        # Remove common non-JSON text
        cleaned = re.sub(r'[^\{}\[\]",:.\-\d\s]', '', text)
        
        # Try to find the JSON structure
        try:
            # Find opening brace
            start = cleaned.find('{')
            if start == -1:
                raise ValueError("No JSON object found")
            
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(cleaned[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = cleaned[start:i+1]
                        json.loads(json_str)  # Validate
                        return json_str
            
            raise ValueError("Incomplete JSON object")
            
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            # Return a minimal valid JSON as fallback
            return '{"error": "Failed to parse response as JSON"}'
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
