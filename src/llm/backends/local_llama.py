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
        self._setup_model_path()
        self._init_gpt4all()
        self._setup_backend_info()
        logger.info(f"Local Llama client initialized with model: {Path(self.model_path).name}")
    
    def _setup_model_path(self) -> None:
        """Setup and validate the model path."""
        self.model_path = os.getenv('LLM_INSTRUCT_MODEL_PATH') or self.config.get('model_path')
        if not self.model_path:
            raise LLMConfigError("LLM_INSTRUCT_MODEL_PATH environment variable or model_path config required")
        
        if not Path(self.model_path).exists():
            raise LLMConfigError(f"Model file not found: {self.model_path}")
    
    def _init_gpt4all(self) -> None:
        """Initialize GPT4All model."""
        try:
            from gpt4all import GPT4All
            
            self.model = GPT4All(
                model_name=self.model_path,
                model_path=self.model_path,
                allow_download=False,
                device="cpu"
            )
            
            logger.info(f"GPT4All model loaded: {Path(self.model_path).name}")
            
        except ImportError:
            raise LLMConfigError(
                "GPT4All not installed. Install with: pip install gpt4all"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to load GPT4All model: {e}")
    
    def _setup_backend_info(self) -> None:
        """Setup backend identification information."""
        self.backend = "local_llama"
    
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
            full_prompt = self._prepare_prompt(prompt, system, json_mode)
            gen_kwargs = self._build_generation_kwargs(max_tokens, seed)
            response = self.model.generate(**gen_kwargs)
            
            response_text = self._process_response(response, json_mode)
            result = self._create_result(full_prompt, response_text, start_time)
            
            return result
            
        except Exception as e:
            self._log_generation_error(e, start_time)
            raise LLMUnknownError(f"Generation failed: {e}")
    
    def _prepare_prompt(self, prompt: str, system: Optional[str], json_mode: bool) -> str:
        """Prepare the full prompt with system message and JSON formatting."""
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"System: {system}")
        
        if json_mode:
            prompt_parts.append("IMPORTANT: Respond with valid JSON only. No additional text or explanations.")
        
        prompt_parts.append(f"User: {prompt}")
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _build_generation_kwargs(self, max_tokens: Optional[int], seed: Optional[int]) -> Dict[str, Any]:
        """Build generation parameters for GPT4All."""
        kwargs = {
            'prompt': '',  # Will be set by caller
            'max_tokens': max_tokens or self.max_tokens,
            'temp': self.temperature,
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'streaming': False
        }
        
        if seed is not None:
            kwargs['seed'] = seed
        
        # Remove None values
        return {k: v for k, v in kwargs.items() if v is not None}
    
    def _process_response(self, response: str, json_mode: bool) -> str:
        """Process the model response based on mode."""
        if json_mode:
            return self._extract_json(response)
        return response.strip()
    
    def _create_result(self, prompt: str, response_text: str, start_time: float) -> LLMResult:
        """Create LLMResult from generation."""
        latency_ms = int((time.time() - start_time) * 1000)
        tokens_in = self._estimate_tokens(prompt)
        tokens_out = self._estimate_tokens(response_text)
        
        return LLMResult(
            text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            finish_reason="stop",
            model=Path(self.model_path).name,
            backend=self.backend
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough approximation)."""
        return int(len(text.split()) * 1.3)
    
    def _log_generation_error(self, error: Exception, start_time: float) -> None:
        """Log generation error with timing information."""
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Generation failed after {latency_ms}ms: {error}")
    
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
        
        prompt = self._messages_to_prompt(messages, tools)
        return self.generate(prompt, json_mode=json_mode)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]]) -> str:
        """Convert chat messages to a single prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            prompt_parts.append(f"{role.capitalize()}: {content}")
        
        if tools:
            prompt_parts.append(f"\nAvailable tools:\n{json.dumps(tools, indent=2)}")
        
        return "\n\n".join(prompt_parts)
    
    def _chat_stream(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False
    ) -> List[LLMChunk]:
        """Stream chat response (simulated for local models)."""
        result = self.chat(messages, tools=tools, json_mode=json_mode, stream=False)
        return self._simulate_streaming(result.text)
    
    def _simulate_streaming(self, text: str) -> List[LLMChunk]:
        """Simulate streaming by splitting text into chunks."""
        words = text.split()
        chunk_size = max(1, len(words) // 5)
        chunks = []
        
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
            return [self._create_simple_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise LLMUnknownError(f"Embedding generation failed: {e}")
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple hash-based embedding (not for production)."""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 128-dimensional vector
        embedding = [float(b) / 255.0 for b in hash_bytes] * 8
        return embedding[:128]
    
    def healthcheck(self) -> LLMHealth:
        """Check health status of the local model."""
        try:
            start_time = time.time()
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
                del self.model
                self.model = None
            
            logger.info("Local Llama client closed")
            
        except Exception as e:
            logger.error(f"Error closing client: {e}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    json.loads(json_str)  # Validate
                    return json_str
                except json.JSONDecodeError:
                    continue
        
        return self._repair_json(text)
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair malformed JSON."""
        try:
            cleaned = re.sub(r'[^\{}\[\]",:.\-\d\s]', '', text)
            json_str = self._find_json_structure(cleaned)
            json.loads(json_str)  # Validate
            return json_str
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            return '{"error": "Failed to parse response as JSON"}'
    
    def _find_json_structure(self, cleaned_text: str) -> str:
        """Find JSON structure in cleaned text."""
        start = cleaned_text.find('{')
        if start == -1:
            raise ValueError("No JSON object found")
        
        brace_count = 0
        for i, char in enumerate(cleaned_text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return cleaned_text[start:i+1]
        
        raise ValueError("Incomplete JSON object")
