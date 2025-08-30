#!/usr/bin/env python3
"""Test script for LLM interface functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import get_llm_client, healthcheck_all_backends
from src.utils.logging import setup_logging, get_logger


def test_llm_interface():
    """Test the LLM interface functionality."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("🧪 Testing LLM Interface")
    logger.info("=" * 50)
    
    # Test 1: Backend Health Check
    logger.info("Test 1: Backend Health Check")
    try:
        health_results = healthcheck_all_backends()
        logger.info(f"Health check results: {health_results}")
        logger.info("✅ Backend health check passed")
    except Exception as e:
        logger.error(f"❌ Backend health check failed: {e}")
        return False
    
    # Test 2: Client Initialization
    logger.info("Test 2: Client Initialization")
    try:
        client = get_llm_client(run_id="test_run")
        logger.info(f"✅ Client initialized: {client.backend}")
    except Exception as e:
        logger.error(f"❌ Client initialization failed: {e}")
        return False
    
    # Test 3: Basic Text Generation
    logger.info("Test 3: Basic Text Generation")
    try:
        result = client.generate("Hello, how are you?")
        logger.info(f"✅ Generation successful: {result.text[:100]}...")
        logger.info(f"   Tokens: {result.tokens_in} → {result.tokens_out}")
        logger.info(f"   Latency: {result.latency_ms}ms")
    except Exception as e:
        logger.error(f"❌ Basic generation failed: {e}")
        return False
    
    # Test 4: JSON Mode Generation
    logger.info("Test 4: JSON Mode Generation")
    try:
        result = client.generate(
            "Generate a simple JSON object with a 'message' field",
            json_mode=True
        )
        logger.info(f"✅ JSON generation successful: {result.text}")
    except Exception as e:
        logger.error(f"❌ JSON generation failed: {e}")
        return False
    
    # Test 5: Chat Interface
    logger.info("Test 5: Chat Interface")
    try:
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        result = client.chat(messages)
        logger.info(f"✅ Chat successful: {result.text}")
    except Exception as e:
        logger.error(f"❌ Chat failed: {e}")
        return False
    
    # Test 6: Health Check
    logger.info("Test 6: Client Health Check")
    try:
        health = client.healthcheck()
        logger.info(f"✅ Health check successful: {health}")
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False
    
    # Cleanup
    client.close()
    logger.info("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_llm_interface()
    sys.exit(0 if success else 1)
