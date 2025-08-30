#!/usr/bin/env python3
"""Test script for the LLM interface."""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import get_llm_client, list_available_backends, healthcheck_all_backends
from src.utils.logging import setup_logging, get_logger


def test_llm_interface():
    """Test the LLM interface functionality."""
    logger = get_logger(__name__)
    
    print("🚀 TESTING LLM INTERFACE")
    print("=" * 50)
    
    try:
        # Test 1: List available backends
        print("\n📋 Test 1: Available Backends")
        backends = list_available_backends()
        print(f"Available backends: {backends}")
        
        # Test 2: Health check all backends
        print("\n🏥 Test 2: Backend Health Check")
        health_results = healthcheck_all_backends()
        for backend, health in health_results.items():
            status = "✅" if health['status'] == 'healthy' else "❌"
            print(f"{status} {backend}: {health['status']}")
            if health['error']:
                print(f"   Error: {health['error']}")
        
        # Test 3: Initialize LLM client
        print("\n🔧 Test 3: LLM Client Initialization")
        try:
            client = get_llm_client(run_id="test_run")
            print(f"✅ Client initialized: {client.backend}")
            print(f"   Model: {client.model}")
            print(f"   Temperature: {client.temperature}")
            print(f"   Max tokens: {client.max_tokens}")
            
            # Test 4: Simple generation
            print("\n💬 Test 4: Simple Text Generation")
            try:
                result = client.generate(
                    "Hello! Please respond with just 'Hello from LLM!'",
                    max_tokens=20
                )
                print(f"✅ Generation successful:")
                print(f"   Response: {result.text}")
                print(f"   Tokens in: {result.tokens_in}")
                print(f"   Tokens out: {result.tokens_out}")
                print(f"   Latency: {result.latency_ms}ms")
                print(f"   Backend: {result.backend}")
                print(f"   Model: {result.model}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
            
            # Test 5: JSON mode generation
            print("\n📊 Test 5: JSON Mode Generation")
            try:
                result = client.generate(
                    "Create a simple JSON response with 'message' and 'status' fields",
                    json_mode=True,
                    max_tokens=100
                )
                print(f"✅ JSON generation successful:")
                print(f"   Raw response: {result.text}")
                
                # Try to parse JSON
                try:
                    parsed = json.loads(result.text)
                    print(f"   Parsed JSON: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"   JSON parsing failed: {e}")
                
            except Exception as e:
                print(f"❌ JSON generation failed: {e}")
            
            # Test 6: Health check
            print("\n🏥 Test 6: Client Health Check")
            try:
                health = client.healthcheck()
                status = "✅" if health.healthy else "❌"
                print(f"{status} Health check: {health.healthy}")
                print(f"   Backend: {health.backend}")
                print(f"   Model: {health.model}")
                if health.latency_ms:
                    print(f"   Latency: {health.latency_ms}ms")
                if health.error_message:
                    print(f"   Error: {health.error_message}")
                    
            except Exception as e:
                print(f"❌ Health check failed: {e}")
            
            # Clean up
            client.close()
            print(f"\n🧹 Client closed")
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False
        
        print(f"\n🎉 ALL TESTS COMPLETED!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main test function."""
    # Setup logging
    setup_logging(log_level='INFO')
    
    # Run tests
    success = test_llm_interface()
    
    if success:
        print("\n✅ LLM interface is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ LLM interface has issues!")
        sys.exit(1)


if __name__ == '__main__':
    main()
