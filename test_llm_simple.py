#!/usr/bin/env python3
"""Simple test script to debug LLM generation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except Exception as e:
    print(f"⚠️ Failed to load .env: {e}")

from src.llm import get_llm_client

def test_simple_generation():
    """Test simple LLM generation."""
    print("🚀 Testing simple LLM generation...")
    
    try:
        # Initialize client
        print("🔧 Initializing LLM client...")
        client = get_llm_client(
            run_id="test_simple",
            config={'temperature': 0.1, 'max_tokens': 50}
        )
        print("✅ LLM client initialized")
        
        # Test different prompt formats
        prompts = [
            "Hello World",
            "Say 'Hello World'",
            "Generate JSON: {\"message\": \"Hello World\"}",
            "Respond with JSON only: {\"greeting\": \"Hello\"}"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🧪 Test {i}: {prompt}")
            print(f"📝 Prompt: {prompt}")
            
            # Generate
            print("🤖 Generating response...")
            start_time = time.time()
            result = client.generate(
                prompt,
                json_mode=False,  # Try without JSON mode first
                max_tokens=20
            )
            end_time = time.time()
            
            print(f"✅ Generation completed in {end_time - start_time:.1f}s!")
            print(f"📄 Result text: {repr(result.text)}")
            print(f"📄 Result text length: {len(result.text)}")
            print(f"⏱️ Latency: {result.latency_ms}ms")
            
            # Try to parse as JSON if it looks like JSON
            if result.text.strip().startswith('{'):
                try:
                    import json
                    parsed = json.loads(result.text)
                    print(f"✅ JSON parsed successfully: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
            else:
                print(f"📄 Raw text response: {result.text}")
        
        # Close client
        client.close()
        print("\n✅ Client closed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import time
    test_simple_generation()
