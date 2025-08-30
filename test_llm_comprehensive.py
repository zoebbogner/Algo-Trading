#!/usr/bin/env python3
"""Comprehensive test script for Local LLM performance and capabilities."""

import sys
import time
import json
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

def test_llm_performance():
    """Test LLM performance with different prompt types."""
    print("🚀 Comprehensive LLM Performance Test")
    print("=" * 50)
    
    try:
        # Initialize client
        print("🔧 Initializing LLM client...")
        start_time = time.time()
        client = get_llm_client(
            run_id="test_comprehensive",
            config={'temperature': 0.1, 'max_tokens': 100}
        )
        init_time = time.time() - start_time
        print(f"✅ LLM client initialized in {init_time:.2f}s")
        
        # Test different prompt types
        test_cases = [
            {
                'name': 'Simple Greeting',
                'prompt': 'Say "Hello World"',
                'json_mode': False,
                'max_tokens': 10
            },
            {
                'name': 'JSON Response',
                'prompt': 'Generate JSON: {"message": "Hello"}',
                'json_mode': True,
                'max_tokens': 50
            },
            {
                'name': 'Trading Analysis',
                'prompt': 'Analyze BTCUSDT. JSON: {"symbol": "BTCUSDT", "action": "hold|buy|sell", "confidence": 0.5}',
                'json_mode': True,
                'max_tokens': 100
            },
            {
                'name': 'Number Generation',
                'prompt': 'Generate a random number between 1 and 10',
                'json_mode': False,
                'max_tokens': 20
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']}")
            print(f"📝 Prompt: {test_case['prompt']}")
            print(f"⚙️ JSON Mode: {test_case['json_mode']}")
            print(f"🔢 Max Tokens: {test_case['max_tokens']}")
            
            # Generate
            print("🤖 Generating response...")
            gen_start = time.time()
            result = client.generate(
                test_case['prompt'],
                json_mode=test_case['json_mode'],
                max_tokens=test_case['max_tokens']
            )
            gen_time = time.time() - gen_start
            
            # Analyze result
            response_text = result.text
            response_length = len(response_text)
            
            print(f"✅ Generation completed in {gen_time:.2f}s")
            print(f"📄 Response: {repr(response_text)}")
            print(f"📏 Length: {response_length} characters")
            print(f"⏱️ Latency: {result.latency_ms}ms")
            
            # Test JSON parsing if applicable
            json_success = False
            if test_case['json_mode']:
                try:
                    parsed = json.loads(response_text)
                    json_success = True
                    print(f"✅ JSON parsed successfully: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"🔍 Raw response: {response_text}")
            
            # Store results
            results.append({
                'test_name': test_case['name'],
                'prompt': test_case['prompt'],
                'json_mode': test_case['json_mode'],
                'max_tokens': test_case['max_tokens'],
                'generation_time': gen_time,
                'latency_ms': result.latency_ms,
                'response_length': response_length,
                'json_success': json_success,
                'response_text': response_text
            })
            
            # Small delay between tests
            time.sleep(1)
        
        # Performance summary
        print(f"\n📊 Performance Summary")
        print("=" * 50)
        
        total_gen_time = sum(r['generation_time'] for r in results)
        avg_gen_time = total_gen_time / len(results)
        total_latency = sum(r['latency_ms'] for r in results)
        avg_latency = total_latency / len(results)
        json_success_rate = sum(1 for r in results if r['json_success']) / len(results) * 100
        
        print(f"📈 Total Generation Time: {total_gen_time:.2f}s")
        print(f"📊 Average Generation Time: {avg_gen_time:.2f}s")
        print(f"⏱️ Total Latency: {total_latency}ms")
        print(f"📊 Average Latency: {avg_latency:.0f}ms")
        print(f"✅ JSON Success Rate: {json_success_rate:.1f}%")
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/llm_test_results_{timestamp}.json"
        Path("logs").mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'summary': {
                    'total_tests': len(results),
                    'total_generation_time': total_gen_time,
                    'average_generation_time': avg_gen_time,
                    'total_latency_ms': total_latency,
                    'average_latency_ms': avg_latency,
                    'json_success_rate': json_success_rate
                },
                'test_results': results
            }, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to: {results_file}")
        
        # Close client
        client.close()
        print("\n✅ Client closed")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    test_llm_performance()
