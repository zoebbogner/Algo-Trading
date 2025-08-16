#!/usr/bin/env python3
"""Test script to verify basic setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import settings, config_manager
from utils.logging import logger


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        configs = config_manager.load_configs()
        print(f"✓ Configuration loaded successfully")
        print(f"  - App name: {configs.get('app.name', 'N/A')}")
        print(f"  - Trading symbols: {configs.get('trading.default_symbols', 'N/A')}")
        print(f"  - Risk max position: {configs.get('risk.max_position_size', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_environment():
    """Test environment variables."""
    print("Testing environment...")
    
    try:
        print(f"✓ Mode: {settings.mode}")
        print(f"✓ Log level: {settings.log_level}")
        print(f"✓ Data cache dir: {settings.data_cache_dir}")
        print(f"✓ State dir: {settings.state_dir}")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_logging():
    """Test logging system."""
    print("Testing logging...")
    
    try:
        test_logger = logger.get_logger("test")
        test_logger.info("Test log message", extra={"test": True})
        print("✓ Logging system working")
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_directories():
    """Test directory structure."""
    print("Testing directory structure...")
    
    required_dirs = [
        "configs",
        "src",
        "data",
        "state", 
        "logs",
        "reports",
        "tests"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories exist")
        return True


def main():
    """Run all tests."""
    print("🧪 Algo-Trading Setup Test")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_configuration),
        ("Environment", test_environment),
        ("Logging", test_logging),
        ("Directory Structure", test_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
