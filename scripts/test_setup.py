#!/usr/bin/env python3
"""
Test Setup Script

Simple script to verify that the data collection system is properly configured
and can load all necessary components.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test that configuration files can be loaded."""
    print("Testing configuration loading...")
    
    try:
        from configs.base import load_config as load_base_config
        from configs.data_history import load_config as load_data_config
        
        base_config = load_base_config()
        data_config = load_data_config()
        
        print("✅ Configuration loading successful")
        print(f"   Base config keys: {list(base_config.keys())}")
        print(f"   Data config keys: {list(data_config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "configs",
        "data/raw/binance",
        "data/processed/binance", 
        "data/features",
        "data/cache",
        "reports/runs",
        "logs",
        "scripts"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist

def test_config_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")
    
    config_files = [
        "configs/base.yaml",
        "configs/data.history.yaml",
        "configs/costs.sim.yaml"
    ]
    
    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} (missing)")
            all_exist = False
    
    return all_exist

def test_script_files():
    """Test that script files exist."""
    print("\nTesting script files...")
    
    script_files = [
        "scripts/download_binance_csv.py",
        "scripts/fetch_binance_rest.py", 
        "scripts/normalize_data.py",
        "scripts/run_data_collection.py"
    ]
    
    all_exist = True
    for script_file in script_files:
        path = Path(script_file)
        if path.exists():
            print(f"✅ {script_file}")
        else:
            print(f"❌ {script_file} (missing)")
            all_exist = False
    
    return all_exist

def test_dependencies():
    """Test that required Python packages can be imported."""
    print("\nTesting Python dependencies...")
    
    required_packages = [
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("aiohttp", "aiohttp")
    ]
    
    all_available = True
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} (not installed)")
            all_available = False
    
    return all_available

def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA COLLECTION SYSTEM SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_config_files),
        ("Script Files", test_script_files),
        ("Python Dependencies", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run data collection: python scripts/run_data_collection.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
