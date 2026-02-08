#!/usr/bin/env python3
"""
Test statelessness of all extensions.

Ensures extensions follow ManifoldOS statelessness principles:
  1. No mutable state accumulated across operations
  2. Idempotent operations (same input → same output)
  3. No hidden side effects
  4. Deterministic behavior
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.extensions.stateless_validator import (
    StatelessnessValidator,
    validate_extension_statelessness
)
from core.extensions.storage import DuckDBStorageExtension


def test_duckdb_storage_statelessness():
    """Test DuckDB storage extension for statelessness."""
    print("\n" + "=" * 70)
    print("Testing DuckDB Storage Extension Statelessness")
    print("=" * 70)
    
    validator = StatelessnessValidator()
    violations = validator.validate_extension(DuckDBStorageExtension, "DuckDBStorage")
    
    # Print report
    report = validator.generate_report(violations)
    print(report)
    
    # Check if stateless (no errors)
    errors = [v for v in violations if v.severity == 'error']
    
    if errors:
        print(f"\n✗ DuckDB Storage has {len(errors)} error(s) - NOT STATELESS")
        return False
    else:
        print("\n✓ DuckDB Storage is STATELESS")
        return True


def test_duckdb_idempotence():
    """Test that DuckDB operations are idempotent."""
    print("\n" + "=" * 70)
    print("Testing DuckDB Storage Idempotence")
    print("=" * 70)
    
    # Create extension
    ext = DuckDBStorageExtension()
    success = ext.initialize({'db_path': ':memory:'})
    
    if not success:
        print("✗ Failed to initialize extension")
        return False
    
    print("✓ Extension initialized")
    
    # Test that get_info() is idempotent
    info1 = ext.get_info()
    info2 = ext.get_info()
    info3 = ext.get_info()
    
    if info1 == info2 == info3:
        print("✓ get_info() is idempotent (same result each time)")
    else:
        print("✗ get_info() is NOT idempotent")
        return False
    
    # Test that get_config_hash() is idempotent
    hash1 = ext.get_config_hash()
    hash2 = ext.get_config_hash()
    hash3 = ext.get_config_hash()
    
    if hash1 == hash2 == hash3:
        print(f"✓ get_config_hash() is idempotent: {hash1[:16]}...")
    else:
        print("✗ get_config_hash() is NOT idempotent")
        return False
    
    # Test that config property is idempotent
    config1 = ext.config
    config2 = ext.config
    config3 = ext.config
    
    if config1 == config2 == config3:
        print(f"✓ config property is idempotent: {config1}")
    else:
        print("✗ config property is NOT idempotent")
        return False
    
    # Test that capabilities are idempotent
    caps1 = ext.get_capabilities()
    caps2 = ext.get_capabilities()
    caps3 = ext.get_capabilities()
    
    if caps1 == caps2 == caps3:
        print(f"✓ get_capabilities() is idempotent ({len(caps1)} capabilities)")
    else:
        print("✗ get_capabilities() is NOT idempotent")
        return False
    
    print("\n✓ All operations are idempotent")
    return True


def test_multiple_instances_independence():
    """Test that multiple extension instances are independent (no shared state)."""
    print("\n" + "=" * 70)
    print("Testing Multiple Instance Independence")
    print("=" * 70)
    
    # Create two instances with different configs
    ext1 = DuckDBStorageExtension()
    ext1.initialize({'db_path': ':memory:', 'threads': 2})
    
    ext2 = DuckDBStorageExtension()
    ext2.initialize({'db_path': ':memory:', 'threads': 4})
    
    # Check they have different hashes
    hash1 = ext1.get_config_hash()
    hash2 = ext2.get_config_hash()
    
    if hash1 != hash2:
        print(f"✓ Instance 1 hash: {hash1[:16]}...")
        print(f"✓ Instance 2 hash: {hash2[:16]}...")
        print("✓ Different configs → different hashes")
    else:
        print("✗ Different configs produced same hash - SHARED STATE DETECTED")
        return False
    
    # Check they have different configs
    config1 = ext1.config
    config2 = ext2.config
    
    if config1 != config2:
        print(f"✓ Instance 1 config: {config1}")
        print(f"✓ Instance 2 config: {config2}")
        print("✓ Instances have independent configurations")
    else:
        print("✗ Instances share configuration - SHARED STATE DETECTED")
        return False
    
    # Verify instance 1 config hasn't changed
    config1_again = ext1.config
    if config1 == config1_again:
        print("✓ Instance 1 config unchanged (no cross-contamination)")
    else:
        print("✗ Instance 1 config changed - SHARED STATE DETECTED")
        return False
    
    print("\n✓ Instances are independent (no shared state)")
    return True


def test_config_isolation():
    """Test that configuration is truly isolated (not shared references)."""
    print("\n" + "=" * 70)
    print("Testing Configuration Isolation")
    print("=" * 70)
    
    ext = DuckDBStorageExtension()
    ext.initialize({'db_path': 'test.db', 'threads': 4})
    
    # Get config multiple times
    config1 = ext.config
    config2 = ext.config
    
    # Modify config1 (should not affect config2 if properly isolated)
    config1['new_key'] = 'new_value'
    
    if 'new_key' not in config2:
        print("✓ Config dict 1: modified")
        print("✓ Config dict 2: unchanged")
        print("✓ Configuration is properly isolated (returns copies)")
    else:
        print("✗ Modification to config1 affected config2 - SHARED REFERENCE")
        return False
    
    # Verify original config is unchanged
    config3 = ext.config
    if 'new_key' not in config3:
        print("✓ Original configuration unchanged")
        print("✓ Each call returns a new copy")
    else:
        print("✗ Original configuration was modified - MUTABLE STATE")
        return False
    
    print("\n✓ Configuration is properly isolated")
    return True


def main():
    """Run all statelessness tests."""
    print("\n" + "=" * 70)
    print("ManifoldOS Extension Statelessness Test Suite")
    print("=" * 70)
    
    results = []
    
    # Test 1: Static analysis for statelessness
    results.append(("Statelessness Analysis", test_duckdb_storage_statelessness()))
    
    # Test 2: Idempotence testing
    results.append(("Idempotence Testing", test_duckdb_idempotence()))
    
    # Test 3: Multiple instance independence
    results.append(("Instance Independence", test_multiple_instances_independence()))
    
    # Test 4: Config isolation
    results.append(("Configuration Isolation", test_config_isolation()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ ALL STATELESSNESS TESTS PASSED")
        print("✓ Extension is certified STATELESS")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed")
        print("✗ Extension is NOT certified stateless")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
