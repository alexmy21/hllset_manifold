#!/usr/bin/env python3
"""
Simple test runner for extension system (doesn't require pytest).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.extensions import (
    ExtensionRegistry,
    DuckDBStorageExtension,
)
from core.manifold_os import ManifoldOS


def test_extension_registry():
    """Test basic registry functionality."""
    print("Testing ExtensionRegistry...")
    
    registry = ExtensionRegistry()
    assert len(registry.list_extensions()) == 0
    
    extension = DuckDBStorageExtension()
    success = registry.register('storage', extension, {'db_path': ':memory:'})
    assert success, "Registration failed"
    assert 'storage' in registry.list_extensions()
    
    assert registry.has_capability('persistent_storage')
    assert registry.has_capability('lut_queries')
    
    registry.cleanup_all()
    print("  ✓ ExtensionRegistry tests passed\n")


def test_duckdb_extension():
    """Test DuckDB storage extension."""
    print("Testing DuckDBStorageExtension...")
    
    extension = DuckDBStorageExtension()
    success = extension.initialize({'db_path': ':memory:'})
    assert success, "Initialization failed"
    assert extension.is_available()
    
    info = extension.get_info()
    assert info.name == "DuckDB Storage"
    
    caps = extension.get_capabilities()
    assert caps['persistent_storage'] is True
    assert caps['lut_queries'] is True
    
    extension.cleanup()
    assert not extension.is_available()
    print("  ✓ DuckDBStorageExtension tests passed\n")


def test_backward_compatibility():
    """Test backward compatibility with old API."""
    print("Testing backward compatibility...")
    
    os = ManifoldOS(lut_db_path=':memory:')
    assert os.lut_store is not None
    assert os.extensions.has('storage')
    
    # Old API should work
    stats = os.lut_store.get_stats()
    assert 'total_token_hashes' in stats
    
    os.extensions.cleanup_all()
    print("  ✓ Backward compatibility tests passed\n")


def test_new_config_style():
    """Test new extension configuration."""
    print("Testing new configuration style...")
    
    os = ManifoldOS(extensions={
        'storage': {
            'type': 'duckdb',
            'db_path': ':memory:'
        }
    })
    
    assert os.extensions.has('storage')
    assert os.lut_store is not None
    
    caps = os.extensions.list_capabilities()
    assert 'persistent_storage' in caps
    
    os.extensions.cleanup_all()
    print("  ✓ New configuration tests passed\n")


def test_ingest_with_storage():
    """Test data ingestion with storage."""
    print("Testing ingestion with storage...")
    
    os = ManifoldOS(lut_db_path=':memory:')
    
    rep = os.ingest("test data example", metadata={'source': 'test'})
    assert rep is not None
    
    stats = os.lut_store.get_stats()
    assert stats['total_token_hashes'] > 0
    
    os.extensions.cleanup_all()
    print("  ✓ Ingestion with storage tests passed\n")


def test_storage_operations():
    """Test storage operations."""
    print("Testing storage operations...")
    
    from core.lut_store import LUTRecord
    
    extension = DuckDBStorageExtension()
    extension.initialize({'db_path': ':memory:'})
    
    # Calculate correct hash for the token
    n = 1
    token = ('test',)
    token_hash = hash(f"__n{n}__" + "__".join(token)) & 0xFFFFFFFFFFFFFFFF
    
    # Create test LUT
    lut = {
        (42, 3): LUTRecord(
            reg=42,
            zeros=3,
            hashes={token_hash},
            tokens=[token]
        )
    }
    
    # Store
    count = extension.store_lut(n=n, lut=lut, hllset_hash='test_hash')
    assert count > 0, f"Store failed, count={count}"
    
    # Query forward
    tokens = extension.query_tokens(n=n, reg=42, zeros=3)
    assert token in tokens, f"Token not found. Got: {tokens}"
    
    # Query reverse
    coords = extension.query_by_token(n=n, token_tuple=token)
    assert (42, 3) in coords, f"Coordinates not found. Got: {coords}"
    
    extension.cleanup()
    print("  ✓ Storage operations tests passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("ManifoldOS Extension System Tests")
    print("="*60 + "\n")
    
    tests = [
        test_extension_registry,
        test_duckdb_extension,
        test_backward_compatibility,
        test_new_config_style,
        test_ingest_with_storage,
        test_storage_operations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} ERROR: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
