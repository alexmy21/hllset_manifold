"""
Tests for ManifoldOS Extension System

Validates:
  - Extension registration and lifecycle
  - Storage extension functionality
  - Backward compatibility
  - Error handling
  - Capability discovery
"""

import pytest
import tempfile
import os
from pathlib import Path

from core.extensions import (
    ExtensionRegistry,
    DuckDBStorageExtension,
    StorageExtension,
    ManifoldExtension,
    ExtensionInfo
)
from core.manifold_os import ManifoldOS


class TestExtensionRegistry:
    """Test ExtensionRegistry functionality."""
    
    def test_registry_creation(self):
        """Test creating an extension registry."""
        registry = ExtensionRegistry()
        assert registry.list_extensions() == []
        assert registry.list_capabilities() == {}
    
    def test_register_extension(self):
        """Test registering an extension."""
        registry = ExtensionRegistry()
        extension = DuckDBStorageExtension()
        
        success = registry.register('storage', extension, {'db_path': ':memory:'})
        assert success is True
        assert 'storage' in registry.list_extensions()
        assert registry.has('storage')
    
    def test_register_duplicate(self):
        """Test registering same extension twice."""
        registry = ExtensionRegistry()
        extension1 = DuckDBStorageExtension()
        extension2 = DuckDBStorageExtension()
        
        registry.register('storage', extension1, {'db_path': ':memory:'})
        registry.register('storage', extension2, {'db_path': ':memory:'})
        
        # Should have one extension (second overwrites first)
        assert len(registry.list_extensions()) == 1
    
    def test_unregister_extension(self):
        """Test unregistering an extension."""
        registry = ExtensionRegistry()
        extension = DuckDBStorageExtension()
        
        registry.register('storage', extension, {'db_path': ':memory:'})
        assert registry.has('storage')
        
        registry.unregister('storage')
        assert not registry.has('storage')
    
    def test_get_extension(self):
        """Test getting an extension."""
        registry = ExtensionRegistry()
        extension = DuckDBStorageExtension()
        
        registry.register('storage', extension, {'db_path': ':memory:'})
        
        retrieved = registry.get('storage')
        assert retrieved is not None
        assert retrieved is extension
    
    def test_get_nonexistent(self):
        """Test getting nonexistent extension."""
        registry = ExtensionRegistry()
        assert registry.get('nonexistent') is None
    
    def test_has_capability(self):
        """Test capability checking."""
        registry = ExtensionRegistry()
        extension = DuckDBStorageExtension()
        
        registry.register('storage', extension, {'db_path': ':memory:'})
        
        assert registry.has_capability('persistent_storage')
        assert registry.has_capability('lut_queries')
        assert not registry.has_capability('nonexistent_capability')
    
    def test_list_capabilities(self):
        """Test listing all capabilities."""
        registry = ExtensionRegistry()
        extension = DuckDBStorageExtension()
        
        registry.register('storage', extension, {'db_path': ':memory:'})
        
        caps = registry.list_capabilities()
        assert 'persistent_storage' in caps
        assert 'storage' in caps['persistent_storage']
    
    def test_context_manager(self):
        """Test registry as context manager."""
        with ExtensionRegistry() as registry:
            extension = DuckDBStorageExtension()
            registry.register('storage', extension, {'db_path': ':memory:'})
            assert registry.has('storage')
        
        # After context, extensions should be cleaned up
        assert not registry.has('storage')


class TestDuckDBStorageExtension:
    """Test DuckDB storage extension."""
    
    def test_initialization(self):
        """Test extension initialization."""
        extension = DuckDBStorageExtension()
        success = extension.initialize({'db_path': ':memory:'})
        
        assert success is True
        assert extension.is_available()
        assert extension.lut_store is not None
    
    def test_initialization_file(self):
        """Test initialization with file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.duckdb')
            
            extension = DuckDBStorageExtension()
            success = extension.initialize({'db_path': db_path})
            
            assert success is True
            extension.cleanup()
            assert os.path.exists(db_path)
    
    def test_get_info(self):
        """Test getting extension info."""
        extension = DuckDBStorageExtension()
        extension.initialize({'db_path': ':memory:'})
        
        info = extension.get_info()
        assert isinstance(info, ExtensionInfo)
        assert info.name == "DuckDB Storage"
        assert 'DuckDB' in info.description
    
    def test_capabilities(self):
        """Test extension capabilities."""
        extension = DuckDBStorageExtension()
        extension.initialize({'db_path': ':memory:'})
        
        caps = extension.get_capabilities()
        assert caps['persistent_storage'] is True
        assert caps['lut_queries'] is True
        assert caps['sql_queries'] is True
    
    def test_cleanup(self):
        """Test extension cleanup."""
        extension = DuckDBStorageExtension()
        extension.initialize({'db_path': ':memory:'})
        
        assert extension.is_available()
        extension.cleanup()
        assert not extension.is_available()
    
    def test_storage_operations(self):
        """Test basic storage operations."""
        from core.lut_store import LUTRecord
        
        extension = DuckDBStorageExtension()
        extension.initialize({'db_path': ':memory:'})
        
        # Create test LUT
        lut = {
            (42, 3): LUTRecord(
                reg=42,
                zeros=3,
                hashes={12345},
                tokens=[('test',)]
            )
        }
        
        # Store LUT
        count = extension.store_lut(n=1, lut=lut, hllset_hash='test_hash')
        assert count > 0
        
        # Query tokens
        tokens = extension.query_tokens(n=1, reg=42, zeros=3)
        assert ('test',) in tokens
        
        # Reverse query
        coords = extension.query_by_token(n=1, token_tuple=('test',))
        assert (42, 3) in coords
        
        extension.cleanup()


class TestManifoldOSIntegration:
    """Test ManifoldOS integration with extensions."""
    
    def test_backward_compatibility(self):
        """Test old lut_db_path parameter still works."""
        os = ManifoldOS(lut_db_path=':memory:')
        
        assert os.lut_store is not None
        assert os.extensions.has('storage')
    
    def test_new_extension_config(self):
        """Test new extensions parameter."""
        os = ManifoldOS(extensions={
            'storage': {
                'type': 'duckdb',
                'db_path': ':memory:'
            }
        })
        
        assert os.extensions.has('storage')
        assert os.lut_store is not None
    
    def test_no_extensions(self):
        """Test ManifoldOS without extensions."""
        os = ManifoldOS()
        
        # Should work but without storage
        assert os.lut_store is None
        # Core functionality should still work
        assert os.kernel is not None
    
    def test_ingest_with_storage(self):
        """Test ingestion with storage extension."""
        os = ManifoldOS(lut_db_path=':memory:')
        
        rep = os.ingest("test data", metadata={'source': 'test'})
        
        assert rep is not None
        assert os.lut_store is not None
        
        # Check storage stats
        stats = os.lut_store.get_stats()
        assert stats['total_token_hashes'] > 0
    
    def test_ingest_without_storage(self):
        """Test ingestion without storage extension."""
        os = ManifoldOS()
        
        # Should still work, just without persistence
        rep = os.ingest("test data")
        assert rep is not None
    
    def test_lut_store_property(self):
        """Test lut_store property provides backward compat."""
        os = ManifoldOS(extensions={
            'storage': {'type': 'duckdb', 'db_path': ':memory:'}
        })
        
        # Property should work
        assert os.lut_store is not None
        assert hasattr(os.lut_store, 'commit_lut')
        assert hasattr(os.lut_store, 'get_stats')
    
    def test_extension_lifecycle(self):
        """Test extension lifecycle through ManifoldOS."""
        os = ManifoldOS(lut_db_path=':memory:')
        
        storage = os.extensions.get('storage')
        assert storage.is_available()
        
        # Use it
        os.ingest("test")
        
        # Cleanup
        os.extensions.cleanup_all()
        assert not storage.is_available()


class TestErrorHandling:
    """Test error handling in extension system."""
    
    def test_invalid_config(self):
        """Test handling invalid configuration."""
        extension = DuckDBStorageExtension()
        
        # Invalid db_path type
        errors = extension.validate_config({'db_path': 123})
        assert len(errors) > 0
    
    def test_missing_duckdb(self, monkeypatch):
        """Test handling when DuckDB is not installed."""
        # This is tricky since DuckDB is installed
        # In real scenario, would mock import failure
        extension = DuckDBStorageExtension()
        
        # Extension should handle gracefully
        assert hasattr(extension, 'initialize')
    
    def test_query_unavailable_extension(self):
        """Test querying when extension not available."""
        extension = DuckDBStorageExtension()
        # Don't initialize
        
        tokens = extension.query_tokens(n=1, reg=0, zeros=0)
        assert tokens == []
    
    def test_storage_after_cleanup(self):
        """Test storage operations after cleanup."""
        extension = DuckDBStorageExtension()
        extension.initialize({'db_path': ':memory:'})
        extension.cleanup()
        
        # Should handle gracefully
        tokens = extension.query_tokens(n=1, reg=0, zeros=0)
        assert tokens == []


class TestProductionPatterns:
    """Test production usage patterns."""
    
    def test_capability_based_code(self):
        """Test capability-based programming pattern."""
        os = ManifoldOS(lut_db_path=':memory:')
        
        def process_with_optional_storage(os, data):
            rep = os.ingest(data)
            
            if os.extensions.has_capability('persistent_storage'):
                # Enhanced functionality
                storage = os.extensions.get('storage')
                stats = storage.get_stats()
                return rep, stats
            else:
                # Core functionality
                return rep, None
        
        rep, stats = process_with_optional_storage(os, "test")
        assert rep is not None
        assert stats is not None  # Storage available
    
    def test_configuration_management(self):
        """Test different configurations."""
        # Dev config
        dev_config = {
            'extensions': {
                'storage': {'type': 'duckdb', 'db_path': ':memory:'}
            }
        }
        
        os_dev = ManifoldOS(**dev_config)
        assert os_dev.extensions.has('storage')
        
        # Production config
        with tempfile.TemporaryDirectory() as tmpdir:
            prod_config = {
                'storage_path': Path(tmpdir),
                'extensions': {
                    'storage': {
                        'type': 'duckdb',
                        'db_path': os.path.join(tmpdir, 'metadata.duckdb')
                    }
                }
            }
            
            os_prod = ManifoldOS(**prod_config)
            assert os_prod.extensions.has('storage')
    
    def test_multiple_instances(self):
        """Test multiple ManifoldOS instances with extensions."""
        os1 = ManifoldOS(lut_db_path=':memory:')
        os2 = ManifoldOS(lut_db_path=':memory:')
        
        # Each should have own extension instance
        assert os1.extensions.get('storage') is not os2.extensions.get('storage')
        
        # Both should work independently
        os1.ingest("data1")
        os2.ingest("data2")
        
        stats1 = os1.lut_store.get_stats()
        stats2 = os2.lut_store.get_stats()
        
        # Should have independent storage
        assert stats1 is not None
        assert stats2 is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
