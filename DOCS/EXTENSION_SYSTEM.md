# ManifoldOS Extension System

The extension system provides a pluggable architecture for integrating external resources and capabilities into ManifoldOS while maintaining core principles of **immutability**, **idempotence**, **content-addressability**, and **separation of concerns**.

## Core Principles

All extensions follow ManifoldOS core principles:

1. **Immutability**: Extension configurations are frozen after initialization
   - Configurations cannot be changed after setup
   - Ensures stable, predictable behavior
   - Enables content-addressability

2. **Idempotence**: Same input → same output
   - Operations can be safely repeated
   - No side effects from multiple calls
   - Critical for knowledge base integration

3. **Content-Addressability**: Everything has a stable hash
   - Extension configs identified by hash
   - Storage records identified by hash
   - Knowledge base integration ready

4. **Statelessness**: No application state in extensions
   - Extensions are pure processors
   - State managed by application
   - See [STATELESSNESS_REQUIREMENTS.md](STATELESSNESS_REQUIREMENTS.md)

5. **Separation of Concerns**: Clear boundaries
   - **Application**: State management, business logic, policy
   - **Extension**: Processing, persistence, resources
   - See [REFERENCE_IMPLEMENTATION.md](REFERENCE_IMPLEMENTATION.md)

## Reference Implementation

**DuckDB Storage** serves as the reference implementation demonstrating all patterns:

- Proper separation of concerns
- Application vs extension responsibilities
- Stateless processing
- Idempotent operations

See [REFERENCE_IMPLEMENTATION.md](REFERENCE_IMPLEMENTATION.md) for detailed patterns and examples.

## Overview

Extensions are **optional components** that enhance ManifoldOS functionality without being hard dependencies. The system is designed around:

- **Loose coupling**: Core doesn't depend on extensions
- **Graceful degradation**: Missing extensions don't break core functionality
- **Progressive enhancement**: Better experience with extensions, but works without them
- **Clear contracts**: Well-defined interfaces for each extension type
- **Immutable design**: Frozen configs, content-addressed storage

## Architecture

```text
ManifoldOS
    ├── ExtensionRegistry (manages all extensions)
    │   ├── StorageExtension (DuckDB, PostgreSQL, Redis)
    │   ├── CacheExtension (Redis, Memcached) [future]
    │   ├── MonitoringExtension (Prometheus, StatsD) [future]
    │   └── VectorStoreExtension (Pinecone, Weaviate) [future]
    └── Core functionality (always available)
```

## Usage

### Backward Compatible (Old Style)

```python
from core.manifold_os import ManifoldOS

# Still works - automatically uses DuckDB storage
os = ManifoldOS(lut_db_path=':memory:')
```

### New Extension Configuration

```python
from core.manifold_os import ManifoldOS

# Explicit extension configuration
os = ManifoldOS(extensions={
    'storage': {
        'type': 'duckdb',
        'db_path': 'metadata.duckdb'
    }
})

# Check capabilities
if os.extensions.has_capability('persistent_storage'):
    print("✓ Persistent storage available")

# Access extension directly
storage = os.extensions.get('storage')
print(storage.get_info())
```

### No Extensions

```python
# Works fine without any extensions
os = ManifoldOS()
# Core functionality available, persistence disabled
```

## Creating New Extensions

### 1. Define Extension Interface

```python
from core.extensions.base import ManifoldExtension, ExtensionInfo

class MyExtension(ManifoldExtension):
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize with IMMUTABLE configuration.
        
        Call _freeze_config() to lock configuration:
          - Makes config content-addressed
          - Ensures idempotence
          - Prepares for knowledge base
        """
        try:
            # Freeze configuration first (REQUIRED)
            self._freeze_config(config, extension_type='my_extension')
            
            # Setup your extension using frozen config
            option = self.config.get('option1', 'default')
            # ... initialization code ...
            
            self._available = True
            return True
        except Exception as e:
            self._available = False
            return False
    
    def is_available(self) -> bool:
        return self._available
    
    def cleanup(self):
        # Release resources
        pass
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'my_capability': True,
            'idempotent': True,          # Operations can be repeated
            'content_addressable': True   # Records identified by hash
        }
    
    def get_info(self) -> ExtensionInfo:
        return ExtensionInfo(
            name="My Extension",
            version="1.0.0",
            description="Does something useful",
            config_hash=self.get_config_hash()  # Include content-addressed ID
        )
```

### 2. Register Extension

```python
from core.extensions import ExtensionRegistry

registry = ExtensionRegistry()
extension = MyExtension()
success = registry.register('my_ext', extension, config={
    'option1': 'value1'
})

# Configuration is now frozen - cannot be changed
# Extension has stable content-addressed hash
print(f"Config hash: {extension.get_config_hash()}")
```

### 3. Use Extension

```python
if registry.has_capability('my_capability'):
    ext = registry.get('my_ext')
    ext.do_something()
```

## Available Extensions

### StorageExtension (DuckDB)

Provides persistent storage for LUTs and metadata with **immutable, content-addressed** design.

**Core Principles:**

- **Immutable**: Configuration frozen at initialization
- **Idempotent**: Same data → same storage result
- **Content-Addressed**: Records identified by hash (token_hash is PRIMARY KEY)
- **Append-Only**: Records never deleted, only updated with new collisions

**Capabilities:**

- `persistent_storage`: Store data to disk
- `lut_queries`: Query token → coordinate mappings
- `metadata_tracking`: Track ingestion provenance
- `sql_queries`: Full SQL query support
- `transactions`: ACID guarantees
- `analytics`: Storage statistics
- `content_addressable`: Records identified by hash
- `append_only`: Immutable storage pattern
- `idempotent`: Same data → same result

**Configuration:**

```python
{
    'type': 'duckdb',
    'db_path': ':memory:' | '/path/to/db.duckdb',
    'read_only': False,
    'threads': None  # Auto-detect
}
```

**Methods:**

- `store_lut(n, lut, hllset_hash, metadata)`: Store lookup table
- `query_tokens(n, reg, zeros, hllset_hash)`: Get tokens at coordinates
- `query_by_token(n, token_tuple)`: Reverse lookup
- `get_metadata(hllset_hash)`: Get ingestion metadata
- `get_stats()`: Storage statistics

## Future Extensions

### CacheExtension (Redis)

Fast in-memory caching for frequently accessed data.

### MonitoringExtension (Prometheus)

Metrics collection and monitoring.

### VectorStoreExtension (Pinecone/Weaviate)

Semantic search and vector similarity.

## Extension Lifecycle

1. **Creation**: `extension = MyExtension()`
2. **Registration**: `registry.register('name', extension, config)`
3. **Initialization**: Extension sets up resources
4. **Use**: Core code calls extension methods
5. **Cleanup**: `registry.cleanup_all()` or context manager

## Best Practices

### For Extension Developers

1. **Never crash**: Handle errors gracefully, return False on failure
2. **Declare capabilities**: Be explicit about what you provide
3. **Validate config**: Check configuration in `validate_config()`
4. **Clean up**: Always release resources in `cleanup()`
5. **Test availability**: Check dependencies in `is_available()`

### For Extension Users

1. **Check availability**: Always check `is_available()` before use
2. **Check capabilities**: Use `has_capability()` for features
3. **Handle missing**: Code should work without extensions
4. **Cleanup**: Use context managers or call `cleanup_all()`

## Testing

```python
# Mock extension for testing
class MockStorageExtension(StorageExtension):
    def initialize(self, config):
        self._data = {}
        return True
    
    def is_available(self):
        return True
    
    # ... implement interface

# Use in tests
registry = ExtensionRegistry()
registry.register('storage', MockStorageExtension())
```

## Migration Guide

### From Old Style

```python
# Old
os = ManifoldOS(lut_db_path='metadata.duckdb')
os.lut_store.commit_lut(...)

# New (same result, using extension system)
os = ManifoldOS(extensions={
    'storage': {'type': 'duckdb', 'db_path': 'metadata.duckdb'}
})
storage = os.extensions.get('storage')
storage.store_lut(...)

# Or use backward compat property
os.lut_store.commit_lut(...)  # Still works!
```

## Knowledge Base Integration

All extensions are designed for eventual integration into the ManifoldOS knowledge base:

### Content-Addressability

Every extension configuration has a stable hash:

```python
ext = DuckDBStorageExtension()
ext.initialize({'db_path': 'test.db'})

# Stable, content-addressed identifier
config_hash = ext.get_config_hash()
# Same config → same hash (idempotent)
```

### Immutable Configurations

Configurations are frozen after initialization:

```python
ext = MyExtension()
ext.initialize({'param': 'value'})

# This will raise ExtensionError
ext.initialize({'param': 'new_value'})  # ✗ Already initialized!

# Access frozen config (read-only)
config_dict = ext.config  # Returns dict for easy access
```

### Idempotent Operations

All extension operations are designed to be repeatable:

```python
# DuckDB storage uses UPSERT - safe to repeat
storage.store_lut(lut_data)  # First call
storage.store_lut(lut_data)  # Safe to repeat

# Same data → same database state
```

### Knowledge Base Storage Pattern

Extensions follow a pattern that enables storage in the knowledge base:

1. **Configuration frozen** → stable content-addressed ID
2. **Operations logged** → immutable event history
3. **State append-only** → no data deletion
4. **Results cached** → by content hash

Future: Extension configs and operation logs will be stored directly in the knowledge base, making the entire system traceable and reproducible.

### Adding New Extension Type

1. Create `core/extensions/my_extension.py`
2. Define `MyExtension(ManifoldExtension)`
3. Call `_freeze_config()` in `initialize()`
4. Ensure all operations are idempotent
5. Export from `core/extensions/__init__.py`
6. Add to ManifoldOS `__init__` if needed
7. Document in this file

## Example: Full Setup

```python
from core.manifold_os import ManifoldOS

# Production setup with multiple extensions
os = ManifoldOS(
    storage_path="./data",
    extensions={
        'storage': {
            'type': 'duckdb',
            'db_path': './data/metadata.duckdb',
            'threads': 4
        },
        # Future:
        # 'cache': {
        #     'type': 'redis',
        #     'host': 'localhost',
        #     'port': 6379
        # },
        # 'monitoring': {
        #     'type': 'prometheus',
        #     'port': 9090
        # }
    }
)

# Check what's available
print("Available capabilities:", os.extensions.list_capabilities())

# Use extensions
if os.extensions.has_capability('persistent_storage'):
    rep = os.ingest("enterprise data", metadata={'source': 'CRM'})
    print("✓ Data persisted to DuckDB")
```

## Troubleshooting

**Extension fails to initialize:**

- Check dependencies are installed
- Verify configuration is valid
- Look for error messages in output

**Extension not available:**

- Verify `is_available()` returns True
- Check extension was registered successfully
- Ensure dependencies didn't fail to import

**Backward compatibility issues:**

- `lut_db_path` parameter still works
- `os.lut_store` property still available
- Old code should run unchanged
