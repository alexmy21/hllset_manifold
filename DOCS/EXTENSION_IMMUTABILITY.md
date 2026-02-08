# Extension System: Immutability & Content-Addressability

## Overview

The extension system has been refactored to align with ManifoldOS core principles:

1. **Immutability**: Configurations frozen after initialization
2. **Idempotence**: Operations can be safely repeated
3. **Content-Addressability**: Everything identified by stable hashes

This ensures extensions can be stored in the knowledge base and operations are reproducible.

## Key Changes

### 1. Immutable Configuration

**Before:**

```python
# Configuration was mutable
ext._config = config
ext._config['new_param'] = 'value'  # Could be changed anytime
```

**After:**

```python
# Configuration is frozen (ExtensionConfig dataclass)
ext._freeze_config(config, extension_type='my_extension')
# ext.config['new_param'] = 'value'  # ✗ Would raise error

# Access via read-only property
params = ext.config  # Returns dict copy
```

### 2. Content-Addressed Identifiers

Every extension configuration has a stable SHA-256 hash:

```python
ext = DuckDBStorageExtension()
ext.initialize({'db_path': 'test.db'})

# Stable identifier
hash1 = ext.get_config_hash()
# '4f3d8e9c7a1b5d6f...'

# Same config → same hash (idempotent)
ext2 = DuckDBStorageExtension()
ext2.initialize({'db_path': 'test.db'})
hash2 = ext2.get_config_hash()

assert hash1 == hash2  # ✓ Idempotent
```

### 3. Frozen ExtensionInfo

Extension metadata is now immutable:

```python
@dataclass(frozen=True)
class ExtensionInfo:
    name: str
    version: str
    description: str
    author: str = "unknown"
    config_hash: Optional[str] = None  # Content-addressed ID
    capabilities: Tuple[Tuple[str, bool], ...] = ()
```

Capabilities are stored as tuple of tuples (immutable) instead of dict.

## Implementation Details

### ManifoldExtension Base Class

```python
class ManifoldExtension(ABC):
    def __init__(self):
        self._config: Optional[ExtensionConfig] = None
        self._config_hash: Optional[str] = None
        self._initialized: bool = False
    
    def _freeze_config(self, config: Dict[str, Any], extension_type: str):
        """
        Freeze configuration (call from initialize()).
        Must be called exactly once during initialization.
        """
        if self._config is not None:
            raise ExtensionError("Already initialized - config is immutable")
        
        # Create immutable config
        params = tuple(sorted(config.items()))
        self._config = ExtensionConfig(
            extension_type=extension_type,
            parameters=params
        )
        self._config_hash = self._config.get_hash()
        self._initialized = True
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration as dictionary (read-only)."""
        if self._config is None:
            return {}
        return dict(self._config.parameters)
```

### ExtensionConfig Dataclass

```python
@dataclass(frozen=True)
class ExtensionConfig:
    """
    Immutable extension configuration.
    
    Frozen to ensure:
      - Content-addressability (stable hash)
      - Idempotence (same config = same behavior)
      - Knowledge base integration
    """
    extension_type: str
    parameters: Tuple[Tuple[str, Any], ...]  # Immutable
    
    def get_hash(self) -> str:
        """Content-addressed hash (SHA-256)."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
```

## Usage in Extensions

### DuckDB Storage Example

```python
class DuckDBStorageExtension(StorageExtension):
    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            # Check dependencies
            import duckdb
            
            # Freeze configuration (REQUIRED)
            self._freeze_config(config, extension_type='duckdb_storage')
            
            # Access frozen config via property
            db_path = self.config.get('db_path', ':memory:')
            
            # Initialize (idempotent - safe to repeat)
            self.lut_store = DuckDBLUTStore(db_path)
            self._available = True
            
            return True
        except Exception as e:
            self._available = False
            return False
    
    def get_info(self) -> ExtensionInfo:
        return ExtensionInfo(
            name="DuckDB Storage",
            version=duckdb.__version__,
            description="Embedded SQL database",
            config_hash=self.get_config_hash()  # Include hash
        )
```

## Benefits

### 1. Reproducibility

Same configuration always produces same extension state:

```python
# Run 1
ext1 = DuckDBStorageExtension()
ext1.initialize({'db_path': 'test.db'})
hash1 = ext1.get_config_hash()

# Run 2 (later, different process)
ext2 = DuckDBStorageExtension()
ext2.initialize({'db_path': 'test.db'})
hash2 = ext2.get_config_hash()

# Guaranteed to match
assert hash1 == hash2
```

### 2. Knowledge Base Integration

Extensions can be stored in the knowledge base:

```python
# Store extension config in KB
kb.store({
    'type': 'extension_config',
    'extension_type': 'duckdb_storage',
    'config_hash': ext.get_config_hash(),
    'parameters': ext.config
})

# Later: Reconstruct exact same extension
stored = kb.retrieve(config_hash)
ext = DuckDBStorageExtension()
ext.initialize(stored['parameters'])

# Same hash = same extension
assert ext.get_config_hash() == stored['config_hash']
```

### 3. Idempotent Operations

Storage operations use content-addressed records:

```python
# DuckDB uses token_hash as PRIMARY KEY
# UPSERT ensures idempotence

# Call 1: Insert
storage.store_lut({'token_hash': 12345, 'data': '...'})

# Call 2: Safe to repeat
storage.store_lut({'token_hash': 12345, 'data': '...'})

# Same result - no duplicates
```

### 4. Audit Trail

Configuration changes are prevented:

```python
ext = MyExtension()
ext.initialize({'param': 'value'})

# Try to change config
try:
    ext.initialize({'param': 'new_value'})
except ExtensionError:
    print("✓ Config is immutable - change prevented")
```

## Testing

All tests pass with immutability:

```text
============================================================
ManifoldOS Extension System Tests
============================================================

Testing ExtensionRegistry...
✓ Extension registered: storage v1.4.4
  ✓ ExtensionRegistry tests passed

Testing DuckDBStorageExtension...
  ✓ DuckDBStorageExtension tests passed

Testing backward compatibility...
  ✓ Backward compatibility tests passed

Testing new configuration style...
  ✓ New configuration tests passed

Testing ingestion with storage...
  ✓ Ingestion with storage tests passed

Testing storage operations...
  ✓ Storage operations tests passed

============================================================
Results: 6 passed, 0 failed
============================================================
```

## Migration Guide

### For Extension Developers

**Old way (mutable):**

```python
class MyExtension(ManifoldExtension):
    def initialize(self, config):
        self._config = config  # Mutable!
        return True
```

**New way (immutable):**

```python
class MyExtension(ManifoldExtension):
    def initialize(self, config):
        # Freeze config first
        self._freeze_config(config, extension_type='my_extension')
        
        # Access via property
        value = self.config.get('param')
        return True
```

### For Extension Users

No changes required! The API remains the same:

```python
# Works exactly as before
os = ManifoldOS(extensions={
    'storage': {'type': 'duckdb', 'db_path': 'test.db'}
})

# But now with immutability guarantees
storage = os.extensions.get('storage')
hash1 = storage.get_config_hash()
# Config is frozen - stable hash
```

## Future Work

1. **Knowledge Base Storage**: Store all extension configs in KB
2. **Operation Logging**: Log all extension operations immutably
3. **Time Travel**: Replay operations from knowledge base
4. **Distributed Extensions**: Share extension configs across nodes

## Summary

The extension system now fully aligns with ManifoldOS principles:

- ✅ **Immutability**: Frozen configurations
- ✅ **Idempotence**: Repeatable operations
- ✅ **Content-Addressability**: Stable hashes
- ✅ **Knowledge Base Ready**: Can be stored and reproduced
- ✅ **Backward Compatible**: Old code still works
- ✅ **Well Tested**: All tests passing

Extensions are now first-class citizens in the ManifoldOS knowledge graph.
