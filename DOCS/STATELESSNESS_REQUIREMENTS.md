# ManifoldOS Extension Statelessness Requirements

## Overview

ManifoldOS extensions must be **stateless** to ensure reproducibility, distributability, and knowledge base integration. This document defines statelessness requirements and validation procedures.

## Core Principle

> **Stateless Extension**: An extension whose behavior depends only on its immutable configuration and the current input, with no internal state accumulated across operations.

## Why Statelessness Matters

### 1. Reproducibility

- Same config + same input → same output (always)
- Operations can be replayed from knowledge base
- Time-travel debugging works correctly

### 2. Distributability

- Extensions can run on any node
- No state synchronization needed
- Horizontal scaling is trivial

### 3. Knowledge Base Integration

- Operations are pure functions of inputs
- State is stored externally (in KB or DB)
- Complete audit trail possible

### 4. Testability

- Tests are deterministic
- No test pollution
- Parallel test execution safe

## Statelessness vs Immutability

| Concept | Definition | Example |
| --------- | ------------ | --------- |
| **Immutability** | Configuration cannot change | `config` is frozen after init |
| **Statelessness** | No accumulated state | No counters, caches, or history |

Both are required! Immutability ensures config doesn't change; statelessness ensures no hidden state accumulates.

## Requirements

### MUST NOT Have

#### ❌ Mutable Class Variables

```python
# BAD: Class variable accumulates state
class BadExtension(ManifoldExtension):
    request_count = 0  # ✗ Shared across instances
    cache = {}         # ✗ Shared mutable state
```

#### ❌ Accumulated Instance State

```python
# BAD: Instance accumulates state
class BadExtension(ManifoldExtension):
    def __init__(self):
        super().__init__()
        self.call_history = []  # ✗ Accumulates state
    
    def process(self, data):
        self.call_history.append(data)  # ✗ State mutation
```

#### ❌ Hidden Side Effects

```python
# BAD: Hidden global state
_global_cache = {}

class BadExtension(ManifoldExtension):
    def process(self, data):
        global _global_cache
        _global_cache[data] = True  # ✗ Global state mutation
```

#### ❌ Non-Idempotent Operations

```python
# BAD: Different results each time
class BadExtension(ManifoldExtension):
    def __init__(self):
        super().__init__()
        self.counter = 0
    
    def get_id(self):
        self.counter += 1  # ✗ Non-idempotent
        return self.counter
```

### MUST Have

#### ✅ Immutable Configuration

```python
# GOOD: Configuration frozen after init
class GoodExtension(ManifoldExtension):
    def initialize(self, config):
        self._freeze_config(config, extension_type='good')  # ✓ Immutable
        return True
```

#### ✅ External State Storage

```python
# GOOD: State stored externally (database, knowledge base)
class GoodExtension(ManifoldExtension):
    def initialize(self, config):
        self._freeze_config(config, extension_type='good')
        # lut_store is external - not accumulated in memory
        self.lut_store = DuckDBLUTStore(self.config['db_path'])
        return True
    
    def store_data(self, data):
        # State goes to database, not instance
        self.lut_store.commit(data)  # ✓ External state
```

#### ✅ Pure Read-Only Methods

```python
# GOOD: Methods don't modify state
class GoodExtension(ManifoldExtension):
    def get_info(self):
        # Always returns same info for same config
        return ExtensionInfo(
            name="Good",
            config_hash=self.get_config_hash()  # ✓ Deterministic
        )
```

#### ✅ Idempotent Operations

```python
# GOOD: Same input → same output
class GoodExtension(ManifoldExtension):
    def process(self, data):
        # Deterministic computation
        result = self._compute(data)  # ✓ Pure function
        return result
```

## Allowed Infrastructure Variables

These instance variables are allowed (extension infrastructure):

- `_config` - Frozen configuration (immutable)
- `_config_hash` - Content-addressed hash (immutable)
- `_initialized` - Initialization flag (set once)
- `_available` - Availability flag (set once)

These are allowed because they're set once during initialization and never modified.

## External Resources

Extensions may use external stateful resources:

- **Databases** (DuckDB, PostgreSQL)
- **File systems**
- **Knowledge base**
- **Message queues**

The key is that the extension itself doesn't accumulate state - it delegates to external systems.

```python
# GOOD: External state
class StorageExtension(ManifoldExtension):
    def initialize(self, config):
        self._freeze_config(config, extension_type='storage')
        # Connection to external stateful system
        self.lut_store = DuckDBLUTStore(self.config['db_path'])
        return True
    
    def store(self, data):
        # State goes to database (external)
        self.lut_store.commit(data)  # ✓ Not stored in instance
```

## Validation

### Automated Validation

Use the `StatelessnessValidator` tool:

```python
from core.extensions.stateless_validator import validate_extension_statelessness

# Validate extension
is_stateless, violations = validate_extension_statelessness(MyExtension)

if not is_stateless:
    for v in violations:
        if v.severity == 'error':
            print(f"ERROR: {v}")
```

### Manual Checklist

- [ ] No mutable class variables
- [ ] No accumulating instance variables
- [ ] No global state mutations
- [ ] All operations are idempotent
- [ ] Configuration is frozen
- [ ] Multiple instances are independent
- [ ] Same input → same output (always)

### Test Suite

Run the statelessness test suite:

```bash
python tests/test_statelessness.py
```

Tests include:

1. Static analysis for stateful patterns
2. Idempotence testing
3. Multiple instance independence
4. Configuration isolation

## Certification Process

To certify an extension as stateless:

1. **Run static validator**

   ```python
   validator = StatelessnessValidator()
   violations = validator.validate_extension(MyExtension)
   # Must have 0 errors
   ```

2. **Run test suite**

   ```bash
   python tests/test_statelessness.py
   # All tests must pass
   ```

3. **Document in code**

   ```python
   class MyExtension(ManifoldExtension):
       """
       MyExtension - CERTIFIED STATELESS
       
       Validated with StatelessnessValidator on 2026-02-07.
       No mutable state, all operations idempotent.
       """
   ```

4. **Add to registry**

   ```python
   # In extension registration
   registry.register('my_ext', MyExtension(), config={
       'certified_stateless': True  # Validated
   })
   ```

## Common Pitfalls

### Pitfall 1: "Innocent" Counters

```python
# Looks harmless but breaks statelessness
class Extension:
    def __init__(self):
        self.call_count = 0  # ✗ Accumulates state
    
    def process(self, data):
        self.call_count += 1  # ✗ Non-idempotent
```

**Solution**: Don't count. If you need metrics, use external monitoring.

### Pitfall 2: Caching Results

```python
# Seems like optimization but breaks statelessness
class Extension:
    def __init__(self):
        self.cache = {}  # ✗ Accumulates state
    
    def compute(self, x):
        if x in self.cache:
            return self.cache[x]  # ✗ Non-deterministic (depends on history)
        result = expensive_computation(x)
        self.cache[x] = result  # ✗ State mutation
        return result
```

**Solution**: Use external cache (Redis, knowledge base) or no cache.

### Pitfall 3: Connection Pooling

```python
# Connection pool seems reasonable but...
class Extension:
    connections = []  # ✗ Class variable shared across instances
    
    def get_connection(self):
        if not self.connections:
            self.connections.append(create_connection())
        return self.connections[0]  # ✗ Shared mutable state
```

**Solution**: Create new connection per instance or use external pool.

### Pitfall 4: Lazy Initialization

```python
# Lazy init seems efficient but...
class Extension:
    def __init__(self):
        self.resource = None  # ✗ Mutable state
    
    def get_resource(self):
        if self.resource is None:
            self.resource = create_resource()  # ✗ State mutation
        return self.resource
```

**Solution**: Initialize everything in `initialize()` method.

## DuckDB Storage Example (Correct)

```python
class DuckDBStorageExtension(StorageExtension):
    """
    CERTIFIED STATELESS - Validated 2026-02-07
    
    Stateless design:
      - Configuration frozen after init
      - No accumulated instance state
      - All operations idempotent (UPSERT)
      - State stored externally (DuckDB)
    """
    
    def __init__(self):
        super().__init__()
        self.lut_store = None  # ✓ Set once, external resource
        self._available = False  # ✓ Set once
    
    def initialize(self, config):
        # Freeze config (immutable)
        self._freeze_config(config, extension_type='duckdb_storage')
        
        # Create external state store
        self.lut_store = DuckDBLUTStore(self.config['db_path'])
        self._available = True
        
        return True
    
    def get_info(self):
        # ✓ Idempotent - same result each time
        return ExtensionInfo(
            name="DuckDB Storage",
            config_hash=self.get_config_hash()
        )
```

## Future Enhancements

1. **Automated CI Checks**: Fail build if extension not stateless
2. **Runtime Monitoring**: Detect state accumulation in production
3. **Distributed Extensions**: Run extensions across multiple nodes
4. **Knowledge Base Logging**: Log all operations for replay

## References

- [EXTENSION_SYSTEM.md](EXTENSION_SYSTEM.md) - Extension architecture
- [EXTENSION_IMMUTABILITY.md](EXTENSION_IMMUTABILITY.md) - Immutability patterns
- [core/extensions/stateless_validator.py](../core/extensions/stateless_validator.py) - Validation tool
- [tests/test_statelessness.py](../tests/test_statelessness.py) - Test suite

## Summary

**Stateless extensions** are essential for ManifoldOS:

✅ No mutable state accumulated  
✅ Idempotent operations  
✅ External state storage  
✅ Deterministic behavior  
✅ Reproducible from configuration  
✅ Knowledge base ready  

Use the validation tools to ensure all extensions meet these requirements.
