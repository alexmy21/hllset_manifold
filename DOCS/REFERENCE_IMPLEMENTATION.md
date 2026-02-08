# DuckDB Storage: Reference Implementation

## Overview

The DuckDB storage extension serves as the **reference implementation** for all ManifoldOS extensions, demonstrating not just functional correctness but proper **architectural patterns** and **separation of concerns**.

## Core Principle

> **"Extensions provide capabilities, Applications manage state"**

Extensions are **pure service providers** - they process requests but don't manage application state or make policy decisions.

### IICA Compliance

All extensions must preserve **IICA** properties:

- **Immutable**: Operations don't modify existing data structures
- **Idempotent**: Repeating operations yields same results
- **Content Addressable**: All artifacts identified by content hash

These properties ensure extensions compose safely and predictably.

## Separation of Concerns

### Application Responsibility (ManifoldOS)

The application layer manages:

1. **State Management**
   - Maintains LUT data structures in memory
   - Tracks HLLSet fingerprints
   - Manages internal mappings

2. **Business Logic**
   - Token ingestion algorithms
   - N-token processing
   - Hash computations

3. **Lifecycle Management**
   - Decides WHEN to persist (commit timing)
   - Determines WHAT to store (data selection)
   - Handles shutdown procedures

4. **Policy Decisions**
   - Storage triggers (size thresholds, time intervals)
   - Retention policies
   - Query strategies

### Extension Responsibility (DuckDBStorage)

The extension layer provides:

1. **Processing Capabilities**
   - Execute storage operations
   - Run SQL queries
   - Handle transactions

2. **Resource Management**
   - Database connections
   - Connection pooling
   - Resource cleanup

3. **Data Persistence**
   - Write to disk/database
   - Read from storage
   - Ensure ACID properties

4. **Error Handling**
   - Graceful degradation
   - Error reporting
   - Recovery strategies

### What Extensions MUST NOT Do

❌ **Make Policy Decisions**

```python
# WRONG: Extension decides when to store
class BadStorage:
    def process(self, data):
        if len(self.buffer) > 100:  # ✗ Policy decision
            self.flush()
```

❌ **Accumulate Application State**

```python
# WRONG: Extension caches application data
class BadStorage:
    def __init__(self):
        self.pending_data = []  # ✗ Application state
```

❌ **Implement Business Logic**

```python
# WRONG: Extension validates tokens
class BadStorage:
    def store(self, tokens):
        if not self.validate_tokens(tokens):  # ✗ Business logic
            return False
```

## Reference Patterns

### Pattern 1: Pure Processing

**Extension as Pure Function**: Input → External Storage

```python
class DuckDBStorageExtension:
    def store_lut(self, lut_data):
        """
        Pure processing function:
          - Takes input (prepared by app)
          - Writes to external storage (database)
          - Returns result
          - NO state accumulated in extension
        """
        # Input → Output (via external system)
        self.lut_store.commit(lut_data)
        return True
```

**Application Manages When/What**:

```python
class ManifoldOS:
    def __init__(self):
        self.lut = {}  # APPLICATION STATE
        self.storage = DuckDBStorageExtension()
    
    def ingest(self, tokens):
        # App updates its state
        for token in tokens:
            self.lut[token] = compute_coord(token)  # APP STATE
        
        # App decides when to persist
        if self.should_commit():  # APP LOGIC
            # Extension processes request
            self.storage.store_lut(self.lut)  # EXTENSION PROCESSING
```

### Pattern 2: Application-Driven Lifecycle

**Correct**: Application controls lifecycle

```python
class ManifoldOS:
    def commit(self):
        """Application method - makes policy decision."""
        if self.has_uncommitted_data():  # APP LOGIC
            # Prepare data (APP RESPONSIBILITY)
            lut_data = self.prepare_lut_for_storage()
            
            # Request processing (EXTENSION RESPONSIBILITY)
            success = self.storage.store_lut(lut_data)
            
            if success:
                self.mark_committed()  # APP STATE UPDATE
```

**Wrong**: Extension drives lifecycle

```python
class BadStorage:
    def auto_commit(self):
        """✗ Extension should not decide when to commit."""
        if time.time() - self.last_commit > 60:  # ✗ Policy
            self.commit()  # ✗ Extension driving app logic
```

### Pattern 3: Stateless Operations

**Correct**: No state accumulation

```python
class DuckDBStorageExtension:
    def store_lut(self, lut_data):
        """
        Stateless: Same input always produces same effect.
        No data accumulated in extension.
        """
        # Write to external storage
        self.lut_store.commit(lut_data)
        # No data cached in self.*
        return True
    
    def retrieve_lut(self, key):
        """
        Stateless: Read from external storage.
        No caching in extension.
        """
        # Read from external storage
        return self.lut_store.query(key)
        # No caching of results
```

**Wrong**: State accumulation

```python
class BadStorage:
    def __init__(self):
        self.cache = {}  # ✗ State accumulation
    
    def retrieve_lut(self, key):
        """✗ Caching violates statelessness."""
        if key in self.cache:  # ✗ State-dependent
            return self.cache[key]
        
        result = self.lut_store.query(key)
        self.cache[key] = result  # ✗ State mutation
        return result
```

### Pattern 4: Configuration vs State

**Configuration** (Immutable, set once):

- Database path
- Connection parameters
- Performance tuning

**State** (Mutable, changes over time):

- LUT data
- Query results
- Operation counts

```python
class DuckDBStorageExtension:
    def initialize(self, config):
        # CONFIGURATION (frozen, immutable)
        self._freeze_config(config, extension_type='storage')
        
        # INFRASTRUCTURE (connection, not state)
        db_path = self.config['db_path']
        self.lut_store = DuckDBLUTStore(db_path)
        
        # WRONG: Don't do this
        # self.data_cache = {}  # ✗ State
        # self.operation_count = 0  # ✗ State
```

## Real-World Example

### Scenario: Token Ingestion with Persistence

**Correct Implementation**:

```python
# APPLICATION LAYER (ManifoldOS)
class ManifoldOS:
    def __init__(self, extensions=None):
        # App state
        self.lut = {}
        self.uncommitted_count = 0
        
        # Extension (processing capability)
        self.storage = self.extensions.get('storage')
    
    def ingest(self, tokens):
        """App method - manages state and logic."""
        # 1. APP LOGIC: Process tokens
        for token in tokens:
            coord = self._compute_coordinate(token)
            self.lut[coord] = token  # APP STATE
            self.uncommitted_count += 1
        
        # 2. APP POLICY: Decide when to persist
        if self.uncommitted_count >= 1000:  # APP DECISION
            self._persist_to_storage()
    
    def _persist_to_storage(self):
        """App method - orchestrates persistence."""
        # 3. APP PREPARATION: Get data ready
        lut_data = {
            'n': self.n,
            'lut': dict(self.lut),
            'hash': self.compute_hash()
        }
        
        # 4. EXTENSION PROCESSING: Execute storage
        success = self.storage.store_lut(**lut_data)
        
        # 5. APP STATE UPDATE: Mark committed
        if success:
            self.uncommitted_count = 0


# EXTENSION LAYER (DuckDBStorage)
class DuckDBStorageExtension:
    def store_lut(self, n, lut, hash, metadata=None):
        """Extension method - pure processing."""
        # 1. VALIDATE INPUT (extension responsibility)
        if not self.is_available():
            return False
        
        # 2. EXECUTE OPERATION (pure processing)
        try:
            # Write to external storage (not internal state)
            self.lut_store.commit_lut({
                'n': n,
                'lut': lut,
                'hash': hash,
                'metadata': metadata
            })
            return True
        except Exception as e:
            # Error handling (extension responsibility)
            print(f"Storage error: {e}")
            return False
```

### Anti-Pattern (Wrong)

```python
# WRONG: Extension manages application state
class BadStorageExtension:
    def __init__(self):
        self.pending_luts = []  # ✗ Application state
        self.uncommitted_count = 0  # ✗ Application state
    
    def store_lut(self, n, lut, hash):
        """✗ Extension making policy decisions."""
        # ✗ Extension accumulating state
        self.pending_luts.append((n, lut, hash))
        self.uncommitted_count += 1
        
        # ✗ Extension making policy decision
        if self.uncommitted_count >= 1000:
            self._flush_to_database()
        
        return True
    
    def _flush_to_database(self):
        """✗ Extension managing lifecycle."""
        for n, lut, hash in self.pending_luts:
            self.lut_store.commit(n, lut, hash)
        self.pending_luts = []
        self.uncommitted_count = 0
```

## Testing the Pattern

### Test 1: Statelessness

```python
def test_extension_is_stateless():
    """Extension should not accumulate state."""
    ext = DuckDBStorageExtension()
    ext.initialize({'db_path': ':memory:'})
    
    # Call twice with same data
    ext.store_lut(n=1, lut={'key': 'val'}, hash='abc123')
    ext.store_lut(n=1, lut={'key': 'val'}, hash='abc123')
    
    # Extension should have no accumulated state
    # (All data went to database)
    assert not hasattr(ext, 'cache')
    assert not hasattr(ext, 'pending_data')
    assert not hasattr(ext, 'operation_count')
```

### Test 2: Application Controls When

```python
def test_application_controls_timing():
    """Application decides when to persist."""
    mos = ManifoldOS()
    storage = mos.extensions.get('storage')
    
    # Application ingests data
    mos.ingest(['token1', 'token2'])
    
    # Extension has NOT been called yet
    # (Application hasn't decided to persist)
    
    # Application decides to commit
    mos.commit()  # NOW extension is called
    
    # Extension processed request when asked
    # (Not before, not on its own schedule)
```

### Test 3: Multiple Instances Independent

```python
def test_instances_independent():
    """Multiple extension instances don't share state."""
    ext1 = DuckDBStorageExtension()
    ext1.initialize({'db_path': 'db1.duckdb'})
    
    ext2 = DuckDBStorageExtension()
    ext2.initialize({'db_path': 'db2.duckdb'})
    
    # Store to ext1
    ext1.store_lut(n=1, lut={'key': 'val1'}, hash='hash1')
    
    # ext2 unaffected (no shared state)
    result = ext2.retrieve_lut('hash1')
    assert result is None  # Not found (different databases)
```

## Summary

The DuckDB storage extension demonstrates:

✅ **Separation of Concerns**

- Application: State management, business logic, policy
- Extension: Processing, persistence, resources

✅ **Statelessness**

- No application state in extension
- Pure processing functions
- External storage only

✅ **Idempotence**

- UPSERT semantics
- Same input → same result
- Safe to repeat

✅ **Immutability**

- Configuration frozen
- Records content-addressed
- Append-only storage

✅ **Clear Boundaries**

- Application calls extension (not vice versa)
- Extension reports results
- No hidden side effects

This pattern enables:

- **Reproducibility**: Same config + input → same result
- **Distributability**: Extensions can run anywhere
- **Testability**: Pure functions, no hidden state
- **Knowledge Base**: All operations can be logged and replayed

Use DuckDB storage as a template for all ManifoldOS extensions.
