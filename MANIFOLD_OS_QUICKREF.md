# ManifoldOS Quick Reference

## Import

```python
from core.manifold_os import (
    ManifoldOS, IngestDriver, TokenizationConfig,
    Driver, DriverState
)
```

## Basic Usage

### Create OS

```python
os = ManifoldOS()  # Auto-creates default ingest driver
```

### Ingest Data

```python
# Single document
hllset = os.ingest("your text here")

# Batch
hllsets = os.ingest_batch(["doc1", "doc2", "doc3"])
```

### Check Drivers

```python
# List all drivers
drivers = os.list_drivers()
for driver_id, info in drivers.items():
    print(f"{driver_id}: {info['state']}")

# Get specific driver
driver = os.get_driver("ingest_default")
print(f"Operations: {driver.stats.operations_count}")
```

## Custom Driver

### Create with Config

```python
config = TokenizationConfig(
    min_token_length=3,
    lowercase=True,
    remove_punctuation=True
)
driver = IngestDriver("my_driver", config)
os.register_driver(driver)
driver.wake()
```

### Use Custom Driver

```python
hllset = os.ingest("text", driver_id="my_driver")
```

## Driver Management

### Lifecycle

```python
os.wake_driver("driver_id")      # Wake to idle
os.idle_driver("driver_id")       # Put to idle
os.restart_driver("driver_id")    # Restart errored
os.unregister_driver("driver_id") # Remove
```

### Health Monitoring

```python
os.start_driver_monitoring()  # Start background monitoring
# ... processing ...
os.stop_driver_monitoring()   # Stop monitoring
```

### Cleanup

```python
removed = os.cleanup_dead_drivers()
print(f"Removed {len(removed)} dead drivers")
```

## ICASRA Pattern

```python
# D - Interface: Ingest
hllset = os.ingest("external data")

# B - Copier: Reproduce
copy = os.kernel.reproduce(hllset)

# C - Controller: Manage
drivers = os.list_drivers()

# A - Constructor: Commit
from core.manifold_os import OSState
state = OSState(
    state_hash="",
    root_hllset_hash=hllset.name,
    hrt_hash="",
    perceptron_states={},
    pipeline_config={}
)
state_hash = os.store.commit(state)
```

## Kernel Operations

```python
# Union
union = os.kernel.union(hllset1, hllset2)

# Intersection
intersection = os.kernel.intersection(hllset1, hllset2)

# Difference
difference = os.kernel.difference(hllset1, hllset2)

# Similarity
sim = hllset1.similarity(hllset2)

# Cardinality
card = hllset.cardinality()
```

## Driver States

```text
CREATED → IDLE ⇄ ACTIVE
           ↓        ↓
        ERROR → DEAD
```

- **CREATED**: Just instantiated
- **IDLE**: Ready to work
- **ACTIVE**: Processing
- **ERROR**: Failed, needs restart
- **DEAD**: Terminated

## Statistics

```python
driver = os.get_driver("ingest_default")
stats = driver.stats

print(f"Operations: {stats.operations_count}")
print(f"Errors: {stats.errors_count}")
print(f"Uptime: {time.time() - stats.created_at:.2f}s")
print(f"Active time: {stats.total_active_time:.4f}s")
```

## TokenizationConfig

```python
@dataclass
class TokenizationConfig:
    min_token_length: int = 1       # Minimum length
    max_token_length: int = 100     # Maximum length
    lowercase: bool = True           # Convert to lowercase
    remove_punctuation: bool = False # Strip punctuation
    split_on: str = " "              # Token delimiter
```

## Common Patterns

### Process Multiple Documents

```python
docs = ["doc1", "doc2", "doc3"]
hllsets = os.ingest_batch(docs)

# Union all
result = hllsets[0]
for h in hllsets[1:]:
    result = os.kernel.union(result, h)

print(f"Total cardinality: {result.cardinality()}")
```

### Custom Processing Pipeline

```python
# Register multiple drivers
for i in range(3):
    driver = IngestDriver(f"worker_{i}")
    os.register_driver(driver)
    driver.wake()

# Process with different drivers
results = []
for i, doc in enumerate(docs):
    hllset = os.ingest(doc, driver_id=f"worker_{i%3}")
    results.append(hllset)
```

### Error Recovery

```python
# Check for errors
drivers = os.list_drivers()
for driver_id, info in drivers.items():
    if info['errors'] > 0:
        print(f"⚠ {driver_id} has errors")
        os.restart_driver(driver_id)
```

### Content Addressability

```python
# Same input = same hash (idempotent)
h1 = os.ingest("hello world")
h2 = os.ingest("hello world")
assert h1.name == h2.name  # Content-addressed
```

## Running Examples

```bash
# Tests (11 tests, all passing)
python tests/test_manifold_drivers.py

# Demos (7 demos)
python examples/demo_manifold_drivers.py

# Quick test
python -c "from core.manifold_os import ManifoldOS; os = ManifoldOS(); print(os.ingest('test').cardinality())"
```

## Key Principles

1. **Immutability** - Data never changes
2. **Idempotence** - Same input → same output
3. **Content Addressability** - Data identified by hash
4. **No Scheduling** - Pure functional design
5. **No Harm** - Processes can't corrupt each other

## Documentation

- [MANIFOLD_OS_DRIVERS.md](MANIFOLD_OS_DRIVERS.md) - Complete guide
- [MANIFOLD_OS_SUCCESS.md](MANIFOLD_OS_SUCCESS.md) - Implementation summary
- [core/manifold_os.py](core/manifold_os.py) - Source code
- [examples/demo_manifold_drivers.py](examples/demo_manifold_drivers.py) - Examples
- [tests/test_manifold_drivers.py](tests/test_manifold_drivers.py) - Tests
