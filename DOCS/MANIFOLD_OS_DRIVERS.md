# ManifoldOS Driver Management

## Overview

ManifoldOS implements the **Universal Constructor** pattern from ICASRA  (Immutable Content-Addressable Self-Reproducing Automata) theory with resource-managed drivers. The design leverages immutability and content-addressability to eliminate the need for complex scheduling while providing robust driver lifecycle management.

## ICASRA Pattern Implementation

The ManifoldOS implements all four ICASRA components:

### A - Constructor (commit/validate)

- **Purpose**: Validates and persists system states
- **Implementation**: `OSState` class with `commit()` method
- **Content Addressable**: States identified by SHA1 hash
- **Immutable**: Once committed, states cannot be modified

### B - Copier (reproduce)

- **Purpose**: Reproduces states with structural preservation
- **Implementation**: `kernel.reproduce()` method
- **Structural Isomorphism**: Preserves relationships via HLLSet morphisms
- **Lossless**: Reproduction maintains all structural properties

### C - Controller (driver lifecycle)

- **Purpose**: Manages driver resources and lifecycle
- **Implementation**: Driver management system with states
- **Resource Management**: Wake, idle, restart, remove drivers
- **No Scheduling**: Immutability eliminates need for complex scheduling

### D - Interface (external data)

- **Purpose**: Gateway between external reality and system
- **Implementation**: `IngestDriver` for tokenization
- **Tokenization**: Converts raw text to meaningful units
- **Content Addressed**: Output is immutable HLLSet

## Design Principles

### 1. Immutability

All data structures are immutable:

- HLLSets cannot be modified after creation
- States are snapshots, not mutable objects
- Drivers produce new data, never modify existing

**Benefit**: Processes cannot harm each other

### 2. Idempotence

Operations can be repeated safely:

- Same input always produces same output
- Union, intersection, difference are idempotent
- Ingestion of same data produces same HLLSet

**Benefit**: No need for complex transaction management

### 3. Content Addressability

Everything identified by content hash:

- HLLSets identified by SHA1 of registers
- States identified by hash of content
- Git-like storage with content-addressed lookup

**Benefit**: Deduplication and integrity verification automatic

### 4. No Scheduling

Immutability eliminates need for scheduling:

- No race conditions (data is immutable)
- No deadlocks (no locks needed for data)
- No priority inversion (pure functional)

**Benefit**: Simple, robust resource management

## Driver Architecture

### Driver States

```text
CREATED → IDLE ⇄ ACTIVE
           ↓        ↓
        ERROR → DEAD
```

- **CREATED**: Just instantiated
- **IDLE**: Ready to work, waiting
- **ACTIVE**: Currently processing
- **ERROR**: Failed, needs restart
- **DEAD**: Terminated, needs removal

### Driver Lifecycle

```python
# Register driver
driver = IngestDriver("my_driver")
os.register_driver(driver)

# Wake to idle
driver.wake()

# Process (idle → active → idle)
result = driver.process(data, kernel)

# Idle when not needed
os.idle_driver("my_driver")

# Restart on error
os.restart_driver("my_driver")

# Remove when done
os.unregister_driver("my_driver")
```

### Resource Management

The OS manages driver resources without scheduling:

1. **Wake**: Bring driver from created/idle to idle state
2. **Activate**: Transition idle → active for processing
3. **Idle**: Return active → idle when done
4. **Restart**: Error → idle recovery
5. **Remove**: Mark dead and cleanup

Since data is immutable, no complex coordination needed!

## IngestDriver

The first and most important driver: ingests external data.

### Features

- **Tokenization**: Splits raw text into tokens
- **Configurable**: Min/max length, case, punctuation
- **Batch Processing**: Process multiple documents
- **Content Addressed**: Produces immutable HLLSets
- **Thread Safe**: Uses driver lifecycle locks

### Usage

```python
# Basic ingestion
os = ManifoldOS()
hllset = os.ingest("Hello world")

# Batch ingestion
documents = ["doc1", "doc2", "doc3"]
hllsets = os.ingest_batch(documents)

# Custom tokenization
config = TokenizationConfig(
    min_token_length=3,
    lowercase=True,
    remove_punctuation=True
)
driver = IngestDriver("custom", config)
os.register_driver(driver)
```

### Tokenization Config

```python
@dataclass
class TokenizationConfig:
    min_token_length: int = 1      # Minimum token length
    max_token_length: int = 100    # Maximum token length
    lowercase: bool = True          # Convert to lowercase
    remove_punctuation: bool = False # Remove punctuation
    split_on: str = " "             # Token delimiter
```

## Health Monitoring

The OS provides automatic driver health monitoring:

```python
# Start monitoring
os.start_driver_monitoring()

# Monitors:
# - Restarts errored drivers
# - Removes dead drivers
# - Tracks statistics

# Check driver health
drivers = os.list_drivers()
for driver_id, info in drivers.items():
    print(f"{driver_id}: {info['state']}")
    print(f"  Operations: {info['operations']}")
    print(f"  Errors: {info['errors']}")
    print(f"  Uptime: {info['uptime']:.2f}s")

# Stop monitoring
os.stop_driver_monitoring()
```

## Statistics and Metrics

Each driver tracks:

```python
@dataclass
class DriverStats:
    operations_count: int      # Successful operations
    errors_count: int          # Failed operations
    created_at: float          # Creation timestamp
    last_active: float         # Last activity
    total_active_time: float   # Total processing time
```

Access via:

```python
driver = os.get_driver("ingest_default")
print(f"Operations: {driver.stats.operations_count}")
print(f"Errors: {driver.stats.errors_count}")
print(f"Uptime: {time.time() - driver.stats.created_at:.2f}s")
```

## API Reference

### ManifoldOS Driver Methods

#### register_driver(driver)

Register a driver with the OS.

#### unregister_driver(driver_id) → bool

Unregister and remove a driver.

#### get_driver(driver_id) → Driver

Get driver by ID.

#### list_drivers() → Dict

List all drivers with status.

#### wake_driver(driver_id) → bool

Wake driver from idle state.

#### idle_driver(driver_id) → bool

Put driver to idle state.

#### restart_driver(driver_id) → bool

Restart errored driver.

#### cleanup_dead_drivers() → List[str]

Remove all dead drivers.

#### start_driver_monitoring()

Start background health monitoring.

#### stop_driver_monitoring()

Stop health monitoring.

### ManifoldOS Ingest Methods

#### ingest(raw_data, driver_id=None) → HLLSet

Ingest raw data, return content-addressed HLLSet.

#### ingest_batch(raw_data_list, driver_id=None) → List[HLLSet]

Batch ingest multiple documents.

## Examples

### Example 1: Basic Usage

```python
from core.manifold_os import ManifoldOS

# Create OS
os = ManifoldOS()

# Ingest data
text = "The quick brown fox jumps over the lazy dog"
hllset = os.ingest(text)

print(f"Cardinality: {hllset.cardinality()}")
# Output: Cardinality: 8.0 (unique words)
```

### Example 2: Batch Processing

```python
documents = [
    "Machine learning is amazing",
    "HyperLogLog is efficient",
    "Immutability is powerful"
]

hllsets = os.ingest_batch(documents)

# Union all
union = hllsets[0]
for h in hllsets[1:]:
    union = os.kernel.union(union, h)

print(f"Total unique tokens: {union.cardinality()}")
```

### Example 3: Custom Driver

```python
from core.manifold_os import IngestDriver, TokenizationConfig

# Custom tokenization
config = TokenizationConfig(
    min_token_length=5,
    lowercase=True,
    remove_punctuation=True
)

driver = IngestDriver("long_tokens", config)
os.register_driver(driver)
driver.wake()

# Use custom driver
hllset = os.ingest("Hello beautiful world!", driver_id="long_tokens")
# Only tokens >= 5 chars: ["hello", "beautiful", "world"]
```

### Example 4: Driver Monitoring

```python
# Start monitoring
os.start_driver_monitoring()

# Simulate processing
for text in documents:
    os.ingest(text)

# Check health
drivers = os.list_drivers()
for driver_id, info in drivers.items():
    if info['errors'] > 0:
        print(f"⚠ {driver_id} has {info['errors']} errors")
        os.restart_driver(driver_id)

os.stop_driver_monitoring()
```

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                       ManifoldOS                            │
│                  (Universal Constructor)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Driver Management (C)                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │ Driver1 │  │ Driver2 │  │ Driver3 │              │    │
│  │  │ (IDLE)  │  │(ACTIVE) │  │ (IDLE)  │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  │         ↕           ↕           ↕                   │    │
│  │    Wake/Idle   Wake/Idle   Wake/Idle                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         IngestDriver (D - Interface)                │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │ Tokenize: "text" → {"token1", "token2", ...} │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  │                          ↓                          │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │ kernel.absorb() → HLLSet (content-addressed) │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Kernel (B - Copier)                         │    │
│  │  • union()      • intersection()   • difference()   │    │
│  │  • reproduce()  • validate()       • morphisms      │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    PersistentStore (A - Constructor)                │    │
│  │  • commit(state) → SHA1 hash                        │    │
│  │  • checkout(hash) → state                           │    │
│  │  • Git-like content-addressed storage               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

        ↑ Immutability ↑
        No process can harm another
        Idempotent operations
        Content-addressed storage
```

## Why This Design Works

### 1. No Scheduling Needed

- **Immutability**: Data never changes → no race conditions
- **Pure Functional**: No side effects → no ordering dependencies
- **Content Addressed**: Same input → same output always

### 2. Processes Can't Harm Each Other

- **Immutable Data**: One process can't corrupt another's data
- **No Shared State**: Each driver operates independently
- **Pure Functions**: No global state to conflict over

### 3. Simple Resource Management

- **Wake/Idle**: Basic states, no complex FSM
- **Error Recovery**: Just restart (idempotent)
- **Cleanup**: Mark dead, remove later

### 4. Scalability

- **Parallel**: Drivers can run in parallel (no locks on data)
- **Batch Processing**: Natural batching support
- **Distributed**: Content-addressed enables distribution

## Comparison to Traditional OS

| Feature | Traditional OS | ManifoldOS |
|---------|---------------|------------|
| Scheduling | Complex algorithms | Not needed |
| Locking | Mutexes, semaphores | Minimal (driver state only) |
| Race Conditions | Must prevent | Impossible (immutable) |
| Deadlocks | Must prevent | Impossible (no data locks) |
| Resource Cleanup | Complex ref counting | Simple (mark dead) |
| State Management | Mutable state machines | Immutable snapshots |
| Rollback | Complex transaction logs | Git-like checkout |
| Parallelism | Careful coordination | Free (no conflicts) |

## Future Extensions

### Additional Drivers

1. **QueryDriver**: Query HLLSet collections
2. **TransformDriver**: Apply transformations
3. **NetworkDriver**: Communicate with other nodes
4. **PersistDriver**: Handle storage operations
5. **AnalyticsDriver**: Compute metrics and statistics

### Enhanced Monitoring

1. **Performance Metrics**: Latency, throughput
2. **Resource Usage**: Memory, CPU
3. **Alert System**: Notify on errors
4. **Dashboard**: Real-time visualization

### Distributed System

1. **Content Distribution**: Share HLLSets across nodes
2. **Driver Migration**: Move drivers between nodes
3. **Consensus**: Agree on state transitions
4. **Replication**: Redundancy for reliability

## References

- [ENTANGLEMENT_SINGULARITY.md](../ENTANGLEMENT_SINGULARITY.md) - ICASRA theory
- [KERNEL_ENTANGLEMENT.md](../KERNEL_ENTANGLEMENT.md) - Kernel operations
- [core/manifold_os.py](../core/manifold_os.py) - Implementation
- [examples/demo_manifold_drivers.py](../examples/demo_manifold_drivers.py) - Examples
