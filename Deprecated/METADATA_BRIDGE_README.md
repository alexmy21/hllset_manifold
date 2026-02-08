# Enterprise-to-AI Metadata Bridge

## Overview

The HLLSet Manifold has been transformed into a **persistent metadata bridge** that connects Enterprise Data (ED) to AI operations through content-addressed fingerprints.

**Key Insight**: *"Metadata is the missing chain between ED and AI"*

## Architecture

### Two-Way Bridge

```text
┌─────────────┐                  ┌──────────────┐                  ┌─────────┐
│ Enterprise  │                  │   Metadata   │                  │   AI    │
│    Data     │ ────────────────>│   Bridge     │─────────────────>│ Systems │
│  (Source)   │  Ingest/Store    │ (HLLSet+LUT) │  Operations      │         │
└─────────────┘                  └──────────────┘                  └─────────┘
       ▲                                  │                             │
       │                                  │                             │
       └──────────────────────────────────┴─────────────────────────────┘
                         Query/Ground (Explainability)
```

### Components

1. **HLLSet Fingerprints**
   - Fixed-size (1.5KB) AI-native representations
   - Structure-preserving (cardinality, intersections)
   - Content-addressed (immutable)

2. **Lookup Tables (LUT)**
   - Maps (reg, zeros) → [token tuples, hashes]
   - Enables grounding: AI coordinates → source tokens
   - Stored persistently with DuckDB

3. **Persistent Storage**
   - **DuckDB**: Embedded, ACID-compliant, SQL interface
   - Schema: `lut_records` + `lut_metadata`
   - Indexes for fast (n, reg, zeros, hash) lookups
   - Future: Redis (distributed), PostgreSQL (enterprise)

## Implementation

### Core Files

1. **`core/lut_store.py`** (~350 lines)
   - `LUTRecord`: Data structure with serialization
   - `LUTPersistentStore`: Abstract base class
   - `DuckDBLUTStore`: Implementation with SQL schema

2. **`core/manifold_os.py`** (enhanced)
   - Auto-initialization of LUT store
   - `ingest()`: Automatic LUT persistence with metadata
   - Query API: bidirectional ED↔AI lookup

### Database Schema

```sql
-- LUT Records Table
CREATE TABLE lut_records (
    id INTEGER PRIMARY KEY,
    n INTEGER NOT NULL,              -- N-gram size
    reg INTEGER NOT NULL,            -- HLL register
    zeros INTEGER NOT NULL,          -- Leading zeros
    hllset_hash TEXT NOT NULL,       -- Content address
    hashes TEXT NOT NULL,            -- JSON array
    tokens TEXT NOT NULL,            -- JSON array  
    commit_id TEXT NOT NULL,         -- Batch identifier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lut_lookup ON lut_records(n, reg, zeros, hllset_hash);
CREATE INDEX idx_hllset ON lut_records(hllset_hash);

-- Metadata Table
CREATE TABLE lut_metadata (
    hllset_hash TEXT PRIMARY KEY,
    n INTEGER NOT NULL,
    metadata TEXT NOT NULL,          -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Usage Examples

### Basic Ingestion with Metadata

```python
from core.manifold_os import ManifoldOS

# Initialize with persistent storage
os = ManifoldOS(lut_db_path="enterprise_metadata.duckdb")

# Ingest with source tracking
metadata = {
    'source': 'CRM_DB',
    'table': 'customers', 
    'record_id': 12345,
    'timestamp': '2024-01-15T10:30:00Z'
}

rep = os.ingest(
    "premium customer revenue growth engagement",
    metadata=metadata
)

# HLLSets stored, LUTs committed automatically
# ✓ LUT committed: n=1, hash=fbc1b7bb..., id=abc123
```

### AI → ED Grounding

```python
# AI has coordinates from HLLSet operation
# Ground decision back to source tokens

tokens = os.query_tokens_from_metadata(
    n=1,
    reg=42,
    zeros=3,
    hllset_hash="fbc1b7bb9584b882..."
)
# Returns: [('premium',), ('customer',), ...]

# Trace to source
metadata = os.get_ingestion_metadata("fbc1b7bb9584b882...")
print(metadata)
# {'source': 'CRM_DB', 'table': 'customers', 'record_id': 12345, ...}
```

### ED → AI Lookup

```python
# Enterprise wants to find where token appears in AI space

keys = os.query_by_token(
    n=1,
    token_tuple=('premium',)
)
# Returns: [(42, 3), (15, 1), ...]  # (reg, zeros) pairs

# These coordinates can be used for AI operations
# (selection, intersection, navigation, etc.)
```

### Storage Statistics

```python
stats = os.get_lut_stats()
print(stats)
# {
#     'total_lut_records': 15432,
#     'unique_hllsets': 287,
#     'n_groups': {1: 12000, 2: 2800, 3: 632},
#     'oldest_record': '2024-01-10T08:15:22Z',
#     'newest_record': '2024-01-15T14:30:45Z'
# }
```

## Benefits

### 1. Explainability

- Trace AI decisions back to source enterprise data
- Full audit trail: metadata → tokens → source records
- Critical for regulated industries (finance, healthcare)

### 2. Grounding

- Bridge abstract AI representations to concrete data
- Verify AI operations reference correct source material
- Enable human validation of AI reasoning

### 3. Persistence

- Metadata survives application restarts
- Query historical ingestions
- Reconstruct data lineage

### 4. Compliance

- Track data provenance
- Maintain GDPR/audit requirements
- Support right-to-explanation

### 5. Integration

- Two-way bridge enables hybrid ED/AI workflows
- Enterprise systems query AI space
- AI systems ground to enterprise reality

## Performance

### Space Complexity

- **HLLSet**: 1.5KB fixed (independent of source size)
- **LUT Record**: ~100 bytes per (reg, zeros) entry
- **Typical**: 16,384 HLLSet registers → ~1.6MB LUT storage

### Time Complexity

- **Ingest**: O(n) tokens + O(1) DuckDB commit
- **Query by coordinates**: O(1) indexed lookup
- **Query by token**: O(log n) B-tree search
- **Batch operations**: Transaction support for efficiency

### Storage Backend

**DuckDB** (Current):

- ✅ Embedded (no server)
- ✅ ACID transactions
- ✅ Fast analytics (columnar)
- ✅ SQL interface
- ✅ Single-file deployment

**Future Backends**:

- **Redis**: Distributed, cache-first, real-time
- **PostgreSQL**: Enterprise-scale, replication, backup

## Testing

### Run Demo

```bash
# Install DuckDB
pip install duckdb

# Run comprehensive demo
python examples/demo_metadata_bridge.py
```

### Demo Coverage

1. **Basic Bridge**: ED → Metadata → AI flow
2. **AI Grounding**: Query tokens from coordinates  
3. **Reverse Lookup**: Find coordinates for tokens
4. **Metadata Tracking**: Full audit trail
5. **Persistence**: Cross-session storage

### Expected Output

```text
======================================================================
DEMO 1: Enterprise Data → Metadata → AI
======================================================================
✓ LUT Store initialized: metadata.duckdb
[Step 1] Ingest enterprise data:
  Data: 'premium customer revenue growth engagement'
  Source: {'source': 'CRM_DB', 'table': 'customers', 'record_id': 12345}
✓ LUT committed: n=1, hash=fbc1b7bb..., id=abc123

[Step 2] Metadata stored:
  Original tokens: ['premium', 'customer', 'revenue', 'growth', 'engagement']
  HLLSets created: [1, 2, 3]
  LUTs committed: [1, 2, 3]

[Step 3] Persistent store stats:
  Total LUT records: 5
  Unique HLLSets: 3
  N-groups: {1: 5, 2: 0, 3: 0}
```

## API Reference

### ManifoldOS Initialization

```python
os = ManifoldOS(lut_db_path="path/to/metadata.duckdb")
```

**Parameters**:

- `lut_db_path` (str): Path to DuckDB file (default: `"metadata.duckdb"`)
  - Use `":memory:"` for in-memory (testing)
  - Relative or absolute paths supported

### Ingestion with Metadata

```python
representation = os.ingest(data, metadata=None)
```

**Parameters**:

- `data` (str): Raw text to ingest
- `metadata` (dict, optional): Source tracking information
  - Arbitrary JSON-serializable structure
  - Common fields: `source`, `table`, `record_id`, `timestamp`

**Returns**:

- `NTokenRepresentation`: Contains `hllsets`, `luts`, `original_tokens`

### Query Methods

#### 1. Query Tokens from Metadata (AI → ED)

```python
tokens = os.query_tokens_from_metadata(n, reg, zeros, hllset_hash=None)
```

**Use Case**: AI has coordinates, needs source tokens

**Parameters**:

- `n` (int): N-gram size
- `reg` (int): HLL register
- `zeros` (int): Leading zeros
- `hllset_hash` (str, optional): Specific HLLSet

**Returns**: List of token tuples

#### 2. Query by Token (ED → AI)

```python
keys = os.query_by_token(n, token_tuple)
```

**Use Case**: Enterprise has token, needs AI coordinates

**Parameters**:

- `n` (int): N-gram size
- `token_tuple` (tuple): Token(s) to find

**Returns**: List of (reg, zeros) pairs

#### 3. Get Ingestion Metadata

```python
metadata = os.get_ingestion_metadata(hllset_hash)
```

**Use Case**: Audit trail, provenance tracking

**Parameters**:

- `hllset_hash` (str): HLLSet content address

**Returns**: Metadata dict with ingestion details

#### 4. Get Storage Statistics

```python
stats = os.get_lut_stats()
```

**Use Case**: Monitoring, capacity planning

**Returns**: Dict with:

- `total_lut_records`: Total entries
- `unique_hllsets`: Number of fingerprints
- `n_groups`: Breakdown by n-gram size
- `oldest_record`, `newest_record`: Timestamps

## Roadmap

### Phase 1: Core Implementation ✅

- [x] LUTPersistentStore abstract base
- [x] DuckDB implementation
- [x] ManifoldOS integration
- [x] Automatic persistence on ingest
- [x] Bidirectional query API

### Phase 2: Validation & Demo 🚧

- [ ] Install DuckDB in environment
- [ ] Run full demo suite
- [ ] Performance benchmarks
- [ ] Integration tests

### Phase 3: Production Hardening ⏳

- [ ] Connection pooling
- [ ] Bulk insert optimization
- [ ] Error recovery strategies
- [ ] Migration scripts
- [ ] Backup/restore utilities

### Phase 4: Alternative Backends ⏳

- [ ] RedisLUTStore (distributed)
- [ ] PostgreSQLLUTStore (enterprise)
- [ ] Performance comparison
- [ ] Backend selection guide

### Phase 5: Advanced Features ⏳

- [ ] LUT versioning
- [ ] Incremental updates
- [ ] Compression strategies
- [ ] Distributed queries
- [ ] Real-time sync

## Architecture Decisions

### Why DuckDB?

1. **Embedded**: No server to manage
2. **ACID**: Reliable transactions
3. **Fast Analytics**: Columnar storage
4. **SQL**: Familiar query interface
5. **Single File**: Easy deployment
6. **Python-First**: Native integration

### Why Content Addressing?

1. **Immutability**: HLLSets never change
2. **Deduplication**: Same content = same hash
3. **Verification**: Detect corruption
4. **Caching**: Hash-based lookups

### Why Separate LUT?

1. **Size**: HLLSet is 1.5KB, LUT can be 1.6MB
2. **Access Pattern**: HLLSet hot, LUT cold (grounding only)
3. **Persistence**: LUT needs durability, HLLSet in-memory fast
4. **Flexibility**: Different backends for different needs

## Related Documentation

- [README.md](README.md): Project overview
- [AM_ARCHITECTURE.md](AM_ARCHITECTURE.md): Adjacency Matrix architecture
- [MANIFOLD_OS_QUICKREF.md](MANIFOLD_OS_QUICKREF.md): ManifoldOS API
- [examples/demo_metadata_bridge.py](examples/demo_metadata_bridge.py): Full demo

## Installation

```bash
# Install package
pip install -e .

# Install DuckDB for persistence
pip install duckdb

# Verify installation
python -c "from core.lut_store import DuckDBLUTStore; print('✓ Ready')"
```

## Contributing

The metadata bridge is designed for extensibility:

1. **New Backend**: Subclass `LUTPersistentStore`
2. **New Queries**: Add methods to `ManifoldOS`
3. **New Metadata**: Extend schema in `_create_schema()`
4. **Optimizations**: Implement in backend-specific code

## License

Same as HLLSet Manifold project (see [LICENSE](LICENSE))

---

**Status**: Architecture complete, ready for validation with DuckDB installation.

**Next Steps**:

1. Install DuckDB: `pip install duckdb`
2. Run demo: `python examples/demo_metadata_bridge.py`
3. Validate end-to-end flow
4. Performance benchmarks
