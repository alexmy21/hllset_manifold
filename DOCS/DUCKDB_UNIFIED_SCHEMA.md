# DuckDB Unified Lattice Schema Design v2

## Overview

Single DuckDB database with **unified lattice schema**. All structures (AM, W, metadata graphs) are represented as lattice instances with typed nodes and edges.

## Design Philosophy

### The Lattice Abstraction

**Key Insight**: AM, W, and metadata graphs are all lattices with different node/edge semantics:

| Structure | Node Type | Edge Meaning | Node Properties |
|-----------|-----------|--------------|-----------------|
| **AM** | Token identifier | Transition frequency | (reg, zeros), token |
| **W** | Basic HLLSet | BSS morphism | HLLSet hash, cardinality |
| **Metadata** | Table/Column/FK | Schema relationship | Name, schema info |

**Solution**: Unified tables with `node_type`/`edge_type` + JSON for specifics.

### Design Principles

1. **Single database**: Easier management, atomic transactions
2. **Perceptron scoping**: Each perceptron has multiple lattices
3. **Unified lattice model**: Same tables for AM, W, metadata
4. **Roaring bitmap compression**: HLLSets compressed 10-50x
5. **Sparse storage**: Only non-zero edges stored
6. **Content addressable**: All artifacts identified by hash
7. **IICA compliant**: Immutable records, idempotent operations
8. **JSON for specifics**: Type-specific properties in JSON
9. **Processor knowledge**: Each processor knows its node/edge types

## Core Schema

### 1. Perceptrons

```sql
CREATE TABLE perceptrons (
    perceptron_id VARCHAR PRIMARY KEY,
    perceptron_type VARCHAR NOT NULL,  -- 'data', 'metadata', 'image', etc.
    hash_function VARCHAR NOT NULL,    -- 'sha1', 'xxhash', etc.
    hash_seed BIGINT,                  -- Seed value for hash function
    config_json VARCHAR,               -- HRTConfig as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR
);
```

### 2. Lattices

Each perceptron can have multiple lattices (AM, W, custom).

```sql
CREATE TABLE lattices (
    lattice_id VARCHAR PRIMARY KEY,    -- Content hash
    perceptron_id VARCHAR NOT NULL,
    lattice_type VARCHAR NOT NULL,     -- 'AM', 'W', 'metadata', 'custom'
    dimension INTEGER,                 -- Size of lattice
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_json VARCHAR,               -- Type-specific config
    
    FOREIGN KEY (perceptron_id) REFERENCES perceptrons(perceptron_id)
);

CREATE INDEX idx_lattice_perceptron ON lattices(perceptron_id);
CREATE INDEX idx_lattice_type ON lattices(perceptron_id, lattice_type);
```

### 3. Lattice Nodes (Unified)

**All node types** in one table with standardized fields + JSON properties.

```sql
CREATE TABLE lattice_nodes (
    node_id VARCHAR PRIMARY KEY,       -- Content hash
    lattice_id VARCHAR NOT NULL,
    node_index INTEGER NOT NULL,       -- Position in lattice (0-based)
    node_type VARCHAR NOT NULL,        -- 'am_token', 'w_hllset', 'meta_table', etc.
    content_hash VARCHAR NOT NULL,     -- Hash of node content
    
    -- STANDARD REQUIRED FIELDS
    cardinality DOUBLE NOT NULL,       -- Estimated size/count
    
    -- TYPE-SPECIFIC PROPERTIES (JSON)
    properties JSON,                   -- Processor-specific data
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (lattice_id) REFERENCES lattices(lattice_id),
    UNIQUE (lattice_id, node_index)
);

CREATE INDEX idx_node_lattice ON lattice_nodes(lattice_id);
CREATE INDEX idx_node_type ON lattice_nodes(lattice_id, node_type);
CREATE INDEX idx_node_hash ON lattice_nodes(content_hash);
CREATE INDEX idx_node_index ON lattice_nodes(lattice_id, node_index);
```

**Node Examples**:

```json
// AM row token node
{
  "node_id": "sha1_abc123...",
  "lattice_id": "am_lattice_001",
  "node_index": 42,
  "node_type": "am_token",
  "content_hash": "token_hash_xyz",
  "cardinality": 1.0,
  "properties": {
    "token": "hello",
    "register": 42,
    "zeros": 3,
    "position": "row"
  }
}

// W basic HLLSet node
{
  "node_id": "sha1_def456...",
  "lattice_id": "w_lattice_001",
  "node_index": 10,
  "node_type": "w_hllset",
  "content_hash": "hllset_hash_uvw",
  "cardinality": 15.7,
  "properties": {
    "hllset_hash": "hllset_abc123",
    "p_bits": 14,
    "position": "row"
  }
}

// Metadata table node
{
  "node_id": "sha1_ghi789...",
  "lattice_id": "meta_lattice_001",
  "node_index": 0,
  "node_type": "meta_table",
  "content_hash": "table_hash_rst",
  "cardinality": 1000000.0,
  "properties": {
    "table_name": "customers",
    "schema": "public",
    "columns": ["id", "name", "email"]
  }
}

// Metadata column node
{
  "node_id": "sha1_jkl012...",
  "lattice_id": "meta_lattice_001",
  "node_index": 1,
  "node_type": "meta_column",
  "content_hash": "column_hash_mno",
  "cardinality": 1000000.0,
  "properties": {
    "column_name": "customer_id",
    "data_type": "INTEGER",
    "nullable": false,
    "parent_table": "customers"
  }
}
```

### 4. Lattice Edges (Unified)

**All edge types** in one table with standardized fields + JSON properties.

```sql
CREATE TABLE lattice_edges (
    edge_id VARCHAR PRIMARY KEY,       -- Content hash
    lattice_id VARCHAR NOT NULL,
    source_node VARCHAR NOT NULL,      -- node_id
    target_node VARCHAR NOT NULL,      -- node_id
    edge_type VARCHAR NOT NULL,        -- 'am_transition', 'w_morphism', 'meta_fk', etc.
    
    -- STANDARD REQUIRED FIELD
    weight DOUBLE NOT NULL,            -- Frequency, similarity, strength, etc.
    
    -- TYPE-SPECIFIC PROPERTIES (JSON)
    properties JSON,                   -- Edge-specific data
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (lattice_id) REFERENCES lattices(lattice_id),
    FOREIGN KEY (source_node) REFERENCES lattice_nodes(node_id),
    FOREIGN KEY (target_node) REFERENCES lattice_nodes(node_id)
);

CREATE INDEX idx_edge_lattice ON lattice_edges(lattice_id);
CREATE INDEX idx_edge_source ON lattice_edges(source_node);
CREATE INDEX idx_edge_target ON lattice_edges(target_node);
CREATE INDEX idx_edge_type ON lattice_edges(lattice_id, edge_type);
CREATE INDEX idx_edge_weight ON lattice_edges(lattice_id, weight DESC);
```

**Edge Examples**:

```json
// AM transition edge
{
  "edge_id": "sha1_edge_001",
  "lattice_id": "am_lattice_001",
  "source_node": "node_abc",
  "target_node": "node_def",
  "edge_type": "am_transition",
  "weight": 15.0,
  "properties": {
    "frequency": 15,
    "window_size": 3,
    "stride": 1
  }
}

// W morphism edge
{
  "edge_id": "sha1_edge_002",
  "lattice_id": "w_lattice_001",
  "source_node": "node_ghi",
  "target_node": "node_jkl",
  "edge_type": "w_morphism",
  "weight": 0.85,
  "properties": {
    "bss_tau": 0.75,
    "bss_rho": 0.20,
    "is_inclusion": true,
    "morphism_type": "row_to_col"
  }
}

// Metadata foreign key edge
{
  "edge_id": "sha1_edge_003",
  "lattice_id": "meta_lattice_001",
  "source_node": "node_mno",
  "target_node": "node_pqr",
  "edge_type": "meta_foreign_key",
  "weight": 1.0,
  "properties": {
    "constraint_name": "fk_orders_customer",
    "on_delete": "CASCADE",
    "on_update": "RESTRICT",
    "cardinality_ratio": 15.3
  }
}
```

### 5. HLLSet Storage (Roaring Bitmap Compressed)

```sql
CREATE TABLE hllsets (
    hllset_hash VARCHAR PRIMARY KEY,   -- Content hash of HLLSet
    p_bits INTEGER NOT NULL,           -- HLL precision
    cardinality DOUBLE NOT NULL,       -- Estimated cardinality
    
    -- Roaring bitmap compressed registers
    -- Encodes position*256 + value for non-zero registers
    registers_roaring BLOB NOT NULL,   -- Roaring bitmap serialization
    
    -- Compression metadata
    original_size INTEGER,             -- Original bytes (2^p_bits)
    compressed_size INTEGER,           -- Compressed bytes
    compression_ratio DOUBLE,          -- original / compressed
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hllset_pbits ON hllsets(p_bits);
CREATE INDEX idx_hllset_card ON hllsets(cardinality);
```

**Roaring Bitmap Encoding**:

```python
# Encoding: position * 256 + value
# Example: register[42] = 7 → encoded as 42 * 256 + 7 = 10759
# Roaring bitmap stores set of encoded integers
# Typical compression: 10-50x for sparse HLLSets
```

**Storage estimates**:
- p_bits=14: 16,384 bytes uncompressed → ~500-2000 bytes compressed (10-30x)
- p_bits=10: 1,024 bytes uncompressed → ~50-200 bytes compressed (5-20x)
- Very sparse: up to 50x compression

### 6. Cross-Perceptron Entanglements

Links between different perceptrons' lattices.

```sql
CREATE TABLE entanglements (
    entanglement_id VARCHAR PRIMARY KEY,  -- Content hash
    source_lattice VARCHAR NOT NULL,      -- lattice_id
    target_lattice VARCHAR NOT NULL,      -- lattice_id
    entanglement_type VARCHAR NOT NULL,   -- 'data_to_metadata', 'cross_modal', etc.
    total_pairs INTEGER,
    avg_strength DOUBLE,
    properties JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (source_lattice) REFERENCES lattices(lattice_id),
    FOREIGN KEY (target_lattice) REFERENCES lattices(lattice_id)
);

CREATE INDEX idx_ent_source ON entanglements(source_lattice);
CREATE INDEX idx_ent_target ON entanglements(target_lattice);
```

### 7. Entanglement Mappings

Detailed node-to-node mappings within an entanglement.

```sql
CREATE TABLE entanglement_mappings (
    entanglement_id VARCHAR NOT NULL,
    source_node VARCHAR NOT NULL,     -- node_id from source lattice
    target_node VARCHAR NOT NULL,     -- node_id from target lattice
    similarity DOUBLE NOT NULL,       -- Entanglement strength
    properties JSON,
    
    PRIMARY KEY (entanglement_id, source_node, target_node),
    FOREIGN KEY (entanglement_id) REFERENCES entanglements(entanglement_id),
    FOREIGN KEY (source_node) REFERENCES lattice_nodes(node_id),
    FOREIGN KEY (target_node) REFERENCES lattice_nodes(node_id)
);

CREATE INDEX idx_emap_source ON entanglement_mappings(entanglement_id, source_node);
CREATE INDEX idx_emap_similarity ON entanglement_mappings(entanglement_id, similarity DESC);
```

## Query Patterns

### 1. Get all lattices for a perceptron

```sql
SELECT * FROM lattices
WHERE perceptron_id = 'data_perceptron'
ORDER BY lattice_type;
```

### 2. Get AM transition graph

```sql
-- All AM transitions with frequency > 10
SELECT 
    src.node_index as from_idx,
    src.properties->>'token' as from_token,
    tgt.node_index as to_idx,
    tgt.properties->>'token' as to_token,
    e.weight as frequency
FROM lattice_edges e
JOIN lattice_nodes src ON e.source_node = src.node_id
JOIN lattice_nodes tgt ON e.target_node = tgt.node_id
WHERE e.lattice_id = 'am_lattice_001'
AND e.edge_type = 'am_transition'
AND e.weight > 10
ORDER BY e.weight DESC;
```

### 3. Get W morphisms above threshold

```sql
SELECT 
    src.node_index as from_basic,
    tgt.node_index as to_basic,
    e.weight as bss_similarity,
    e.properties->>'bss_tau' as inclusion,
    e.properties->>'bss_rho' as exclusion
FROM lattice_edges e
JOIN lattice_nodes src ON e.source_node = src.node_id
JOIN lattice_nodes tgt ON e.target_node = tgt.node_id
WHERE e.lattice_id = 'w_lattice_001'
AND e.edge_type = 'w_morphism'
AND e.weight >= 0.7;
```

### 4. Get metadata graph structure

```sql
-- Tables and their columns
SELECT 
    t.properties->>'table_name' as table_name,
    c.properties->>'column_name' as column_name,
    c.properties->>'data_type' as data_type
FROM lattice_nodes t
JOIN lattice_edges e ON t.node_id = e.source_node
JOIN lattice_nodes c ON e.target_node = c.node_id
WHERE t.lattice_id = 'meta_lattice_001'
AND t.node_type = 'meta_table'
AND c.node_type = 'meta_column'
AND e.edge_type = 'meta_has_column';
```

### 5. Cross-perceptron entanglement queries

```sql
-- Find data nodes that map to metadata structures
SELECT 
    data_node.node_index as data_idx,
    meta_node.properties->>'table_name' as maps_to_table,
    em.similarity
FROM entanglement_mappings em
JOIN entanglements ent ON em.entanglement_id = ent.entanglement_id
JOIN lattice_nodes data_node ON em.source_node = data_node.node_id
JOIN lattice_nodes meta_node ON em.target_node = meta_node.node_id
WHERE ent.entanglement_type = 'data_to_metadata'
AND em.similarity > 0.8
ORDER BY em.similarity DESC;
```

## Processor Pattern

Each processor knows its node/edge types and how to interpret properties:

```python
class AMProcessor:
    node_type = "am_token"
    edge_type = "am_transition"
    
    def create_node(self, token: str, reg: int, zeros: int) -> dict:
        return {
            "node_type": self.node_type,
            "cardinality": 1.0,
            "properties": {
                "token": token,
                "register": reg,
                "zeros": zeros
            }
        }
    
    def create_edge(self, freq: int, window: int) -> dict:
        return {
            "edge_type": self.edge_type,
            "weight": float(freq),
            "properties": {
                "frequency": freq,
                "window_size": window
            }
        }

class WProcessor:
    node_type = "w_hllset"
    edge_type = "w_morphism"
    
    def create_node(self, hllset: HLLSet) -> dict:
        return {
            "node_type": self.node_type,
            "cardinality": hllset.cardinality(),
            "properties": {
                "hllset_hash": hllset.name,
                "p_bits": hllset.p_bits
            }
        }
    
    def create_edge(self, bss_tau: float, bss_rho: float) -> dict:
        return {
            "edge_type": self.edge_type,
            "weight": bss_tau,  # Use tau as primary weight
            "properties": {
                "bss_tau": bss_tau,
                "bss_rho": bss_rho,
                "is_inclusion": bss_tau >= 0.7
            }
        }

class MetadataProcessor:
    def create_table_node(self, table_name: str, row_count: int) -> dict:
        return {
            "node_type": "meta_table",
            "cardinality": float(row_count),
            "properties": {
                "table_name": table_name,
                "row_count": row_count
            }
        }
    
    def create_fk_edge(self, constraint_name: str) -> dict:
        return {
            "edge_type": "meta_foreign_key",
            "weight": 1.0,
            "properties": {
                "constraint_name": constraint_name
            }
        }
```

## Benefits of Unified Schema

1. **Simplicity**: Fewer tables, consistent patterns
2. **Extensibility**: New lattice types = new node/edge types
3. **Query flexibility**: JOIN across different lattice types
4. **Storage efficiency**: Roaring compression + sparse edges
5. **IICA compliant**: Content-addressed, immutable
6. **Processor autonomy**: Each knows its types/properties
7. **No special maintenance**: Generic lattice operations

## Storage Estimates

100-table metadata + data perceptron:

- **Perceptrons**: 2 × 200 bytes = 400 bytes
- **Lattices**: 4 × 200 bytes = 800 bytes
- **Nodes**: 50K × 150 bytes = 7.5 MB
- **Edges**: 100K × 100 bytes = 10 MB
- **HLLSets** (compressed): 1K × 1KB = 1 MB
- **Entanglements**: 10K × 100 bytes = 1 MB

**Total**: ~20 MB (vs ~50MB uncompressed)

## Migration Notes

For existing DuckDB storage:

1. Create new unified tables
2. Migrate `am_cells` → `lattice_edges` with `edge_type='am_transition'`
3. Migrate `lattice_nodes` → new unified `lattice_nodes` with `node_type='w_hllset'`
4. Compress `hllset_registers` → `hllsets` with Roaring bitmaps
5. Drop old tables after validation
