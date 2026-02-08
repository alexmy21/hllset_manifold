# DuckDB Multi-Perceptron Schema Design

## Overview

Single DuckDB database with unified lattice schema supporting multiple perceptrons. All structures (AM, W, metadata graphs) are represented as lattices with typed nodes and edges.

## Design Principles

1. **Single database**: Easier management, atomic transactions
2. **Perceptron identification**: `perceptron_id` as part of primary keys
3. **Unified lattice model**: AM, W, and metadata graphs as lattice instances
4. **Sparse storage**: Only non-zero edges stored
5. **Roaring bitmap compression**: HLLSet registers compressed for storage
6. **Content addressable**: All artifacts identified by hash
7. **IICA compliant**: Immutable records, idempotent operations
8. **JSON for specifics**: Type-specific properties in JSON fields
9. **Optimized for queries**: Indexes on common query patterns

## Key Insight: Everything is a Lattice

| Structure | Node Type | Edge Meaning | Node Properties |
|-----------|-----------|--------------|-----------------|
| **AM** | Token identifier | Transition frequency | (reg, zeros), token |
| **W** | Basic HLLSet | BSS morphism | HLLSet hash, cardinality |
| **Metadata** | Table/Column/FK | Schema relationship | Name, schema info |

All three share the same underlying lattice structure!

## Core Tables

### 1. Perceptrons Registry

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

-- Example rows:
-- ('data_perceptron', 'data', 'sha1', 0x1234..., '{"p_bits": 8, ...}', ...)
-- ('metadata_perceptron', 'metadata', 'sha1', 0x5678..., '{"p_bits": 6, ...}', ...)
```

### 2. Adjacency Matrix (AM) Cells

Sparse representation - only non-zero cells stored.

```sql
CREATE TABLE am_cells (
    perceptron_id VARCHAR NOT NULL,
    row_idx INTEGER NOT NULL,
    col_idx INTEGER NOT NULL,
    frequency INTEGER NOT NULL,
    row_hash VARCHAR,              -- (reg, zeros) hash for row
    col_hash VARCHAR,              -- (reg, zeros) hash for col
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (perceptron_id, row_idx, col_idx),
    FOREIGN KEY (perceptron_id) REFERENCES perceptrons(perceptron_id)
);

CREATE INDEX idx_am_row ON am_cells(perceptron_id, row_idx);
CREATE INDEX idx_am_col ON am_cells(perceptron_id, col_idx);
CREATE INDEX idx_am_hash ON am_cells(perceptron_id, row_hash, col_hash);
```

**Storage estimate**:

- 8 bytes (perceptron_id) + 8 bytes (indices) + 4 bytes (frequency) = ~20 bytes/cell
- 1M non-zero cells = ~20MB

### 3. Token Lookup (LUT)

Maps (reg, zeros) identifiers to actual tokens.

```sql
CREATE TABLE token_lut (
    perceptron_id VARCHAR NOT NULL,
    token_hash VARCHAR NOT NULL,       -- SHA1 of (reg, zeros)
    register INTEGER NOT NULL,          -- HLL register value
    zeros INTEGER NOT NULL,             -- Leading zeros count
    token VARCHAR,                      -- Original token (optional)
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (perceptron_id, token_hash),
    FOREIGN KEY (perceptron_id) REFERENCES perceptrons(perceptron_id)
);

CREATE INDEX idx_lut_regzero ON token_lut(perceptron_id, register, zeros);
CREATE INDEX idx_lut_token ON token_lut(perceptron_id, token);
```

### 4. Lattice Nodes (W)

Basic HLLSets that form the lattice.

```sql
CREATE TABLE lattice_nodes (
    perceptron_id VARCHAR NOT NULL,
    node_type VARCHAR NOT NULL,        -- 'row' or 'col'
    node_idx INTEGER NOT NULL,         -- Index in lattice
    hllset_hash VARCHAR NOT NULL,      -- Content hash of HLLSet
    cardinality DOUBLE,                -- Estimated cardinality
    p_bits INTEGER,                    -- HLL precision
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (perceptron_id, node_type, node_idx),
    FOREIGN KEY (perceptron_id) REFERENCES perceptrons(perceptron_id)
);

CREATE INDEX idx_lattice_hash ON lattice_nodes(perceptron_id, hllset_hash);
CREATE INDEX idx_lattice_type ON lattice_nodes(perceptron_id, node_type);
```

### 5. HLLSet Registers

Actual HLL register data for reconstruction.

```sql
CREATE TABLE hllset_registers (
    hllset_hash VARCHAR NOT NULL,
    register_idx INTEGER NOT NULL,
    register_value TINYINT NOT NULL,   -- 0-255
    
    PRIMARY KEY (hllset_hash, register_idx)
);

-- Note: Not tied to perceptron_id since HLLSets are content-addressed
-- Multiple perceptrons might reference same HLLSet
```

**Storage estimate**:
- p_bits=14: 16,384 registers/HLLSet
- 1 byte/register + overhead = ~20KB/HLLSet

### 6. Entanglement Morphisms

Cross-perceptron or intra-perceptron entanglements.

```sql
CREATE TABLE entanglements (
    entanglement_id VARCHAR PRIMARY KEY,  -- Content hash
    source_perceptron VARCHAR NOT NULL,
    target_perceptron VARCHAR NOT NULL,
    source_lattice_hash VARCHAR NOT NULL,
    target_lattice_hash VARCHAR NOT NULL,
    total_pairs INTEGER,
    max_degree_diff DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (source_perceptron) REFERENCES perceptrons(perceptron_id),
    FOREIGN KEY (target_perceptron) REFERENCES perceptrons(perceptron_id)
);

CREATE INDEX idx_ent_source ON entanglements(source_perceptron);
CREATE INDEX idx_ent_target ON entanglements(target_perceptron);
```

### 7. Entanglement Mappings

Detailed node-to-node mappings within an entanglement.

```sql
CREATE TABLE entanglement_mappings (
    entanglement_id VARCHAR NOT NULL,
    source_node_idx INTEGER NOT NULL,
    target_node_idx INTEGER NOT NULL,
    source_hash VARCHAR,               -- HLLSet hash
    target_hash VARCHAR,               -- HLLSet hash
    degree_similarity DOUBLE,
    register_similarity DOUBLE,
    strength DOUBLE,                   -- Overall entanglement strength
    
    PRIMARY KEY (entanglement_id, source_node_idx, target_node_idx),
    FOREIGN KEY (entanglement_id) REFERENCES entanglements(entanglement_id)
);

CREATE INDEX idx_emap_source ON entanglement_mappings(entanglement_id, source_node_idx);
CREATE INDEX idx_emap_strength ON entanglement_mappings(entanglement_id, strength);
```

## Metadata-Specific Tables

For ED-AI bridge metadata.

### 8. Metadata Graph Nodes

```sql
CREATE TABLE metadata_nodes (
    node_id VARCHAR PRIMARY KEY,       -- Content hash
    node_type VARCHAR NOT NULL,        -- 'table', 'column', 'relationship'
    name VARCHAR NOT NULL,             -- e.g., 'customers', 'customer_id'
    hllset_hash VARCHAR,               -- Associated HLLSet fingerprint
    parent_node VARCHAR,               -- For hierarchy (table -> columns)
    metadata_json VARCHAR,             -- Schema info, constraints, etc.
    
    FOREIGN KEY (parent_node) REFERENCES metadata_nodes(node_id)
);

CREATE INDEX idx_meta_type ON metadata_nodes(node_type);
CREATE INDEX idx_meta_name ON metadata_nodes(name);
CREATE INDEX idx_meta_parent ON metadata_nodes(parent_node);
```

### 9. Metadata Edges

```sql
CREATE TABLE metadata_edges (
    edge_id VARCHAR PRIMARY KEY,
    source_node VARCHAR NOT NULL,
    target_node VARCHAR NOT NULL,
    edge_type VARCHAR NOT NULL,        -- 'foreign_key', 'one_to_many', etc.
    entanglement_id VARCHAR,           -- Link to lattice entanglement
    metadata_json VARCHAR,             -- Additional edge properties
    
    FOREIGN KEY (source_node) REFERENCES metadata_nodes(node_id),
    FOREIGN KEY (target_node) REFERENCES metadata_nodes(node_id),
    FOREIGN KEY (entanglement_id) REFERENCES entanglements(entanglement_id)
);

CREATE INDEX idx_edge_source ON metadata_edges(source_node);
CREATE INDEX idx_edge_target ON metadata_edges(target_node);
CREATE INDEX idx_edge_type ON metadata_edges(edge_type);
```

## Query Patterns

### 1. Get AM for specific perceptron

```sql
SELECT row_idx, col_idx, frequency
FROM am_cells
WHERE perceptron_id = 'data_perceptron'
AND frequency > 0
ORDER BY row_idx, col_idx;
```

### 2. Get lattice structure

```sql
SELECT 
    node_type,
    node_idx,
    hllset_hash,
    cardinality
FROM lattice_nodes
WHERE perceptron_id = 'metadata_perceptron'
ORDER BY node_type, node_idx;
```

### 3. Find entanglements between two perceptrons

```sql
SELECT 
    e.*,
    COUNT(em.target_node_idx) as mapping_count,
    AVG(em.strength) as avg_strength
FROM entanglements e
JOIN entanglement_mappings em ON e.entanglement_id = em.entanglement_id
WHERE e.source_perceptron = 'data_perceptron'
AND e.target_perceptron = 'metadata_perceptron'
GROUP BY e.entanglement_id;
```

### 4. Traverse metadata graph

```sql
-- Find all columns for a table
SELECT c.*
FROM metadata_nodes t
JOIN metadata_nodes c ON c.parent_node = t.node_id
WHERE t.name = 'customers'
AND t.node_type = 'table'
AND c.node_type = 'column';

-- Find foreign key relationships
SELECT 
    me.edge_type,
    src.name as source_table,
    tgt.name as target_table
FROM metadata_edges me
JOIN metadata_nodes src ON me.source_node = src.node_id
JOIN metadata_nodes tgt ON me.target_node = tgt.node_id
WHERE me.edge_type = 'foreign_key';
```

### 5. Cross-perceptron queries

```sql
-- Find data HLLSets that map to metadata structures
SELECT 
    ln_data.node_idx as data_node,
    ln_meta.node_idx as metadata_node,
    mn.name as metadata_name,
    em.strength
FROM entanglement_mappings em
JOIN entanglements e ON em.entanglement_id = e.entanglement_id
JOIN lattice_nodes ln_data 
    ON ln_data.perceptron_id = e.source_perceptron
    AND ln_data.node_idx = em.source_node_idx
JOIN lattice_nodes ln_meta 
    ON ln_meta.perceptron_id = e.target_perceptron
    AND ln_meta.node_idx = em.target_node_idx
LEFT JOIN metadata_nodes mn 
    ON mn.hllset_hash = ln_meta.hllset_hash
WHERE e.source_perceptron = 'data_perceptron'
AND e.target_perceptron = 'metadata_perceptron'
AND em.strength > 0.7
ORDER BY em.strength DESC;
```

## Migration Strategy

### Phase 1: Extend Current Schema

Add `perceptron_id` to existing tables:

```sql
-- Add perceptron support to existing tables
ALTER TABLE am_cells ADD COLUMN perceptron_id VARCHAR DEFAULT 'default';
ALTER TABLE lattice_nodes ADD COLUMN perceptron_id VARCHAR DEFAULT 'default';

-- Update primary keys
ALTER TABLE am_cells DROP CONSTRAINT am_cells_pkey;
ALTER TABLE am_cells ADD PRIMARY KEY (perceptron_id, row_idx, col_idx);
```

### Phase 2: Add New Tables

```sql
-- Create new tables for multi-perceptron support
-- (perceptrons, entanglements, metadata_nodes, metadata_edges)
```

### Phase 3: Populate

```sql
-- Insert default perceptron
INSERT INTO perceptrons VALUES (
    'default',
    'data',
    'sha1',
    0x1234567890ABCDEF,
    '{"p_bits": 8, "h_bits": 16, "tau": 0.7, "rho": 0.3}',
    CURRENT_TIMESTAMP,
    'Default data perceptron'
);
```

## Storage Estimates

Typical metadata perceptron for 100-table database:

- **Perceptrons**: 2 rows × 200 bytes = 400 bytes
- **AM cells**: 50K cells × 20 bytes = 1 MB
- **Token LUT**: 10K tokens × 50 bytes = 500 KB
- **Lattice nodes**: 8K nodes × 100 bytes = 800 KB
- **HLLSet registers**: 1K HLLSets × 20 KB = 20 MB
- **Entanglements**: 1 × 200 bytes = 200 bytes
- **Entanglement mappings**: 10K mappings × 50 bytes = 500 KB
- **Metadata nodes**: 1K nodes × 200 bytes = 200 KB
- **Metadata edges**: 500 edges × 100 bytes = 50 KB

**Total**: ~23 MB for complete metadata representation

## IICA Compliance

### Immutability

- Never UPDATE existing records
- Only INSERT new versions
- Use timestamps for versioning

### Idempotency

```sql
-- Idempotent insert
INSERT INTO am_cells (perceptron_id, row_idx, col_idx, frequency)
VALUES ('data', 10, 20, 5)
ON CONFLICT (perceptron_id, row_idx, col_idx) 
DO UPDATE SET frequency = am_cells.frequency + EXCLUDED.frequency;
```

### Content Addressability

- All hashes computed from content
- No auto-incrementing IDs for content
- Use SHA1 hashes as primary keys where appropriate

## Next Steps

1. Create migration script for existing DuckDB schema
2. Implement perceptron registration in ManifoldOS
3. Update storage extension to handle multi-perceptron writes
4. Add metadata perceptron initialization
5. Implement metadata graph construction from ED schemas
6. Add cross-perceptron entanglement computation
