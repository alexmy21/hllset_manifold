# Columnar Database Ingestion System

## Overview

This system implements **columnar ingestion** of tabular data from DuckDB databases into HLLSet-based representations. It creates a three-tier architecture:

1. **Data Hierarchy**: DB HLLSet → Table HLLSets → Column HLLSets
2. **Metadata Hierarchy**: Hybrid approach (raw metadata + HLLSet fingerprints + SHA1 IDs)
3. **Entanglement Network**: Explicit data ↔ metadata mappings

## Design Rationale

### Why Columnar Representation?

**Key Insight**: Thousands of columns vs billions of rows = massive compression

- Most databases: few hundred tables, few thousand columns total
- But millions/billions of rows
- **Columnar HLLSets**: One HLLSet per column (all distinct values)
- **Row Reconstruction**: Row HLLSet ∩ Column HLLSet = Cell HLLSet

### Benefits

1. **Compression**: Store thousands of column HLLSets instead of billions of row records
2. **Semantic Search**: Find columns by content similarity, not just name
3. **Lazy Reconstruction**: Only reconstruct cells/rows that are actually needed
4. **Schema Discovery**: Automatically detect relationships and patterns
5. **Privacy**: Data exists as HLLSets, not plaintext (disambiguation required)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Database HLLSet                        │
│            (Union of all table data)                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ Table HLLSet │        │ Table HLLSet │
│   (Union of  │        │   (Union of  │
│   columns)   │        │   columns)   │
└──────┬───────┘        └──────┬───────┘
       │                       │
   ┌───┴───┐               ┌───┴───┐
   ▼       ▼               ▼       ▼
┌──────┐ ┌──────┐       ┌──────┐ ┌──────┐
│Col 1 │ │Col 2 │       │Col A │ │Col B │
│HLLSet│ │HLLSet│       │HLLSet│ │HLLSet│
└──────┘ └──────┘       └──────┘ └──────┘
```

### Metadata Hierarchy (Hybrid Approach)

For each level, we store:

**Raw Metadata** (JSON):
- Small, precise, human-readable
- Schema information, statistics
- SHA1 ID for universal reference

**HLLSet Fingerprint**:
- Enables semantic search
- Column name + data type + semantic tokens
- Example: `["revenue", "decimal", "currency", "numeric"]`

**Explicit Entanglement**:
- Data HLLSet ↔ Metadata (strength=1.0)
- Data HLLSet ↔ Fingerprint HLLSet (strength=0.8)

## Components

### 1. CSV to DuckDB Converter (`tools/csv2db.py`)

Converts directory of CSV files into single DuckDB database.

```bash
python tools/csv2db.py /path/to/csvs ./data/business.duckdb
```

Features:
- Auto-detection of schema
- Batch processing
- Error handling
- Statistics reporting

### 2. Columnar Ingestion System (`core/db_ingestion.py`)

Main ingestion engine that:
- Connects to DuckDB database
- Extracts metadata at all levels
- Creates HLLSets for all columns
- Builds hierarchy and entanglements
- Stores in ManifoldOS unified storage

Classes:
- `ColumnMetadata`: Column-level metadata
- `TableMetadata`: Table-level metadata
- `DatabaseMetadata`: Database-level metadata
- `DatabaseIngestionSystem`: Main ingestion engine

### 3. Query Helper (`tools/db_query_helper.py`)

Utilities for querying ingested data:

```python
from tools.db_query_helper import DatabaseQueryHelper

helper = DatabaseQueryHelper(manifold, './data/ingestion_result.json')

# Search for columns
matches = helper.search_columns(['revenue', 'sales'])

# Find related columns (foreign keys, similar data)
related = helper.find_related_columns('sales', 'customer_id')

# Get sample values
samples = helper.get_column_sample_values('products', 'name')
```

## Usage Workflow

### Step 1: Convert CSVs to DuckDB (One Time)

```bash
# Convert all CSV files to DuckDB
python tools/csv2db.py ./data/csv_files/ ./data/business.duckdb

# Verify
duckdb ./data/business.duckdb "SHOW TABLES;"
```

### Step 2: Ingest Database

```python
from core.manifold_os import ManifoldOS
from core.db_ingestion import DatabaseIngestionSystem

# Initialize
manifold = ManifoldOS()
db_ingestion = DatabaseIngestionSystem(manifold)

# Ingest
result = db_ingestion.ingest_database('./data/business.duckdb')

# Save result
import json
with open('./data/ingestion_result.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### Step 3: Query Ingested Data

```python
from tools.db_query_helper import DatabaseQueryHelper

helper = DatabaseQueryHelper(manifold, './data/ingestion_result.json')

# Semantic search for columns
matches = helper.search_columns(['revenue', 'income', 'sales'])

for match in matches[:10]:
    print(f"{match['similarity']:.3f} | {match['table']} | {match['column']}")
    print(f"  Type: {match['data_type']}")
    print(f"  Distinct values: {match['distinct_count']:,}")
```

### Step 4: Reconstruct Data

```python
# Get column data
col_info = helper.get_column_info('sales', 'revenue')

# Get sample values (disambiguation)
samples = helper.get_column_sample_values('sales', 'revenue', limit=10)
print(f"Sample revenue values: {samples}")

# Find foreign key relationships
related = helper.find_related_columns('sales', 'customer_id', threshold=0.5)
for rel in related:
    print(f"{rel['similarity']:.3f} | {rel['table']}.{rel['column']}")
```

## Data Types Handling

### Numeric Values

Stored as strings in HLLSet:
```python
# Example: revenue column
tokens = ["100.50", "250.00", "1500.00", ...]
hllset = HLLSet.from_tokens(tokens)
```

Consider binning for very large numeric ranges:
```python
# Alternative: bin numeric values
def bin_numeric(value):
    if value < 100: return "0-100"
    elif value < 1000: return "100-1000"
    # ... etc
```

### Dates/Times

Normalized to ISO format:
```python
tokens = ["2025-01-15", "2025-02-20", ...]
```

### Text

Direct storage:
```python
tokens = ["Customer A", "Customer B", ...]
```

### NULLs

Special token:
```python
if null_count > 0:
    tokens.append("__NULL__")
```

### Booleans

String representation:
```python
tokens = ["true", "false"]
```

## Scalability

### Memory Considerations

**Scenario**: 200 tables × 50 columns avg = **10,000 column HLLSets**

Memory per HLLSet:
- Precision 14 (default): ~16KB per HLLSet
- Total: ~156 MB for all column HLLSets

**Optimization**:
- Lazy loading: Load HLLSets on demand
- Cache frequently accessed columns
- Use lower precision for large cardinality columns

### Query Performance

**Column Search** (by keyword):
- Load metadata fingerprints only (~16KB each)
- Calculate Jaccard similarity
- O(N) where N = number of columns (~10,000)
- Fast: ~0.1-1 second for full scan

**Data Reconstruction**:
- Small result (<1000 rows): Full reconstruction
- Medium result (<100K rows): Progressive disambiguation
- Large result: Statistics only (min/max/avg/count)

## Use Cases

### 1. Schema Discovery

Find all currency-related columns:
```python
matches = helper.search_columns(['currency', 'money', 'price', 'cost'])
```

### 2. Foreign Key Detection

Find potential foreign key relationships:
```python
related = helper.find_related_columns('orders', 'customer_id', threshold=0.7)
# High similarity = likely FK relationship
```

### 3. Data Quality Analysis

Check for duplicates across tables:
```python
# Compare column from two tables
col1_hllset = load_column('table1', 'email')
col2_hllset = load_column('table2', 'email')
overlap = col1_hllset.intersection(col2_hllset)
print(f"Duplicate emails: {overlap.cardinality()}")
```

### 4. Natural Language to SQL

Analyst: "Show me sales by region"

System:
1. Convert query to HLLSet: `["sales", "region", "show", ...]`
2. Search columns: Find `sales.revenue`, `sales.region_id`, `regions.name`
3. Detect relationship: `sales.region_id` ↔ `regions.id` (high similarity)
4. Generate SQL: `SELECT r.name, SUM(s.revenue) FROM sales s JOIN regions r ...`

### 5. Cross-Database Search

"Find all tables with customer email addresses across multiple databases"

```python
# Search across all ingested databases
for db_result in [db1_result, db2_result, db3_result]:
    helper = DatabaseQueryHelper(manifold, db_result)
    matches = helper.search_columns(['email', 'contact'])
    # Show matches from each database
```

## Performance Tips

### 1. Start Small

Ingest a subset first:
```bash
# Test with 5-10 tables
python tools/csv2db.py ./data/sample_csvs/ ./data/test.duckdb
```

### 2. Monitor Statistics

Track during ingestion:
- Cardinality per column (compression effectiveness)
- Distinct values (potential for binning)
- Null counts (data quality)

### 3. Optimize Precision

For columns with huge cardinality (>1M distinct values):
```python
# Use lower precision
hllset = HLLSet(precision=12)  # Instead of 14
# Smaller size, slightly less accurate
```

### 4. Index Metadata

For fast lookup:
```python
# Build index after ingestion
column_index = {
    f"{table}.{column}": col_data['data_id']
    for table, tbl_data in result['tables'].items()
    for column, col_data in tbl_data['columns'].items()
}
```

## Future Enhancements

### 1. Incremental Updates

Support for updating ingested data:
- Delta ingestion (only changed rows)
- Append-only mode (for log data)

### 2. Distributed Ingestion

Process large databases in parallel:
- Multiple workers, one table per worker
- Merge results at end

### 3. Query Optimizer

Use column cardinality for query planning:
```python
# Estimate query selectivity
query_result_size = query_hllset.intersection(table_hllset).cardinality()
```

### 4. Data Lineage

Track data transformations:
- Ingestion timestamp
- Source file SHA256
- Column dependencies

## Comparison with Traditional Approaches

| Aspect | Traditional DB | Columnar HLLSets |
|--------|----------------|------------------|
| Storage | Row-based | Column-based |
| Size | Proportional to rows | Proportional to columns |
| Semantic search | Limited (SQL) | Native (HLLSet similarity) |
| Privacy | Plaintext | Compressed/obfuscated |
| Reconstruction | Direct access | Disambiguation required |
| Schema discovery | Manual | Automatic |
| Cross-database | Complex | Natural (HLLSet unions) |

## Conclusion

The columnar ingestion system provides:

✅ **Efficient storage**: Compress millions of rows into thousands of column HLLSets  
✅ **Semantic capabilities**: Search data by meaning, not just structure  
✅ **Flexible reconstruction**: Retrieve only what's needed  
✅ **Relationship discovery**: Automatically find correlations and FKs  
✅ **Hybrid metadata**: Best of both worlds (precision + semantics)

This approach transforms traditional databases into a **semantic knowledge graph** where data relationships emerge naturally from HLLSet operations.
