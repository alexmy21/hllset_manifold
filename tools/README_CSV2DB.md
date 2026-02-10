# CSV to DuckDB Conversion Utility

## Overview

`csv2db.py` converts a directory of CSV files into a single DuckDB database file, with one table per CSV file.

## Features

- **Automatic schema detection**: DuckDB analyzes CSV files and infers optimal data types
- **Batch processing**: Converts entire directory in one command
- **Smart naming**: Converts CSV filenames to valid SQL table names
- **Statistics**: Reports row counts, column counts, and database size
- **Error handling**: Continues processing even if some files fail

## Usage

### Basic Usage

```bash
python tools/csv2db.py <csv_directory> <output_db_path>
```

### Example

```bash
# Convert all CSV files in ./data/csv/ to ./data/business.duckdb
python tools/csv2db.py ./data/csv/ ./data/business.duckdb
```

### Custom Pattern

```bash
# Only process files matching specific pattern
python tools/csv2db.py ./data/csv/ ./data/business.duckdb --pattern "sales_*.csv"
```

## Output

The tool provides detailed progress information:

```
Found 200 CSV files
Creating database: ./data/business.duckdb

Ingesting: customers.csv → customers... ✓ (10,500 rows)
Ingesting: sales_2023.csv → sales_2023... ✓ (45,230 rows)
Ingesting: products.csv → products... ✓ (1,250 rows)
...

======================================================================
Successfully created 200 tables:
  • customers                    (customers.csv)              10,500 rows
  • sales_2023                   (sales_2023.csv)             45,230 rows
  • products                     (products.csv)                1,250 rows
  ...

======================================================================
Database Statistics:
  Total tables: 200
  Estimated size: 125,433,856 bytes (119.62 MB)

Top 10 tables by column count:
  • customer_demographics                                    25 columns
  • product_details                                          18 columns
  • transaction_history                                      15 columns
  ...

Database saved to: ./data/business.duckdb
======================================================================
```

## Table Naming

CSV filenames are converted to valid SQL table names:

| CSV Filename | Table Name |
|-------------|------------|
| `sales_2023.csv` | `sales_2023` |
| `Customer Data.csv` | `customer_data` |
| `Q1-Revenue.csv` | `q1_revenue` |
| `2023-products.csv` | `table_2023_products` |

Rules:
- Spaces, hyphens, dots → underscores
- Non-alphanumeric characters removed
- Names starting with digit get `table_` prefix
- All lowercase

## Requirements

```bash
pip install duckdb
```

## Integration with Columnar Ingestion

After creating the DuckDB database, use it with the columnar ingestion system:

```python
from core.db_ingestion import DatabaseIngestionSystem
from core.manifold_os import ManifoldOS

manifold = ManifoldOS()
db_ingestion = DatabaseIngestionSystem(manifold)

# Ingest database column by column
result = db_ingestion.ingest_database("./data/business.duckdb")
```

This creates:
1. **Data hierarchy**: DB HLLSet → Table HLLSets → Column HLLSets
2. **Metadata hierarchy**: Raw metadata + HLLSet fingerprints + SHA1 IDs
3. **Explicit entanglements**: Data ↔ Metadata mappings

## Troubleshooting

### CSV Format Issues

If some files fail to load:
- Check CSV has headers
- Verify encoding (UTF-8 recommended)
- Look for malformed rows
- Check for inconsistent delimiters

### Large Files

For very large CSV files (>1GB):
- DuckDB handles them efficiently
- Process may take longer
- Monitor disk space (database will be smaller than sum of CSVs due to compression)

### Memory

The tool uses DuckDB's streaming capabilities:
- Memory usage is modest even for large datasets
- DuckDB processes data in chunks
- No need to load entire CSVs into memory

## Advanced Usage

### Verify Database Contents

```bash
# Use DuckDB CLI to explore
duckdb ./data/business.duckdb

# List all tables
D SHOW TABLES;

# Describe table schema
D DESCRIBE customers;

# Query data
D SELECT * FROM customers LIMIT 10;
```

### Re-run Conversion

To update database with new CSV files:
```bash
# Remove old database
rm ./data/business.duckdb

# Re-run conversion
python tools/csv2db.py ./data/csv/ ./data/business.duckdb
```

Note: This completely recreates the database. For incremental updates, use SQL INSERT statements directly.
