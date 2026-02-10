# Columnar Database Ingestion - Quick Start

## TL;DR

Convert 200 CSV files into semantic, queryable HLLSet representations in 3 steps:

```bash
# 1. Convert CSVs to DuckDB (one time)
python tools/csv2db.py /path/to/your/csvs ./data/business.duckdb

# 2. Ingest to HLLSets (run in Python/Jupyter)
from core.manifold_os import ManifoldOS
from core.db_ingestion import DatabaseIngestionSystem
import json

manifold = ManifoldOS()
db_ingestion = DatabaseIngestionSystem(manifold)
result = db_ingestion.ingest_database('./data/business.duckdb')

with open('./data/ingestion_result.json', 'w') as f:
    json.dump(result, f, indent=2)

# 3. Query the data
from tools.db_query_helper import DatabaseQueryHelper

helper = DatabaseQueryHelper(manifold, './data/ingestion_result.json')
matches = helper.search_columns(['revenue', 'sales'])

for match in matches[:10]:
    print(f"{match['similarity']:.3f} | {match['table']}.{match['column']}")
```

## What You Get

After ingestion, you have:

✅ **Semantic Search**: Find columns by meaning, not just name  
✅ **Relationship Discovery**: Automatic foreign key detection  
✅ **Compact Storage**: Millions of rows → thousands of column HLLSets  
✅ **Privacy**: Data stored as HLLSets (requires disambiguation)  
✅ **Cross-Table Analysis**: Natural similarity comparisons

## Example Queries

### Find All Revenue Columns

```python
matches = helper.search_columns(['revenue', 'income', 'sales', 'earnings'])

# Output:
# 0.850 | financials.total_revenue
# 0.720 | sales.quarterly_revenue  
# 0.650 | reports.gross_income
```

### Detect Foreign Keys

```python
related = helper.find_related_columns('orders', 'customer_id', threshold=0.7)

# Output:
# 0.950 | customers.id               (likely FK!)
# 0.880 | customer_addresses.customer_id
# 0.450 | products.supplier_id
```

### Get Sample Data

```python
samples = helper.get_column_sample_values('products', 'name', limit=5)

# Output: ['Widget A', 'Gadget B', 'Tool C', 'Device D', 'Kit E']
```

### Database Overview

```python
helper.print_database_summary()

# Output:
# ======================================================================
# DATABASE SUMMARY
# ======================================================================
# Database HLLSet cardinality: 1,245,678
# Number of tables: 200
# Total columns: 8,450
# 
# Tables:
#   • customers                    ( 15 columns)
#   • orders                       ( 12 columns)
#   • products                     ( 20 columns)
#   ...
```

## Integration with demo_analyst_workflow.ipynb

The demo notebook now includes real data ingestion:

1. **Setup Phase**: Convert CSVs, ingest to HLLSets
2. **Query Phase**: Semantic search for relevant columns
3. **Analysis Phase**: Natural language → SQL using discovered schema
4. **Result Phase**: Reconstruct data via disambiguation

Open [demo_analyst_workflow.ipynb](demo_analyst_workflow.ipynb) and run cells sequentially.

## File Structure

```text
hllset_manifold/
├── tools/
│   ├── csv2db.py              # CSV → DuckDB converter
│   ├── db_query_helper.py     # Query utilities
│   └── README_CSV2DB.md       # Detailed CSV converter docs
├── core/
│   ├── db_ingestion.py        # Columnar ingestion engine
│   └── ...
├── DOCS/
│   └── COLUMNAR_INGESTION.md  # Full architecture docs
├── data/
│   ├── business.duckdb        # Your DuckDB database (created)
│   └── ingestion_result.json  # Ingestion metadata (created)
└── demo_analyst_workflow.ipynb
```

## Next Steps

1. **Point to your CSVs**: Update path in step 1 above
2. **Run conversion**: Creates DuckDB database (~1-5 min for 200 files)
3. **Run ingestion**: Creates HLLSet hierarchy (~5-15 min for 200 tables)
4. **Explore data**: Use helper utilities or notebook

## Need Help?

- CSV conversion issues: See [tools/README_CSV2DB.md](tools/README_CSV2DB.md)
- Architecture details: See [DOCS/COLUMNAR_INGESTION.md](DOCS/COLUMNAR_INGESTION.md)
- Query examples: See [demo_analyst_workflow.ipynb](demo_analyst_workflow.ipynb)

## Performance Expectations

**200 CSV files** (typical business scenario):

- CSV → DuckDB: 1-5 minutes
- DuckDB → HLLSets: 5-15 minutes
- Storage: ~200-500 MB for all HLLSets
- Query speed: <1 second for column search
- Reconstruction: Varies by result size

Scales to thousands of tables with proper optimization.
