#!/usr/bin/env python3
"""
CSV to DuckDB Conversion Utility

Converts a directory of CSV files into a single DuckDB database,
with one table per CSV file.

Usage:
    python csv2db.py <csv_directory> <output_db_path> [--pattern "*.csv"]
"""

import argparse
import duckdb
from pathlib import Path
import sys


def csv_to_table_name(csv_path: Path) -> str:
    """Convert CSV filename to valid table name."""
    # Remove extension and replace special chars with underscores
    name = csv_path.stem
    # Replace spaces, hyphens, dots with underscores
    name = name.replace(' ', '_').replace('-', '_').replace('.', '_')
    # Remove any non-alphanumeric characters except underscore
    name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
    # Ensure it starts with letter or underscore
    if name[0].isdigit():
        name = f'table_{name}'
    return name.lower()


def ingest_csv_files(csv_dir: Path, db_path: Path, pattern: str = "*.csv"):
    """
    Ingest all CSV files from directory into DuckDB.
    
    Args:
        csv_dir: Directory containing CSV files
        db_path: Output DuckDB database path
        pattern: Glob pattern for CSV files (default: "*.csv")
    """
    csv_dir = Path(csv_dir)
    db_path = Path(db_path)
    
    if not csv_dir.exists():
        print(f"Error: Directory '{csv_dir}' does not exist")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = sorted(csv_dir.glob(pattern))
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern '{pattern}' in '{csv_dir}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Creating database: {db_path}")
    
    # Create/connect to database
    conn = duckdb.connect(str(db_path))
    
    # Process each CSV file
    tables_created = []
    tables_failed = []
    
    for csv_file in csv_files:
        table_name = csv_to_table_name(csv_file)
        
        try:
            # DuckDB can read CSV directly with auto-detection
            print(f"Ingesting: {csv_file.name} → {table_name}...", end=' ')
            
            # Create table from CSV with auto-detection
            conn.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_file}', 
                    header=true,
                    auto_detect=true,
                    sample_size=-1
                )
            """)
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"✓ ({row_count} rows)")
            
            tables_created.append((table_name, csv_file.name, row_count))
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            tables_failed.append((csv_file.name, str(e)))
    
    # Print summary
    print("\n" + "="*70)
    print(f"Successfully created {len(tables_created)} tables:")
    for table_name, csv_name, row_count in tables_created:
        print(f"  • {table_name:30s} ({csv_name:30s}) {row_count:>10,} rows")
    
    if tables_failed:
        print(f"\nFailed to create {len(tables_failed)} tables:")
        for csv_name, error in tables_failed:
            print(f"  • {csv_name}: {error}")
    
    # Get database statistics
    print("\n" + "="*70)
    print("Database Statistics:")
    
    result = conn.execute("""
        SELECT 
            COUNT(*) as table_count,
            SUM(estimated_size) as total_size
        FROM duckdb_tables()
        WHERE schema_name = 'main'
    """).fetchone()
    
    table_count, total_size = result
    print(f"  Total tables: {table_count}")
    print(f"  Estimated size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    # Show column statistics
    result = conn.execute("""
        SELECT 
            table_name,
            COUNT(*) as column_count
        FROM duckdb_columns()
        WHERE schema_name = 'main'
        GROUP BY table_name
        ORDER BY column_count DESC
        LIMIT 10
    """).fetchall()
    
    print(f"\nTop 10 tables by column count:")
    for table_name, col_count in result:
        print(f"  • {table_name:30s} {col_count:>3} columns")
    
    conn.close()
    print(f"\nDatabase saved to: {db_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert directory of CSV files to DuckDB database"
    )
    parser.add_argument(
        "csv_directory",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "output_db",
        help="Output DuckDB database file path"
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)"
    )
    
    args = parser.parse_args()
    
    ingest_csv_files(
        Path(args.csv_directory),
        Path(args.output_db),
        args.pattern
    )


if __name__ == "__main__":
    main()
