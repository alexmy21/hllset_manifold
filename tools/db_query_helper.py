#!/usr/bin/env python3
"""
Database Query Utility

Helper tool to query ingested database using the columnar HLLSet representation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import duckdb
from core.hllset import HLLSet
from core.manifold_os import ManifoldOS


class DatabaseQueryHelper:
    """
    Helper for querying ingested database via HLLSets.
    """
    
    def __init__(self, manifold: ManifoldOS, ingestion_result_path: str, db_path: Optional[str] = None):
        """
        Initialize with manifold and path to ingestion result JSON.
        
        Args:
            manifold: ManifoldOS instance
            ingestion_result_path: Path to ingestion_result.json
            db_path: Optional path to DuckDB database (for sample value queries)
        """
        self.manifold = manifold
        self.db_path = db_path
        
        # Load ingestion result
        with open(ingestion_result_path, 'r') as f:
            self.ingestion_result = json.load(f)
    
    def list_tables(self) -> List[str]:
        """List all tables in database."""
        return list(self.ingestion_result['tables'].keys())
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        if table_name not in self.ingestion_result['tables']:
            raise ValueError(f"Table '{table_name}' not found")
        
        table_data = self.ingestion_result['tables'][table_name]
        
        # Load metadata
        meta_bytes = self.manifold.retrieve_artifact(table_data['metadata_id'])
        metadata = json.loads(meta_bytes.decode())
        
        return {
            'name': table_name,
            'columns': list(table_data['columns'].keys()),
            'metadata': metadata
        }
    
    def get_column_info(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get detailed information about a column."""
        if table_name not in self.ingestion_result['tables']:
            raise ValueError(f"Table '{table_name}' not found")
        
        table_data = self.ingestion_result['tables'][table_name]
        
        if column_name not in table_data['columns']:
            raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")
        
        col_data = table_data['columns'][column_name]
        
        # Load metadata
        meta_bytes = self.manifold.retrieve_artifact(col_data['metadata_id'])
        metadata = json.loads(meta_bytes.decode())
        
        # Load data HLLSet
        data_bytes = self.manifold.retrieve_artifact(col_data['data_id'])
        data_hllset = HLLSet.from_roaring(data_bytes)
        
        return {
            'table': table_name,
            'column': column_name,
            'metadata': metadata,
            'cardinality': data_hllset.cardinality()
        }
    
    def search_columns(self, keywords: List[str], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for columns matching keywords.
        
        Args:
            keywords: List of keywords to search for
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of matching columns with similarity scores
        """
        # Create query HLLSet
        query_hllset = HLLSet.from_batch(keywords)
        
        matches = []
        
        for table_name, table_data in self.ingestion_result['tables'].items():
            for col_name, col_data in table_data['columns'].items():
                # Load column metadata fingerprint
                fp_bytes = self.manifold.retrieve_artifact(col_data['fingerprint_id'])
                col_fingerprint = HLLSet.from_roaring(fp_bytes)
                
                # Calculate similarity (Jaccard index)
                intersection = query_hllset.intersect(col_fingerprint)
                union = query_hllset.union(col_fingerprint)
                similarity = intersection.cardinality() / union.cardinality()
                
                if similarity > threshold:
                    # Load metadata
                    meta_bytes = self.manifold.retrieve_artifact(col_data['metadata_id'])
                    metadata = json.loads(meta_bytes.decode())
                    
                    matches.append({
                        'similarity': similarity,
                        'table': table_name,
                        'column': col_name,
                        'data_type': metadata['data_type'],
                        'distinct_count': metadata.get('distinct_count'),
                        'metadata': metadata
                    })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def get_column_sample_values(self, table_name: str, column_name: str, limit: int = 10) -> List[Any]:
        """
        Get sample values from column (via database query).
        
        Args:
            table_name: Table name
            column_name: Column name
            limit: Maximum number of values to return
        
        Returns:
            List of sample values
        """
        if table_name not in self.ingestion_result['tables']:
            raise ValueError(f"Table '{table_name}' not found")
        
        table_data = self.ingestion_result['tables'][table_name]
        
        if column_name not in table_data['columns']:
            raise ValueError(f"Column '{column_name}' not found")
        
        # Need database connection to get actual values
        if not self.db_path:
            raise ValueError("Database path required for sample value queries. Pass db_path to __init__.")
        
        # Query database for sample values
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            # Use DISTINCT and LIMIT to get sample values
            query = f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL LIMIT {limit}'
            result = conn.execute(query).fetchall()
            return [row[0] for row in result]
        finally:
            conn.close()

    
    def find_related_columns(self, table_name: str, column_name: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find columns related to given column (via HLLSet similarity).
        
        Useful for finding:
        - Foreign key relationships
        - Similar data patterns
        - Duplicate columns
        
        Args:
            table_name: Source table name
            column_name: Source column name
            threshold: Similarity threshold
        
        Returns:
            List of related columns with similarity scores
        """
        if table_name not in self.ingestion_result['tables']:
            raise ValueError(f"Table '{table_name}' not found")
        
        table_data = self.ingestion_result['tables'][table_name]
        
        if column_name not in table_data['columns']:
            raise ValueError(f"Column '{column_name}' not found")
        
        # Load source column data
        source_col_data = table_data['columns'][column_name]
        source_bytes = self.manifold.retrieve_artifact(source_col_data['data_id'])
        source_hllset = HLLSet.from_roaring(source_bytes)
        
        related = []
        
        # Compare with all other columns
        for tbl_name, tbl_data in self.ingestion_result['tables'].items():
            for col_name, col_data in tbl_data['columns'].items():
                # Skip self
                if tbl_name == table_name and col_name == column_name:
                    continue
                
                # Load column data
                col_bytes = self.manifold.retrieve_artifact(col_data['data_id'])
                col_hllset = HLLSet.from_roaring(col_bytes)
                
                # Calculate similarity
                intersection = source_hllset.intersect(col_hllset)
                union = source_hllset.union(col_hllset)
                similarity = intersection.cardinality() / union.cardinality()
                
                if similarity > threshold:
                    # Load metadata
                    meta_bytes = self.manifold.retrieve_artifact(col_data['metadata_id'])
                    metadata = json.loads(meta_bytes.decode())
                    
                    related.append({
                        'similarity': similarity,
                        'table': tbl_name,
                        'column': col_name,
                        'data_type': metadata['data_type'],
                        'overlap_cardinality': intersection.cardinality()
                    })
        
        # Sort by similarity
        related.sort(key=lambda x: x['similarity'], reverse=True)
        
        return related
    
    def print_database_summary(self):
        """Print summary of ingested database."""
        print("="*70)
        print("DATABASE SUMMARY")
        print("="*70)
        
        db_info = self.ingestion_result['database']
        print(f"Database HLLSet cardinality: {db_info['cardinality']:,}")
        print(f"Number of tables: {len(self.ingestion_result['tables'])}")
        
        total_columns = sum(
            len(t['columns']) 
            for t in self.ingestion_result['tables'].values()
        )
        print(f"Total columns: {total_columns}")
        
        print("\nTables:")
        for table_name in sorted(self.ingestion_result['tables'].keys()):
            table_data = self.ingestion_result['tables'][table_name]
            col_count = len(table_data['columns'])
            print(f"  • {table_name:30s} ({col_count:3d} columns)")
        
        print("="*70)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query ingested database")
    parser.add_argument("ingestion_result", help="Path to ingestion_result.json")
    parser.add_argument("--list-tables", action="store_true", help="List all tables")
    parser.add_argument("--table-info", help="Get info about table")
    parser.add_argument("--search", nargs="+", help="Search for columns by keywords")
    
    args = parser.parse_args()
    
    # Initialize
    manifold = ManifoldOS()
    helper = DatabaseQueryHelper(manifold, args.ingestion_result)
    
    if args.list_tables:
        print("Tables:")
        for table in helper.list_tables():
            print(f"  • {table}")
    
    elif args.table_info:
        info = helper.get_table_info(args.table_info)
        print(f"Table: {info['name']}")
        print(f"Columns ({len(info['columns'])}):")
        for col in info['columns']:
            print(f"  • {col}")
    
    elif args.search:
        matches = helper.search_columns(args.search)
        print(f"Searching for: {args.search}")
        print(f"Found {len(matches)} matches:\n")
        for match in matches[:20]:
            print(f"{match['similarity']:.3f} | {match['table']:25s} | {match['column']:20s}")
            print(f"       Type: {match['data_type']}, Distinct: {match['distinct_count']:,}")
            print()
    
    else:
        helper.print_database_summary()


if __name__ == "__main__":
    main()
