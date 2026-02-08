#!/usr/bin/env python3
"""
Demo: Enterprise-to-AI Metadata Bridge with Persistent LUT

Demonstrates:
1. Ingesting enterprise data → HLLSet fingerprints
2. Committing LUTs to persistent metadata store (DuckDB)
3. Querying tokens from metadata (AI → ED grounding)
4. Reverse lookups (ED → AI coordinates)
5. Metadata tracking for explainability
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manifold_os import ManifoldOS


def demo_basic_metadata_bridge():
    """Demo 1: Basic ingestion with persistent metadata."""
    print("\n" + "="*70)
    print("DEMO 1: Enterprise Data → Metadata → AI")
    print("="*70)
    
    # Create OS with persistent LUT store
    os = ManifoldOS(lut_db_path=":memory:")  # In-memory for demo
    
    print("\n[Step 1] Ingest enterprise data:")
    
    # Simulate CRM data
    customer_data = "premium customer revenue growth engagement"
    metadata = {
        'source': 'CRM_DB',
        'table': 'customers',
        'record_id': 12345
    }
    
    print(f"  Data: '{customer_data}'")
    print(f"  Source: {metadata}")
    
    # Ingest with metadata tracking
    representation = os.ingest(customer_data, metadata=metadata)
    
    print(f"\n[Step 2] Metadata stored:")
    print(f"  Original tokens: {representation.original_tokens}")
    print(f"  HLLSets created: {list(representation.hllsets.keys())}")
    print(f"  LUTs committed: {list(representation.luts.keys())}")
    
    # Get LUT stats
    stats = os.get_lut_stats()
    print(f"\n[Step 3] Persistent store stats:")
    print(f"  Total LUT records: {stats['total_lut_records']}")
    print(f"  Unique HLLSets: {stats['unique_hllsets']}")
    print(f"  N-groups: {stats['n_groups']}")


def demo_metadata_grounding():
    """Demo 2: AI → ED Grounding (query tokens from coordinates)."""
    print("\n" + "="*70)
    print("DEMO 2: AI → Enterprise Data Grounding")
    print("="*70)
    
    os = ManifoldOS(lut_db_path=":memory:")
    
    # Ingest data
    print("\n[Step 1] Ingest customer data:")
    data = "high value customer lifetime subscription"
    rep = os.ingest(data, metadata={'source': 'analytics_db'})
    
    hllset_hash = rep.hllsets[1].name
    print(f"  HLLSet hash: {hllset_hash[:20]}...")
    
    # AI has a coordinate from HLLSet - ground it back to tokens
    print("\n[Step 2] AI has HLLSet coordinates (reg, zeros):")
    print("  Querying metadata to ground AI decision...")
    
    # Get first few LUT entries
    lut = rep.luts[1]
    sample_keys = list(lut.keys())[:3]
    
    for reg, zeros in sample_keys:
        tokens = os.query_tokens_from_metadata(
            n=1,
            reg=reg,
            zeros=zeros,
            hllset_hash=hllset_hash
        )
        print(f"  (reg={reg}, zeros={zeros}) → Tokens: {tokens}")
    
    print("\n  ✓ AI decision grounded to source enterprise data!")


def demo_reverse_lookup():
    """Demo 3: ED → AI Lookup (find coordinates for token)."""
    print("\n" + "="*70)
    print("DEMO 3: Enterprise Data → AI Coordinates")
    print("="*70)
    
    os = ManifoldOS(lut_db_path=":memory:")
    
    # Ingest
    print("\n[Step 1] Ingest product data:")
    data = "enterprise software cloud platform subscription"
    rep = os.ingest(data, metadata={'source': 'product_catalog'})
    
    print(f"  Tokens: {rep.original_tokens}")
    
    # Enterprise wants to find where "enterprise" appears in AI space
    print("\n[Step 2] Find AI coordinates for enterprise token 'enterprise':")
    
    token = ('enterprise',)
    keys = os.query_by_token(n=1, token_tuple=token)
    
    print(f"  Token: {token}")
    print(f"  AI coordinates (reg, zeros): {keys}")
    print(f"\n  ✓ Enterprise data mapped to AI representation!")


def demo_metadata_tracking():
    """Demo 4: Metadata tracking for explainability."""
    print("\n" + "="*70)
    print("DEMO 4: Metadata Tracking & Explainability")
    print("="*70)
    
    os = ManifoldOS(lut_db_path=":memory:")
    
    print("\n[Step 1] Ingest multiple sources:")
    
    # Different sources
    sources = [
        ("customer purchase history data", {'source': 'ERP', 'dept': 'Sales'}),
        ("product inventory analytics", {'source': 'WMS', 'dept': 'Operations'}),
        ("employee performance metrics", {'source': 'HR', 'dept': 'People'}),
    ]
    
    hashes = []
    for data, meta in sources:
        rep = os.ingest(data, metadata=meta)
        hashes.append(rep.hllsets[1].name)
        print(f"  ✓ {meta['source']}: {rep.hllsets[1].name[:16]}...")
    
    print("\n[Step 2] Query metadata for each HLLSet:")
    
    for i, hash_val in enumerate(hashes, 1):
        metadata = os.get_ingestion_metadata(hash_val)
        print(f"\n  HLLSet {i}: {hash_val[:16]}...")
        print(f"    Source: {metadata['source']}")
        print(f"    Department: {metadata['dept']}")
        print(f"    Original tokens: {metadata['original_length']}")
        print(f"    Ingested: {metadata['ingested_at']}")
    
    print("\n  ✓ Full audit trail maintained for compliance!")


def demo_persistent_storage():
    """Demo 5: Persistent storage across sessions."""
    print("\n" + "="*70)
    print("DEMO 5: Persistent Metadata Storage")
    print("="*70)
    
    import tempfile
    import os as os_module
    
    # Create temp file for DuckDB
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
        db_path = f.name
    
    print(f"\n[Step 1] Create OS with persistent DB: {db_path}")
    
    # First session: Write data
    print("\n[Session 1] Ingest data:")
    os1 = ManifoldOS(lut_db_path=db_path)
    
    data = "persistent metadata storage example"
    rep1 = os1.ingest(data, metadata={'session': 1})
    hllset_hash = rep1.hllsets[1].name
    
    print(f"  Ingested: '{data}'")
    print(f"  HLLSet: {hllset_hash[:16]}...")
    
    stats1 = os1.get_lut_stats()
    print(f"  LUT records: {stats1['total_lut_records']}")
    
    # Close first session
    if os1.lut_store:
        os1.lut_store.close()
    
    # Second session: Read data
    print("\n[Session 2] Reconnect and query:")
    os2 = ManifoldOS(lut_db_path=db_path)
    
    stats2 = os2.get_lut_stats()
    print(f"  LUT records persisted: {stats2['total_lut_records']}")
    
    # Query metadata from first session
    metadata = os2.get_ingestion_metadata(hllset_hash)
    print(f"  Metadata from session 1: {metadata}")
    
    # Query tokens
    lut = rep1.luts[1]
    reg, zeros = list(lut.keys())[0]
    tokens = os2.query_tokens_from_metadata(1, reg, zeros, hllset_hash)
    print(f"  Tokens recovered: {tokens}")
    
    print("\n  ✓ Metadata persists across sessions!")
    
    # Cleanup
    if os2.lut_store:
        os2.lut_store.close()
    os_module.remove(db_path)


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Enterprise-to-AI Metadata Bridge")
    print("Persistent LUT Storage with DuckDB")
    print("="*70)
    
    demos = [
        demo_basic_metadata_bridge,
        demo_metadata_grounding,
        demo_reverse_lookup,
        demo_metadata_tracking,
        demo_persistent_storage,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ All demos complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • HLLSets = AI-native fingerprints (fixed 1.5KB)")
    print("  • LUT = Metadata bridge (tokens ↔ coordinates)")
    print("  • DuckDB = Persistent, queryable storage")
    print("  • Two-way: ED → AI and AI → ED")
    print("  • Audit trail for compliance/explainability")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
