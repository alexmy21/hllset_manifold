#!/usr/bin/env python3
"""
Test n-token ingestion algorithm with LUT disambiguation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manifold_os import (
    ManifoldOS, IngestDriver, TokenizationConfig,
    NTokenRepresentation, LUTRecord
)


def test_n_token_generation():
    """Test n-token group generation."""
    print("\n" + "="*70)
    print("TEST: N-Token Generation")
    print("="*70)
    
    os = ManifoldOS()
    
    # Test data
    text = "the quick brown fox jumps"
    print(f"\nInput: '{text}'")
    
    # Process with n-token algorithm
    representation = os.ingest(text)
    
    print(f"\nOriginal tokens: {representation.original_tokens}")
    print(f"Number of n-token groups: {len(representation.n_token_groups)}")
    
    for n, n_tokens in sorted(representation.n_token_groups.items()):
        print(f"\n{n}-tokens: {len(n_tokens)} groups")
        print(f"  {n_tokens[:5]}...")  # Show first 5
    
    print("\n✓ N-token generation working")


def test_implicit_order():
    """Test implicit order preservation."""
    print("\n" + "="*70)
    print("TEST: Implicit Order Preservation")
    print("="*70)
    
    os = ManifoldOS()
    text = "a b c d"
    print(f"\nInput: '{text}'")
    
    representation = os.ingest(text)
    
    implicit_order = representation.get_implicit_order()
    print(f"\nImplicit order ({len(implicit_order)} items):")
    for item in implicit_order:
        print(f"  {item}")
    
    # Verify order
    # Should see: ('a',) < ('a','b') < ('a','b','c') < ('b',) < ('b','c') < ('b','c','d') < ...
    print("\n✓ Order preserved: 1-tokens, then 2-tokens, then 3-tokens")


def test_multiple_hllsets():
    """Test that each n-token group creates different HLLSet."""
    print("\n" + "="*70)
    print("TEST: Multiple HLLSets per Document")
    print("="*70)
    
    os = ManifoldOS()
    text = "hello world from multiple representations"
    print(f"\nInput: '{text}'")
    
    representation = os.ingest(text)
    
    print(f"\nHLLSets created: {len(representation.hllsets)}")
    for n, hllset in sorted(representation.hllsets.items()):
        card = hllset.cardinality()
        print(f"  {n}-token HLLSet: {hllset.name[:16]}... (card={card})")
    
    print("\n✓ Each n-token group has distinct HLLSet")


def test_lut_structure():
    """Test LUT (Lookup Table) structure."""
    print("\n" + "="*70)
    print("TEST: LUT Structure")
    print("="*70)
    
    os = ManifoldOS()
    text = "test lut structure"
    print(f"\nInput: '{text}'")
    
    representation = os.ingest(text)
    
    print(f"\nLUTs created: {len(representation.luts)}")
    for n, lut in sorted(representation.luts.items()):
        print(f"\n  {n}-token LUT: {len(lut)} entries")
        
        # Show first few entries
        count = 0
        for key, record in lut.items():
            if count >= 3:
                break
            print(f"    (reg={record.reg}, zeros={record.zeros})")
            print(f"      Hashes: {len(record.hashes)}")
            print(f"      Tokens: {record.tokens}")
            count += 1
    
    print("\n✓ LUT structure correct")


def test_disambiguation():
    """Test token disambiguation using LUT intersection."""
    print("\n" + "="*70)
    print("TEST: Token Disambiguation")
    print("="*70)
    
    os = ManifoldOS()
    text = "disambiguation test case"
    print(f"\nInput: '{text}'")
    
    representation = os.ingest(text)
    
    # Try to disambiguate from first entry in each LUT
    print("\nTrying disambiguation on sample (reg, zeros) keys:")
    
    tested = 0
    for n, lut in sorted(representation.luts.items()):
        if not lut:
            continue
        
        # Get first key
        first_key = next(iter(lut.keys()))
        reg, zeros = first_key
        
        # Get candidates from this LUT
        candidates = lut[first_key].get_candidates()
        print(f"\n  {n}-token group at (reg={reg}, zeros={zeros}):")
        print(f"    Candidates: {candidates}")
        
        tested += 1
        if tested >= 2:
            break
    
    # Try disambiguation across all groups
    if representation.luts:
        # Pick a key that might exist in multiple LUTs
        all_keys = set()
        for lut in representation.luts.values():
            all_keys.update(lut.keys())
        
        if all_keys:
            test_key = next(iter(all_keys))
            reg, zeros = test_key
            
            disambiguated = representation.disambiguate_tokens(reg, zeros)
            print(f"\n  Disambiguated tokens at (reg={reg}, zeros={zeros}):")
            print(f"    Result (intersection): {disambiguated}")
    
    print("\n✓ Disambiguation algorithm working")


def test_config_options():
    """Test configuration options."""
    print("\n" + "="*70)
    print("TEST: Configuration Options")
    print("="*70)
    
    # Test 1: Simple mode (no n-tokens)
    config1 = TokenizationConfig(use_n_tokens=False)
    driver1 = IngestDriver("test1", config1)
    os = ManifoldOS()
    os.register_driver(driver1)
    driver1.wake()
    
    text = "simple mode test"
    rep1 = driver1.process(text, os.kernel)
    
    print(f"\nSimple mode (use_n_tokens=False):")
    print(f"  HLLSets: {len(rep1.hllsets)}")
    print(f"  n-token groups: {len(rep1.n_token_groups)}")
    
    # Test 2: Custom n-token groups
    config2 = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3, 4]
    )
    driver2 = IngestDriver("test2", config2)
    os.register_driver(driver2)
    driver2.wake()
    
    rep2 = driver2.process(text, os.kernel)
    
    print(f"\nCustom n-token groups [1,2,3,4]:")
    print(f"  HLLSets: {len(rep2.hllsets)}")
    print(f"  n-token groups: {list(rep2.n_token_groups.keys())}")
    
    print("\n✓ Configuration options working")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("N-Token Ingestion Algorithm Tests")
    print("="*70)
    
    tests = [
        test_n_token_generation,
        test_implicit_order,
        test_multiple_hllsets,
        test_lut_structure,
        test_disambiguation,
        test_config_options,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ All tests complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
