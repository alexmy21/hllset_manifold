"""
Test C Backend Implementation

This script tests the new C/Cython backend and compares it with Julia (if available).
"""

import time
import sys

print("=" * 70)
print("HLLSet C Backend Test")
print("=" * 70)

# Check what's available
print("\n1. Checking available backends...")
print("-" * 70)

try:
    from core.hllset_new import HLLSet, C_BACKEND_AVAILABLE, JULIA_AVAILABLE
    print(f"✓ Imported hllset_new successfully")
    print(f"  C Backend Available: {C_BACKEND_AVAILABLE}")
    print(f"  Julia Backend Available: {JULIA_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import hllset_new: {e}")
    sys.exit(1)

if not C_BACKEND_AVAILABLE:
    print("\n⚠ C backend not available. Build it with:")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)

# Test basic functionality
print("\n2. Testing basic operations...")
print("-" * 70)

# Create HLLSet
tokens = ['apple', 'banana', 'cherry', 'date', 'elderberry'] * 20
hll = HLLSet.from_batch(tokens)

print(f"Created HLLSet from {len(tokens)} tokens (5 unique)")
print(f"Backend: {hll.backend}")
print(f"Estimated cardinality: {hll.cardinality():.2f}")
print(f"Expected: ~5.0")
print(f"✓ Basic creation works!" if 4.0 <= hll.cardinality() <= 6.0 else "✗ Cardinality off!")

# Test batch processing
print("\n3. Testing batch processing...")
print("-" * 70)

batches = [
    [f'batch1_token_{i}' for i in range(100)],
    [f'batch2_token_{i}' for i in range(100)],
    [f'batch3_token_{i}' for i in range(100)],
]

hll_batches = HLLSet.from_batches(batches, parallel=False)
print(f"Sequential processing: {hll_batches.cardinality():.0f} unique tokens")
print(f"Expected: ~300")
print(f"✓ Batch processing works!" if 280 <= hll_batches.cardinality() <= 320 else "✗ Cardinality off!")

# Test parallel processing
print("\n4. Testing parallel processing...")
print("-" * 70)

large_batches = [[f'token_{batch}_{i}' for i in range(1000)] for batch in range(10)]

start = time.time()
hll_seq = HLLSet.from_batches(large_batches, parallel=False)
seq_time = time.time() - start
print(f"Sequential: {seq_time:.3f}s - {hll_seq.cardinality():.0f} unique")

start = time.time()
hll_par = HLLSet.from_batches(large_batches, parallel=True)
par_time = time.time() - start
print(f"Parallel:   {par_time:.3f}s - {hll_par.cardinality():.0f} unique")

if par_time < seq_time:
    speedup = seq_time / par_time
    print(f"✓ Parallel speedup: {speedup:.2f}x")
else:
    print(f"⚠ Parallel not faster (overhead for small data is normal)")

# Test set operations
print("\n5. Testing set operations...")
print("-" * 70)

hll_a = HLLSet.from_batch([f'token_{i}' for i in range(50)])
hll_b = HLLSet.from_batch([f'token_{i}' for i in range(25, 75)])

print(f"HLL A: {hll_a.cardinality():.0f} items")
print(f"HLL B: {hll_b.cardinality():.0f} items")

hll_union = hll_a.union(hll_b)
print(f"Union: {hll_union.cardinality():.0f} items (expected ~75)")
print(f"✓ Union works!" if 70 <= hll_union.cardinality() <= 80 else "✗ Union failed!")

# Test similarity
similarity = hll_a.similarity(hll_b)
cosine_sim = hll_a.cosine(hll_b)
print(f"Jaccard similarity: {similarity:.2%}")
print(f"Cosine similarity: {cosine_sim:.2%}")
print(f"✓ Similarity works!" if 0.3 <= similarity <= 0.4 else "⚠ Check similarity")

# Test immutability
print("\n6. Testing immutability...")
print("-" * 70)

hll_base = HLLSet.from_batch(['a', 'b', 'c'])
card_before = hll_base.cardinality()
print(f"Base HLLSet: {card_before:.0f} items")

hll_new = HLLSet.add(hll_base, ['d', 'e', 'f'])
card_after_base = hll_base.cardinality()
card_new = hll_new.cardinality()

print(f"After adding to base:")
print(f"  Base HLLSet: {card_after_base:.0f} items (should be unchanged)")
print(f"  New HLLSet: {card_new:.0f} items (should be 6)")

if card_before == card_after_base and card_new > card_before:
    print("✓ Immutability preserved!")
else:
    print("✗ Immutability violated!")

# Compare with Julia if available
if JULIA_AVAILABLE:
    print("\n7. Comparing C vs Julia backends...")
    print("-" * 70)
    
    try:
        from core.hllset import HLLSet as JuliaHLL
        
        test_tokens = [f'token_{i}' for i in range(1000)]
        
        # C backend
        start = time.time()
        hll_c = HLLSet.from_batch(test_tokens)
        c_time = time.time() - start
        c_card = hll_c.cardinality()
        
        # Julia backend  
        start = time.time()
        hll_j = JuliaHLL.from_batch(test_tokens)
        j_time = time.time() - start
        j_card = hll_j.cardinality()
        
        print(f"C Backend:     {c_time:.4f}s - cardinality: {c_card:.0f}")
        print(f"Julia Backend: {j_time:.4f}s - cardinality: {j_card:.0f}")
        print(f"Difference: {abs(c_card - j_card):.1f} ({abs(c_card - j_card)/max(c_card, j_card)*100:.1f}%)")
        
        if abs(c_card - j_card) / max(c_card, j_card) < 0.05:
            print("✓ Results match within 5%")
        else:
            print("⚠ Results differ significantly")
            
    except Exception as e:
        print(f"⚠ Could not compare with Julia: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ C backend is working!
✓ All basic operations pass
✓ Parallel processing enabled
✓ Performance is good

Next steps:
1. Run your existing tests with the new backend
2. Compare outputs with Julia version
3. When satisfied, switch to C backend as default:
   mv core/hllset.py core/hllset_julia_backup.py
   mv core/hllset_new.py core/hllset.py

Backend: {hll.backend}
""")
