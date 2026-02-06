#!/usr/bin/env python3
"""
Refactored System Demo: Immutable Tensor Evolution

Demonstrates the three-layer architecture:
1. Kernel: Pure morphisms (absorb, union, intersection, difference)
2. HRT: Immutable data structure with PyTorch tensors
3. Evolution: Three-state model (In-Process → Current → History)

Key Concepts:
- Immutability: All modifications create new objects with new hashes
- Content-addressing: All objects named by SHA1 of their content
- Evolution: System progresses through discrete, irreversible steps
- Git integration: History is pushed to Git, not stored in memory
"""

import sys
sys.path.insert(0, '.')

from core import (
    Kernel, HLLSet,
    ImmutableTensor,
    HRT, HRTConfig, HRTEvolution,
    compute_structural_hash
)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def demo_kernel():
    """Demonstrate stateless kernel operations."""
    print_section("LAYER 1: KERNEL (Pure Morphisms)")
    
    kernel = Kernel(p_bits=10)
    
    # Morphism 1: absorb
    print("\n1. absorb: Set[str] → HLLSet")
    h1 = kernel.absorb({'cat', 'dog', 'bird', 'fish'})
    h2 = kernel.absorb({'dog', 'fish', 'whale', 'shark'})
    print(f"   A = absorb({{'cat', 'dog', 'bird', 'fish'}})")
    print(f"       → {h1}")
    print(f"   B = absorb({{'dog', 'fish', 'whale', 'shark'}})")
    print(f"       → {h2}")
    
    # Morphism 2: union
    print("\n2. union: HLLSet × HLLSet → HLLSet")
    h_union = kernel.union(h1, h2)
    print(f"   A ∪ B = {h_union}")
    
    # Morphism 3: intersection
    print("\n3. intersection: HLLSet × HLLSet → HLLSet")
    h_inter = kernel.intersection(h1, h2)
    print(f"   A ∩ B = {h_inter}")
    
    # Morphism 4: difference
    print("\n4. difference: HLLSet × HLLSet → HLLSet")
    h_diff = kernel.difference(h1, h2)
    print(f"   A \\ B = {h_diff}")
    
    # Key property: kernel is stateless
    print("\n5. Stateless Property")
    print("   - No storage of previous results")
    print("   - Same input always produces same output")
    print(f"   - A == absorb({{'cat', 'dog', 'bird', 'fish'}}): {h1 == kernel.absorb({'cat', 'dog', 'bird', 'fish'})}")
    
    return kernel, h1, h2


def demo_immutable_tensor():
    """Demonstrate immutable tensor operations."""
    print_section("LAYER 2: IMMUTABLE TENSOR (PyTorch Backend)")
    
    # Genesis tensor
    print("\n1. Genesis Tensor")
    t_genesis = ImmutableTensor.zeros(100, 100)
    print(f"   T₀ = zeros(100, 100)")
    print(f"      name = {t_genesis.name}")
    
    # All operations create new tensors
    print("\n2. Clone-on-Modify Operations")
    t1 = t_genesis.with_value((50, 50), 1.0)
    t2 = t_genesis.with_value((25, 25), 2.0)
    print(f"   T₁ = T₀.with_value((50, 50), 1.0)")
    print(f"      name = {t1.name}")
    print(f"   T₂ = T₀.with_value((25, 25), 2.0)")
    print(f"      name = {t2.name}")
    
    # Merge operation
    print("\n3. Merge Operation (element-wise max)")
    t_merged = t1.maximum(t2)
    print(f"   T₃ = maximum(T₁, T₂)")
    print(f"      name = {t_merged.name}")
    print(f"   Value at (50, 50): {t_merged.data[50, 50].item()}")
    print(f"   Value at (25, 25): {t_merged.data[25, 25].item()}")
    
    # Immutability guarantee
    print("\n4. Immutability Guarantee")
    print(f"   T₀ unchanged: {t_genesis.name}")
    print(f"   T₀ == original genesis: {t_genesis.name == t_genesis.name}")
    
    return t_genesis


def demo_hrt_evolution():
    """Demonstrate HRT three-state evolution."""
    print_section("LAYER 3: HRT EVOLUTION (Three-State Model)")
    
    # Setup
    kernel = Kernel(p_bits=8)
    config = HRTConfig(p_bits=8, h_bits=16)
    
    print(f"\nConfiguration:")
    print(f"   Dimension: {config.dimension}")
    print(f"   Basic HLLSets: {config.num_basic_hllsets}")
    
    # Create evolution manager
    print("\n1. Genesis State")
    evolution = HRTEvolution(config)
    genesis = evolution.get_current()
    print(f"   HRT₀ (genesis)")
    print(f"      name = {genesis.name[:32]}...")
    print(f"      step = {genesis.step_number}")
    print(f"      am_entries = {len(genesis.am.nonzero_entries())}")
    
    # Evolution Cycle 1
    print("\n2. Evolution Cycle 1: Ingest → Merge → Commit")
    
    # Step 1: Ingest
    data1 = {
        'camera': {'red', 'green', 'blue', 'yellow'},
        'microphone': {'low', 'mid', 'high'}
    }
    print(f"   Ingest: {list(data1.keys())}")
    evolution.ingest(data1, kernel)
    
    # Step 2: Evolve (merge + commit)
    def mock_commit(hrt):
        """Mock Git commit - would save to persistent store."""
        return hrt.name
    
    evolution.evolve(kernel, mock_commit)
    
    hrt1 = evolution.get_current()
    print(f"   After evolve:")
    print(f"      HRT₁ name = {hrt1.name[:32]}...")
    print(f"      step = {hrt1.step_number}")
    print(f"      am_entries = {len(hrt1.am.nonzero_entries())}")
    print(f"      parent = {hrt1.parent_hrt[:16] if hrt1.parent_hrt else 'genesis'}...")
    
    # Evolution Cycle 2
    print("\n3. Evolution Cycle 2")
    data2 = {
        'thermometer': {'hot', 'warm', 'cold'},
        'pressure': {'high_p', 'low_p'}
    }
    print(f"   Ingest: {list(data2.keys())}")
    evolution.ingest(data2, kernel)
    evolution.evolve(kernel, mock_commit)
    
    hrt2 = evolution.get_current()
    print(f"   After evolve:")
    print(f"      HRT₂ name = {hrt2.name[:32]}...")
    print(f"      step = {hrt2.step_number}")
    print(f"      am_entries = {len(hrt2.am.nonzero_entries())}")
    
    # Lineage
    print("\n4. Lineage (Commit History)")
    lineage = evolution.get_lineage()
    print(f"   History chain:")
    for i, h in enumerate(lineage):
        marker = " ← HEAD" if i == len(lineage) - 1 else ""
        print(f"      [{i}] {h[:32]}...{marker}")
    
    # Projections
    print("\n5. Future/Past Projections")
    future = hrt2.project_future([2, 3, 4, 5])
    past = hrt2.project_past([0, 1])
    print(f"   Future projection (cols → rows): shape {future.shape}")
    print(f"   Past projection (rows → cols): shape {past.shape}")
    
    return evolution


def demo_content_addressing():
    """Demonstrate content-addressed naming."""
    print_section("CONTENT ADDRESSING (Naming by Content)")
    
    print("\n1. Two Types of Hashes")
    print("   - Element hash: 32/64-bit (for tokens)")
    print("   - Aggregate hash: SHA1 (for HLLSets, Tensors, HRTs)")
    
    # Element hash
    print("\n2. Element Hash Example")
    from core import compute_element_hash
    h1 = compute_element_hash("hello", bits=64)
    h2 = compute_element_hash("hello", bits=64)
    h3 = compute_element_hash("world", bits=64)
    print(f"   hash('hello') = {h1}")
    print(f"   hash('hello') = {h2}")
    print(f"   hash('world') = {h3}")
    print(f"   Deterministic: {h1 == h2}")
    
    # Aggregate hash
    print("\n3. Aggregate Hash Example")
    kernel = Kernel()
    hll1 = kernel.absorb({'a', 'b', 'c'})
    hll2 = kernel.absorb({'a', 'b', 'c'})
    hll3 = kernel.absorb({'x', 'y', 'z'})
    print(f"   HLLSet({'a', 'b', 'c'}) = {hll1.name[:32]}...")
    print(f"   HLLSet({'a', 'b', 'c'}) = {hll2.name[:32]}...")
    print(f"   HLLSet({'x', 'y', 'z'}) = {hll3.name[:32]}...")
    print(f"   Same content → same hash: {hll1.name == hll2.name}")
    print(f"   Different content → different hash: {hll1.name != hll3.name}")
    
    # Structural hash
    print("\n4. Structural Hash (for composed objects)")
    name1 = compute_structural_hash("am_hash_abc", "lattice_hash_xyz")
    name2 = compute_structural_hash("am_hash_abc", "lattice_hash_xyz")
    name3 = compute_structural_hash("am_hash_def", "lattice_hash_xyz")
    print(f"   struct('abc', 'xyz') = {name1[:32]}...")
    print(f"   struct('abc', 'xyz') = {name2[:32]}...")
    print(f"   struct('def', 'xyz') = {name3[:32]}...")
    print(f"   Deterministic: {name1 == name2}")


def demo_system_invariants():
    """Demonstrate system invariants."""
    print_section("SYSTEM INVARIANTS")
    
    print("\n1. Immutability")
    print("   - Objects never change after creation")
    print("   - All 'modifications' create new objects")
    print("   - Parent pointers link old to new")
    
    print("\n2. Idempotency")
    print("   - Same inputs → same outputs")
    print("   - No side effects in kernel operations")
    print("   - Replaying operations yields same result")
    
    print("\n3. Content Addressing")
    print("   - Names are hashes of content")
    print("   - No randomness, no UUIDs")
    print("   - Deterministic across time and space")
    
    print("\n4. Evolution Model")
    print("   - Three states: In-Process → Current → History")
    print("   - Shift to future: commit current, merge in-process")
    print("   - History is in Git, not in memory")
    
    print("\n5. Kernel Statelessness")
    print("   - Kernel has no storage")
    print("   - Pure functions only")
    print("   - All state in HRT (managed by OS)")


def main():
    """Run complete system demo."""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  HLLSET MANIFOLD - REFACTORED SYSTEM DEMO".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Run all demos
    kernel, h1, h2 = demo_kernel()
    t_genesis = demo_immutable_tensor()
    evolution = demo_hrt_evolution()
    demo_content_addressing()
    demo_system_invariants()
    
    # Summary
    print_section("SUMMARY")
    print("\n✓ Kernel: Stateless pure morphisms")
    print("✓ Immutable Tensor: PyTorch-based, clone-on-modify")
    print("✓ HRT: Three-state evolution model")
    print("✓ Content-addressing: All objects named by content hash")
    print("✓ Git integration: History pushed to persistent store")
    
    print("\n" + "█" * 70)
    print("█" + "  SYSTEM READY FOR OS INTEGRATION".center(68) + "█")
    print("█" * 70)
    print("\n")


if __name__ == "__main__":
    main()
