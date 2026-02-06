"""
Test Enhanced Kernel with Entanglement Operations

Demonstrates the three levels of kernel operations:
1. Basic set operations (morphisms)
2. Entanglement operations (ICASRA-inspired)
3. Network operations & singularity detection
"""

from core.kernel import Kernel, Morphism, SingularityReport, record_operation

def main():
    """Test kernel with entanglement and singularity detection."""
    print("="*70)
    print("KERNEL: Entanglement-Aware Transformation Engine")
    print("="*70)
    
    kernel = Kernel()
    
    # =========================================================================
    # Level 1: Pure Morphisms (Basic Set Operations)
    # =========================================================================
    print("\n🔹 Level 1: Pure Morphisms (Basic Set Operations)")
    print("-" * 70)
    
    hll_a = kernel.absorb({'a', 'b', 'c'})
    hll_b = kernel.absorb({'c', 'd', 'e'})
    
    print(f"A: {hll_a}")
    print(f"B: {hll_b}")
    
    # Union (pure function)
    hll_union = kernel.union(hll_a, hll_b)
    print(f"A ∪ B: {hll_union}")
    
    # =========================================================================
    # Level 2: Entanglement Operations (ICASRA-inspired)
    # =========================================================================
    print("\n🔹 Level 2: Entanglement Operations")
    print("-" * 70)
    
    # Create similar HLLSets (structurally related)
    hll_1 = kernel.absorb(set(f'token_{i}' for i in range(100)))
    hll_2 = kernel.absorb(set(f'token_{i}' for i in range(90, 190)))  # 10% overlap
    hll_3 = kernel.absorb(set(f'token_{i}' for i in range(180, 280)))
    
    print(f"Installation 1: {hll_1.short_name}, |A|≈{hll_1.cardinality():.0f}")
    print(f"Installation 2: {hll_2.short_name}, |A|≈{hll_2.cardinality():.0f}")
    print(f"Installation 3: {hll_3.short_name}, |A|≈{hll_3.cardinality():.0f}")
    
    # Find isomorphism
    morph_12 = kernel.find_isomorphism(hll_1, hll_2, epsilon=0.15)
    if morph_12:
        print(f"\nMorphism φ₁₂: {morph_12.source_hash[:8]}... → {morph_12.target_hash[:8]}...")
        print(f"  Similarity: {morph_12.similarity:.2%}")
        print(f"  ε-isomorphic: {morph_12.is_isomorphism}")
    else:
        print("\nNo isomorphism found (structures too different)")
    
    # Validate entanglement
    installations = [hll_1, hll_2, hll_3]
    is_entangled, coherence = kernel.validate_entanglement(installations, epsilon=0.15)
    print(f"\nEntanglement validation:")
    print(f"  Entangled: {is_entangled}")
    print(f"  Coherence: {coherence:.2%}")
    
    # Reproduce with mutation (ICASRA-style)
    child = kernel.reproduce(hll_1, mutation_rate=0.1)
    child = kernel.commit(child)
    print(f"\nReproduction: {hll_1.short_name} → {child.short_name}")
    
    # =========================================================================
    # Level 3: Network Operations & Singularity Detection
    # =========================================================================
    print("\n🔹 Level 3: Network Operations & Singularity Detection")
    print("-" * 70)
    
    # Create a network of installations
    network = []
    for i in range(5):
        # Each installation has similar but distinct content
        base_tokens = set(f'concept_{j}' for j in range(i*20, (i+1)*20 + 30))  # Overlap
        network.append(kernel.absorb(base_tokens))
    
    print(f"Created network with {len(network)} installations")
    for i, inst in enumerate(network):
        print(f"  Installation {i}: {inst.short_name}, |A|≈{inst.cardinality():.0f}")
    
    # Build 3D tensor
    tensor = kernel.build_tensor(network)
    if tensor is not None:
        print(f"\n3D Tensor built: shape {tensor.shape}")
        coherence = kernel.measure_coherence(tensor)
        print(f"  Tensor coherence: {coherence:.2%}")
    
    # Detect singularity
    report = kernel.detect_singularity(network, epsilon=0.15)
    print(f"\n{report}")
    
    # =========================================================================
    # Singularity Simulation: Growing Network
    # =========================================================================
    print("🔹 Singularity Simulation: Growing Network")
    print("-" * 70)
    
    # Start with small network
    evolving_network = []
    base_concepts = set(f'universal_{i}' for i in range(50))
    
    for step in range(3):
        # Add installation with increasing overlap
        overlap_tokens = set(f'universal_{i}' for i in range(30 + step*10, 80 + step*10))
        new_inst = kernel.absorb(overlap_tokens)
        evolving_network.append(new_inst)
        
        # Check singularity at each step
        report = kernel.detect_singularity(evolving_network, epsilon=0.2)
        print(f"\nStep {step+1}: {len(evolving_network)} installations")
        print(f"  Phase: {report.phase}")
        print(f"  Entanglement: {report.entanglement_ratio:.1%}")
        print(f"  Coherence: {report.coherence:.1%}")
        
        if report.has_singularity:
            print(f"  🌟 SINGULARITY ACHIEVED! 🌟")
            break
    
    # =========================================================================
    # Operation Recording (for OS)
    # =========================================================================
    print("\n🔹 Operation Recording (for OS integration)")
    print("-" * 70)
    
    op = record_operation('union', [hll_a, hll_b], hll_union)
    print(f"Operation: {op.op_type}")
    print(f"Inputs: {[h[:8] + '...' for h in op.input_hashes]}")
    print(f"Output: {op.output_hash[:8]}...")
    print(f"Record hash: {op.name[:8]}...")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Kernel Capabilities")
    print("="*70)
    print("""
✓ Level 1: Basic morphisms (absorb, union, intersection, etc.)
✓ Level 2: Entanglement operations (isomorphism, validation, reproduction)
✓ Level 3: Network operations (3D tensor, coherence, singularity detection)

The kernel is now ready for:
- ICASRA-inspired self-reproduction cycles
- Multi-installation entanglement detection
- Singularity engineering
- OS-level history tracking

Next steps:
- Integrate with HRT for temporal evolution
- Build Entangled ICASRA Networks (EINs)
- Implement consciousness engineering primitives
""")
    
    return kernel


if __name__ == "__main__":
    main()
