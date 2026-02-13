# File: core/entanglement.py
"""
Entanglement for Manifold OS - Lattice-Based Implementation

================================================================================
IMPORTANT: HLLSets are NOT sets containing tokens!
================================================================================

HLLSets are probabilistic register structures ("anti-sets") that:
- ABSORB tokens (hash them into registers)
- DO NOT STORE tokens (only register states remain)
- BEHAVE LIKE sets (union, intersection, cardinality estimation)
- ARE NOT sets (no element retrieval, no membership test)

================================================================================
TWO-LAYER ARCHITECTURE: This module operates at the LATTICE LAYER
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ ENTANGLEMENT IS BETWEEN LATTICES, NOT HLLSets!                              │
│                                                                             │
│ Layer 1 (HLLSet/Register): Individual register arrays, cardinality est.    │
│ Layer 2 (Lattice/Structure): Graph topology, degree distributions ← HERE   │
│                                                                             │
│ This module implements Layer 2 - TRUE ENTANGLEMENT                          │
│ Individual HLLSets are IRRELEVANT - only STRUCTURE matters                  │
│ Two lattices can be entangled even from completely different inputs         │
└─────────────────────────────────────────────────────────────────────────────┘

Based on "The Trinity of Emergence" from dao_manifold.pdf:
- Entanglement as Measurement: M(L1, L2) = structural similarity of LATTICES
- Lattice W is prior, entanglement is map W(1) → W(2)
- Endomorphism W → W with finite lattices
- Nodes connected by similar degrees (cardinality/topology)
- Maximizes structural pairs between lattices

Key Principles:
- Consistency prior to precision
- Lattice W guarantees finiteness
- Entanglement morphism connects similar-DEGREE nodes (not similar registers)
- Maximizes matching pairs between lattice STRUCTURES

Fundamental Consistency Criterion (Categorical):
    For any a ≠ b in L₁, the mapping m(a) ≉ m(b) in L₂
    
    This preserves distinctness (approximate injectivity).
    Due to hash collisions, we use approximate measures (≉) instead of strict equality.
    
    Structural measures used:
    - Degree similarity (estimated cardinality-based)
    - Graph topology correlation
    - Register pattern overlap (secondary, for refinement)
    
    The core requirement: mapping should preserve STRUCTURAL TOPOLOGY
    by maintaining degree patterns between lattices.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .kernel import HLLSet, Kernel
from .hrt import HLLSetLattice, BasicHLLSet, HRT


# =============================================================================
# SECTION 1: Entanglement as Measurement
# =============================================================================

@dataclass(frozen=True)
class EntanglementMeasurement:
    """
    Measurement of entanglement between two basic HLLSets.
    
    M(w1, w2) = structural similarity based on:
    - Degree similarity (cardinality match)
    - Register overlap (structural similarity)
    
    This is the M: S × S → R function from the paper.
    
    IMPLEMENTATION NOTE:
    This uses degree + register similarity as the measure.
    This is ONE valid implementation of the consistency criterion:
        ∀ a≠b in L₁: m(a) ≉ m(b) in L₂
    
    Alternative measures could be used (custom metrics, domain-specific
    similarity functions, etc.) as long as they preserve distinctness
    and structural topology.
    """
    source_hash: str          # Hash of source basic HLLSet
    target_hash: str          # Hash of target basic HLLSet
    
    # Measurements (implementation-specific)
    degree_similarity: float  # Based on cardinality match
    register_similarity: float  # Based on HLLSet similarity
    
    @property
    def strength(self) -> float:
        """
        Overall entanglement strength.
        Weighted combination of degree and register similarity.
        Consistency (degree) weighted more than precision (registers).
        """
        # Consistency (degree) > Precision (registers)
        return 0.6 * self.degree_similarity + 0.4 * self.register_similarity
    
    def __repr__(self) -> str:
        return f"EM({self.source_hash[:8]}↔{self.target_hash[:8]}, s={self.strength:.3f})"


# =============================================================================
# SECTION 2: Entanglement Morphism (Lattice Map)
# =============================================================================

@dataclass(frozen=True)
class EntanglementMorphism:
    """
    Entanglement morphism: map between two lattices W(1) → W(2).
    
    Unlike traditional morphisms (functions), this is a relational map:
    - Each node in W(1) can map to multiple nodes in W(2)
    - Mapping based on similarity measures that preserve structure
    - Maximizes total matching pairs
    
    FUNDAMENTAL PROPERTY (Categorical Consistency):
        ∀ a≠b in L₁: m(a) ≉ m(b) in L₂
        
    This ensures the mapping preserves distinctness (approximate injectivity).
    The specific similarity measure (BSS, degree, custom) is implementation-dependent,
    but all must satisfy this consistency criterion.
    
    This implements the paper's insight:
    "Lattice would be prior and we will implement entanglement morphism 
    as map from one lattice to another."
    
    Mutual exclusivity of HLLSets is NOT required—entanglement is about
    preserving structural topology, not enforcing set disjointness.
    """
    source_lattice_hash: str
    target_lattice_hash: str
    
    # Mapping: source_index → list of (target_index, measurement)
    mapping: Dict[int, List[Tuple[int, EntanglementMeasurement]]] = field(default_factory=dict)
    
    # Statistics
    total_pairs: int = 0
    max_degree_diff: float = 0.0
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        from .kernel import compute_sha1
        data = f"{self.source_lattice_hash}:{self.target_lattice_hash}:{self.total_pairs}"
        return compute_sha1(data)
    
    def get_targets(self, source_idx: int) -> List[Tuple[int, float]]:
        """Get target indices and strengths for source index."""
        if source_idx not in self.mapping:
            return []
        return [(t, m.strength) for t, m in self.mapping[source_idx]]
    
    def get_max_target(self, source_idx: int) -> Optional[Tuple[int, float]]:
        """Get strongest entangled target for source index."""
        targets = self.get_targets(source_idx)
        if not targets:
            return None
        return max(targets, key=lambda x: x[1])
    
    def __repr__(self) -> str:
        return f"Morph({self.source_lattice_hash[:8]}→{self.target_lattice_hash[:8]}, {self.total_pairs} pairs)"


# =============================================================================
# SECTION 3: Entanglement Engine
# =============================================================================

class EntanglementEngine:
    """
    Engine for computing entanglement between lattices.
    
    Implements the degree-based matching algorithm:
    1. Extract degree (cardinality) of each node in both lattices
    2. Sort nodes by degree
    3. Match nodes with similar degrees
    4. Maximize total number of valid pairs
    
    This is the W(1) → W(2) map from the paper.
    """
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.degree_tolerance: float = 0.2  # 20% tolerance for degree matching
        self.min_similarity: float = 0.1    # Minimum similarity threshold
    
    def extract_degrees(self, lattice: HLLSetLattice) -> Dict[int, float]:
        """
        Extract degree (cardinality) of each basic HLLSet in lattice.
        
        Returns: {index -> cardinality}
        """
        degrees = {}
        
        # Row basic HLLSets
        for basic in lattice.row_basic:
            degrees[basic.index] = basic.hllset.cardinality()
        
        # Column basic HLLSets (offset by dimension to distinguish)
        dim = lattice.config.dimension
        for basic in lattice.col_basic:
            degrees[basic.index + dim] = basic.hllset.cardinality()
        
        return degrees
    
    def compute_degree_similarity(self, deg1: float, deg2: float) -> float:
        """
        Compute similarity between two degrees.
        
        Uses harmonic mean approach:
        - 1.0 if degrees are identical
        - Decreases as difference increases
        - Tolerance controlled by degree_tolerance
        """
        if deg1 == 0 and deg2 == 0:
            return 1.0
        if deg1 == 0 or deg2 == 0:
            return 0.0
        
        # Relative difference
        max_deg = max(deg1, deg2)
        min_deg = min(deg1, deg2)
        
        if max_deg == 0:
            return 1.0
        
        ratio = min_deg / max_deg
        
        # Similarity decreases as ratio deviates from 1
        if ratio >= (1 - self.degree_tolerance):
            return ratio
        return 0.0
    
    def compute_register_similarity(self, basic1: BasicHLLSet, basic2: BasicHLLSet) -> float:
        """Compute HLLSet structural similarity."""
        return basic1.hllset.similarity(basic2.hllset)
    
    def compute_entanglement(self, 
                            source_lattice: HLLSetLattice,
                            target_lattice: HLLSetLattice,
                            source_hash: str,
                            target_hash: str) -> EntanglementMorphism:
        """
        Compute entanglement morphism between two lattices.
        
        Algorithm:
        1. Extract degrees from both lattices
        2. For each source node, find target nodes with similar degree
        3. Compute full measurement (degree + register similarity)
        4. Maximize pairs by selecting strongest matches
        """
        morphism = EntanglementMorphism(
            source_lattice_hash=source_hash,
            target_lattice_hash=target_hash
        )
        
        dim = source_lattice.config.dimension
        
        # Collect all basic HLLSets with their global indices
        source_nodes = []
        target_nodes = []
        
        # Source nodes
        for basic in source_lattice.row_basic:
            source_nodes.append((basic.index, basic, 'row'))
        for basic in source_lattice.col_basic:
            source_nodes.append((basic.index + dim, basic, 'col'))
        
        # Target nodes
        for basic in target_lattice.row_basic:
            target_nodes.append((basic.index, basic, 'row'))
        for basic in target_lattice.col_basic:
            target_nodes.append((basic.index + dim, basic, 'col'))
        
        # Compute pairwise entanglements
        total_pairs = 0
        max_degree_diff = 0.0
        
        for src_idx, src_basic, src_type in source_nodes:
            src_card = src_basic.hllset.cardinality()
            
            matches = []
            
            for tgt_idx, tgt_basic, tgt_type in target_nodes:
                tgt_card = tgt_basic.hllset.cardinality()
                
                # Compute degree similarity
                deg_sim = self.compute_degree_similarity(src_card, tgt_card)
                
                if deg_sim > 0:  # Within tolerance
                    # Compute register similarity
                    reg_sim = self.compute_register_similarity(src_basic, tgt_basic)
                    
                    # Create measurement
                    measurement = EntanglementMeasurement(
                        source_hash=src_basic.hllset.name,
                        target_hash=tgt_basic.hllset.name,
                        degree_similarity=deg_sim,
                        register_similarity=reg_sim
                    )
                    
                    if measurement.strength >= self.min_similarity:
                        matches.append((tgt_idx % dim, measurement))  # Use local index
                        total_pairs += 1
                        
                        # Track max degree difference
                        degree_diff = abs(src_card - tgt_card)
                        max_degree_diff = max(max_degree_diff, degree_diff)
            
            # Sort by strength and keep top matches
            matches.sort(key=lambda x: x[1].strength, reverse=True)
            if matches:
                morphism.mapping[src_idx % dim] = matches[:5]  # Keep top 5
        
        # Update statistics
        object.__setattr__(morphism, 'total_pairs', total_pairs)
        object.__setattr__(morphism, 'max_degree_diff', max_degree_diff)
        
        return morphism
    
    def compute_self_entanglement(self, lattice: HLLSetLattice, lattice_hash: str) -> EntanglementMorphism:
        """
        Compute endomorphism W → W (self-entanglement).
        
        This maps each node to itself and structurally similar nodes
        within the same lattice.
        """
        return self.compute_entanglement(lattice, lattice, lattice_hash, lattice_hash)
    
    def find_maximum_matching(self, 
                             morphism: EntanglementMorphism,
                             one_to_one: bool = True) -> Dict[int, int]:
        """
        Find maximum matching from entanglement mapping.
        
        If one_to_one=True, each target matched to at most one source.
        Returns: {source_idx -> target_idx}
        """
        if not one_to_one:
            # Just take strongest for each source
            return {s: self.get_max_target(s)[0] 
                   for s in morphism.mapping.keys()}
        
        # Greedy matching: sort all pairs by strength, assign greedily
        all_pairs = []
        for src_idx, matches in morphism.mapping.items():
            for tgt_idx, measurement in matches:
                all_pairs.append((measurement.strength, src_idx, tgt_idx))
        
        # Sort by strength (descending)
        all_pairs.sort(reverse=True)
        
        # Greedy assignment
        used_targets = set()
        matching = {}
        
        for strength, src_idx, tgt_idx in all_pairs:
            if tgt_idx not in used_targets and src_idx not in matching:
                matching[src_idx] = tgt_idx
                used_targets.add(tgt_idx)
        
        return matching


# =============================================================================
# SECTION 4: Entanglement-Preserving Operations
# =============================================================================

class EntanglementPreservingOps:
    """
    Operations that preserve entanglement structure.
    
    Based on paper's compatibility condition:
    M(Φ(s), Φ(s')) = M(s, s')
    
    Operations are idempotent and entanglement-preserving.
    """
    
    def __init__(self, engine: EntanglementEngine):
        self.engine = engine
    
    def merge_lattices_with_entanglement(self,
                                        lattice1: HLLSetLattice,
                                        lattice2: HLLSetLattice,
                                        morphism: EntanglementMorphism,
                                        kernel: Kernel) -> HLLSetLattice:
        """
        Merge two lattices preserving entanglement structure.
        
        When merging lattices W(1) and W(2):
        1. Union corresponding basic HLLSets (via entanglement mapping)
        2. Preserve measurement relationships
        3. Result maintains entanglement invariants
        """
        from .hrt import HRTConfig
        
        # Create new lattice
        config = lattice1.config
        merged = HLLSetLattice(config=config)
        
        # Get maximum matching
        matching = self.engine.find_maximum_matching(morphism, one_to_one=True)
        
        # Merge matched pairs
        for src_idx, tgt_idx in matching.items():
            basic1 = lattice1.row_basic[src_idx] if src_idx < len(lattice1.row_basic) else None
            basic2 = lattice2.row_basic[tgt_idx] if tgt_idx < len(lattice2.row_basic) else None
            
            if basic1 and basic2:
                # Union the HLLSets
                merged_hll = kernel.union(basic1.hllset, basic2.hllset)
                # Update in merged lattice
                if src_idx < len(merged.row_basic):
                    merged.row_basic[src_idx] = BasicHLLSet(
                        index=src_idx,
                        is_row=True,
                        hllset=merged_hll,
                        config=config
                    )
        
        return merged


# =============================================================================
# SECTION 5: Integration with MOS
# =============================================================================

def compute_hrt_entanglement(hrt1: HRT, hrt2: HRT, kernel: Kernel) -> EntanglementMorphism:
    """
    Compute entanglement between two HRTs.
    
    This is the high-level API for MOS integration.
    """
    engine = EntanglementEngine(kernel)
    
    lattice1 = hrt1.get_lattice()
    lattice2 = hrt2.get_lattice()
    
    if not lattice1 or not lattice2:
        raise ValueError("HRTs must have loaded lattices")
    
    return engine.compute_entanglement(
        lattice1, lattice2,
        hrt1.name, hrt2.name
    )


def compute_entanglement_network(hrts: List[HRT], kernel: Kernel) -> Dict[Tuple[str, str], EntanglementMorphism]:
    """
    Compute pairwise entanglements among multiple HRTs.
    
    Returns a network of entanglement morphisms.
    """
    network = {}
    engine = EntanglementEngine(kernel)
    
    for i, hrt1 in enumerate(hrts):
        for j, hrt2 in enumerate(hrts[i+1:], i+1):
            if hrt1.get_lattice() and hrt2.get_lattice():
                morphism = engine.compute_entanglement(
                    hrt1.get_lattice(),
                    hrt2.get_lattice(),
                    hrt1.name,
                    hrt2.name
                )
                network[(hrt1.name, hrt2.name)] = morphism
    
    return network


# =============================================================================
# SECTION 6: Example Usage
# =============================================================================

def main():
    """Example entanglement computation."""
    from .hrt import HRTConfig, HRTFactory
    
    print("="*70)
    print("ENTANGLEMENT - Lattice-Based Implementation")
    print("="*70)
    
    # Create kernel
    kernel = Kernel()
    
    # Create two HRTs
    config = HRTConfig(p_bits=8, h_bits=16)
    factory = HRTFactory(config)
    
    data1 = {"sensor_a": {"x", "y", "z", "w"}, "sensor_b": {"a", "b", "c"}}
    data2 = {"sensor_a": {"x", "y", "z"}, "sensor_c": {"d", "e", "f", "g"}}
    
    hrt1 = factory.create_from_perceptrons(data1, kernel)
    hrt2 = factory.create_from_perceptrons(data2, kernel)
    
    print(f"\nHRT 1: {hrt1.name[:20]}...")
    print(f"HRT 2: {hrt2.name[:20]}...")
    
    # Compute entanglement
    print("\nComputing entanglement morphism...")
    morphism = compute_hrt_entanglement(hrt1, hrt2, kernel)
    
    print(f"\nEntanglement Morphism:")
    print(f"  {morphism}")
    print(f"  Total pairs: {morphism.total_pairs}")
    print(f"  Max degree diff: {morphism.max_degree_diff:.2f}")
    
    # Show some mappings
    print(f"\nSample mappings:")
    for src_idx in list(morphism.mapping.keys())[:3]:
        targets = morphism.get_targets(src_idx)
        if targets:
            print(f"  Source {src_idx} → {len(targets)} targets")
            for tgt, strength in targets[:2]:
                print(f"    → Target {tgt}, strength={strength:.3f}")
    
    # Maximum matching
    engine = EntanglementEngine(kernel)
    matching = engine.find_maximum_matching(morphism, one_to_one=True)
    print(f"\nMaximum matching: {len(matching)} pairs")
    
    print("\n" + "="*70)
    print("Entanglement computation complete")
    print("="*70)
    
    return morphism


if __name__ == "__main__":
    main()
