"""
HLLSet Manifold Core Module

This module provides the core components of the HLLSet Manifold system:
- hllset: HLLSet class with C/Cython backend
- immutable_tensor: Generic immutable tensor foundation with PyTorch backend
- kernel: Stateless transformation engine (pure morphisms)
- hrt: Hash Relational Tensor with three-state evolution model

================================================================================
IMPORTANT: HLLSets are NOT sets containing tokens!
================================================================================

HLLSets are probabilistic register structures ("anti-sets") that:
- ABSORB tokens (hash them into registers)
- DO NOT STORE tokens (only register states remain)
- BEHAVE LIKE sets (union, intersection, cardinality estimation)
- ARE NOT sets (no element retrieval, no membership test)

================================================================================
TWO-LAYER ARCHITECTURE: HLLSets vs Lattices
================================================================================

LAYER 1: HLLSet (Register Layer)
- Works with individual HLLSets (register arrays)
- Compares: Register states, estimated cardinalities
- Morphism: Register-level comparison (NOT entanglement)

LAYER 2: Lattice (Structure Layer) - TRUE ENTANGLEMENT
- Works with HLLSetLattice objects
- Compares STRUCTURE (degree distributions, graph topology)
- LatticeMorphism: Structure-level comparison
- Individual HLLSets are IRRELEVANT - only topology matters
- Two lattices can be entangled even from completely different inputs!

Architecture:
1. HLLSet: Named, immutable register array (C/Cython backend)
2. Kernel: Pure operations (absorb, union, intersection, difference)
3. HRT: Operation data structure (AM, Lattice, Covers) - immutable
4. OS: Reality interface (evolution orchestration, persistent storage)

Evolution Model (Three-State):
- In-Process: Newly ingested data
- Current: Active state in memory
- History: Committed states in Git/persistent store

Evolution Step (Shift to Future):
    In-Process → Current → History
       ↑                        
   ingest → merge → commit
"""

from .hllset import HLLSet, compute_sha1

from .immutable_tensor import (
    ImmutableTensor,
    TensorEvolution,
    TensorEvolutionTriple,
    compute_element_hash,
    compute_aggregate_hash,
    compute_structural_hash,
)

from .kernel import (
    Kernel,
    Operation,
    record_operation,
    Morphism,
    LatticeMorphism,
    SingularityReport,
)

from .hrt import (
    HRT,
    HRTConfig,
    HRTEvolution,
    HRTEvolutionTriple,
    AdjacencyMatrix,
    HLLSetLattice,
    BasicHLLSet,
    Cover,
)

from .algebra import (
    HLLCatalog,
    RelAlgebra,
    QueryResult,
    ColumnProfile,
    TableProfile,
)

from .entanglement import (
    # Fragment-based (structural)
    EntanglementFragment,
    EntanglementSubgraph,
    ExtendedEntanglement,
    CommonSubgraphExtractor,
    extract_entanglement,
    # Morphism-based
    EntanglementMeasurement,
    EntanglementMorphism,
    EntanglementEngine,
    # N-Edge based (stochastic) - NEW
    EdgeSignature,
    NEdgePath,
    EdgeLUT,
    NEdgeEntanglement,
    NEdgeExtractor,
    compute_nedge_entanglement,
)

__all__ = [
    # HLLSet
    'HLLSet',
    'compute_sha1',
    
    # Immutable Tensor
    'ImmutableTensor',
    'TensorEvolution',
    'TensorEvolutionTriple',
    'compute_element_hash',
    'compute_aggregate_hash',
    'compute_structural_hash',
    
    # Kernel
    'Kernel',
    'Operation',
    'record_operation',
    'Morphism',
    'LatticeMorphism',
    'SingularityReport',
    
    # HRT
    'HRT',
    'HRTConfig',
    'HRTEvolution',
    'HRTEvolutionTriple',
    'AdjacencyMatrix',
    'HLLSetLattice',
    'BasicHLLSet',
    'Cover',
    
    # Algebra
    'HLLCatalog',
    'RelAlgebra',
    'QueryResult',
    'ColumnProfile',
    'TableProfile',
    
    # Entanglement (Common Subgraph Extraction)
    'EntanglementFragment',
    'EntanglementSubgraph',
    'ExtendedEntanglement',
    'CommonSubgraphExtractor',
    'extract_entanglement',
    'EntanglementMeasurement',
    'EntanglementMorphism',
    'EntanglementEngine',
    # N-Edge Entanglement (Stochastic)
    'EdgeSignature',
    'NEdgePath',
    'EdgeLUT',
    'NEdgeEntanglement',
    'NEdgeExtractor',
    'compute_nedge_entanglement',
]

__version__ = "0.2.0"
