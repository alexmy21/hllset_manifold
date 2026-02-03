"""
HLLSet Manifold Core Module

This module provides the core components of the HLLSet Manifold system:
- hllset: HLLSet class with direct Julia integration
- immutable_tensor: Generic immutable tensor foundation with PyTorch backend
- kernel: Stateless transformation engine (pure morphisms)
- hrt: Hash Relational Tensor with three-state evolution model

Architecture:
1. HLLSet: Named, immutable probabilistic set (direct Julia interface)
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
    
    # HRT
    'HRT',
    'HRTConfig',
    'HRTEvolution',
    'HRTEvolutionTriple',
    'AdjacencyMatrix',
    'HLLSetLattice',
    'BasicHLLSet',
    'Cover',
]

__version__ = "0.2.0"
