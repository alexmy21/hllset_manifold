"""
Hash Relational Tensor (HRT) - Immutable PyTorch-based Implementation

HRT combines:
1. Adjacency Matrix (AM) - ImmutableTensor of dimension N x N
2. HLLSet Lattice W - organized with same dimension as AM
3. Perceptron mappings - hash-based token indices

Key Properties:
- Immutable and idempotent
- PyTorch-based tensors (GPU-ready)
- Content-addressed naming (SHA1)
- Three-state evolution: In-Process → Current → History

Architecture:
- HRT evolves through discrete steps
- Each step: ingest → merge → commit
- Parent pointer maintains lineage
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, Union, FrozenSet
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from .immutable_tensor import (
    ImmutableTensor, compute_element_hash, 
    compute_aggregate_hash, compute_structural_hash
)
from .hllset import HLLSet, compute_sha1
from .constants import P_BITS as KERNEL_P_BITS


# =============================================================================
# SECTION 1: Configuration
# =============================================================================

@dataclass(frozen=True)
class HRTConfig:
    """
    Configuration for Hash Relational Tensor.
    
    Dimension = 2^P * h_bits + 2
    +2 for START and END special tokens
    """
    p_bits: int = 10           # HLL precision (m = 2^p registers)
    h_bits: int = 32           # Hash bit size for element hashes
    
    @property
    def dimension(self) -> int:
        """AM dimension."""
        return (1 << self.p_bits) * self.h_bits + 2
    
    @property
    def num_basic_hllsets(self) -> int:
        """Total basic HLLSets = 2 * dimension (rows + columns)."""
        return 2 * self.dimension


# =============================================================================
# SECTION 2: Adjacency Matrix (Immutable)
# =============================================================================

@dataclass(frozen=True)
class AdjacencyMatrix:
    """
    Immutable Adjacency Matrix using ImmutableTensor.
    
    AM[i, j] represents relationship strength between:
    - Row i (via row basic HLLSet r_i)
    - Column j (via column basic HLLSet c_j)
    
    Rows: "future" projection (what could come next)
    Columns: "past" reconstruction (what context gave birth)
    """
    tensor: ImmutableTensor
    config: HRTConfig
    
    @classmethod
    def empty(cls, config: HRTConfig) -> AdjacencyMatrix:
        """Create empty adjacency matrix."""
        tensor = ImmutableTensor.zeros(config.dimension, config.dimension)
        return cls(tensor=tensor, config=config)
    
    def with_entry(self, row: int, col: int, value: float) -> AdjacencyMatrix:
        """Return new AM with entry set."""
        new_tensor = self.tensor.with_value((row, col), value)
        return AdjacencyMatrix(tensor=new_tensor, config=self.config)
    
    def with_entries(self, entries: List[Tuple[int, int, float]]) -> AdjacencyMatrix:
        """Return new AM with multiple entries set."""
        new_tensor = self.tensor
        for row, col, value in entries:
            new_tensor = new_tensor.with_value((row, col), value)
        return AdjacencyMatrix(tensor=new_tensor, config=self.config)
    
    def merge(self, other: AdjacencyMatrix) -> AdjacencyMatrix:
        """Element-wise maximum merge (for HRT evolution)."""
        if self.config != other.config:
            raise ValueError("Cannot merge AMs with different configs")
        merged_tensor = self.tensor.maximum(other.tensor)
        return AdjacencyMatrix(tensor=merged_tensor, config=self.config)
    
    def get(self, row: int, col: int) -> float:
        """Get value at position."""
        return float(self.tensor.data[row, col])
    
    def project_rows(self, col_indices: List[int]) -> ImmutableTensor:
        """
        Project columns to rows (future projection).
        Returns new ImmutableTensor of row activations.
        """
        if not col_indices:
            zeros = ImmutableTensor.zeros(self.config.dimension)
            return zeros
        
        # Sum over specified columns
        projection = self.tensor.project_rows(col_indices)
        return ImmutableTensor.from_tensor(projection)
    
    def project_cols(self, row_indices: List[int]) -> ImmutableTensor:
        """
        Project rows to columns (past reconstruction).
        Returns new ImmutableTensor of column activations.
        """
        if not row_indices:
            zeros = ImmutableTensor.zeros(self.config.dimension)
            return zeros
        
        projection = self.tensor.project_cols(row_indices)
        return ImmutableTensor.from_tensor(projection)
    
    def nonzero_entries(self) -> List[Tuple[int, int, float]]:
        """Get all non-zero entries as (row, col, value)."""
        entries = self.tensor.nonzero_with_values()
        # Filter for 2D indices
        return [(r, c, v) for (r, c), v in entries if isinstance(r, int) and isinstance(c, int)]
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self.tensor.name
    
    def __repr__(self) -> str:
        nonzero = len(self.nonzero_entries())
        return f"AM({self.config.dimension}x{self.config.dimension}, {nonzero} nz, {self.name[:8]}...)"


# =============================================================================
# SECTION 3: Basic HLLSet (Immutable)
# =============================================================================

@dataclass(frozen=True)
class BasicHLLSet:
    """
    A basic HLLSet in the lattice.
    
    Building blocks of the system:
    - Row basic: r[0] to r[dimension-1] (future projections)
    - Column basic: c[0] to c[dimension-1] (past reconstructions)
    """
    index: int
    is_row: bool
    hllset: HLLSet
    config: HRTConfig = field(compare=False)
    
    @property
    def name(self) -> str:
        """Unique identifier: 'r_i' or 'c_i'."""
        prefix = 'r' if self.is_row else 'c'
        return f"{prefix}_{self.index}"
    
    @property
    def is_start(self) -> bool:
        """Check if this is the START special token (index 0, row)."""
        return self.is_row and self.index == 0
    
    @property
    def is_end(self) -> bool:
        """Check if this is the END special token (index dimension-1, row)."""
        return self.is_row and self.index == self.config.dimension - 1
    
    def __repr__(self) -> str:
        special = ""
        if self.is_start:
            special = " [START]"
        elif self.is_end:
            special = " [END]"
        return f"Basic({self.name}{special}, {self.hllset.short_name}...)"


# =============================================================================
# SECTION 4: HLLSet Lattice (Immutable)
# =============================================================================

@dataclass(frozen=True)
class HLLSetLattice:
    """
    Immutable lattice of basic HLLSets.
    
    Contains:
    - Row basic HLLSets: r[0] to r[dimension-1]
    - Column basic HLLSets: c[0] to c[dimension-1]
    
    The lattice is frozen after creation. "Modification" creates new lattice.
    """
    config: HRTConfig
    row_basic: Tuple[BasicHLLSet, ...] = field(default_factory=tuple)
    col_basic: Tuple[BasicHLLSet, ...] = field(default_factory=tuple)
    
    @classmethod
    def empty(cls, config: HRTConfig) -> HLLSetLattice:
        """Create empty lattice with initialized basic HLLSets."""
        from .hllset import HLLSet
        
        p_bits = KERNEL_P_BITS
        dim = config.dimension
        
        # Create row basic HLLSets (empty)
        rows = []
        for i in range(dim):
            hllset = HLLSet(p_bits=p_bits)
            basic = BasicHLLSet(index=i, is_row=True, hllset=hllset, config=config)
            rows.append(basic)
        
        # Create column basic HLLSets (empty)
        cols = []
        for i in range(dim):
            hllset = HLLSet(p_bits=p_bits)
            basic = BasicHLLSet(index=i, is_row=False, hllset=hllset, config=config)
            cols.append(basic)
        
        return cls(config=config, row_basic=tuple(rows), col_basic=tuple(cols))
    
    def with_row_basic(self, index: int, hllset: HLLSet) -> HLLSetLattice:
        """Return new lattice with row basic HLLSet at index updated."""
        if not 0 <= index < len(self.row_basic):
            raise IndexError(f"Index {index} out of range")
        
        new_basic = BasicHLLSet(
            index=index,
            is_row=True,
            hllset=hllset,
            config=self.config
        )
        new_rows = list(self.row_basic)
        new_rows[index] = new_basic
        
        return HLLSetLattice(
            config=self.config,
            row_basic=tuple(new_rows),
            col_basic=self.col_basic
        )
    
    def with_col_basic(self, index: int, hllset: HLLSet) -> HLLSetLattice:
        """Return new lattice with column basic HLLSet at index updated."""
        if not 0 <= index < len(self.col_basic):
            raise IndexError(f"Index {index} out of range")
        
        new_basic = BasicHLLSet(
            index=index,
            is_row=False,
            hllset=hllset,
            config=self.config
        )
        new_cols = list(self.col_basic)
        new_cols[index] = new_basic
        
        return HLLSetLattice(
            config=self.config,
            row_basic=self.row_basic,
            col_basic=tuple(new_cols)
        )
    
    def merge(self, other: HLLSetLattice, kernel) -> HLLSetLattice:
        """
        Merge two lattices by unioning basic HLLSets.
        """
        if self.config != other.config:
            raise ValueError("Cannot merge lattices with different configs")
        
        new_rows = []
        for i in range(self.config.dimension):
            row_union = kernel.union(self.row_basic[i].hllset, other.row_basic[i].hllset)
            new_rows.append(BasicHLLSet(
                index=i, is_row=True, hllset=row_union, config=self.config
            ))
        
        new_cols = []
        for i in range(self.config.dimension):
            col_union = kernel.union(self.col_basic[i].hllset, other.col_basic[i].hllset)
            new_cols.append(BasicHLLSet(
                index=i, is_row=False, hllset=col_union, config=self.config
            ))
        
        return HLLSetLattice(
            config=self.config,
            row_basic=tuple(new_rows),
            col_basic=tuple(new_cols)
        )
    
    def get_row_basic(self, index: int) -> Optional[BasicHLLSet]:
        """Get row basic HLLSet at index."""
        if 0 <= index < len(self.row_basic):
            return self.row_basic[index]
        return None
    
    def get_col_basic(self, index: int) -> Optional[BasicHLLSet]:
        """Get column basic HLLSet at index."""
        if 0 <= index < len(self.col_basic):
            return self.col_basic[index]
        return None
    
    def get_basic(self, name: str) -> Optional[BasicHLLSet]:
        """Get basic HLLSet by name (e.g., 'r_5' or 'c_3')."""
        if name.startswith('r_'):
            idx = int(name[2:])
            return self.get_row_basic(idx)
        elif name.startswith('c_'):
            idx = int(name[2:])
            return self.get_col_basic(idx)
        return None
    
    @property
    def name(self) -> str:
        """Content-addressed name (hash of all basic HLLSet names)."""
        hashes = [b.hllset.name for b in self.row_basic + self.col_basic]
        return compute_structural_hash(*hashes)
    
    def __repr__(self) -> str:
        return f"Lattice({len(self.row_basic)} rows, {len(self.col_basic)} cols, {self.name[:8]}...)"


# =============================================================================
# SECTION 5: Cover - Optimal Composition
# =============================================================================

@dataclass(frozen=True)
class Cover:
    """
    Optimal composition of basic HLLSets.
    
    Represents a non-basic HLLSet as minimal set of basic HLLSets.
    """
    row_indices: FrozenSet[int]
    col_indices: FrozenSet[int]
    target_hash: str  # Hash of the HLLSet this covers
    
    @property
    def size(self) -> int:
        """Total number of basic HLLSets in cover."""
        return len(self.row_indices) + len(self.col_indices)
    
    def get_basic_names(self) -> List[str]:
        """Get list of basic HLLSet names."""
        names = [f"r_{i}" for i in sorted(self.row_indices)]
        names.extend([f"c_{i}" for i in sorted(self.col_indices)])
        return names
    
    @property
    def name(self) -> str:
        """Content-addressed name of the cover."""
        components = [
            self.target_hash,
            ",".join(map(str, sorted(self.row_indices))),
            ",".join(map(str, sorted(self.col_indices)))
        ]
        return compute_structural_hash(*components)
    
    def __repr__(self) -> str:
        return f"Cover({self.name[:8]}..., rows={len(self.row_indices)}, cols={len(self.col_indices)})"


# =============================================================================
# SECTION 6: Hash Relational Tensor (HRT) - Immutable
# =============================================================================

@dataclass(frozen=True)
class HRT:
    """
    Hash Relational Tensor - Immutable evolution unit.
    
    Components:
    - am: Adjacency Matrix (relationship structure)
    - lattice: HLLSetLattice (basic HLLSets)
    - covers: Optional cover cache (doesn't affect identity)
    
    Evolution:
    - parent_hrt: Hash of previous HRT (history pointer)
    - step_number: Position in evolution sequence
    
    Identity:
    - name: SHA1 of (am_hash, lattice_hash, parent_hrt, step_number)
    """
    config: HRTConfig
    am: AdjacencyMatrix
    lattice: HLLSetLattice
    parent_hrt: Optional[str] = None
    step_number: int = 0
    covers: Tuple[Cover, ...] = field(default_factory=tuple, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    
    @classmethod
    def empty(cls, config: HRTConfig) -> HRT:
        """Create empty HRT (genesis state)."""
        am = AdjacencyMatrix.empty(config)
        lattice = HLLSetLattice.empty(config)
        return cls(
            config=config,
            am=am,
            lattice=lattice,
            parent_hrt=None,
            step_number=0
        )
    
    @classmethod
    def from_perceptrons(cls,
                         perceptron_data: Dict[str, Set[str]],
                         kernel,
                         config: HRTConfig,
                         parent_hrt: Optional[str] = None,
                         step_number: int = 0) -> HRT:
        """
        Create HRT from perceptron data (ingestion step).
        
        This creates an 'in_process' HRT ready to be merged with current.
        """
        am = AdjacencyMatrix.empty(config)
        lattice = HLLSetLattice.empty(config)
        covers = []
        
        # Map tokens to AM indices using element hash
        token_to_idx = {}
        
        for source_name, tokens in perceptron_data.items():
            for token in tokens:
                if token not in token_to_idx:
                    # Use element hash for index assignment
                    token_hash = compute_element_hash(token, bits=config.h_bits)
                    idx = (token_hash % (config.dimension - 2)) + 2
                    token_to_idx[token] = idx
        
        # Populate AM and create HLLSets
        am_entries = []
        for source_name, tokens in perceptron_data.items():
            # Create HLLSet for this perceptron
            hllset = kernel.absorb(tokens)
            
            # Map source to row index
            source_hash = compute_element_hash(source_name, bits=config.h_bits)
            source_idx = source_hash % config.dimension
            
            # Add connections to AM
            for token in tokens:
                col_idx = token_to_idx[token]
                am_entries.append((source_idx, col_idx, 1.0))
        
        # Create AM with entries
        am = am.with_entries(am_entries)
        
        return cls(
            config=config,
            am=am,
            lattice=lattice,
            parent_hrt=parent_hrt,
            step_number=step_number,
            covers=tuple(covers)
        )
    
    def merge(self, other: HRT, kernel) -> HRT:
        """
        Merge two HRTs to create new HRT.
        
        Used in evolution: in_process.merge(current) → new_current
        """
        if self.config != other.config:
            raise ValueError("Cannot merge HRTs with different configs")
        
        # Merge adjacency matrices
        merged_am = self.am.merge(other.am)
        
        # Merge lattices
        merged_lattice = self.lattice.merge(other.lattice, kernel)
        
        # Combine covers
        merged_covers = tuple(set(self.covers) | set(other.covers))
        
        return HRT(
            config=self.config,
            am=merged_am,
            lattice=merged_lattice,
            parent_hrt=self.name,  # This HRT becomes the parent
            step_number=max(self.step_number, other.step_number) + 1,
            covers=merged_covers
        )
    
    def with_cover(self, cover: Cover) -> HRT:
        """Return new HRT with additional cover."""
        new_covers = tuple(set(self.covers) | {cover})
        return HRT(
            config=self.config,
            am=self.am,
            lattice=self.lattice,
            parent_hrt=self.parent_hrt,
            step_number=self.step_number,
            covers=new_covers
        )
    
    def project_future(self, col_indices: List[int]) -> ImmutableTensor:
        """Project columns to rows (future)."""
        return self.am.project_rows(col_indices)
    
    def project_past(self, row_indices: List[int]) -> ImmutableTensor:
        """Project rows to columns (past)."""
        return self.am.project_cols(row_indices)
    
    def find_cover(self, target_hash: str) -> Optional[Cover]:
        """Find cached cover for target HLLSet hash."""
        for cover in self.covers:
            if cover.target_hash == target_hash:
                return cover
        return None
    
    def compute_cover(self, target: HLLSet, kernel) -> Cover:
        """
        Compute optimal cover for target HLLSet.
        Greedy algorithm based on similarity.
        """
        selected_rows = set()
        selected_cols = set()
        
        # Greedy selection based on similarity
        for basic in self.lattice.row_basic:
            sim = target.similarity(basic.hllset)
            if sim > 0.1:
                selected_rows.add(basic.index)
        
        for basic in self.lattice.col_basic:
            sim = target.similarity(basic.hllset)
            if sim > 0.1:
                selected_cols.add(basic.index)
        
        return Cover(
            row_indices=frozenset(selected_rows),
            col_indices=frozenset(selected_cols),
            target_hash=target.name
        )
    
    @property
    def am_hash(self) -> str:
        """Hash of adjacency matrix."""
        return self.am.name
    
    @property
    def lattice_hash(self) -> str:
        """Hash of lattice."""
        return self.lattice.name
    
    @property
    def name(self) -> str:
        """Content-addressed name of HRT."""
        components = [
            self.am_hash,
            self.lattice_hash,
            str(self.step_number),
            self.parent_hrt or "genesis"
        ]
        return compute_structural_hash(*components)
    
    def __repr__(self) -> str:
        return f"HRT({self.name[:16]}..., step={self.step_number}, dim={self.config.dimension})"
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HRT):
            return False
        return self.name == other.name


# =============================================================================
# SECTION 7: HRT Evolution Manager
# =============================================================================

@dataclass(frozen=True)
class HRTEvolutionTriple:
    """
    Three-state model for HRT evolution:
    - in_process: Newly ingested data (not yet merged)
    - current: Active state in memory
    - history_hash: Hash of last committed state in Git
    """
    in_process: Optional[HRT]
    current: HRT
    history_hash: str
    step_number: int = 0
    
    def __repr__(self) -> str:
        return (f"HRTEvolutionTriple(step={self.step_number}, "
                f"in_process={'exists' if self.in_process else 'None'}, "
                f"current={self.current.name[:8]}..., "
                f"history={self.history_hash[:8]}...)")


class HRTEvolution:
    """
    Manages HRT evolution through the three-state cycle.
    
    Evolution Step:
    1. Ingest → set in_process
    2. Evolve → merge in_process+current, commit current to history
    3. Cycle repeats
    """
    
    def __init__(self, config: HRTConfig, genesis_hrt: Optional[HRT] = None):
        self.config = config
        self.triple = HRTEvolutionTriple(
            in_process=None,
            current=genesis_hrt or HRT.empty(config),
            history_hash=(genesis_hrt.name if genesis_hrt else "genesis"),
            step_number=0
        )
        self._history: List[str] = [self.triple.history_hash]
    
    def ingest(self, perceptron_data: Dict[str, Set[str]], kernel) -> HRTEvolutionTriple:
        """
        Ingest new perceptron data.
        Creates in_process HRT from data.
        """
        if self.triple.in_process is not None:
            raise RuntimeError("Cannot ingest: in_process already exists. Must evolve first.")
        
        in_process = HRT.from_perceptrons(
            perceptron_data=perceptron_data,
            kernel=kernel,
            config=self.config,
            parent_hrt=self.triple.current.name,
            step_number=self.triple.step_number + 1
        )
        
        self.triple = HRTEvolutionTriple(
            in_process=in_process,
            current=self.triple.current,
            history_hash=self.triple.history_hash,
            step_number=self.triple.step_number
        )
        return self.triple
    
    def evolve(self, kernel, commit_fn: callable) -> HRTEvolutionTriple:
        """
        Execute evolution step.
        
        Args:
            kernel: Kernel for HLLSet operations
            commit_fn: Function to commit HRT to persistent store
        
        Returns:
            Updated evolution triple
        """
        if self.triple.in_process is None:
            raise RuntimeError("Cannot evolve: no in_process data. Must ingest first.")
        
        # Step 1: Merge in_process with current
        new_current = self.triple.in_process.merge(self.triple.current, kernel)
        
        # Step 2: Commit current to history
        new_history_hash = commit_fn(self.triple.current)
        
        # Step 3: Update triple
        self.triple = HRTEvolutionTriple(
            in_process=None,
            current=new_current,
            history_hash=new_history_hash,
            step_number=self.triple.step_number + 1
        )
        self._history.append(new_history_hash)
        
        return self.triple
    
    def get_lineage(self) -> List[str]:
        """Get full commit history."""
        return list(self._history)
    
    def get_current(self) -> HRT:
        """Get current HRT."""
        return self.triple.current
    
    def get_in_process(self) -> Optional[HRT]:
        """Get in_process HRT if exists."""
        return self.triple.in_process


# =============================================================================
# SECTION 8: Example Usage
# =============================================================================

def main():
    """Example HRT usage with immutable tensors."""
    from .kernel import Kernel
    
    print("="*70)
    print("HASH RELATIONAL TENSOR (HRT) - Immutable PyTorch Implementation")
    print("="*70)
    
    # Create kernel
    kernel = Kernel()
    
    # Create config
    config = HRTConfig(p_bits=8, h_bits=16)
    print(f"\nConfig: dim={config.dimension}, basic_sets={config.num_basic_hllsets}")
    
    # Create genesis HRT
    print("\n1. Genesis State")
    print("-" * 40)
    
    genesis = HRT.empty(config)
    print(f"Genesis: {genesis}")
    print(f"AM: {genesis.am}")
    print(f"Lattice: {genesis.lattice}")
    
    # Create evolution manager
    print("\n2. Evolution Manager")
    print("-" * 40)
    
    evolution = HRTEvolution(config, genesis)
    print(f"Initial: {evolution.triple}")
    
    # Ingest data
    print("\n3. Ingest Perceptron Data")
    print("-" * 40)
    
    perceptron_data = {
        "visual": {"red", "green", "blue"},
        "audio": {"low", "mid", "high"},
    }
    
    evolution.ingest(perceptron_data, kernel)
    print(f"After ingest: {evolution.triple}")
    print(f"In-process HRT: {evolution.triple.in_process}")
    print(f"AM entries: {len(evolution.triple.in_process.am.nonzero_entries())}")
    
    # Evolve
    print("\n4. Evolution Step")
    print("-" * 40)
    
    def commit_fn(hrt: HRT) -> str:
        """Mock commit - would save to Git."""
        print(f"  Committing HRT {hrt.name[:16]}... to history")
        return hrt.name
    
    evolution.evolve(kernel, commit_fn)
    print(f"After evolve: {evolution.triple}")
    print(f"Lineage: {[h[:8] for h in evolution.get_lineage()]}")
    
    # Another cycle
    print("\n5. Another Evolution Cycle")
    print("-" * 40)
    
    more_data = {"tactile": {"soft", "hard", "rough"}}
    evolution.ingest(more_data, kernel)
    evolution.evolve(kernel, commit_fn)
    print(f"After second evolve: {evolution.triple}")
    
    # Test immutability
    print("\n6. Immutability Verification")
    print("-" * 40)
    
    current = evolution.get_current()
    modified_am = current.am.with_entry(0, 0, 5.0)
    
    print(f"Original AM: {current.am.name[:16]}...")
    print(f"Modified AM: {modified_am.name[:16]}...")
    print(f"Same? {current.am.name == modified_am.name}")
    print(f"Current unchanged? {evolution.get_current().am.name == current.am.name}")
    
    # Projections
    print("\n7. Future/Past Projections")
    print("-" * 40)
    
    future = evolution.triple.current.project_future([2, 3, 4])
    past = evolution.triple.current.project_past([0, 1])
    print(f"Future projection shape: {future.shape}")
    print(f"Past projection shape: {past.shape}")
    
    print("\n" + "="*70)
    print("HRT Evolution Ready")
    print("="*70)
    
    return evolution


if __name__ == "__main__":
    main()
