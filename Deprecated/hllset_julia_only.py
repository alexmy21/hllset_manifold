"""
HLLSet - Immutable Julia Interface with Batch Processing

Design principles:
- HLLSet instances are fully immutable
- All operations return new instances
- Batch processing is the primary mode for token ingestion
- Multi-batch processing can be parallelized and results merged via union
- No in-place modifications

Batch Processing Pattern:
    # Single batch
    hll = HLLSet.from_batch(['token1', 'token2', ...])
    
    # Multi-batch with parallel processing (conceptual)
    batches = [batch1, batch2, batch3]
    hlls = [HLLSet.from_batch(b) for b in batches]  # Can parallelize
    final_hll = HLLSet.merge(hlls)  # Union all batches
    
    # Accumulating pattern
    hll1 = HLLSet.from_batch(batch1)
    hll2 = HLLSet.from_batch(batch2)
    hll_combined = hll1.union(hll2)  # Immutable merge
"""

from __future__ import annotations
from typing import Set, Union, List, Optional, Iterable
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

# Try to import and initialize Julia
try:
    from julia import Main
    import os
    from pathlib import Path
    
    # Load HllSets.jl module
    hllsets_path = os.getenv("HLLSETS_PATH")
    if not hllsets_path:
        current_dir = Path(__file__).parent
        hllsets_jl = current_dir / "HllSets.jl"
        if hllsets_jl.exists():
            hllsets_path = str(hllsets_jl)
    
    if hllsets_path:
        Main.include(hllsets_path)
        # Don't use Main.using() to avoid namespace conflicts
        # Access functions via Main.HllSets instead
        JULIA_AVAILABLE = True
    else:
        JULIA_AVAILABLE = False
        
except Exception:
    JULIA_AVAILABLE = False

from .constants import P_BITS, SHARED_SEED


def compute_sha1(data: Union[str, bytes, np.ndarray]) -> str:
    """Compute SHA1 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha1(data).hexdigest()


class HLLSet:
    """
    HLLSet with direct Julia backend.
    
    Treat instances as immutable. Methods return new instances.
    """
    
    def __init__(self, p_bits: int = P_BITS, _jl_hll: Optional[object] = None):
        """
        Create HLLSet.
        
        Args:
            p_bits: Precision bits
            _jl_hll: Existing Julia HLL (for internal use)
        """
        self.p_bits = p_bits
        self._jl_hll = _jl_hll
        self._name: Optional[str] = None
        
        # Create Julia HLL if not provided
        if self._jl_hll is None and JULIA_AVAILABLE:
            self._jl_hll = Main.HllSets.HllSet(self.p_bits)
        
        # Compute name from content
        self._compute_name()
    
    def _compute_name(self):
        """Compute content-addressed name from registers."""
        registers = self.dump_numpy()
        self._name = compute_sha1(registers)
    
    # -------------------------------------------------------------------------
    # Class Methods - Primary API (Immutable Batch Processing)
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_batch(cls, tokens: Union[List[str], Set[str], Iterable[str]], 
                   p_bits: int = P_BITS, seed: int = SHARED_SEED) -> HLLSet:
        """
        Create HLLSet from a batch of tokens (PRIMARY FACTORY METHOD).
        
        This is the recommended way to create HLLSets. All tokens in the batch
        are processed together into a new immutable HLLSet instance.
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            p_bits: Precision bits for HLL
            seed: Hash seed for consistency
            
        Returns:
            New immutable HLLSet containing all tokens
            
        Example:
            >>> hll = HLLSet.from_batch(['token1', 'token2', 'token3'])
            >>> print(hll.cardinality())
        """
        # Convert to list if needed
        if not isinstance(tokens, (list, set)):
            tokens = list(tokens)
        
        hll = cls(p_bits=p_bits)
        if tokens and JULIA_AVAILABLE and hll._jl_hll is not None:
            add_func = getattr(Main.HllSets, "add!")
            add_func(hll._jl_hll, list(tokens), seed=seed)
            hll._compute_name()  # Compute content-based name
        return hll
    
    @classmethod
    def from_batches(cls, batches: List[Union[List[str], Set[str]]], 
                     p_bits: int = P_BITS, seed: int = SHARED_SEED,
                     parallel: bool = False, max_workers: Optional[int] = None) -> HLLSet:
        """
        Create HLLSet from multiple batches with optional parallel processing.
        
        Each batch is processed independently (can be parallelized), then all
        results are merged via union operation. This is efficient for large
        datasets that can be split into chunks.
        
        NOTE: Parallel processing is disabled when Julia backend is available
        due to thread-safety constraints with Julia's runtime. Sequential
        processing is still very fast with Julia backend.
        
        Args:
            batches: List of token batches
            p_bits: Precision bits for HLL
            seed: Hash seed (must be same for all batches)
            parallel: If True, process batches in parallel (only without Julia)
            max_workers: Number of parallel workers (None = CPU count)
            
        Returns:
            New immutable HLLSet containing union of all batches
            
        Example:
            >>> batches = [['a', 'b'], ['c', 'd'], ['e', 'f']]
            >>> hll = HLLSet.from_batches(batches, parallel=True)
        """
        if not batches:
            return cls(p_bits=p_bits)
        
        # Julia backend is not thread-safe, so force sequential processing
        # TODO: Future C implementation will support true parallel processing
        if JULIA_AVAILABLE:
            parallel = False
        
        if parallel:
            # Process batches in parallel (only when Julia not available)
            max_workers = max_workers or os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                hlls = list(executor.map(
                    lambda b: cls.from_batch(b, p_bits=p_bits, seed=seed),
                    batches
                ))
        else:
            # Sequential processing
            hlls = [cls.from_batch(b, p_bits=p_bits, seed=seed) for b in batches]
        
        # Merge all HLLSets via union
        return cls.merge(hlls)
    
    @classmethod
    def merge(cls, hlls: List[HLLSet]) -> HLLSet:
        """
        Merge multiple HLLSets into one via union operation.
        
        This is the recommended way to combine multiple HLLSets. All input
        HLLSets must have the same p_bits.
        
        Args:
            hlls: List of HLLSet instances to merge
            
        Returns:
            New HLLSet containing union of all inputs
            
        Example:
            >>> hll1 = HLLSet.from_batch(['a', 'b'])
            >>> hll2 = HLLSet.from_batch(['c', 'd'])
            >>> merged = HLLSet.merge([hll1, hll2])
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        # Start with first HLL and union with rest
        result = hlls[0]
        for hll in hlls[1:]:
            result = result.union(hll)
        
        return result
    
    @classmethod
    def absorb(cls, tokens: Set[str], p_bits: int = P_BITS, seed: int = SHARED_SEED) -> HLLSet:
        """
        Create HLLSet from tokens (legacy method, use from_batch instead).
        
        Kept for backward compatibility. Prefer from_batch() for new code.
        """
        return cls.from_batch(tokens, p_bits=p_bits, seed=seed)
    
    @classmethod
    def add(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Add tokens to an HLLSet, return new HLLSet (legacy method).
        
        Note: For better performance with large datasets, use from_batch()
        or from_batches() instead.
        
        Usage:
            h1 = HLLSet.from_batch(['a', 'b'])
            h2 = HLLSet.add(h1, 'c')  # h2 contains a,b,c; h1 unchanged
            h3 = HLLSet.add(h1, ['d', 'e'])  # batch add
        
        For accumulating batches, prefer:
            h1 = HLLSet.from_batch(batch1)
            h2 = HLLSet.from_batch(batch2)
            h_combined = h1.union(h2)
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        
        if not tokens:
            return base  # No change needed
        
        # Create new HLLSet from tokens, then union with base
        # This is pure - no mutation of base
        tokens_hll = cls.from_batch(tokens, p_bits=base.p_bits, seed=seed)
        return base.union(tokens_hll)
    
    @classmethod
    def append(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Append tokens to an HLLSet (alias for add).
        
        Same as add() - provided for API consistency.
        """
        return cls.add(base, tokens, seed)
    
    # -------------------------------------------------------------------------
    # Instance Methods - Return new instances
    # -------------------------------------------------------------------------
    
    def union(self, other: HLLSet) -> HLLSet:
        """Union with another HLLSet (returns new instance)."""
        if not JULIA_AVAILABLE or self._jl_hll is None or other._jl_hll is None:
            return HLLSet(p_bits=self.p_bits)
        
        result_jl = Main.HllSets.union(self._jl_hll, other._jl_hll)
        return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
    
    def intersect(self, other: HLLSet) -> HLLSet:
        """Intersection with another HLLSet (returns new instance)."""
        if not JULIA_AVAILABLE or self._jl_hll is None or other._jl_hll is None:
            return HLLSet(p_bits=self.p_bits)
        
        result_jl = Main.HllSets.intersect(self._jl_hll, other._jl_hll)
        return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
    
    def diff(self, other: HLLSet) -> HLLSet:
        """Difference with another HLLSet (returns new instance)."""
        if not JULIA_AVAILABLE or self._jl_hll is None or other._jl_hll is None:
            return HLLSet(p_bits=self.p_bits)
        
        result_jl = Main.HllSets.diff(self._jl_hll, other._jl_hll)
        return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def cardinality(self) -> float:
        """Estimated cardinality."""
        if not JULIA_AVAILABLE or self._jl_hll is None:
            return 0.0
        try:
            return float(Main.HllSets.count(self._jl_hll))
        except:
            return 0.0
    
    def similarity(self, other: HLLSet) -> float:
        """Compute Jaccard similarity with another HLLSet (0.0 to 1.0)."""
        if not JULIA_AVAILABLE or self._jl_hll is None or other._jl_hll is None:
            if self._name == other._name:
                return 1.0
            return 0.0
        # Julia match() returns integer percentage (0-100), convert to float (0.0-1.0)
        return float(Main.HllSets.match(self._jl_hll, other._jl_hll)) / 100.0
    
    def cosine(self, other: HLLSet) -> float:
        """Cosine similarity."""
        if not JULIA_AVAILABLE or self._jl_hll is None or other._jl_hll is None:
            return self.similarity(other)
        return float(Main.HllSets.cosine(self._jl_hll, other._jl_hll))
    
    def dump_numpy(self) -> np.ndarray:
        """Get register vector as numpy array."""
        if not JULIA_AVAILABLE or self._jl_hll is None:
            return np.array([], dtype=np.uint32)
        try:
            # Access the counts field directly from the Julia struct
            counts = self._jl_hll.counts
            return np.array(list(counts), dtype=np.uint32)
        except:
            try:
                hll_str = str(self._jl_hll)
                return np.array([ord(c) for c in hll_str], dtype=np.uint32)
            except:
                return np.array([], dtype=np.uint32)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name if self._name is not None else ""
    
    @property
    def short_name(self) -> str:
        """Short name for display."""
        return self._name[:8] if self._name else "unknown"
    
    # -------------------------------------------------------------------------
    # Python Protocols
    # -------------------------------------------------------------------------
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HLLSet):
            return False
        return self.name == other.name
    
    def __repr__(self) -> str:
        return f"HLLSet({self.short_name}..., |A|≈{self.cardinality():.1f})"


# =============================================================================
# Mock HLLSet for when Julia is not available
# =============================================================================

class MockHLLSet:
    """Mock HLLSet for testing without Julia."""
    
    def __init__(self, p_bits: int = P_BITS, _registers=None):
        self.p_bits = p_bits
        self.m = 1 << p_bits
        self._count = 0
        self._registers = _registers if _registers is not None else [0] * self.m
        self._name: Optional[str] = None
        self._compute_name()
    
    def _compute_name(self):
        import numpy as np
        registers = np.array(self._registers, dtype=np.uint32)
        self._name = compute_sha1(registers.tobytes())
    
    @property
    def name(self):
        return self._name if self._name is not None else ""
    
    @property
    def short_name(self):
        return self.name[:8]
    
    @classmethod
    def from_batch(cls, tokens: Union[List[str], Set[str], Iterable[str]], 
                   p_bits: int = P_BITS, seed: int = SHARED_SEED):
        """Create HLLSet from batch of tokens."""
        if not isinstance(tokens, (list, set)):
            tokens = list(tokens)
        h = cls(p_bits)
        for token in tokens:
            h._count += 1
            idx = hash((token, seed)) % h.m
            h._registers[idx] = max(h._registers[idx], 1)
        h._compute_name()
        return h
    
    @classmethod
    def from_batches(cls, batches: List[Union[List[str], Set[str]]], 
                     p_bits: int = P_BITS, seed: int = SHARED_SEED,
                     parallel: bool = False, max_workers: Optional[int] = None):
        """Create HLLSet from multiple batches."""
        if not batches:
            return cls(p_bits=p_bits)
        
        if parallel:
            import os
            from concurrent.futures import ThreadPoolExecutor
            max_workers = max_workers or os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                hlls = list(executor.map(
                    lambda b: cls.from_batch(b, p_bits=p_bits, seed=seed),
                    batches
                ))
        else:
            hlls = [cls.from_batch(b, p_bits=p_bits, seed=seed) for b in batches]
        
        return cls.merge(hlls)
    
    @classmethod
    def merge(cls, hlls: List):
        """Merge multiple HLLSets."""
        if not hlls:
            return cls()
        if len(hlls) == 1:
            return hlls[0]
        result = hlls[0]
        for hll in hlls[1:]:
            result = result.union(hll)
        return result
    
    @classmethod
    def absorb(cls, tokens: Set[str], p_bits: int = P_BITS, seed: int = SHARED_SEED):
        """Legacy method."""
        return cls.from_batch(tokens, p_bits=p_bits, seed=seed)
    
    @classmethod
    def add(cls, base, tokens, seed=SHARED_SEED):
        if isinstance(tokens, str):
            tokens = [tokens]
        if not tokens:
            return base
        
        tokens_hll = cls.from_batch(tokens, p_bits=base.p_bits, seed=seed)
        return base.union(tokens_hll)
    
    @classmethod
    def append(cls, base, tokens, seed=SHARED_SEED):
        return cls.add(base, tokens, seed)
    
    def union(self, other):
        h = MockHLLSet(self.p_bits, _registers=[max(a, b) for a, b in zip(self._registers, other._registers)])
        h._count = self._count + other._count
        h._compute_name()
        return h
    
    def intersect(self, other):
        h = MockHLLSet(self.p_bits, _registers=[min(a, b) for a, b in zip(self._registers, other._registers)])
        h._count = max(0, self._count + other._count - (self.m * 2))
        h._compute_name()
        return h
    
    def diff(self, other):
        h = MockHLLSet(self.p_bits, _registers=[max(0, a - b) for a, b in zip(self._registers, other._registers)])
        h._count = max(0, self._count - other._count)
        h._compute_name()
        return h
    
    def cardinality(self):
        return float(self._count)
    
    def similarity(self, other):
        if self._count == 0 and other._count == 0:
            return 1.0
        max_count = max(self._count, other._count)
        return min(self._count, other._count) / max_count if max_count > 0 else 0.0
    
    def cosine(self, other):
        return self.similarity(other)
    
    def dump_numpy(self):
        import numpy as np
        return np.array(self._registers, dtype=np.uint32)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, MockHLLSet):
            return False
        return self.name == other.name
    
    def __repr__(self):
        return f"HLLSet({self.short_name}..., |A|≈{self.cardinality():.1f})"


# Export the appropriate implementation
if JULIA_AVAILABLE:
    __all__ = ['HLLSet', 'compute_sha1']
else:
    HLLSet = MockHLLSet
    __all__ = ['HLLSet', 'compute_sha1']
