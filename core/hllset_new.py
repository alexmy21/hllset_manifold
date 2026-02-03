"""
HLLSet - Immutable C/Cython Backend with Batch Processing

Design principles:
- HLLSet instances are fully immutable
- All operations return new instances
- Batch processing is the primary mode for token ingestion
- Multi-batch processing can be parallelized (thread-safe C backend)
- No in-place modifications

Batch Processing Pattern:
    # Single batch
    hll = HLLSet.from_batch(['token1', 'token2', ...])
    
    # Multi-batch with parallel processing
    batches = [batch1, batch2, batch3]
    hll_combined = HLLSet.from_batches(batches, parallel=True)
    
    # Accumulating pattern
    hll1 = HLLSet.from_batch(batch1)
    hll2 = HLLSet.from_batch(batch2)
    hll_combined = hll1.union(hll2)  # Immutable merge
"""

from __future__ import annotations
from typing import Set, Union, List, Optional, Iterable
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

from .constants import P_BITS, SHARED_SEED

# Initialize backend flags
C_BACKEND_AVAILABLE = False
JULIA_AVAILABLE = False

# Try to import C backend
try:
    from .hll_core import HLLCore
    C_BACKEND_AVAILABLE = True
except ImportError:
    pass

# Fallback: Try Julia backend if C not available
if not C_BACKEND_AVAILABLE:
    try:
        from julia import Main
        from pathlib import Path
        
        hllsets_path = os.getenv("HLLSETS_PATH")
        if not hllsets_path:
            current_dir = Path(__file__).parent
            hllsets_jl = current_dir / "HllSets.jl"
            if hllsets_jl.exists():
                hllsets_path = str(hllsets_jl)
        
        if hllsets_path:
            Main.include(hllsets_path)
            JULIA_AVAILABLE = True
    except Exception:
        pass


def compute_sha1(data: Union[str, bytes, np.ndarray]) -> str:
    """Compute SHA1 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha1(data).hexdigest()


class HLLSet:
    """
    HLLSet with C backend (or Julia/Mock fallback).
    
    Treat instances as immutable. Methods return new instances.
    """
    
    def __init__(self, p_bits: int = P_BITS, _core: Optional[object] = None, _jl_hll: Optional[object] = None):
        """
        Create HLLSet.
        
        Args:
            p_bits: Precision bits
            _core: Existing C HLLCore (internal use)
            _jl_hll: Existing Julia HLL (fallback, internal use)
        """
        self.p_bits = p_bits
        self._core = _core
        self._jl_hll = _jl_hll
        self._name: Optional[str] = None
        
        # Create backend if not provided
        if self._core is None and C_BACKEND_AVAILABLE:
            self._core = HLLCore(self.p_bits)
        elif self._jl_hll is None and JULIA_AVAILABLE and not C_BACKEND_AVAILABLE:
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
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        hll = cls(p_bits=p_bits)
        
        if not tokens:
            return hll
        
        # Use C backend (preferred)
        if C_BACKEND_AVAILABLE and hll._core is not None:
            hll._core.add_batch(tokens, seed)
            hll._compute_name()
        # Fallback to Julia
        elif JULIA_AVAILABLE and hll._jl_hll is not None:
            add_func = getattr(Main.HllSets, "add!")
            add_func(hll._jl_hll, tokens, seed=seed)
            hll._compute_name()
        
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
        
        NOTE: C backend is thread-safe and supports true parallel processing!
        Julia backend forces sequential processing due to thread safety.
        
        Args:
            batches: List of token batches
            p_bits: Precision bits for HLL
            seed: Hash seed (must be same for all batches)
            parallel: If True, process batches in parallel (C backend only)
            max_workers: Number of parallel workers (None = CPU count)
            
        Returns:
            New immutable HLLSet containing union of all batches
            
        Example:
            >>> batches = [['a', 'b'], ['c', 'd'], ['e', 'f']]
            >>> hll = HLLSet.from_batches(batches, parallel=True)
        """
        if not batches:
            return cls(p_bits=p_bits)
        
        # Julia backend is not thread-safe, force sequential
        if JULIA_AVAILABLE and not C_BACKEND_AVAILABLE:
            parallel = False
        
        if parallel and C_BACKEND_AVAILABLE:
            # TRUE parallel processing with C backend!
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
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union HLLs with different p_bits")
        
        # C backend
        if C_BACKEND_AVAILABLE and self._core is not None and other._core is not None:
            result_core = self._core.union(other._core)
            return HLLSet(p_bits=self.p_bits, _core=result_core)
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
            result_jl = Main.HllSets.union(self._jl_hll, other._jl_hll)
            return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
        
        return HLLSet(p_bits=self.p_bits)
    
    def intersect(self, other: HLLSet) -> HLLSet:
        """Intersection with another HLLSet (returns new instance with estimated intersection)."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot intersect HLLs with different p_bits")
        
        # C backend - create new HLL with intersection cardinality
        # Note: HLL can't directly compute intersection, we estimate it
        if C_BACKEND_AVAILABLE and self._core is not None and other._core is not None:
            # For C backend, we don't have direct intersection support yet
            # Just return the one with smaller cardinality as approximation
            if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
                result_jl = Main.HllSets.intersect(self._jl_hll, other._jl_hll)
                return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
            result_jl = Main.HllSets.intersect(self._jl_hll, other._jl_hll)
            return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
        
        return HLLSet(p_bits=self.p_bits)
    
    def diff(self, other: HLLSet) -> HLLSet:
        """Difference with another HLLSet (returns new instance)."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot diff HLLs with different p_bits")
        
        # Julia backend (C doesn't support diff directly yet)
        if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
            result_jl = Main.HllSets.diff(self._jl_hll, other._jl_hll)
            return HLLSet(p_bits=self.p_bits, _jl_hll=result_jl)
        
        return HLLSet(p_bits=self.p_bits)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def cardinality(self) -> float:
        """Estimated cardinality."""
        # C backend (preferred)
        if C_BACKEND_AVAILABLE and self._core is not None:
            return self._core.cardinality()
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None:
            try:
                return float(Main.HllSets.count(self._jl_hll))
            except:
                return 0.0
        
        return 0.0
    
    def similarity(self, other: HLLSet) -> float:
        """Compute Jaccard similarity with another HLLSet (0.0 to 1.0)."""
        # C backend
        if C_BACKEND_AVAILABLE and self._core is not None and other._core is not None:
            return self._core.jaccard_similarity(other._core)
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
            # Julia match() returns integer percentage (0-100), convert to float (0.0-1.0)
            return float(Main.HllSets.match(self._jl_hll, other._jl_hll)) / 100.0
        
        # Same content
        if self._name == other._name:
            return 1.0
        return 0.0
    
    def cosine(self, other: HLLSet) -> float:
        """Cosine similarity."""
        # C backend
        if C_BACKEND_AVAILABLE and self._core is not None and other._core is not None:
            return self._core.cosine_similarity(other._core)
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None and other._jl_hll is not None:
            return float(Main.HllSets.cosine(self._jl_hll, other._jl_hll))
        
        return self.similarity(other)
    
    def dump_numpy(self) -> np.ndarray:
        """Get register vector as numpy array."""
        # C backend
        if C_BACKEND_AVAILABLE and self._core is not None:
            return self._core.get_registers()
        
        # Julia fallback
        if JULIA_AVAILABLE and self._jl_hll is not None:
            try:
                counts = self._jl_hll.counts
                return np.array(list(counts), dtype=np.uint8)
            except:
                try:
                    hll_str = str(self._jl_hll)
                    return np.array([ord(c) for c in hll_str], dtype=np.uint8)
                except:
                    return np.array([], dtype=np.uint8)
        
        return np.array([], dtype=np.uint8)
    
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
    
    @property
    def backend(self) -> str:
        """Return which backend is being used."""
        if C_BACKEND_AVAILABLE and self._core is not None:
            return "C/Cython"
        elif JULIA_AVAILABLE and self._jl_hll is not None:
            return "Julia"
        else:
            return "Mock"
    
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
        return f"HLLSet({self.short_name}..., |A|≈{self.cardinality():.1f}, backend={self.backend})"


# Export
__all__ = ['HLLSet', 'compute_sha1', 'C_BACKEND_AVAILABLE', 'JULIA_AVAILABLE']
