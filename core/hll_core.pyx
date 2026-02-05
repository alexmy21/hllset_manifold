# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
HLL Core Operations in Cython

Fast C implementation of HyperLogLog operations with Python-like syntax.
This replaces the Julia backend with a native C extension that:
- Is fully thread-safe (no GIL during critical operations)
- Supports true parallel processing
- Has minimal dependencies (just numpy)
- Is easy to build and distribute
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport log, pow, sqrt
from libc.string cimport memcpy
import hashlib

# Initialize numpy
cnp.import_array()


cdef extern from "Python.h":
    void PyEval_InitThreads()


# Constants
cdef double ALPHA_16 = 0.673
cdef double ALPHA_32 = 0.697
cdef double ALPHA_64 = 0.709
cdef double ALPHA_INF = 0.7213 / (1.0 + 1.079 / 65536.0)  # For m >= 128


cdef inline uint64_t murmur_hash64(const char* data, int length, uint64_t seed) nogil:
    """
    Fast MurmurHash64A implementation for token hashing.
    Thread-safe (no GIL required).
    """
    cdef uint64_t h = seed ^ (length * 0xc6a4a7935bd1e995ULL)
    cdef const uint64_t* data64 = <const uint64_t*>data
    cdef int nblocks = length // 8
    cdef uint64_t k
    cdef int i
    
    # Process 8-byte chunks
    for i in range(nblocks):
        k = data64[i]
        k *= 0xc6a4a7935bd1e995ULL
        k ^= k >> 47
        k *= 0xc6a4a7935bd1e995ULL
        h ^= k
        h *= 0xc6a4a7935bd1e995ULL
    
    # Process remaining bytes
    cdef const uint8_t* tail = <const uint8_t*>(data + nblocks * 8)
    cdef int remaining = length & 7
    
    if remaining >= 7:
        h ^= <uint64_t>tail[6] << 48
    if remaining >= 6:
        h ^= <uint64_t>tail[5] << 40
    if remaining >= 5:
        h ^= <uint64_t>tail[4] << 32
    if remaining >= 4:
        h ^= <uint64_t>tail[3] << 24
    if remaining >= 3:
        h ^= <uint64_t>tail[2] << 16
    if remaining >= 2:
        h ^= <uint64_t>tail[1] << 8
    if remaining >= 1:
        h ^= <uint64_t>tail[0]
        h *= 0xc6a4a7935bd1e995ULL
    
    # Finalize
    h ^= h >> 47
    h *= 0xc6a4a7935bd1e995ULL
    h ^= h >> 47
    
    return h


cdef inline int leading_zeros(uint64_t value) nogil:
    """Count leading zeros in 64-bit integer. Thread-safe."""
    if value == 0:
        return 64
    
    cdef int count = 0
    cdef uint64_t mask = 1ULL << 63
    
    while (value & mask) == 0:
        count += 1
        mask >>= 1
    
    return count


cdef class HLLCore:
    """
    Core HyperLogLog implementation in Cython.
    
    This is the fast C backend that replaces Julia. Operations release
    the GIL where possible for true parallel processing.
    """
    
    cdef public int p_bits
    cdef public int m  # Number of registers (2^p_bits)
    cdef public cnp.ndarray registers
    cdef uint8_t[::1] registers_view  # Fast memoryview
    
    def __init__(self, int p_bits=12):
        """
        Create HLL core.
        
        Args:
            p_bits: Precision (4-16). Default 12 = 4096 registers
        """
        if p_bits < 4 or p_bits > 16:
            raise ValueError("p_bits must be between 4 and 16")
        
        self.p_bits = p_bits
        self.m = 1 << p_bits  # 2^p_bits
        
        # Allocate registers (uint8 array)
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self.registers_view = self.registers
    
    def add_token(self, str token, uint64_t seed=0):
        """Add a single token. For small batches."""
        cdef bytes token_bytes = token.encode('utf-8')
        cdef const char* data = token_bytes
        cdef int length = len(token_bytes)
        self._add_token_c(data, length, seed)
    
    cdef void _add_token_c(self, const char* data, int length, uint64_t seed) nogil:
        """Internal C implementation. Thread-safe, no GIL."""
        # Hash the token
        cdef uint64_t hash_val = murmur_hash64(data, length, seed)
        
        # Extract bucket index (first p_bits)
        cdef uint32_t bucket = hash_val & ((1 << self.p_bits) - 1)
        
        # Extract remaining bits
        cdef uint64_t remaining = hash_val >> self.p_bits
        
        # Count leading zeros + 1
        cdef int lz = leading_zeros(remaining) - self.p_bits + 1
        
        # Update register (take maximum)
        if lz > self.registers_view[bucket]:
            self.registers_view[bucket] = lz
    
    def add_batch(self, list tokens, uint64_t seed=0):
        """
        Add batch of tokens efficiently.
        Encodes tokens first, then releases GIL for processing.
        """
        cdef int i, n = len(tokens)
        cdef bytes token_bytes
        cdef const char* data
        cdef int length
        
        # Process each token
        for i in range(n):
            token_bytes = tokens[i].encode('utf-8')
            data = token_bytes
            length = len(token_bytes)
            
            # Process with GIL released
            with nogil:
                self._add_token_c(data, length, seed)
    
    def compute_reg_zeros_batch(self, list tokens, uint64_t seed=0):
        """
        Compute (reg, zeros) pairs for a batch of tokens WITHOUT adding to registers.
        Returns list of (reg, zeros) tuples for each token.
        
        This is used to avoid double calculation when building adjacency matrices:
        - HLLSet computes these values internally when adding tokens
        - AM needs these values to build transition matrix
        - This method exposes the calculation without duplicate work
        
        Args:
            tokens: List of token strings
            seed: Hash seed (must match seed used for add_batch)
        
        Returns:
            List of (reg, zeros) tuples, one per token
        """
        cdef int i, n = len(tokens)
        cdef bytes token_bytes
        cdef const char* data
        cdef int length
        cdef uint64_t hash_val
        cdef uint32_t bucket
        cdef uint64_t remaining
        cdef int lz
        
        result = []
        
        for i in range(n):
            token_bytes = tokens[i].encode('utf-8')
            data = token_bytes
            length = len(token_bytes)
            
            # Compute hash
            hash_val = murmur_hash64(data, length, seed)
            
            # Extract bucket (reg)
            bucket = hash_val & ((1 << self.p_bits) - 1)
            
            # Extract remaining bits
            remaining = hash_val >> self.p_bits
            
            # Count leading zeros + 1
            lz = leading_zeros(remaining) - self.p_bits + 1
            
            # Zeros is lz - 1 (since we add 1 in the calculation)
            result.append((int(bucket), int(lz - 1)))
        
        return result
    
    def cardinality(self):
        """
        Estimate cardinality using HLL algorithm.
        Uses bias correction and small/large range corrections.
        """
        cdef double raw_estimate = self._compute_raw_estimate()
        cdef int m = self.m
        cdef double alpha
        cdef double estimate
        cdef int zero_count
        cdef double pow_2_32
        
        # Apply bias correction based on m
        if m == 16:
            alpha = ALPHA_16
        elif m == 32:
            alpha = ALPHA_32
        elif m == 64:
            alpha = ALPHA_64
        else:
            alpha = ALPHA_INF
        
        estimate = alpha * m * m / raw_estimate
        
        # Small range correction
        if estimate <= 2.5 * m:
            zero_count = self._count_zeros()
            if zero_count > 0:
                estimate = m * log(m / <double>zero_count)
        
        # Large range correction
        pow_2_32 = pow(2.0, 32)
        if estimate > pow_2_32 / 30.0:
            estimate = -pow_2_32 * log(1.0 - estimate / pow_2_32)
        
        return max(0.0, estimate)
    
    cdef double _compute_raw_estimate(self) nogil:
        """Compute raw HLL estimate. Thread-safe."""
        cdef double sum_val = 0.0
        cdef int i
        
        for i in range(self.m):
            sum_val += pow(2.0, -<double>self.registers_view[i])
        
        return sum_val
    
    cdef int _count_zeros(self) nogil:
        """Count zero registers. Thread-safe."""
        cdef int count = 0
        cdef int i
        
        for i in range(self.m):
            if self.registers_view[i] == 0:
                count += 1
        
        return count
    
    def union(self, HLLCore other):
        """
        Create union of two HLL cores.
        Returns new HLLCore with max of each register.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union HLLs with different p_bits")
        
        cdef HLLCore result = HLLCore(self.p_bits)
        cdef int i
        
        # Take maximum of each register
        with nogil:
            for i in range(self.m):
                result.registers_view[i] = max(
                    self.registers_view[i],
                    other.registers_view[i]
                )
        
        return result
    
    def intersect_cardinality(self, HLLCore other):
        """
        Estimate intersection cardinality using inclusion-exclusion.
        |A ∩ B| = |A| + |B| - |A ∪ B|
        """
        cdef double card_a = self.cardinality()
        cdef double card_b = other.cardinality()
        cdef double card_union = self.union(other).cardinality()
        
        return max(0.0, card_a + card_b - card_union)
    
    def jaccard_similarity(self, HLLCore other):
        """
        Estimate Jaccard similarity: |A ∩ B| / |A ∪ B|
        """
        cdef double card_union = self.union(other).cardinality()
        if card_union == 0:
            return 0.0
        
        cdef double card_intersect = self.intersect_cardinality(other)
        return card_intersect / card_union
    
    def cosine_similarity(self, HLLCore other):
        """
        Estimate cosine similarity: |A ∩ B| / sqrt(|A| * |B|)
        """
        cdef double card_a = self.cardinality()
        cdef double card_b = other.cardinality()
        
        if card_a == 0 or card_b == 0:
            return 0.0
        
        cdef double card_intersect = self.intersect_cardinality(other)
        return card_intersect / sqrt(card_a * card_b)
    
    def get_registers(self):
        """Get registers as numpy array (copy for safety)."""
        return self.registers.copy()
    
    def set_registers(self, cnp.ndarray[uint8_t, ndim=1] new_registers):
        """Set registers from numpy array."""
        if len(new_registers) != self.m:
            raise ValueError(f"Expected {self.m} registers, got {len(new_registers)}")
        
        self.registers[:] = new_registers
    
    def copy(self):
        """Create a deep copy."""
        cdef HLLCore result = HLLCore(self.p_bits)
        result.registers[:] = self.registers
        return result
    
    def __reduce__(self):
        """Support for pickling (needed for multiprocessing)."""
        return (
            HLLCore,
            (self.p_bits,),
            {'registers': self.registers}
        )
    
    def __setstate__(self, state):
        """Restore from pickle."""
        self.registers[:] = state['registers']
