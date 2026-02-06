# HLLSet Manifold - Tutorial Notebooks

This directory contains interactive Jupyter notebooks demonstrating the HLLSet Manifold system.

## Notebooks Overview

### [01_quick_start.ipynb](01_quick_start.ipynb)
**Quick introduction to HLLSet basics**

Learn fundamental concepts:
- Creating HLLSets from tokens
- Basic set operations (union, intersection, similarity)
- Batch and parallel processing
- Kernel operations
- Immutability and content addressing

**Recommended**: Start here if you're new to HLLSet Manifold

---

### [02_n_token_algorithm.ipynb](02_n_token_algorithm.ipynb)
**N-token generation and disambiguation**

Deep dive into n-token ingestion:
- Sliding window n-token generation
- Multiple HLLSets per document (1-tokens, 2-tokens, 3-tokens)
- Lookup Tables (LUTs) for disambiguation
- Implicit order preservation
- Custom tokenization configuration

**Based on**: [NTOKEN_SUMMARY.md](NTOKEN_SUMMARY.md) and [NTOKEN_ALGORITHM.md](NTOKEN_ALGORITHM.md)

---

### [03_adjacency_matrix.ipynb](03_adjacency_matrix.ipynb)
**Order preservation with Adjacency Matrices**

Understanding how order is preserved and reconstructed:
- Why HLLSets lose order (and why that's okay)
- Building Adjacency Matrix during ingestion
- (reg, zeros) identifiers for compact storage
- START/END markers for sequence boundaries
- Query-time order reconstruction via AM traversal
- Two-phase architecture (ingestion vs query)

**Based on**: [AM_SUMMARY.md](AM_SUMMARY.md) and [AM_ARCHITECTURE.md](AM_ARCHITECTURE.md)

---

### [04_kernel_entanglement.ipynb](04_kernel_entanglement.ipynb)
**Kernel operations and entanglement concepts**

Explore the stateless transformation engine:
- Level 1: Basic set operations (absorb, union, intersection)
- Level 2: Entanglement operations (isomorphism, validation)
- Level 3: Network operations (fold, coherence)
- Reproduction with mutation
- Composability and pure functions

**Based on**: [KERNEL_ENTANGLEMENT.md](KERNEL_ENTANGLEMENT.md) and [ENTANGLEMENT_SINGULARITY.md](ENTANGLEMENT_SINGULARITY.md)

---

### [05_manifold_os.ipynb](05_manifold_os.ipynb)
**ManifoldOS and the Universal Constructor pattern**

Complete OS-level orchestration:
- ICASRA pattern (A, B, C, D components)
- Driver lifecycle management (wake, idle, restart, remove)
- Immutability and idempotence
- Content addressability
- Custom driver configuration
- Why no scheduling is needed

**Based on**: [MANIFOLD_OS_QUICKREF.md](MANIFOLD_OS_QUICKREF.md), [MANIFOLD_OS_DRIVERS.md](MANIFOLD_OS_DRIVERS.md), and [MANIFOLD_OS_SUCCESS.md](MANIFOLD_OS_SUCCESS.md)

---

## Learning Path

**Beginner → Advanced**:

1. **Start with**: `01_quick_start.ipynb` - Get familiar with basic concepts
2. **Then**: `02_n_token_algorithm.ipynb` - Understand how data is ingested
3. **Next**: `03_adjacency_matrix.ipynb` - Learn about order preservation
4. **Advanced**: `04_kernel_entanglement.ipynb` - Explore kernel operations
5. **Complete**: `05_manifold_os.ipynb` - See the full system in action

**By Topic**:

- **Data Ingestion**: → Notebooks 02, 05
- **Order Preservation**: → Notebook 03
- **Set Operations**: → Notebooks 01, 04
- **System Architecture**: → Notebooks 04, 05

---

## Running the Notebooks

### Prerequisites

```bash
# Ensure C/Cython backend is built
python setup.py build_ext --inplace

# Or use uv
uv sync
```

### Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### Quick Test

```python
from core import HLLSet

# Create and test
hll = HLLSet.from_batch(['hello', 'world'])
print(f"Backend: {hll.backend}")
print(f"Cardinality: {hll.cardinality()}")
```

Expected output:
```
Backend: C/Cython
Cardinality: 2.00
```

---

## Key Features Demonstrated

### ✓ C/Cython Backend
All notebooks use the high-performance C/Cython backend (Julia removed).

### ✓ Immutability
All operations return new instances - no in-place modifications.

### ✓ Parallel Processing
Thread-safe C backend enables true parallel batch processing.

### ✓ Content Addressing
Every HLLSet is named by its content hash.

### ✓ N-Token Algorithm
Multiple representations with disambiguation via LUTs.

### ✓ Adjacency Matrix
Order preservation and reconstruction for query processing.

---

## Deprecated Notebooks

Old notebooks have been moved to `deprecated/notebooks/`:
- `demo_kernel.ipynb` - Replaced by `04_kernel_entanglement.ipynb`
- `demo_manifold_os.ipynb` - Replaced by `05_manifold_os.ipynb`
- `demo_refactored.ipynb` - Split into multiple focused notebooks

These are kept for reference but may contain outdated code.

---

## Documentation References

Each notebook is based on comprehensive markdown documentation:

- **N-Token**: `NTOKEN_SUMMARY.md`, `NTOKEN_ALGORITHM.md`
- **Adjacency Matrix**: `AM_SUMMARY.md`, `AM_ARCHITECTURE.md`
- **Kernel**: `KERNEL_ENTANGLEMENT.md`, `ENTANGLEMENT_SINGULARITY.md`
- **ManifoldOS**: `MANIFOLD_OS_QUICKREF.md`, `MANIFOLD_OS_DRIVERS.md`, `MANIFOLD_OS_SUCCESS.md`
- **General**: `README.md`, `C_BACKEND_README.md`, `SUCCESS.md`

---

## Questions or Issues?

- Check the markdown documentation for detailed explanations
- Run test files: `python test_c_backend.py`, `python test_kernel_entanglement.py`
- Review examples in `examples/` directory

---

**Last Updated**: February 4, 2026  
**System Version**: 0.2.0 (Julia-free, C/Cython only)
