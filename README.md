# HLLSet Manifold

>**Probabilistic Knowledge Representation with HyperLogLog Lattices**

A Python library for building evolving knowledge structures using HyperLogLog sketches, lattice morphisms, and predictive projection.

---

## What is this?

HLLSet Manifold implements a novel approach to knowledge representation where:

- **Knowledge is probabilistic** - Using HyperLogLog (HLL) sketches as compact "fingerprints" of token sets
- **Structure evolves** - Hash Relational Tensors (HRT) track state transitions with full history
- **Predictions are explicit** - Project future states from current lattice structure via morphisms
- **Everything is immutable** - Content-addressed structures enable fearless parallelization

Think of it as a way to represent knowledge that naturally handles:

- Uncertainty (via probabilistic sketches)
- Evolution (via tracked state transitions)
- Prediction (via lattice projections)
- Scale (via parallelization)

---

## Enterprise Application: Metadata as the AI-Enterprise Bridge

### The Problem: The Metadata Gap

Enterprises face a critical disconnect between **Enterprise Data (ED)** and **AI**:

- **ED is exact, relational, schema-bound**: SAP, Oracle, 1C store millions of structured records
- **AI needs probabilistic patterns and semantic relationships**: Can't efficiently process raw tables
- **The missing link is metadata**: Not schema descriptions, but **structural fingerprints**

Current metadata tools fail AI integration:

- **Data catalogs** describe schemas (column types, row counts) but miss **content relationships**
- **Statistics** (cardinality, distributions) lose **structural patterns** 
- **Samples** risk missing rare but critical cases
- **Result**: AI operates blind to enterprise context, enterprises can't ground AI solutions

### The HLLSet Solution: Intelligent Metadata Management

**HLLSets transform Enterprise Data into metadata fingerprints that bridge both worlds.**

#### The Two-Way Bridge

**1. ED → Metadata → AI** (Upward Transformation):

- Each **row** (customer, transaction, product) → HLLSet fingerprint (1.5KB)
- Each **column** (attribute, feature) → HLLSet fingerprint (1.5KB)  
- **10M-row CRM table** → ~15MB of metadata (preserves structure, not raw values)
- AI reasons about **relationships** via BSS morphisms: "Which customers resemble top performers?"
- **Privacy preserved**: AI never sees PII, only structural fingerprints

**2. AI → Metadata → ED** (Downward Grounding):

- AI produces solution: "Target customers with BSS > 0.8 to segment X"
- Metadata links to **source fingerprints** (which specific customers?)
- Execute in ED: `SELECT * FROM customers WHERE fingerprint_id IN (...)`
- Verify results, close the loop

**This solves explainability**: Every AI decision traces through metadata back to source ED records.

#### HLLSets as Intelligent Metadata

Unlike traditional metadata, HLLSets preserve **structural relationships**:

- **Fixed size**: 1.5KB per fingerprint, whether table has 10 rows or 10M
- **Set algebra**: Union, intersection, difference work on metadata directly
- **BSS morphisms**: Measure relationships (tau inclusion, rho exclusion, sigma similarity)
- **Content-addressed**: SHA1 immutability ensures audit trails

**Key insight**: Your enterprise data becomes **queryable metadata** that AI can understand:

- Rows/columns naturally map to HLLSet collections (W lattice)
- Schema changes → re-ingest → metadata adapts automatically (no ETL breakage)
- Relationships captured mathematically (not hardcoded mappings)

#### Self-Generative Core: Metadata That Adapts

Current metadata is **static** (manually curated). We need **adaptive metadata**:

The **Self-Generative Core** treats metadata as living (von Neumann universal constructor pattern):

- **Constructor**: Ingests new ED patterns (new columns, relationships)
- **Copier**: Propagates proven metadata patterns across lattice
- **Controller**: Predicts which metadata projections matter (strategies: conservative, growth, decay)
- **Environment Interface**: Observes real outcomes, adjusts metadata

**Example**: CRM adds "CustomerLifetimeValue" column:

- Traditional: Breaks integrations, manual ETL updates
- Self-Generative: Re-ingest → new column fingerprint → projection adjusts → AI sees new relationship automatically

#### The Missing Infrastructure

Enterprises don't need another database. They need:

1. **Metadata that summarizes structure**, not just schema
2. **A two-way bridge** that grounds AI solutions back to source ED
3. **Adaptive metadata** that evolves with business logic

**HLLSet Manifold provides intelligent metadata management**: the missing chain between Enterprise Data and AI.

### Why This Matters

Traditional approaches require:

- Custom ETL for each data source
- Separate models for each business domain  
- Manual updates when systems change

HLLSet Manifold enables:

- **Universal representation** across all enterprise data
- **Single unified model** that evolves with your business
- **Automatic adaptation** as systems and requirements change

We're not another AI tool. We're the **connective infrastructure** that makes enterprise AI possible.

---

## Key Concepts in Plain English

### 1. HLLSets are Fingerprints

An HLLSet is like a compact fingerprint of a set of tokens:

```python
tokens = {"dog", "cat", "elephant", "whale"}
fingerprint = HLLSet.absorb(tokens, p_bits=14)

# You can estimate how many items: ~4
print(fingerprint.cardinality())  

# But you CANNOT get the original tokens back!
# It's a one-way sketch, not a storage container
```

**Key insight**: HLLSets are probabilistic data structures for cardinality estimation, not containers.

### 2. Hash Relational Tensor (HRT) = AM + W

An HRT combines two structures:

- **AM** (Adjacency Matrix) - Tracks which tokens connect to which
- **W** (Lattice) - HLLSet basics organized as row/column structures

Together they form a complete snapshot of knowledge state.

### 3. Evolution = (D, R, N) Triples

When knowledge evolves, we track what changed:

- **D** (Deleted) - Tokens removed from previous state
- **R** (Retained) - Tokens that stayed
- **N** (New) - Tokens added

This gives us precise change tracking without storing full history.

### 4. Projection ≠ Iteration

**OLD incorrect thinking**: Iterate W matrix transformations on HLLSets  
**CORRECT approach**: Predict what tokens will be ingested NEXT

Projection asks: "Given current state, what tokens are likely to arrive next?"

### 5. Bell State Similarity (BSS)

Two metrics for comparing HLLSets:

- **BSS_τ** (inclusion): How much of B is covered by A?
- **BSS_ρ** (exclusion): How much of A is outside B?

A morphism f: A → B exists when both thresholds are met:

- `BSS_τ(A→B) ≥ τ` (enough inclusion)
- `BSS_ρ(A→B) ≤ ρ` (not too much exclusion)

---

## Quick Start

### Installation

```bash
# Clone repository
cd hllset_manifold

# Install with uv (recommended)
uv sync

# Build C backend for performance
python setup.py build_ext --inplace
```

### First Steps

Check out the interactive notebooks:

1. **[10_lattice_evolution.ipynb](10_lattice_evolution.ipynb)** ⭐ - Complete demonstration
2. [01_quick_start.ipynb](01_quick_start.ipynb) - HLLSet basics
3. [03_adjacency_matrix.ipynb](03_adjacency_matrix.ipynb) - Adjacency matrices
4. [04_kernel_entanglement.ipynb](04_kernel_entanglement.ipynb) - Kernel operations

### Minimal Example

```python
from core.kernel import Kernel
from core.hrt import HRTConfig, HRTEvolution

# Create kernel for HLL operations
kernel = Kernel(p_bits=14)

# Configure HRT (keep small for manageable memory)
config = HRTConfig(
    p_bits=8,       # Controls AM dimension, not HLL precision!
    h_bits=16,
    tau=0.7,        # Morphism threshold (inclusion)
    rho=0.3,        # Morphism threshold (exclusion)
    epsilon=0.1
)

# Start evolution
evolution = HRTEvolution(config=config)

# Ingest knowledge
knowledge = {
    "animals": {"dog", "cat", "bird"},
    "properties": {"fur", "feathers", "warm_blooded"}
}
evolution.ingest(knowledge, kernel)

# Commit to history
state = evolution.evolve(
    kernel=kernel,
    commit_fn=lambda hrt: f"genesis_{hrt.name[:8]}"
)

print(f"Created HRT: {state.current.name}")
print(f"Lattice dimension: {len(state.current.lattice.row_basic)}")
```

---

## Core Architecture

### Immutability

Everything is immutable:

- HLLSets never change (operations return new instances)
- HRTs are snapshots (evolution creates new HRT with parent pointer)
- Content-addressed (SHA1 hash of structure = name)

This enables:

- Safe parallel processing
- Time-travel (any past state accessible via hash)
- Reproducible results

### The Kernel

The Kernel is your interface to HLL operations:

```python
kernel = Kernel(p_bits=14)  # 2^14 = 16,384 HLL registers

# Core operations (all return new HLLSets)
hllset = kernel.absorb(tokens)           # tokens → HLLSet
hllset = kernel.add(hllset, new_tokens)  # add tokens
union = kernel.union(hll1, hll2)         # combine sets
inter = kernel.intersection(hll1, hll2)  # intersection
```

**Important**: `p_bits` here controls HLL precision (sketch size), not AM dimension!

### HRT Configuration

```python
config = HRTConfig(
    p_bits=8,       # AM dimension parameter (NOT HLL precision!)
    h_bits=16,      # Hash bits for token indices
    tau=0.7,        # BSS inclusion threshold
    rho=0.3,        # BSS exclusion threshold
    epsilon=0.1     # ε-isomorphism tolerance
)

# Computed properties
dimension = config.dimension  # (2^8) * 16 + 2 = 4,098
```

**Critical**: `HRTConfig.p_bits` controls Adjacency Matrix dimension, NOT HyperLogLog precision!

- Small values (6-8) → Manageable memory
- Large values (14+) → Massive memory requirements (avoid!)

### Evolution Lifecycle

```python
evolution = HRTEvolution(config=config)

# 1. Ingest - Creates ephemeral "in_process" state
evolution.ingest(perceptron_data, kernel)

# 2. Evolve - Commits ephemeral → current (with parent pointer)
triple = evolution.evolve(kernel, commit_fn)

# Access states
current = triple.current        # Current committed state
in_process = triple.in_process  # Ephemeral (None after evolve)
step = triple.step_number       # Evolution step counter
```

---

## Projection: Predicting Next Ingestion

**The key insight**: Projection predicts what tokens will arrive next.

### How It Works

1. **Select row basic** from current W lattice
2. **Find referenced columns** via BSS connections (τ, ρ thresholds)
3. **Use evolution history** - (D, R, N) for each column
4. **Apply strategy** to predict tokens
5. **Return predicted token set**

### Strategies

- **Conservative**: Only retained tokens (R)
- **Growth**: Retained + new (R ∪ N)
- **Decay**: Retained minus deleted (R \ D)
- **Balanced**: Retained + half new

### Example

```python
def project_from_row_basic(row_basic, lattice, history, strategy, tau, rho):
    predicted_tokens = set()
    
    # Find columns this row references
    for col_idx, col_basic in enumerate(lattice.col_basic):
        if row_basic.bss_tau(col_basic) >= tau and \
           row_basic.bss_rho(col_basic) <= rho:
            
            # Apply strategy to column's (D, R, N)
            D, R, N = history[col_idx]['D'], history[col_idx]['R'], history[col_idx]['N']
            
            if strategy == 'conservative':
                predicted_tokens.update(R)
            elif strategy == 'growth':
                predicted_tokens.update(R | N)
            # ... other strategies
    
    return predicted_tokens
```

### Projection Workspace

Projections are **hypothetical** - store them separately:

```python
workspace = ProjectionWorkspace(base_state=current.name)

# Try multiple strategies
for strategy in ['conservative', 'growth', 'balanced']:
    predicted = project_from_row_basic(row, lattice, history, strategy, 0.3, 0.7)
    workspace.add_projection(f"proj_{strategy}", predicted, strategy, confidence)

# Select best
best_id, best_proj = workspace.select_best()

# Execute: hypothesis → reality
evolution.ingest({"category": best_proj['tokens']}, kernel)
next_state = evolution.evolve(kernel, commit_fn)
```

**Key principle**: Projections live in workspace (temporary), execution creates actual evolution (permanent).

---

## Performance & Parallelization

### Why It's Fast

1. **C backend**: Cython HLL implementation (10-100x speedup)
2. **No O(n²) bottlenecks**: Direct token access from (D,R,N)
3. **Immutable structures**: No locking needed
4. **Parallel projections**: Each (row, strategy) runs independently

### Complexity

**Single projection**: `O(C·k + C_ref·T·k)`

- C = total columns
- C_ref = referenced columns
- T = tokens per column
- k = HLL operations

**Parallel speedup**: R rows × S strategies = R·S independent projections

Example: 20 rows × 5 strategies = 100x speedup if fully parallelized!

### Memory Considerations

HRTConfig `p_bits` controls memory:

- `p_bits=6`: dimension=390, AM=152K elements (~0.6MB)
- `p_bits=8`: dimension=4,098, AM=16.8M elements (~67MB) ✓ Good
- `p_bits=10`: dimension=16,386, AM=268M elements (~1.1GB)
- `p_bits=14`: dimension=262,146, AM=68.7B elements (~275GB) ✗ Too large!

**Rule of thumb**: Use `p_bits=8` for development, `p_bits=6-7` for constrained environments.

---

## Project Structure

```
hllset_manifold/
├── core/                          # Core library
│   ├── hllset.py                 # HLLSet (Python wrapper)
│   ├── hll_core.pyx              # HLL implementation (Cython)
│   ├── hll_core.c                # Generated C code
│   ├── kernel.py                 # Kernel operations
│   ├── hrt.py                    # HRT, Lattice, Evolution
│   ├── immutable_tensor.py       # Immutable tensors for AM
│   ├── constants.py              # Shared constants
│   └── entanglement.py           # Entanglement operations
│
├── 10_lattice_evolution.ipynb    # ⭐ Main demonstration
├── 01_quick_start.ipynb          # Quick start guide  
├── 03_adjacency_matrix.ipynb     # AM fundamentals
├── 04_kernel_entanglement.ipynb  # Kernel & BSS
│
├── DOCS/                          # Extended documentation
│   ├── MANIFOLD_EVOLUTION.md     # Evolution & projection theory
│   ├── NTOKEN_ALGORITHM.md       # N-token ingestion details
│   └── ...
│
├── tests/                         # Test suite
├── examples/                      # Example scripts
└── deprecated/                    # Old implementations
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific tests
pytest tests/test_adjacency_matrix.py -v
pytest tests/test_manifold_drivers.py -v

# Test C backend
python test_c_backend.py
```

---

## Important Concepts to Remember

### HLLSets Are Fingerprints

❌ **Wrong**: "Store tokens in HLLSet, retrieve later"  
✓ **Right**: "Create compact sketch, estimate cardinality, compare with other sketches"

You **cannot** extract tokens from an HLLSet. It's a one-way probabilistic structure.

### Evolution History Must Be Tracked

❌ **Wrong**: "Extract (D,R,N) from HLLSets during projection"  
✓ **Right**: "Track (D,R,N) separately during actual ingestion"

Evolution triples come from observing real ingestion events, not from decomposing HLLSets.

### Projection vs Evolution

- **Projection** = Hypothesis (what might happen next)
- **Evolution** = Reality (what actually happened)

Store projections in workspace (temporary), execute selected projection to create evolution (permanent).

### Two Different p_bits

- **Kernel `p_bits`** = HLL precision (sketch size: 2^p registers)
- **HRTConfig `p_bits`** = AM dimension parameter (matrix size: ~2^(2p) elements)

They're different parameters for different purposes!

---

## Common Pitfalls

### Memory Explosion

```python
# ❌ BAD: Will try to allocate ~275GB!
config = HRTConfig(p_bits=14, ...)  

# ✓ GOOD: Manageable ~67MB
config = HRTConfig(p_bits=8, ...)
```

### Trying to Extract Tokens

```python
# ❌ BAD: No .data attribute!
tokens = hllset.data  

# ✓ GOOD: Track tokens separately
evolution_history[col_idx] = {'D': deleted, 'R': retained, 'N': new}
```

### Wrong BSS Signature

```python
# ❌ BAD: BSS methods don't take kernel!
bss = row.bss_tau(col, kernel)  

# ✓ GOOD: Just the other BasicHLLSet
bss = row.bss_tau(col)
```

---

## Documentation

- **[10_lattice_evolution.ipynb](10_lattice_evolution.ipynb)** - Working demonstration (start here!)
- **[DOCS/MANIFOLD_EVOLUTION.md](DOCS/MANIFOLD_EVOLUTION.md)** - Detailed theory & algorithms
- **[SUCCESS.md](SUCCESS.md)** - Implementation notes

---

## Requirements

- Python 3.10+
- NumPy
- PyTorch (for immutable tensors)
- Cython (for C backend)

Install everything:

```bash
uv sync
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## What Makes This Different?

Traditional knowledge graphs store explicit relationships. HLLSet Manifold:

- Uses **probabilistic sketches** instead of exact storage
- Tracks **evolution explicitly** with (D,R,N) triples
- Supports **predictive projection** from current state
- Enables **massive parallelization** via immutability
- Provides **full history** via content-addressed snapshots

It's designed for scenarios where:

- Uncertainty is inherent
- Knowledge evolves over time
- Predictions matter
- Scale is important
- Reproducibility is critical

---

## Current Status

✅ **Working demonstration** - See [10_lattice_evolution.ipynb](10_lattice_evolution.ipynb)  
✅ **Fast C backend** - Cython HLL implementation  
✅ **Correct projection** - Predicts next ingestion (not W iteration)  
✅ **Evolution tracking** - (D,R,N) triples with parent pointers  
✅ **Immutable architecture** - Content-addressed structures  
✅ **Parallel-ready** - Independent projections  
✅ **Tested** - Core functionality verified

🚧 **In progress**:

- Noether current computation (has bugs in core library)
- Distributed projection execution
- GPU acceleration for AM operations

---

## Contributing

The project is under active development. Key areas:

1. Fix `compute_noether_current()` in core library
2. Implement true parallel projection (ProcessPoolExecutor)
3. Add confidence scoring for projections
4. Performance benchmarks
5. More example notebooks

---

## Questions?

Check the [documentation](DOCS/) or run the [demonstration notebook](10_lattice_evolution.ipynb).

The key to understanding this project: **HLLSets are fingerprints, not containers**.
