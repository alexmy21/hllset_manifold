# Enhanced Kernel: Two-Layer Architecture

## Overview

The enhanced `kernel.py` implements a **two-layer architecture** that cleanly separates register operations from structural entanglement.

## ⚠️ IMPORTANT: HLLSets are NOT sets containing tokens

HLLSets are probabilistic register structures ("anti-sets") that:

- **ABSORB** tokens (hash them into registers)
- **DO NOT STORE** tokens (only register states remain)
- **BEHAVE LIKE** sets (union, intersection, cardinality estimation)
- **ARE NOT** sets (no element retrieval, no membership test)

The registers encode a probabilistic fingerprint of what was absorbed,
but the original tokens are **irretrievably lost**.

## ⚠️ CRITICAL DISTINCTION: HLLSets vs Lattices

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-LAYER ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 1: HLLSet (Register Layer)                                           │
│  ├─ Compares: Register states, estimated cardinalities                      │
│  ├─ Methods: find_isomorphism(), similarity()                               │
│  ├─ Returns: Morphism (register-level comparison)                           │
│  └─ NOT true entanglement - just register similarity!                       │
│                                                                             │
│  LAYER 2: Lattice (Structure Layer) ← TRUE ENTANGLEMENT                     │
│  ├─ Compares: Degree distributions, graph topology                          │
│  ├─ Methods: find_lattice_isomorphism(), validate_lattice_entanglement()    │
│  ├─ Returns: LatticeMorphism (structure-level comparison)                   │
│  ├─ Individual HLLSets are IRRELEVANT - only STRUCTURE matters              │
│  └─ Two lattices can be entangled even from completely different inputs!    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Concept | Layer | Compares | Method |
| ------------------ | ------------ | ----------------- | -------------------------------- |
| Similarity | HLLSet | Registers | `find_isomorphism()` |
| Cardinality | HLLSet | Estimated count | `cardinality()` |
| **ENTANGLEMENT** | **Lattice** | **Structure** | `find_lattice_isomorphism()` |
| Structural Match | Lattice | Topology | `validate_lattice_entanglement()` |

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    KERNEL (Stateless)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: Basic Operations (HLLSet/Register Layer)          │
│  ├─ absorb: tokens → HLLSet (tokens hashed, then lost)      │
│  ├─ union, intersect, diff: HLLSet × HLLSet → HLLSet        │
│  ├─ add: HLLSet × tokens → HLLSet                           │
│  └─ find_isomorphism: HLLSet × HLLSet → Morphism            │
│                                                             │
│  Level 2: Entanglement (Lattice/Structure Layer)            │
│  ├─ find_lattice_isomorphism: Lattice × Lattice → LMorphism │
│  ├─ validate_lattice_entanglement: [Lattice] → (bool, coh)  │
│  ├─ reproduce: HLLSet → HLLSet (with mutation)              │
│  └─ commit: HLLSet → HLLSet (stabilize)                     │
│                                                             │
│  Level 3: Network Operations                                │
│  ├─ build_tensor: [HLLSet] → 3D Tensor                      │
│  ├─ measure_coherence: Tensor → [0,1]                       │
│  └─ detect_singularity: [HLLSet] → SingularityReport        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Morphism (Content-Level) - NOT True Entanglement

A `Morphism` between two HLLSets represents **content overlap**:

```python
morph = kernel.find_isomorphism(hll_a, hll_b, epsilon=0.05)
# Returns Morphism if HLLSets have similar cardinality and content overlap
# This is CONTENT comparison, NOT true entanglement!
```

**Properties:**

- `source_hash`: Content address of source HLLSet
- `target_hash`: Content address of target HLLSet
- `similarity`: Jaccard similarity score (content overlap)
- `epsilon`: Tolerance for approximate matching
- `is_isomorphism`: Whether content is ε-similar

### 2. LatticeMorphism (Structure-Level) - TRUE Entanglement

A `LatticeMorphism` between two HLLSetLattices represents **structural similarity**:

```python
lattice_morph = kernel.find_lattice_isomorphism(lattice_a, lattice_b, epsilon=0.05)
# Returns LatticeMorphism if STRUCTURES are similar
# Nodes (HLLSets) are IRRELEVANT - only topology matters!
```

**Properties:**

- `source_lattice_hash`: Hash of source lattice
- `target_lattice_hash`: Hash of target lattice
- `row_degree_correlation`: Correlation of row degree sequences
- `col_degree_correlation`: Correlation of column degree sequences
- `overall_structure_match`: Combined structural similarity
- `epsilon_isomorphic_prob`: Probability of structural ε-isomorphism

### 3. True Entanglement (Lattice-Level)

Multiple **lattices** are entangled when their **structures** match:

```python
is_entangled, coherence = kernel.validate_lattice_entanglement(
    [lattice_1, lattice_2, lattice_3], 
    epsilon=0.15
)
# Checks: pairwise STRUCTURAL morphisms, degree correlations
# Two lattices can be entangled with ZERO shared tokens!
```

**Entanglement Criteria:**

- **>90% pairs** have structural ε-isomorphisms
- **Coherence >50%** (average similarity)
- Morphisms compose properly

### 3. ICASRA Operations

Inspired by **Immutable Content-Addressable Self-Reproducing Automata**:

#### A. Constructor (commit)

```python
stable = kernel.commit(candidate)
# Validates and stabilizes HLLSet
# Idempotent: commit(commit(x)) = commit(x)
```

#### B. Copier (reproduce)

```python
child = kernel.reproduce(parent, mutation_rate=0.1)
# Creates structurally similar child with mutations
# Mimics evolutionary reproduction
```

#### C. Controller

- Implemented at OS/HRT level (temporal coordination)

#### D. Interface (reproduce mutations)

- Environmental interaction through mutation rate

### 4. 3D Tensor Architecture

For a network of N installations:

```text
T[i, j, k] = relationship between concept i and j in installation k

Dimensions:
- Axis 0 (i): Concept space (from HLL registers)
- Axis 1 (j): Concept space (from HLL registers)  
- Axis 2 (k): Installation index
```

```python
tensor = kernel.build_tensor([hll_1, hll_2, hll_3])
# Shape: (concept_dim, concept_dim, n_installations)
```

**Decomposition reveals:**

- Universal patterns across installations
- Installation-specific variations
- Entanglement structure

### 5. Singularity Detection

An **Entanglement Singularity** occurs when:

1. **Complete entanglement** (>95% pairs isomorphic)
2. **High coherence** (>70%)
3. **Emergent properties** (system-level behavior)
4. **Stable phase** ("Singularity" phase)

```python
report = kernel.detect_singularity(network, epsilon=0.15)
print(report.has_singularity)  # True/False
print(report.phase)  # "Disordered", "Critical", "Ordered", "Singularity"
print(report.entanglement_ratio)  # 0.0 to 1.0
print(report.coherence)  # 0.0 to 1.0
```

## Phase Transitions

As a network grows, it passes through phases:

```text
Phase 0: Disordered
├─ Entanglement ratio: <30%
├─ No systematic structure
└─ Independent installations

Phase 1: Critical
├─ Entanglement ratio: 30-70%
├─ Scale-free partial entanglement
└─ Emergent patterns begin

Phase 2: Ordered
├─ Entanglement ratio: 70-95%
├─ Strong structural coherence
└─ Universal lattice forming

Phase 3: Singularity 🌟
├─ Entanglement ratio: >95%
├─ Coherence: >70%
├─ Complete entanglement achieved
└─ System consciousness emerges
```

## Example Usage

### Basic Operations

```python
from core.kernel import Kernel

kernel = Kernel()

# Level 1: Basic morphisms
hll_a = kernel.absorb({'a', 'b', 'c'})
hll_b = kernel.absorb({'c', 'd', 'e'})
hll_union = kernel.union(hll_a, hll_b)
```

### Entanglement Detection

```python
# Create installations
installations = [
    kernel.absorb(set(f'token_{i}' for i in range(0, 100))),
    kernel.absorb(set(f'token_{i}' for i in range(50, 150))),
    kernel.absorb(set(f'token_{i}' for i in range(100, 200)))
]

# Find morphisms
morph_01 = kernel.find_isomorphism(
    installations[0], 
    installations[1], 
    epsilon=0.10
)

if morph_01:
    print(f"Isomorphism found: {morph_01.similarity:.1%} similar")

# Validate mutual entanglement
is_entangled, coherence = kernel.validate_entanglement(
    installations, 
    epsilon=0.10
)
print(f"Network entangled: {is_entangled}, coherence: {coherence:.1%}")
```

### Reproduction Cycle (ICASRA)

```python
# Create parent installation
parent = kernel.absorb(set(f'concept_{i}' for i in range(100)))

# Reproduce with mutation (B + D operations)
child = kernel.reproduce(parent, mutation_rate=0.1)

# Commit/stabilize (A operation)
stable_child = kernel.commit(child)

print(f"Parent: {parent.short_name}")
print(f"Child: {stable_child.short_name}")
```

### Singularity Engineering

```python
# Build evolving network
network = []

for step in range(5):
    # Add installation with overlapping content
    tokens = set(f'universal_{i}' for i in range(step*30, (step+1)*30 + 40))
    network.append(kernel.absorb(tokens))
    
    # Check singularity at each step
    report = kernel.detect_singularity(network, epsilon=0.15)
    
    print(f"Step {step+1}:")
    print(f"  Phase: {report.phase}")
    print(f"  Entanglement: {report.entanglement_ratio:.1%}")
    
    if report.has_singularity:
        print("  🌟 SINGULARITY ACHIEVED!")
        break
```

### 3D Tensor Analysis

```python
# Build network
network = [kernel.absorb(...) for _ in range(5)]

# Construct 3D tensor
tensor = kernel.build_tensor(network)
print(f"Tensor shape: {tensor.shape}")

# Measure coherence
coherence = kernel.measure_coherence(tensor)
print(f"Network coherence: {coherence:.1%}")

# Analyze per-installation slices
for k in range(tensor.shape[2]):
    slice_k = tensor[:, :, k]
    print(f"Installation {k} relationship matrix norm: {np.linalg.norm(slice_k):.2f}")
```

## Integration Points

### With HRT (Hash Relational Tensor)

The kernel provides the **transformation operations** that HRT uses for temporal evolution:

```python
# HRT creates snapshots at t-1, t, t+1
# Kernel provides transformations between states
W_t = kernel.union(W_t_minus_1, new_hll)

# Entanglement morphism between temporal states
φ_temporal = kernel.find_isomorphism(W_t_minus_1, W_t)
```

### With Entanglement Engine

The kernel's `find_isomorphism` and `validate_entanglement` methods provide the foundation for:

- Cross-installation communication
- Structural synchronization
- Emergence detection

### With Manifold OS

- **Stateless transformations** - Kernel has no state
- **Operation recording** - OS tracks history via `record_operation()`
- **Content addressing** - All outputs have deterministic hashes

## Mathematical Foundation

### Morphism Composition

For morphisms `φ: A → B` and `ψ: B → C`:

```python
morph_ab = kernel.find_isomorphism(hll_a, hll_b)
morph_bc = kernel.find_isomorphism(hll_b, hll_c)

# Should have: ψ ∘ φ: A → C
morph_ac = kernel.find_isomorphism(hll_a, hll_c)

# Commuting diagram check
assert morph_ac.similarity ≈ morph_ab.similarity * morph_bc.similarity
```

### Category Theory Foundation

```text
HLLSet forms a category:
- Objects: HLLSets
- Morphisms: ε-isomorphisms
- Identity: id_A = find_isomorphism(A, A) with similarity=1.0
- Composition: Associative morphism composition
```

### Information Geometry

The network coherence measure is related to:

- **Fisher information metric** on the space of HLLSets
- **Riemannian distance** between installations
- **Curvature** of the entanglement manifold

## Performance Characteristics

- **Basic operations**: O(m) where m = number of HLL registers (typically 2^12 = 4096)
- **Isomorphism finding**: O(m) similarity computation
- **Entanglement validation**: O(n²) for n installations (all pairs)
- **3D tensor construction**: O(m² × n) for n installations
- **Singularity detection**: O(n² + m²n) combined overhead
- **Common subgraph extraction**: O(n² + e) where n = nodes, e = edges

For typical networks:

- n = 5-10 installations
- m = 4096 registers
- Detection time: <1 second

## Common Subgraph Extraction (Entanglement as Structure)

### Concept

**Entanglement = Common Subgraph** of two (or more) lattices.

**Key Insight**: Entanglement doesn't need to be a single connected subgraph.
It can be a **collection of disconnected matching fragments** ("entanglement seeds").

The `CommonSubgraphExtractor` finds the **structural overlap** between lattices:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│             ENTANGLEMENT = COLLECTION OF MATCHING FRAGMENTS                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Lattice L1:       Lattice L2:       Entanglement (Multiple Fragments):     │
│                                                                             │
│  [START]──→A       [START]──→X       Fragment 1: [START]──→●                │
│      │     │           │     │                        │                     │
│      ▼     ▼           ▼     ▼                        ▼                     │
│      B──→C             Y──→Z                          ●                     │
│      │                 │                                                    │
│      ▼                 ▼             Fragment 2:  ●──→●                     │
│   [END]  D          [END]  W                         │                     │
│          │                 │                         ▼                      │
│          ▼                 ▼                      [END]                     │
│          E                 V                                                │
│                                                                             │
│  Fragments can be DISCONNECTED - each is a "seed" of entanglement           │
│  Extension operation: propagate seeds to neighboring nodes                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Properties

- **Directed graphs** (not necessarily acyclic)
- **START/END markers** respected and prioritized
- **Approximate solution** (polynomial time, not optimal MCS)
- **"Close enough" matching** - degrees don't need exact match
- **Multiple fragments** - returns disconnected matching regions
- **Extension/closure** - propagate entanglement to neighbors

### API Usage

```python
from core.entanglement import extract_entanglement, CommonSubgraphExtractor

# High-level API
subgraph = extract_entanglement(lattice1, lattice2, degree_tolerance=0.3)

print(f"Total fragments: {subgraph.fragment_count}")
print(f"Nodes matched: {subgraph.size}")
print(f"Structural similarity: {subgraph.structural_similarity:.1%}")

# Access individual fragments
for i, fragment in enumerate(subgraph.fragments):
    print(f"Fragment {i}: {fragment.size} nodes, {fragment.edge_count} edges")
    print(f"  Mapping: {fragment.node_mapping}")

# Extension (closure) operation
extractor = CommonSubgraphExtractor()
extended = extractor.extend_in_lattice(subgraph, lattice1, lattice2, depth=1)
print(f"Extended L1 nodes: {extended.extended_l1_nodes}")
print(f"Extended L2 nodes: {extended.extended_l2_nodes}")
```

### Data Classes

#### `EntanglementFragment`

A single disconnected matching component:

| Property | Type | Description |
|----------|------|-------------|
| `node_mapping` | `Dict[int, int]` | Node mapping within this fragment |
| `matched_edges` | `List[...]` | Edges matched in this fragment |
| `start_nodes` | `Set[int]` | START nodes in this fragment |
| `end_nodes` | `Set[int]` | END nodes in this fragment |
| `size` | `int` | Number of nodes |
| `edge_count` | `int` | Number of edges |

#### `EntanglementSubgraph`

Collection of fragments representing full entanglement:

| Property | Type | Description |
|----------|------|-------------|
| `fragments` | `List[EntanglementFragment]` | Disconnected matching regions |
| `matched_nodes` | `Dict[int, int]` | Aggregate node mapping L1 → L2 |
| `matched_edges` | `List[...]` | All edge pairs preserved under mapping |
| `start_nodes` | `Set[int]` | START-marked nodes in entanglement |
| `end_nodes` | `Set[int]` | END-marked nodes in entanglement |
| `node_coverage` | `float` | Fraction of nodes matched |
| `edge_coverage` | `float` | Fraction of edges matched |
| `structural_similarity` | `float` | Overall similarity score |
| `fragment_count` | `int` | Number of disconnected fragments |

#### `ExtendedEntanglement`

Entanglement extended to neighboring nodes:

| Property | Type | Description |
|----------|------|-------------|
| `base_entanglement` | `EntanglementSubgraph` | Original entanglement |
| `extended_l1_nodes` | `Set[int]` | Neighbors discovered in L1 |
| `extended_l2_nodes` | `Set[int]` | Neighbors discovered in L2 |
| `l1_extension_edges` | `Set[...]` | Edges to L1 neighbors |
| `l2_extension_edges` | `Set[...]` | Edges to L2 neighbors |
| `total_l1_nodes` | `int` | Base + extended in L1 |
| `total_l2_nodes` | `int` | Base + extended in L2 |

### Algorithm (Approximate)

```text
1. Build adjacency lists (row→col→row composition)
2. Compute node signatures: (out_degree, in_degree, is_start, is_end, cardinality)
3. Match nodes by signature similarity (greedy, "close enough" degrees)
4. Verify edges: keep edges that exist in both lattices under mapping
5. Identify disconnected fragments (connected components)
6. Optionally extend to neighbors (closure operation)
7. Return EntanglementSubgraph with fragments
```

**Complexity**: O(n² + e) where n = nodes, e = edges

### Example: Cross-Modal Entanglement with Extension

```python
# Two lattices from different modalities (completely different tokens!)
sensory_lattice = build_lattice_from_visual_features(images)
language_lattice = build_lattice_from_text(documents)

# Find structural entanglement (common subgraph fragments)
extractor = CommonSubgraphExtractor(degree_tolerance=0.4)
entanglement = extractor.extract(sensory_lattice, language_lattice)

print(f"Found {entanglement.fragment_count} entanglement fragments")

# Extend entanglement to neighbors
extended = extractor.extend_in_lattice(
    entanglement, sensory_lattice, language_lattice, depth=2
)

print(f"Total reach in sensory lattice: {extended.total_l1_nodes}")
print(f"Total reach in language lattice: {extended.total_l2_nodes}")
```

## Testing

Run the comprehensive test:

```bash
.venv/bin/python test_kernel_entanglement.py
```

Or explore the demo notebook:

```bash
jupyter notebook notebooks/entanglement_subgraph_demo.ipynb
```

## Future Enhancements

### 1. Proper Register Reconstruction

Currently, `reproduce()` returns the parent unchanged. Future work:

- Rebuild HLLSet from mutated registers
- Preserve ε-isomorphism with parent
- Enable true evolutionary cycles

### 2. Advanced Tensor Decomposition

- Tucker decomposition for universal patterns
- CP decomposition for factor analysis
- Attention-like mechanisms for installation interaction

### 3. Curvature Computation

Implement the entanglement curvature tensor:

```python
R_ijkl = compute_curvature(network)
# R = 0 → flat connection (perfect entanglement)
# R ≠ 0 → curvature obstructions
```

### 4. Meta-Entanglement

Networks of networks:

```python
ein_1 = [kernel.absorb(...) for _ in range(5)]
ein_2 = [kernel.absorb(...) for _ in range(5)]

# Find isomorphism between entire networks
meta_morph = kernel.find_network_isomorphism(ein_1, ein_2)
```

## References

- **ENTANGLEMENT_SINGULARITY.md** - Theoretical foundation
- **PDF/Entanglement-singularity.pdf** - Detailed mathematics
- **PDF/United_HLLSet_Framework.pdf** - Integration architecture
- **core/hllset.py** - HLLSet implementation (now with C backend!)
- **core/hrt.py** - Temporal evolution framework
- **core/entanglement.py** - Entanglement morphism engine

## Summary

The enhanced kernel now provides:

✅ **Three-level architecture** - Basic, entanglement, network operations  
✅ **Two-layer distinction** - Register Layer (HLLSet) vs Structure Layer (Lattice)  
✅ **ICASRA operations** - Reproduce, commit, validate  
✅ **Morphism detection** - Find ε-isomorphisms between HLLSets  
✅ **Entanglement validation** - Check mutual structural coherence  
✅ **Common subgraph extraction** - Extract entanglement as shared structure  
✅ **START/END marker support** - Directed graph structure preserved  
✅ **3D tensor construction** - Multi-installation representation  
✅ **Singularity detection** - Identify emergent system consciousness  
✅ **Phase transitions** - Track network evolution through states  

The kernel is **stateless**, **pure**, **composable**, and **entanglement-aware**, ready for building Entangled ICASRA Networks that exhibit emergent intelligence! 🌟
