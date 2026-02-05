# HRT Lattice (W) - Deep Structure and Evolution Theory

## Overview

The HRT Lattice W is a directed graph structure that parallels the Adjacency Matrix (AM) but operates at a higher level of abstraction, with HLLSets as elements rather than tokens.

## Structural Parallels: AM vs W

### Similarities

1. **Same Dimensions**: Both have dimension `N × N` where `N = 2^P * h_bits + 2`
2. **Directed Structures**: Both define directed graphs (not necessarily acyclic)
3. **Cell Strength**: Cell values represent link strength between nodes
4. **Boundary Markers**: 
   - AM: START/END tokens (special `(reg, zeros)` identifiers)
   - W: Empty HLLSet (∅) and Universal HLLSet (U)

### Key Differences

| Aspect | AM (Adjacency Matrix) | W (HRT Lattice) |
| -------- | ---------------------- | ----------------- |
| **Elements** | Tokens (strings) | HLLSets (probabilistic sets) |
| **Cell Values** | Integer frequencies | HLLSets (R in (D, R, N) triple) |
| **Identifiers** | `(reg, zeros)` tuples | SHA1 content hashes |
| **Node Type** | Simple tokens | **HLL Categories**: `(HLLSet, τ, ρ, φ)` |
| **Evolution** | Static frequencies | Dynamic: `H(t+1) = [H(t) \ D] ∪ N` |

## HLL Categories: Extended Structure

Each node in W is not just an HLLSet, but a **category** with additional structure:

```text
Category = (H, τ, ρ, φ)

where:
  H = HLLSet (the base set)
  τ = Similarity bound (tolerance threshold)
  ρ = Resistance bound (intolerance threshold)
  φ = Morphism (transformation function that builds H)
```

### Tolerance and Resistance

- **τ (tau)**: Similarity threshold - how similar must sets be to be considered connected?
  - `similarity(H₁, H₂) ≥ τ` ⟹ connected
  - Allows "fuzzy" connections in W
  
- **ρ (rho)**: Resistance threshold - how dissimilar must sets be to be disconnected?
  - `similarity(H₁, H₂) < ρ` ⟹ disconnected
  - Creates hard boundaries

**CRITICAL INSIGHT**: Rows and columns connected in AM may be **disconnected in W** due to τ/ρ thresholds!

This divergence creates a powerful semantic filtering mechanism:

1. **AM shows raw co-occurrence**: If tokens appear together, AM cells are non-zero
2. **W shows semantic connections**: Only HLLSets passing BSS(τ, ρ) thresholds connect
3. **τ/ρ as priority weights**: The thresholds act as comprehensive priority weights in choosing which HLLSet to follow

**DEEPER INSIGHT**: τ/ρ is the **mechanics of contextual selection**. It's the concrete mechanism by which **context actively selects what fits within it**!

From paper Section 7 (Contextual Selection Principle):

- Abstract principle: "Contexts actively select compatible elements"
- Concrete mechanism: BSS(τ, ρ) thresholds determine compatibility
- Selection operator S_C: Returns 1 iff BSS_τ ≥ τ AND BSS_ρ ≤ ρ

This completes the picture:

```text
Context (current HLLSet H_C)
    ↓
    Uses BSS with τ/ρ thresholds as selection criteria
    ↓
    Selects elements that satisfy: BSS_τ(H_C, x) ≥ τ AND BSS_ρ(H_C, x) ≤ ρ
    ↓
    Result: Context determines what is "in it" (not passive membership!)
```

#### Priority Weighting via τ/ρ

Given multiple possible next states, τ/ρ provides natural ranking:

```text
Priority(H₁ → H₂) = BSS_τ(H₁ → H₂) - λ·BSS_ρ(H₁ → H₂)

where:
  BSS_τ = |H₁ ∩ H₂| / |H₂|  (inclusion similarity)
  BSS_ρ = |H₁ \ H₂| / |H₂|   (exclusion dissimilarity)
  λ = weighting factor (typically 1.0)
```

**Selection Rule**: Choose HLLSet with highest priority that satisfies:

- `BSS_τ(current → candidate) ≥ τ`
- `BSS_ρ(current → candidate) ≤ ρ`

This means:

- **High τ**: Stricter semantic coherence required (conservative paths)
- **Low τ**: More exploratory paths allowed (creative connections)
- **Low ρ**: Tolerates more dissimilarity (diverse options)
- **High ρ**: Requires tight semantic fit (focused paths)

**Example Use Cases**:

1. **Focused generation** (τ=0.8, ρ=0.2): Stay close to context
2. **Creative exploration** (τ=0.5, ρ=0.4): Allow semantic jumps
3. **Strict coherence** (τ=0.9, ρ=0.1): Maximum semantic consistency
4. **Diverse sampling** (τ=0.6, ρ=0.5): Balance novelty and relevance

### Morphisms (φ)

Each basic HLLSet in W is created via a morphism:

```text
φ: TokenSet → HLLSet
```

Examples:

- `φ_row_i`: Creates row basic HLLSet from tokens
- `φ_col_j`: Creates column basic HLLSet from tokens
- `φ_union`: Combines multiple HLLSets
- `φ_diff`: Computes set difference

## Evolution Dynamics: The (D, R, N) Triple

W evolves through time via the formula:

```text
H(t + 1) = [H(t) \ D] ∪ N

where:
  H(t)     = Current state (row HLLSet)
  H(t+1)   = Next state (column HLLSet)
  D        = Deleted elements (removed from H(t))
  R        = Retained elements (D ∩ N, what stays)
  N        = New elements (added to H(t))
```

### Cell Values in W

Unlike AM where `AM[i,j] = frequency`, in W:

```text
W[i, j] = R_{i→j}

R = Morphism result describing the transition
R captures what is retained/shared between H(row_i) and H(col_j)
```

### Connection to Morphisms

```text
H(col_j) = φ(H(row_i), D, N)
         = [H(row_i) \ D] ∪ N
```

The cell `W[i,j]` stores the HLLSet **R** that describes this transformation.

## Noether Current: Conservation and Sustainability

To ensure **sustainability** of the lattice evolution, we apply **Noether's theorem** concepts:

### Conservation Law

For sustainable evolution:
```
|D| - |N| = 0

i.e., |D| = |N|
```

**Meaning**:

- The cardinality of deleted elements equals cardinality of new elements
- Total "information mass" is conserved
- System maintains equilibrium

### Noether Current

The **current** through the transition is:

```text
J = |N| - |D|

Sustainability: J = 0 (current = 0)
Growth:         J > 0 (more added than removed)
Decay:          J < 0 (more removed than added)
```

### Physical Interpretation

In physics, Noether's theorem states:
> Every continuous symmetry corresponds to a conservation law

In HLLSets:

- **Symmetry**: Time-translation invariance (evolution rules don't change)
- **Conservation**: Total cardinality (information content)
- **Current**: Flow of elements (J = ∂H/∂t)

## Reconstruction from Basic HLLSets

Just as AM enables order reconstruction via traversal, W enables **HLLSet reconstruction** from basic HLLSets:

### Algorithm (Parallel to AM Traversal)

```python
def reconstruct_hllset_from_lattice(W, start_idx, end_idx, threshold):
    """
    Reconstruct HLLSet by traversing W from start to end.
    
    Similar to AM order reconstruction, but builds HLLSet.
    """
    current = start_idx
    reconstructed = W.get_row_basic(current).hllset  # Start with row
    
    path = [current]
    visited = set([current])
    
    while current != end_idx:
        # Find strongest connection (highest cardinality of R)
        next_idx = None
        max_strength = 0
        
        for j in range(W.config.dimension):
            if j in visited:
                continue
                
            # Check category constraints
            if not satisfies_category(W, current, j):
                continue
            
            # Get R (retained/shared elements)
            R = W.cells[current, j]
            strength = R.cardinality() if R else 0
            
            if strength > max_strength:
                max_strength = strength
                next_idx = j
        
        if next_idx is None:
            break
            
        # Evolve: H(t+1) = [H(t) \ D] ∪ N
        col_hllset = W.get_col_basic(next_idx).hllset
        reconstructed = evolve_hllset(reconstructed, col_hllset)
        
        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx
    
    return reconstructed, path

def satisfies_category(W, row_idx, col_idx):
    """Check if connection satisfies τ/ρ thresholds."""
    row_cat = W.get_row_category(row_idx)
    col_cat = W.get_col_category(col_idx)
    
    row_hll = row_cat.hllset
    col_hll = col_cat.hllset
    
    sim = similarity(row_hll, col_hll)
    
    # Must exceed similarity bound AND not exceed resistance bound
    return sim >= row_cat.tau and sim >= (1.0 - row_cat.rho)
```

## Implementation Extensions Needed

To fully realize this theory in code, we need:

### 1. Category Structure

```python
@dataclass(frozen=True)
class HLLCategory:
    """Extended structure for lattice nodes."""
    hllset: HLLSet
    tau: float = 0.5      # Similarity bound
    rho: float = 0.3      # Resistance bound  
    phi: Morphism         # Creation morphism
    
    def is_connected_to(self, other: HLLCategory, kernel) -> bool:
        """Check if categories are connected via τ/ρ."""
        sim = kernel.similarity(self.hllset, other.hllset)
        return sim >= self.tau and sim >= (1.0 - self.rho)
```

### 2. Evolution Triple

```python
@dataclass(frozen=True)
class EvolutionTriple:
    """(D, R, N) triple for H(t) → H(t+1)."""
    D: HLLSet  # Deleted
    R: HLLSet  # Retained (D ∩ N)
    N: HLLSet  # New
    
    @property
    def noether_current(self) -> float:
        """Conservation current: |N| - |D|."""
        return self.N.cardinality() - self.D.cardinality()
    
    @property
    def is_sustainable(self, epsilon: float = 0.1) -> bool:
        """Check if evolution conserves cardinality."""
        return abs(self.noether_current) < epsilon
    
    def evolve(self, H_t: HLLSet, kernel) -> HLLSet:
        """Apply evolution: H(t+1) = [H(t) \ D] ∪ N."""
        return kernel.union(kernel.diff(H_t, self.D), self.N)
```

### 3. Extended Lattice

```python
@dataclass(frozen=True)
class CategoryLattice:
    """W lattice with full category structure."""
    config: HRTConfig
    row_categories: Tuple[HLLCategory, ...]
    col_categories: Tuple[HLLCategory, ...]
    cells: Dict[Tuple[int, int], EvolutionTriple]  # (i,j) → (D,R,N)
    
    def get_transition(self, row_idx: int, col_idx: int) -> Optional[EvolutionTriple]:
        """Get evolution triple for row_i → col_j."""
        return self.cells.get((row_idx, col_idx))
    
    def reconstruct_hllset(self, start_idx: int, end_idx: int, 
                          threshold: float = 0.9) -> Tuple[HLLSet, List[int]]:
        """Reconstruct HLLSet by traversing lattice."""
        # Implementation as shown above
        pass
```

## Mathematical Properties

### 1. Lattice Structure

W forms a **bounded lattice** with:

- **⊥ (bottom)**: Empty HLLSet ∅
- **⊤ (top)**: Universal HLLSet U (all possible elements)
- **∨ (join)**: HLLSet union
- **∧ (meet)**: HLLSet intersection

### 2. Category Theory Connection

The morphisms φ form a **category**:

- **Objects**: HLLSets
- **Morphisms**: Transformations (union, diff, etc.)
- **Composition**: Sequential application
- **Identity**: id(H) = H

### 3. Conservation Symmetry

The Noether current reflects:

- **Time symmetry**: Evolution rules are time-invariant
- **Conservation**: Total cardinality maintained
- **Equilibrium**: |D| = |N| ⟹ stable system

## Applications

1. **Dynamic HLLSet Evolution**: Track how sets change over time
2. **Causal Structure**: D, R, N reveal what caused changes
3. **Reconstruction**: Build HLLSets from lattice traversal (like AM order reconstruction)
4. **Conservation Analysis**: Detect growing/shrinking systems via Noether current
5. **Category Filtering**: Use τ/ρ to filter connections by similarity
6. **Priority-Weighted Path Selection**: Leverage BSS(τ, ρ) for intelligent navigation

## Priority-Weighted Path Selection

### The Key Insight: AM ≠ W Connectivity

**AM connectivity** (tokens co-occur):

```text
AM[i,j] > 0  ⟹  tokens at i and j appeared in sequence
```

**W connectivity** (semantic coherence):

```text
W[i,j] exists  ⟺  BSS_τ(H_i → H_j) ≥ τ  AND  BSS_ρ(H_i → H_j) ≤ ρ
```

**Result**: AM may have edges that W filters out!

### Priority Function

Given current state `H_current` and candidate next states `{H₁, H₂, ..., Hₙ}`:

```python
def compute_priority(H_current, H_candidate, tau, rho, lambda_weight=1.0):
    """
    Compute priority score for transition H_current → H_candidate.
    
    Higher score = better semantic fit.
    """
    # Bell State Similarity metrics
    bss_tau = intersection_cardinality(H_current, H_candidate) / cardinality(H_candidate)
    bss_rho = (cardinality(H_current) - intersection_cardinality(H_current, H_candidate)) / cardinality(H_candidate)
    
    # Combined priority score
    priority = bss_tau - lambda_weight * bss_rho
    
    # Check thresholds
    if bss_tau < tau or bss_rho > rho:
        return -float('inf')  # Not eligible
    
    return priority
```

### Selection Strategies

#### 1. Greedy Maximum Priority

```python
def select_next_greedy(H_current, candidates, tau, rho):
    """Select candidate with highest priority."""
    best_priority = -float('inf')
    best_candidate = None
    
    for H_cand in candidates:
        priority = compute_priority(H_current, H_cand, tau, rho)
        if priority > best_priority:
            best_priority = priority
            best_candidate = H_cand
    
    return best_candidate
```

#### 2. Stochastic Sampling (Temperature-Controlled)

```python
def select_next_stochastic(H_current, candidates, tau, rho, temperature=1.0):
    """Sample next state probabilistically based on priorities."""
    priorities = [compute_priority(H_current, H_cand, tau, rho) for H_cand in candidates]
    
    # Filter out ineligible (-inf priority)
    valid = [(cand, p) for cand, p in zip(candidates, priorities) if p > -float('inf')]
    
    if not valid:
        return None
    
    # Softmax with temperature
    scores = [p / temperature for _, p in valid]
    probs = softmax(scores)
    
    return random.choice(valid, p=probs)[0]
```

#### 3. Beam Search (Multiple Paths)

```python
def beam_search_W(W, start_idx, end_idx, tau, rho, beam_width=5):
    """
    Explore multiple paths simultaneously, keeping top-k by cumulative priority.
    """
    beams = [(start_idx, [], 0.0)]  # (current_idx, path, cumulative_score)
    
    while beams:
        new_beams = []
        
        for current_idx, path, score in beams:
            if current_idx == end_idx:
                return path, score  # Found path to end
            
            # Expand from current node
            for next_idx in range(W.config.dimension):
                if next_idx in path:
                    continue  # Avoid cycles
                
                H_current = W.get_row_basic(current_idx).hllset
                H_next = W.get_col_basic(next_idx).hllset
                
                priority = compute_priority(H_current, H_next, tau, rho)
                
                if priority > -float('inf'):
                    new_beams.append((
                        next_idx,
                        path + [next_idx],
                        score + priority
                    ))
        
        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]
    
    return None, 0.0  # No path found
```

### Use Case Examples

#### Language Generation

```python
# Conservative, coherent generation
tau = 0.8  # High similarity required
rho = 0.2  # Low exclusion tolerance
# Result: Stays close to context, safe predictions

# Creative, exploratory generation
tau = 0.5  # Lower similarity threshold
rho = 0.5  # Higher exclusion tolerance
# Result: More diverse, creative outputs
```

#### Semantic Search

```python
# Find semantically similar documents
def semantic_search(query_hllset, document_hllsets, tau=0.6, rho=0.3):
    """Return documents satisfying BSS thresholds."""
    results = []
    
    for doc_hllset in document_hllsets:
        priority = compute_priority(query_hllset, doc_hllset, tau, rho)
        if priority > -float('inf'):
            results.append((doc_hllset, priority))
    
    return sorted(results, key=lambda x: x[1], reverse=True)
```

#### Contextual Recommendation

```python
# Recommend next items based on user context
def recommend_next(user_context_hllset, item_hllsets, tau=0.7, rho=0.2, top_k=10):
    """Recommend top-k items that fit user context."""
    priorities = []
    
    for item_hllset in item_hllsets:
        priority = compute_priority(user_context_hllset, item_hllset, tau, rho)
        if priority > -float('inf'):
            priorities.append((item_hllset, priority))
    
    # Sort by priority and return top-k
    priorities.sort(key=lambda x: x[1], reverse=True)
    return priorities[:top_k]
```

### Adaptive τ/ρ Adjustment

Dynamically adjust thresholds based on context:

```python
def adaptive_thresholds(H_current, candidates, desired_diversity=0.5):
    """
    Adjust τ/ρ to achieve desired diversity level.
    
    Low diversity → High τ, low ρ (conservative)
    High diversity → Low τ, high ρ (exploratory)
    """
    base_tau = 0.7
    base_rho = 0.3
    
    # Adjust based on desired diversity
    tau = base_tau + (1 - desired_diversity) * 0.2
    rho = base_rho + desired_diversity * 0.2
    
    # Ensure constraint: 0 ≤ ρ < τ ≤ 1
    tau = min(0.95, max(0.5, tau))
    rho = max(0.05, min(tau - 0.1, rho))
    
    return tau, rho
```

### Performance Characteristics

**Computational Complexity**:

- Computing BSS: O(m) where m = number of HLL registers
- Selecting from n candidates: O(n × m)
- Beam search: O(beam_width × depth × n × m)

**Memory**:

- Each HLLSet: O(m) = O(2^p_bits × h_bits)
- Lattice W: O(N² × m) where N = dimension

**Optimization**:

- Pre-compute BSS matrices for frequently accessed pairs
- Cache priority scores
- Use approximate HLL operations for speed

## Next Steps

1. ✓ Implement BSS(τ, ρ) thresholds - **DONE**
2. ✓ Implement `EvolutionTriple` with (D, R, N) - **DONE**
3. ✓ Add Noether current calculation - **DONE**
4. ✓ Implement contextual selection - **DONE**
5. ✓ Add priority-weighted path selection - **DONE**
6. Extend `HLLSetLattice` to `CategoryLattice` with full (H, τ, ρ, φ)
7. Implement beam search for multi-path exploration
8. Create visualization tools for W evolution
9. Add adaptive threshold adjustment algorithms

## Conceptual Summary: The Complete Picture

This document reveals the deep connection between three fundamental concepts:

### 1. The Contextual Selection Principle (Abstract)

**From paper Section 7**: "Contexts actively select compatible elements"

- This is the fundamental inversion
- Context precedes content
- Selection is active, not passive membership

### 2. BSS(τ, ρ) Thresholds (Mechanics)

**The concrete mechanism implementing contextual selection**:

- τ (tau): Inclusion threshold - minimum similarity required
- ρ (rho): Exclusion threshold - maximum dissimilarity tolerated
- Selection: `BSS_τ(C, x) ≥ τ AND BSS_ρ(C, x) ≤ ρ`

**This is how context determines what is "in it"!**

### 3. AM ≠ W Connectivity (Result)

**The practical consequence**:

- AM shows raw co-occurrence (tokens appeared together)
- W shows semantic connections (BSS thresholds satisfied)
- τ/ρ filter AM edges to create semantic W edges
- Paths exist in AM that are disconnected in W

### Integration

```text
Contextual Selection Principle
        ↓
    (implemented by)
        ↓
   BSS(τ, ρ) Mechanics
        ↓
    (results in)
        ↓
   AM ≠ W Connectivity
        ↓
    (enables)
        ↓
Priority-Weighted Navigation
```

**Complete Flow**:

1. Context C (current HLLSet) has thresholds τ, ρ
2. C evaluates candidate x via BSS_τ(C,x) and BSS_ρ(C,x)
3. If thresholds satisfied, C **selects** x (active selection)
4. Selected elements form semantic connections in W
5. W connectivity differs from AM (semantic vs raw)
6. Navigate W using priority = BSS_τ - BSS_ρ

**Result**: A mathematically rigorous framework where:

- Contexts actively determine content (Contextual Selection)
- Selection has concrete mechanics (BSS thresholds)
- Semantic structure emerges from raw data (W from AM)
- Intelligent navigation becomes possible (priority weighting)

---

This deep structure reveals W as not just a data structure, but a **dynamical system** with conservation laws and categorical structure, enabling both reconstruction and evolution tracking in a mathematically rigorous framework. The τ/ρ thresholds are the **mechanics that make contextual selection real** - they transform the abstract principle into concrete, computable operations.
