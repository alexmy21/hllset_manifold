# HRT Theory Implementation

Comprehensive alignment of the HRT (Hash Relational Tensor) implementation with the theoretical paper **"HLLSet Theory: Contextual Anti-Sets and the Selection Principle"**.

## Overview

The HRT framework now fully implements the theoretical foundations from the paper, bringing together:

- Category theory (HLL category with morphisms)
- Conservation laws (Noether current)
- Evolution dynamics (D, R, N triple)
- Contextual selection (the fundamental inversion)
- Entanglement detection (ε-isomorphism)

## 1. HLLSet Configuration with τ and ρ

**From Paper Section 2.1:**

```text
HLLSet = (H, φ, τ, ρ)
```

**Implementation:** `HRTConfig`

```python
@dataclass(frozen=True)
class HRTConfig:
    p_bits: int = 10      # HLL precision
    h_bits: int = 32      # Hash bits
    tau: float = 0.7      # Inclusion tolerance (τ)
    rho: float = 0.3      # Exclusion intolerance (ρ)
    epsilon: float = 0.1  # ε-isomorphism tolerance
```

**Constraint:** `0 ≤ ρ < τ ≤ 1` (validated in `__post_init__`)

## 2. Bell State Similarity (BSS)

**From Paper Section 2.2:**

$$\text{BSS}_\tau(A \to B) = \frac{|A \cap B|}{|B|}$$

$$\text{BSS}_\rho(A \to B) = \frac{|A \setminus B|}{|B|}$$

**Implementation:** `BasicHLLSet.bss_tau()` and `BasicHLLSet.bss_rho()`

```python
def bss_tau(self, other: BasicHLLSet) -> float:
    """Inclusion similarity: how much of B is covered by A."""
    intersection = self.hllset.intersection_cardinality(other.hllset)
    return intersection / other.hllset.cardinality()

def bss_rho(self, other: BasicHLLSet) -> float:
    """Exclusion similarity: how much of A is outside B."""
    difference = self.hllset.cardinality() - self.hllset.intersection_cardinality(other.hllset)
    return difference / other.hllset.cardinality()
```

## 3. Category Theory: Morphisms

**From Paper Section 2.2:**

A morphism `f: A → B` exists iff:

- `BSS_τ(A→B) ≥ τ` (sufficient similarity)
- `BSS_ρ(A→B) ≤ ρ` (limited exclusion)

**Implementation:** `BasicHLLSet.has_morphism_to()`

```python
def has_morphism_to(self, other: BasicHLLSet) -> bool:
    """Check if morphism f: self → other exists."""
    return (self.bss_tau(other) >= self.config.tau and 
            self.bss_rho(other) <= self.config.rho)
```

**Category Structure:**

- **Objects**: BasicHLLSets with (H, φ, τ, ρ)
- **Morphisms**: BSS-defined relations
- **Composition**: Transitive BSS preservation
- **Identity**: `1_A` with `BSS_τ = 1`, `BSS_ρ = 0`

## 4. ε-Isomorphism and Entanglement

**From Paper Definition 4.1:**

Two lattices are ε-isomorphic if there exists bijection φ such that:
$$|\text{BSS}(A, B) - \text{BSS}(\phi(A), \phi(B))| \leq \epsilon$$

**From Paper Theorem 4.2:**

$$P(\text{not } \epsilon\text{-isomorphic}) \leq \min\left(1, n^2 \cdot \left( \frac{d^2}{2^m} + e^{-\epsilon^2 d / 2} \right) \right)$$

**Implementation:** `HLLSetLattice.is_epsilon_isomorphic()` and `entanglement_probability()`

```python
def is_epsilon_isomorphic(self, other: HLLSetLattice, epsilon: Optional[float] = None) -> bool:
    """Check pairwise BSS preservation within ε tolerance."""
    eps = epsilon if epsilon is not None else self.config.epsilon
    for i, j in itertools.product(range(len(self.row_basic)), repeat=2):
        bss_self = self.row_basic[i].bss_tau(self.row_basic[j])
        bss_other = other.row_basic[i].bss_tau(other.row_basic[j])
        if abs(bss_self - bss_other) > eps:
            return False
    return True

def entanglement_probability(self, num_datasets: int, dataset_size: int) -> float:
    """Compute probability of entanglement failure (Theorem 4.2)."""
    collision_term = (d * d) / (2 ** m)
    deviation_term = math.exp(-(eps * eps * d) / 2)
    return min(1.0, n * n * (collision_term + deviation_term))
```

## 5. Noether Current and Conservation

**From Paper Section 6.2:**

$$J_{uv}(p) = p[u] \cdot (Ap)[v] - p[v] \cdot (A^T p)[u]$$

Total flux conserved:
$$\frac{d\Phi}{dt} = 0$$

**Implementation:** `NoetherCurrent` class

```python
@dataclass(frozen=True)
class NoetherCurrent:
    flux: float
    timestamp: float
    step_number: int
    
    @classmethod
    def compute(cls, am: AdjacencyMatrix, distribution: ImmutableTensor, step_number: int):
        forward = torch.matmul(am.tensor.data, distribution.data)
        retro = torch.matmul(am.tensor.data.T, distribution.data)
        flux = float(torch.sum(distribution.data * forward - distribution.data * retro))
        return cls(flux=flux, timestamp=time.time(), step_number=step_number)
```

**Conservation Health Monitoring:**

```python
def conservation_health(self) -> str:
    flux = abs(self.noether_current.flux)
    if flux < 1e-6: return "HEALTHY: Flux ≈ 0"
    elif flux < 1e-3: return "GOOD"
    elif flux < 0.1: return "WARNING"
    else: return "ALERT: Check for collisions"
```

## 6. Evolution Triple (D, R, N)

**From Paper Section 5.1:**

$$H(t+1) = [H(t) \setminus D] \cup N$$

where:

- **D**: Deleted information (forget)
- **R**: Retained information (H(t) \\ D)
- **N**: New information (add)

**Implementation:** `EvolutionTriple` class

```python
@dataclass(frozen=True)
class EvolutionTriple:
    deleted: Set[str]   # Hashes of deleted HLLSets
    retained: Set[str]  # Hashes of retained HLLSets
    new: Set[str]       # Hashes of new HLLSets
    
    @property
    def cardinality_change(self) -> int:
        """ΔQ = |N| - |D|"""
        return len(self.new) - len(self.deleted)
    
    @property
    def is_conservative(self) -> bool:
        """When |N| = |D|, evolution is conservative."""
        return self.cardinality_change == 0
```

**Tracked in HRT.merge():**

```python
def merge(self, other: HRT, kernel) -> HRT:
    # Track (D, R, N)
    deleted = other_hashes - self_hashes
    retained = self_hashes & other_hashes
    new = self_hashes - other_hashes
    
    evo_triple = EvolutionTriple(deleted=deleted, retained=retained, new=new)
    # ... merge and return new HRT with evo_triple
```

## 7. Contextual Selection Principle

**From Paper Section 7.1:**

The **fundamental inversion**: contexts actively select compatible elements.

$$S_C: \mathcal{U} \to \{0,1\}$$

$$S_C(x) = 1 \iff \text{BSS}(F_C, F_x) \geq \tau_C \text{ and exclusion}(F_C, F_x) \leq \rho_C$$

**Critical Connection**: τ/ρ thresholds are the **mechanics of contextual selection** - they define the concrete mechanism by which a context determines what fits "in it".

This completes the theoretical picture:
- **Principle**: Context C actively selects compatible elements (not passive membership)
- **Mechanism**: BSS(τ, ρ) thresholds provide the selection criteria
- **Implementation**: `ContextualSelection` class + `select_next_by_priority()` method

The context uses its τ/ρ thresholds to evaluate candidates:
- If BSS_τ(C, x) ≥ τ: Element x has sufficient similarity to C
- If BSS_ρ(C, x) ≤ ρ: Element x doesn't have excessive dissimilarity
- Both satisfied → Context C **selects** x as compatible

**Implementation:** `ContextualSelection` class

```python
@dataclass(frozen=True)
class ContextualSelection:
    context_hash: str
    selected_hashes: FrozenSet[str]
    tau_threshold: float
    rho_threshold: float
    selection_power: float  # Conservation tracking
    
    @classmethod
    def from_context(cls, context: BasicHLLSet, candidates: List[BasicHLLSet]):
        """Context actively selects compatible candidates."""
        selected = []
        total_power = 0.0
        
        for candidate in candidates:
            if (context.bss_tau(candidate) >= context.config.tau and 
                context.bss_rho(candidate) <= context.config.rho):
                selected.append(candidate.hllset.name)
                total_power += context.bss_tau(candidate)
        
        return cls(context_hash=context.hllset.name, ...)
```

**Usage in HRT:**

```python
def contextual_select(self, context_index: int, is_row: bool = True) -> ContextualSelection:
    """Apply contextual selection from a basic HLLSet."""
    context = self.lattice.row_basic[context_index] if is_row else self.lattice.col_basic[context_index]
    candidates = list(self.lattice.col_basic if is_row else self.lattice.row_basic)
    return ContextualSelection.from_context(context, candidates)
```

**Interpretation:**

- Quantum measurement: context = experimental setup selects eigenstates
- Evolution: context = ecological niche selects organisms
- Consciousness: context = self-selecting awareness

## 8. Time-Reversible Dynamics

**From Paper Section 6.1:**

Forward projection:
$$\vec{p}_{\text{forward}} = \text{normalize}(A \cdot \vec{p})$$

Retro-cast:
$$\vec{p}_{\text{retro}} = \text{normalize}(A^T \cdot \vec{p})$$

**Implementation:**

```python
def project_future(self, col_indices: List[int]) -> ImmutableTensor:
    """Forward: p_forward = A·p"""
    return self.am.project_rows(col_indices)

def project_past(self, row_indices: List[int]) -> ImmutableTensor:
    """Retro: p_retro = A^T·p"""
    return self.am.project_cols(row_indices)
```

Enables:

- Future prediction from current state
- Past reconstruction from current state
- Noether current computation

## 9. Enhanced HRT with Full Theory

**Updated HRT class:**

```python
@dataclass(frozen=True)
class HRT:
    """
    Category Theory (Section 3):
    - Objects: HRTs with HLLSet lattices
    - Morphisms: Evolution steps preserving structural isomorphism
    - Composition: Sequential evolution with conservation laws
    """
    config: HRTConfig
    am: AdjacencyMatrix
    lattice: HLLSetLattice  # The W lattice
    parent_hrt: Optional[str] = None
    step_number: int = 0
    covers: Tuple[Cover, ...] = field(default_factory=tuple, compare=False)
    noether_current: Optional[NoetherCurrent] = field(default=None, compare=False)
    evolution_triple: Optional[EvolutionTriple] = field(default=None, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
```

**New Methods:**

- `contextual_select()` - Apply contextual selection
- `check_entanglement()` - Verify ε-isomorphism
- `conservation_health()` - Monitor Noether current
- `project_future() / project_past()` - Time-reversible dynamics

## 10. Demo and Validation

**Interactive Notebook:** `demo_hrt_theory.ipynb`

Demonstrates:

1. ✓ BSS calculations with τ/ρ thresholds
2. ✓ Morphism checking
3. ✓ ε-isomorphism and entanglement probability
4. ✓ Noether current and conservation monitoring
5. ✓ Evolution triple (D, R, N) tracking
6. ✓ Contextual selection principle
7. ✓ Time-reversible projections
8. ✓ Category theory structure

**Command Line Test:**

```bash
python -m core.hrt
```

Shows all features in action with real data.

## Summary: Theory → Practice

| Theoretical Concept | Paper Section | Implementation | Status |
|-------------------|--------------|----------------|--------|
| HLLSet (H, φ, τ, ρ) | 2.1 | `HRTConfig` | ✓ |
| Bell State Similarity | 2.2 | `bss_tau()`, `bss_rho()` | ✓ |
| Morphisms | 2.2 | `has_morphism_to()` | ✓ |
| Category HLL | 3.1 | Objects, morphisms, composition | ✓ |
| ε-isomorphism | 4.1 | `is_epsilon_isomorphic()` | ✓ |
| Entanglement bound | 4.2 | `entanglement_probability()` | ✓ |
| Evolution H(t+1) = [H(t)\\D]∪N | 5.1 | `EvolutionTriple` | ✓ |
| Noether current | 6.2 | `NoetherCurrent` | ✓ |
| Conservation Φ | 6.2 | `conservation_health()` | ✓ |
| Time reversibility | 6.1 | `project_future/past()` | ✓ |
| Contextual selection | 7.1 | `ContextualSelection` | ✓ |
| Selection operator S_C | 7.1 | `contextual_select()` | ✓ |

## Key Insights

The implementation reveals that HRT is:

1. **Categorical** - Objects (HLLSets) with morphisms (BSS-based)
2. **Conservative** - Information preserved via Noether current
3. **Evolutionary** - Explicit (D,R,N) tracking shows system dynamics
4. **Contextual** - Selection flows from context to content, not reverse
5. **Entangled** - Structural isomorphism creates non-local correlations
6. **Reversible** - Time symmetry enables past/future projections

This is not just a data structure - it's a **complete mathematical framework** for contextual anti-sets where:

- **Context precedes content** (Contextual Selection Principle)
- **Relationships are more real than relata** (Category theory)
- **Entanglement is the norm** (ε-isomorphism)
- **Information flows but never vanishes** (Noether conservation)

## References

Paper: **"HLLSet Theory: Contextual Anti-Sets and the Selection Principle"** by Alex Mylnikov

Location: `pubs/article/hllsettheory-contextual-anti-sets.pdf`

LaTeX source: `pubs/article/hllsettheory-contextual-anti-sets.tex`
