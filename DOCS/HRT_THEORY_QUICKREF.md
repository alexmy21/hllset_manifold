# HRT Theory Quick Reference

Fast reference for theoretical concepts and their implementations.

## Paper → Code Mapping

### HLLSet Configuration

```python
# Paper: HLLSet = (H, φ, τ, ρ)
config = HRTConfig(
    p_bits=10,      # |H| = 2^10 registers
    tau=0.7,        # τ: inclusion threshold
    rho=0.3,        # ρ: exclusion threshold
    epsilon=0.1     # ε: isomorphism tolerance
)
```

### Bell State Similarity

```python
# Paper: BSS_τ(A→B) = |A∩B|/|B|
bss_inclusion = basic_a.bss_tau(basic_b)

# Paper: BSS_ρ(A→B) = |A\B|/|B|
bss_exclusion = basic_a.bss_rho(basic_b)

# Morphism f: A→B exists?
has_morphism = basic_a.has_morphism_to(basic_b)
# True iff bss_tau ≥ τ and bss_rho ≤ ρ
```

### Entanglement

```python
# Paper: |BSS(A,B) - BSS(φ(A),φ(B))| ≤ ε
is_entangled = lattice1.is_epsilon_isomorphic(lattice2)

# Paper: P(not ε-iso) ≤ min(1, n²·(d²/2^m + exp(-ε²d/2)))
prob_failure = lattice.entanglement_probability(n_datasets, dataset_size)
```

### Noether Current

```python
# Paper: J_uv(p) = p[u]·(Ap)[v] - p[v]·(A^T·p)[u]
noether = NoetherCurrent.compute(am, distribution, step)

# Paper: dΦ/dt = 0 (conserved)
health = hrt.conservation_health()
# Returns: "HEALTHY" if |Φ| < 1e-6
```

### Evolution Triple

```python
# Paper: H(t+1) = [H(t)\D] ∪ N
triple = hrt.evolution_triple

# Access components
deleted = triple.deleted      # D: forgotten
retained = triple.retained    # R: kept
new = triple.new             # N: added

# Conservation check
delta_q = triple.cardinality_change  # ΔQ = |N| - |D|
is_conservative = triple.is_conservative  # True if ΔQ = 0
```

### Contextual Selection

```python
# Paper: S_C(x) = 1 iff BSS(F_C,F_x) ≥ τ and exclusion ≤ ρ
selection = hrt.contextual_select(context_index=2, is_row=True)

# Results
selected_elements = selection.selected_hashes
selection_power = selection.selection_power
```

### Time Reversibility

```python
# Paper: p_forward = normalize(A·p)
future = hrt.project_future(col_indices)

# Paper: p_retro = normalize(A^T·p)
past = hrt.project_past(row_indices)
```

## Formula Reference

### 1. BSS Morphisms

```text
BSS_τ(A→B) = |A ∩ B| / |B|     (inclusion)
BSS_ρ(A→B) = |A \ B| / |B|     (exclusion)

Morphism exists ⟺ BSS_τ ≥ τ ∧ BSS_ρ ≤ ρ
```

### 2. Entanglement Bound

```text
P(not ε-isomorphic) ≤ min(1, n² · (d²/2^m + e^(-ε²d/2)))

where:
  n = number of datasets
  d = dataset size
  m = 2^p_bits (registers)
  ε = epsilon tolerance
```

### 3. Noether Current

```text
J_uv(p) = p[u]·(Ap)[v] - p[v]·(A^T·p)[u]
Φ = Σ J_uv (total flux)
dΦ/dt = 0 (conservation law)
```

### 4. Evolution Dynamics

```text
H(t+1) = [H(t) \ D] ∪ N

D = deleted information
R = H(t) \ D = retained
N = new information
ΔQ = |N| - |D| (cardinality change)
```

### 5. Contextual Selection

```text
S_C: U → {0,1}
S_C(x) = 1 ⟺ BSS_τ(F_C,F_x) ≥ τ ∧ BSS_ρ(F_C,F_x) ≤ ρ
```

## 6. Category Theory Structure

```text
Category HLL:
  Objects:    HLLSets with (H, φ, τ, ρ)
  Morphisms:  f: A→B where BSS conditions hold
  Composition: g∘f when intermediate conditions propagate
  Identity:    1_A: A→A with BSS_τ=1, BSS_ρ=0
  
Karoubi Equivalence:
  HLL ≃ Karoubi(IdempotentHashes)
```

## Common Patterns

### Check if two HLLSets are compatible

```python
basic_a = hrt.lattice.row_basic[i]
basic_b = hrt.lattice.row_basic[j]

if basic_a.has_morphism_to(basic_b):
    print("Morphism A→B exists!")
```

### Monitor system health

```python
if hrt.noether_current:
    print(hrt.conservation_health())
    # "HEALTHY" if flux ≈ 0
    # "ALERT" if large drift (collision/error)
```

### Analyze evolution

```python
if hrt.evolution_triple:
    if hrt.evolution_triple.is_conservative:
        print("Balanced evolution: |N| = |D|")
    elif hrt.evolution_triple.cardinality_change > 0:
        print("Growing: learning new patterns")
    else:
        print("Contracting: forgetting patterns")
```

### Apply contextual selection

```python
# Context at index 5 selects compatible elements
selection = hrt.contextual_select(context_index=5)
print(f"Selected {len(selection.selected_hashes)} elements")
print(f"Selection power: {selection.selection_power}")
```

### Check entanglement between HRTs

```python
is_entangled, prob_fail = hrt1.check_entanglement(hrt2)
if is_entangled:
    confidence = (1 - prob_fail) * 100
    print(f"Entangled! Confidence: {confidence:.2f}%")
```

## Theoretical Insights

### The Fundamental Inversion

**Traditional:** Elements belong to sets (extensional)
**HRT:** Contexts select elements (contextual)

### Conservation Laws

- **Noether current** tracks information flow
- **ΔQ = 0** ⟹ conservative evolution
- **Φ ≈ 0** ⟹ system health

### The Entanglement

- Not spooky action
- **Structural isomorphism** between lattices
- Probability bound shows it's almost certain

### Time Symmetry

- **Forward:** predict future states
- **Retro:** reconstruct past contexts
- Both preserve information

## Performance Notes

### Computational Complexity

- BSS calculation: O(m) where m = 2^p_bits
- Morphism check: O(m) × 2
- Entanglement: O(n² × m²) for full lattice
- Noether current: O(m²) matrix multiply

### Memory

- HLLSet: O(m × b) bits
- Lattice: O(2 × dimension × m × b)
- HRT: Lattice + AM + metadata

### Optimization Tips

- Use smaller p_bits for demos (8-10)
- Use larger p_bits for production (12-16)
- Entanglement check can be sampled
- Noether current computed on merge only

## Integration Examples

### With ManifoldOS

```python
from core.manifold_os import ManifoldOS, IngestDriver

os = ManifoldOS()
driver = IngestDriver(os.kernel)
os.register_driver("ingest", driver)

# Ingest creates HRT with all theory features
n_token = os.wake("ingest", text="Hello world")
hrt = n_token.hrt  # Has BSS, Noether, selection, etc.
```

### With Adjacency Matrix

```python
# AM and HRT W lattice are aligned
am = hrt.am
lattice = hrt.lattice

# AM tracks transitions
# W lattice tracks HLLSet structure
# Both evolve together
```

## References

**Paper:** "HLLSet Theory: Contextual Anti-Sets and the Selection Principle"

- Location: `pubs/article/hllsettheory-contextual-anti-sets.pdf`
- LaTeX: `pubs/article/hllsettheory-contextual-anti-sets.tex`

**Implementation:** `core/hrt.py`

**Demo:** `demo_hrt_theory.ipynb`

**Documentation:** `HRT_THEORY_IMPLEMENTATION.md`
