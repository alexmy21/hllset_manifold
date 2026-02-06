# Priority-Weighted Path Selection in HRT

## The Fundamental Connection

>**τ/ρ thresholds are the mechanics of Contextual Selection**

From the paper's Contextual Selection Principle (Section 7):
> "Contexts actively select compatible elements, rather than elements passively belonging to contexts"

**How this works in practice:**

- **Abstract**: Context C selects elements that fit within it
- **Concrete**: BSS(τ, ρ) thresholds determine what "fits"
- **Mechanism**: Selection operator S_C uses τ/ρ as criteria

```text
S_C(x) = 1  ⟺  BSS_τ(C, x) ≥ τ  AND  BSS_ρ(C, x) ≤ ρ
```

This is the **fundamental inversion**: Context determines content, not the reverse!

## The Key Insight

>**AM connectivity ≠ W connectivity**

This fundamental difference enables intelligent semantic navigation:

```text
AM[i,j] > 0   ⟹  Tokens co-occurred in sequence
W[i,j] exists ⟺  BSS_τ(H_i → H_j) ≥ τ  AND  BSS_ρ(H_i → H_j) ≤ ρ
```

**Result**: AM may have edges that W filters out due to semantic thresholds!

## Priority Function

```text
Priority(H_current → H_candidate) = BSS_τ - λ·BSS_ρ

where:
  BSS_τ = |H_current ∩ H_candidate| / |H_candidate|  (inclusion)
  BSS_ρ = |H_current \ H_candidate| / |H_candidate|  (exclusion)
  λ = weight factor (default 1.0)
```

**Higher priority = Better semantic fit**.

## Threshold Effects

### Conservative Navigation (τ=0.8, ρ=0.2)

- High similarity required
- Low exclusion tolerance
- Stays close to context
- Safe, coherent paths

### Exploratory Navigation (τ=0.5, ρ=0.5)

- Lower similarity threshold
- Higher exclusion tolerance
- More diverse paths
- Creative, varied outputs

### Strict Coherence (τ=0.9, ρ=0.1)

- Maximum semantic consistency
- Very tight semantic fit
- Minimal diversity
- Highly focused paths

### Diverse Sampling (τ=0.6, ρ=0.5)

- Balance novelty and relevance
- Moderate thresholds
- Controlled exploration

## API Usage

### 1. Select Next by Priority

```python
from core.hrt import HRT, HRTConfig
from core.kernel import Kernel

# Create HRT with thresholds
config = HRTConfig(tau=0.7, rho=0.3)
kernel = Kernel()
hrt = HRT.empty(config)

# Select next node by priority
current_index = 5
result = hrt.select_next_by_priority(
    current_index=current_index,
    is_row=True,
    strategy='greedy'  # or 'stochastic'
)

if result:
    next_index, priority = result
    print(f"Selected index {next_index} with priority {priority:.3f}")
else:
    print("No valid candidates (filtered by τ/ρ)")
```

### 2. Find Path by Priority

```python
# Find path from start to end
path = hrt.find_path_by_priority(
    start_index=2,
    end_index=10,
    max_steps=100
)

if path:
    print(f"Found path with {len(path)} steps")
    for idx, priority in path:
        print(f"  Step to index {idx}, priority={priority:.3f}")
else:
    print("No path found - W disconnected by τ/ρ thresholds")
```

### 3. Selection Strategies

**Greedy** (deterministic):

```python
result = hrt.select_next_by_priority(
    current_index=idx,
    strategy='greedy'  # Select highest priority
)
```

**Stochastic** (probabilistic):

```python
result = hrt.select_next_by_priority(
    current_index=idx,
    strategy='stochastic'  # Sample by softmax(priority)
)
```

## Use Cases

### 1. Language Generation

```python
# Conservative, coherent text
config_conservative = HRTConfig(tau=0.8, rho=0.2)

# Creative, diverse text  
config_creative = HRTConfig(tau=0.5, rho=0.5)
```

### 2. Semantic Search

```python
def semantic_search(query_hllset, candidates, tau=0.6, rho=0.3):
    """Find documents satisfying BSS thresholds."""
    config = HRTConfig(tau=tau, rho=rho)
    # ... compute priorities and rank
```

### 3. Recommendation Systems

```python
def recommend_next(user_context, items, tau=0.7, rho=0.2, top_k=10):
    """Recommend items that fit user context."""
    # Use priority weighting to rank items
    # Filter by τ/ρ for semantic fit
```

### 4. Path Planning

```python
# Find semantically coherent path through knowledge graph
path = hrt.find_path_by_priority(
    start_index=concept_start,
    end_index=concept_target,
    max_steps=50
)
```

## Implementation Details

### Computational Complexity

- **BSS computation**: **O(m/w)** where m = number of HLL registers, w = word size (64 bits)
  - Uses **bitwise operations** on binary vectors (AND, AND NOT, popcount)
  - Intersection: `H1 ∩ H2` = bitwise AND → O(m/w)
  - Difference: `H1 \ H2` = bitwise AND NOT → O(m/w)
  - **Effectively O(1) for typical m=2^14=16,384** (256 words @ 64-bit)
- **Selecting from n candidates**: **O(n)** (constant-time BSS per candidate)
- **Path finding**: **O(max_steps × dimension)** (vectorized BSS operations)

### Memory Requirements

- **Each HLLSet**: O(2^p_bits × h_bits) bits
- **Full lattice W**: O(dimension² × m)

### Optimizations

1. **Pre-compute BSS matrices** for frequently accessed pairs
2. **Cache priority scores** to avoid recomputation
3. **Use approximate HLL operations** for speed/memory tradeoff
4. **Beam search** for multiple path exploration

## Adaptive Thresholds

Dynamically adjust τ/ρ based on context:

```python
def adaptive_thresholds(desired_diversity=0.5):
    """
    Adjust τ/ρ for desired diversity level.
    
    Low diversity (0.0)  → High τ, low ρ (conservative)
    High diversity (1.0) → Low τ, high ρ (exploratory)
    """
    base_tau = 0.7
    base_rho = 0.3
    
    tau = base_tau + (1 - desired_diversity) * 0.2
    rho = base_rho + desired_diversity * 0.2
    
    # Ensure 0 ≤ ρ < τ ≤ 1
    tau = min(0.95, max(0.5, tau))
    rho = max(0.05, min(tau - 0.1, rho))
    
    return HRTConfig(tau=tau, rho=rho)
```

## Key Properties

1. **Semantic Filtering**: τ/ρ filter raw AM connectivity to semantic W connectivity
2. **Priority Ordering**: BSS provides natural ranking of candidates
3. **Threshold Control**: Adjust exploration vs exploitation trade-off
4. **Path Disconnection**: W may disconnect paths that exist in AM
5. **Conservation**: Noether current tracks information flow

## Theoretical Foundation

From **"HLLSet Theory: Contextual Anti-Sets and the Selection Principle"**:

- **Section 2.2**: Bell State Similarity morphism conditions
- **Section 7**: Contextual Selection Principle
- **HRT_LATTICE_THEORY.md**: AM vs W connectivity divergence

## Examples

See:

- `core/hrt.py` - Section 11 of main() demonstration
- `demo_hrt_theory.ipynb` - Interactive examples
- `HRT_LATTICE_THEORY.md` - Extended algorithms and theory

---

**Summary**: τ/ρ thresholds create a comprehensive priority weighting system that transforms raw AM co-occurrence into semantically meaningful W connectivity, enabling intelligent navigation through the HLLSet lattice.
