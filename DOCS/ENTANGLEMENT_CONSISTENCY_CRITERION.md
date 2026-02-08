# Entanglement Consistency Criterion

## Fundamental Principle

**Date**: February 8, 2026

The core requirement for entanglement morphisms is **structural consistency**, not a specific measurement approach.

## Categorical Definition

For an entanglement morphism `φ: L₁ → L₂` to be valid, it must satisfy:

```text
∀ a ≠ b in L₁: φ(a) ≉ φ(b) in L₂
```

**Translation**: If two elements are distinct in the source lattice, their images must remain approximately distinct in the target lattice.

This is **approximate injectivity**—the mapping preserves structural distinctness.

## Why Approximate (≉)?

Due to hash collisions in HLLSets, we cannot use strict equality (`≠`). We must use approximate measures:

- **Similarity thresholds**: `sim(φ(a), φ(b)) < threshold`
- **Distance metrics**: `dist(φ(a), φ(b)) > min_distance`
- **Structural measures**: Preserve relative relationships

## Implementation Freedom

The consistency criterion allows multiple valid implementations:

### 1. BSS-Based (Current)

```python
# Uses Basic Similarity Score between HLLSets
degree_sim + register_sim → strength
```

### 2. Degree-Only

```python
# Uses only cardinality difference
|card(a) - card(b)| → similarity
```

### 3. Custom Domain Metrics

```python
# For metadata: schema relationships
# For ED: foreign key preservation
# For images: visual similarity
```

### 4. Hybrid Approaches

```python
# Combine multiple metrics with domain-specific weights
0.6 * metric1 + 0.4 * metric2
```

## Key Insights

1. **BSS is not mandatory**: It's one valid approach among many

2. **Mutual exclusivity NOT required**: Elements can overlap; what matters is that distinct elements map to distinct targets (approximately)

3. **Auto-entanglement is valid**: A lattice can entangle with itself (`φ: L → L`) to discover self-similar patterns

4. **Manual mapping is valid**: For ED metadata, we can explicitly define `φ: L_data → L_meta` based on domain knowledge

## IICA Compliance

All entanglement implementations must preserve IICA properties:

- **Immutable**: Morphisms never change once computed
- **Idempotent**: Re-computing gives same result
- **Content Addressable**: Morphisms identified by content hash

## Practical Implications

### For AM (Adjacency Matrix)

- Can use different tokenization strategies
- Window sizes can vary
- Token overlap is allowed
- Core requirement: same hash morphism for all HLLSets **within a perceptron**
- Different perceptrons can use different hash morphisms

### For W (Lattice)

- Can build lattice from various HLLSet collections
- Structural topology preservation is automatic if:
  - Same hash morphism used (within perceptron)
  - Same transformation algorithm applied
  - Consistency criterion satisfied

### For Metadata Bridge

- Create metadata lattice with specialized perceptrons
- Establish manual entanglement: `φ: L_data → L_meta`
- Verify: distinct data structures → distinct metadata structures
- Use domain-specific similarity measures

## Theoretical Foundation

This aligns with **category theory**: a morphism preserves structure, and the minimal structure preservation is **distinctness**.

```text
Category of Lattices
- Objects: HLLSet lattices
- Morphisms: Structure-preserving maps
- Composition: φ₂ ∘ φ₁ preserves consistency
- Identity: id_L maps each element to itself
```

The consistency criterion ensures we have a proper categorical structure.

## References

- `core/entanglement.py`: Implementation
- `DOCS/ENTANGLEMENT_SINGULARITY.md`: Theoretical framework
- `DOCS/KERNEL_ENTANGLEMENT.md`: ICASRA integration
