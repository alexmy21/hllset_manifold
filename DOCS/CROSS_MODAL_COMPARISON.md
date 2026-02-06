# Cross-Modal Lattice Comparison: Structural vs Content-Based

## The Problem

When comparing two lattices representing the same reality through different channels (e.g., sensory vs language), we face a fundamental challenge:

**The token spaces are mutually exclusive (disjoint).**

### Example

**Sensory lattice** (visual perception):

- Tokens: `['wavelength_630nm', 'bright', 'saturated_color', 'red_hue']`

**Language lattice** (semantic representation):

- Tokens: `['color_descriptor', 'warm_color', 'chromatic_term', 'hue_word']`

**Result**:

```text
Sensory_HLLSet ∩ Language_HLLSet = ∅ (empty intersection)
```

## Why BSS Fails for Cross-Modal Comparison

BSS (Bell State Similarity) is defined by set intersection:

```text
BSS_τ(A → B) = |A ∩ B| / |B|
BSS_ρ(A → B) = |A \ B| / |B|
```

For disjoint sets:

```text
|A ∩ B| = 0
|A \ B| = |A|

Therefore:
BSS_τ = 0 / |B| = 0 (always!)
BSS_ρ = |A| / |B| ≈ 1 (always!)
```

**Conclusion**: BSS is meaningless for cross-modal comparison. We always get τ=0, ρ=1 regardless of semantic relationship.

## The Solution: Structural Comparison

Instead of comparing node content (HLLSet intersections), we compare **graph structure** (topology).

### Key Insight

Two lattices representing the same reality through different modalities should have **similar graph structures** even though their node contents are disjoint.

>**Same reality → Similar structural patterns → High ε-isomorphism probability**

## Algorithm: Degree-Based Structural Matching

### Step 1: Compute Node Degrees

For each node (BasicHLLSet) in a lattice, compute its degree:

```python
degree(node) = |{v : node → v is morphism}|
```

where morphism exists if:

```text
BSS_τ(node → v) ≥ τ  AND  BSS_ρ(node → v) ≤ ρ
```

### Step 2: Extract Degree Sequences

For each lattice:

```python
row_degrees = [degree(r_0), degree(r_1), ..., degree(r_n)]
col_degrees = [degree(c_0), degree(c_1), ..., degree(c_n)]
```

### Step 3: Compute Structural Similarity

**Pearson Correlation** (measures structural similarity):

```python
ρ(deg₁, deg₂) = cov(deg₁, deg₂) / (σ₁ · σ₂)
```

- High correlation → Similar connectivity patterns
- Low/negative correlation → Different structures

**Normalized L1 Distance** (measures degree distribution difference):

```python
distance = Σ|deg₁[i] - deg₂[i]| / Σmax(deg₁[i], deg₂[i])
```

- Low distance → Similar degree distributions
- High distance → Different degree distributions

### Step 4: Compute ε-Isomorphism Probability

```python
ε_prob = correlation × (1 - distance)
```

Clamped to [0, 1].

## Implementation

### HLLSetLattice Methods

```python
def _compute_node_degrees(self) -> Tuple[List[int], List[int]]:
    """Compute morphism counts for each node."""
    row_degrees = []
    for r in self.row_basic:
        degree = sum(1 for c in self.col_basic if r.has_morphism_to(c))
        row_degrees.append(degree)
    
    col_degrees = []
    for c in self.col_basic:
        degree = sum(1 for r in self.row_basic if c.has_morphism_to(r))
        col_degrees.append(degree)
    
    return row_degrees, col_degrees

def compare_lattices(self, other: 'HLLSetLattice') -> Dict[str, float]:
    """
    Structural comparison of two lattices.
    
    Returns:
        - row_degree_correlation: Correlation of row degrees
        - col_degree_correlation: Correlation of col degrees
        - overall_structure_match: Average correlation
        - epsilon_isomorphic_prob: Structural ε-isomorphism
        - row_degree_distance: Normalized L1 distance (rows)
        - col_degree_distance: Normalized L1 distance (cols)
    """
    # Compute degrees
    self_row_deg, self_col_deg = self._compute_node_degrees()
    other_row_deg, other_col_deg = other._compute_node_degrees()
    
    # Compute correlations and distances
    row_corr = self._degree_correlation(self_row_deg, other_row_deg)
    col_corr = self._degree_correlation(self_col_deg, other_col_deg)
    row_dist = self._degree_distance(self_row_deg, other_row_deg)
    col_dist = self._degree_distance(self_col_deg, other_col_deg)
    
    # Overall metrics
    overall_match = (row_corr + col_corr) / 2.0
    epsilon_prob = overall_match * (1.0 - (row_dist + col_dist) / 2.0)
    
    return {
        'row_degree_correlation': row_corr,
        'col_degree_correlation': col_corr,
        'overall_structure_match': overall_match,
        'epsilon_isomorphic_prob': max(0.0, min(1.0, epsilon_prob)),
        'row_degree_distance': row_dist,
        'col_degree_distance': col_dist
    }
```

## Semantic Grounding Levels

Based on structural ε-isomorphism probability:

- **P ≥ 0.7**: Strong grounding (similar graph structures)
- **0.5 ≤ P < 0.7**: Moderate grounding (partial structural similarity)
- **0.3 ≤ P < 0.5**: Weak grounding (different structures)
- **P < 0.3**: Disconnected (no structural correspondence)

## Example: Sensory vs Language Lattices

```python
# Create sensory lattice (visual features)
sensory_lattice = create_lattice_from_tokens([
    ['red_wavelength', 'bright_intensity'],
    ['round_contour', 'circular_shape'],
    # ...
])

# Create language lattice (semantic features)
language_lattice = create_lattice_from_tokens([
    ['color_word', 'hue_descriptor'],
    ['shape_word', 'geometry_term'],
    # ...
])

# Structural comparison
metrics = sensory_lattice.compare_lattices(language_lattice)

# Results:
# - row_degree_correlation: 0.85 (high structural similarity!)
# - epsilon_isomorphic_prob: 0.72 (strong grounding)
# - semantic_grounding_level: "Strong (well-grounded)"
```

Even though token spaces are disjoint (BSS_τ = 0), structural similarity reveals semantic grounding!

## When to Use Each Approach

### Use BSS (Content-Based):

- **Within a single lattice** (same token vocabulary)
- Measuring morphisms between nodes
- Computing BSS_τ/BSS_ρ for contextual selection
- Priority weighting for path finding

### Use Structural Comparison:

- **Between different lattices** (disjoint token vocabularies)
- Cross-modal semantic grounding (sensory ↔ language)
- Multimodal alignment (vision ↔ text ↔ audio)
- Translation quality (English ↔ French lattices)
- Concept mapping across domains

## Theoretical Foundation

From **DOCS/SENSES_SIGNS_SAUSSURE.md**:

> "The connection between reality and language is not direct mapping but **entanglement** between the two lattice structures."

Entanglement is measured by **graph isomorphism**, not by node content overlap.

### Saussure's Duality Revisited

- **Signified** (object/concept) → W_sensory lattice
- **Signifier** (linguistic sign) → W_language lattice
- **Meaning** = structural correspondence (ε-isomorphism)

The choice of signs (language vocabulary) is indeed arbitrary by Saussure. However, relationship is not arbitrary (as Saussure suggested) but **emergent from structural alignment (entanglement)**.

## Advanced: Beyond Degree-Based Matching

The simplified algorithm uses node degrees. More sophisticated approaches:

1. **Spectral Methods**: Compare eigenvalues of adjacency matrices
2. **Graph Kernels**: Compute structural similarity via kernel functions
3. **Graph Neural Networks**: Learn optimal node alignments
4. **Subgraph Isomorphism**: Find common substructures
5. **Graph Edit Distance**: Measure structural transformation cost

All share the same principle: **compare structure, not content**.

## Summary

**Key Takeaway**: When comparing cross-modal lattices, BSS fails because token spaces are disjoint. Use structural comparison (degree-based or more sophisticated) to measure semantic grounding through graph topology.

**API**:
```python
# WRONG (meaningless for cross-modal):
bss_tau = sensory_basic.bss_tau(language_basic)  # Always 0!

# CORRECT (structural comparison):
metrics = sensory_lattice.compare_lattices(language_lattice)
grounding = sensory_lattice.semantic_grounding_level(language_lattice)
```

**References**:

- **DOCS/SENSES_SIGNS_SAUSSURE.md**: Theoretical framework
- **DOCS/HRT_LATTICE_THEORY.md**: W lattice structure
- **09_senses_and_signs.ipynb**: Demonstration notebook
