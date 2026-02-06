# Manifold Evolution: Sustainable Self-Organization

## Core Principle: Manifold Over Elements

**Key Insight**: We work with **multiples of everything**, not singular instances:

- Not separate tokens → **Contexts** (collections of tokens)
- Not single representation → **Variety** of mutually exclusive but topologically aligned representations
- Not one AM or W → **Multiple AMs and W lattices**
- Not element commonality → **Relational commonality** (topology over content)

### Why Manifold?

Different parts may:

- **Overlap**: Share some elements/structure
- **Mutually Exclusive**: Disjoint content spaces (sensory vs language)
- **Topologically Aligned**: Similar graph structure despite disjoint elements

We treat all as a **manifold** where what matters is **relations between elements**, not the elements themselves.

## Evolution as Swarm Management

### From PSO to Manifold Management

**Traditional PSO** (Particle Swarm Optimization):

- Goal: Find global optimum
- Method: Particles explore search space
- Criteria: Fitness function minimization/maximization

**Manifold Management** (Our Approach):

- Goal: **Sustainable evolution** (not optimization)
- Method: Multiple lattices co-evolve without destruction
- Criteria: **Noether current** (conservation principle)

### Key Difference

We're not looking for **optima** — we're looking for **sustainable management** where all parts organize without destroying each other. Like a swarm that maintains coherence without colliding.

## Self-Reproductive Evolution Loop

Evolution is an **infinite loop** of repeating steps. Steps are **not synchronized** between different parts (they don't need to be).

### Single Lattice Evolution

At any point in the evolution trajectory:

1. **Current State**: W lattice (committed)
2. **Ephemeral State**: Lattice in ingestion process (being built)
3. **Evolution Step**: Merge ephemeral into current → New current state

```text
W_current + W_ephemeral → W_next
```

### Evolution Triple (D, R, N)

Each step explicitly tracks:

- **D (Deleted)**: Information forgotten/removed
- **R (Retained)**: Information preserved from current
- **N (New)**: Information added from ephemeral

```text
W_next = (W_current \ D) ∪ N
```

## Evolution Projection: Predicting Next Ingestion

### The Core Problem

**What are we projecting?** The **next ingestion** - what tokens will arrive.

### Critical Understanding: HLLSets are Fingerprints

**HLLSets ≠ Sets**. They are probabilistic fingerprints that behave like sets but:

- Can't extract actual tokens from HLLSet
- Can only get HLLSets from HLLSets (union, intersection)
- Need actual tokens to update AM and regenerate W

**What works for AM doesn't work for W** - we can't just manipulate HLLSets in W space.

### How We Got to Current State

1. **Ingestion**: Got new tokens (actual data)
2. **AM Update**: Built HLLSet from tokens + updated AM simultaneously
3. **W Generation**: Extracted basic HLLSets from AM, built W lattice
4. **Commit**: Saved current state

### What to Preserve at Commit

To enable projection, we must save:

1. **Original ingested HLLSet**: The fingerprint of what was ingested
2. **Decomposition**: Its breakdown into basic HLLSets
3. **Evolution Triple (D, R, N)**: For each column basic HLLSet - what was deleted, retained, new

**Token mappings exist in two places** (no need to save separately):

- **AM (reg, zeros)**: Actual tokens already stored
- **LUT**: Clarified mappings already maintained

**Key insight**: HLLSet operations don't create new tokens, so LUT is unaffected. We can use current AM + LUT for disambiguation.

### Correct Projection Flow

```text
Step 1: Select Basic Row HLLSet for Projection
    - Choose from current row basics
    - This basic has references to column basics (via W)

Step 2: For Each Referenced Column Basic
    - Has evolution triple (D, R, N) from history
    - Has current BSS relationship with row basic
    - Create candidate contributions to next ingestion
    
Step 3: Aggregate Candidates → Predicted HLLSet
    - Combine column contributions
    - Apply strategy (conservative/exploratory/balanced)
    - Result: HLLSet fingerprint for next ingestion
    
Step 4: Disambiguate HLLSet → Tokens (using AM + LUT)
    - HLLSet is fingerprint - need actual tokens
    - Use current AM: look up (reg, zeros) for matching tokens
    - Use LUT: resolve any ambiguities
    - No new tokens created (HLLSet ops preserve existing tokens)
    
Step 5: Update AM with Predicted Tokens
    - Treat predicted tokens as ephemeral ingestion
    - Update AM: add tokens, recompute basic HLLSets
    
Step 6: Regenerate W from Updated AM
    - Extract new basic HLLSets from updated AM
    - Build new W lattice
    - This is the projected next state
```

### Projection Parameters for Single Basic Row HLLSet

For a specific basic row HLLSet r, we have:

**Row Parameters**:

- τ_row: Inclusion threshold for this row
- ρ_row: Exclusion threshold for this row

**For Each Column Basic c that r references**:

- **(D, R, N)_c**: Evolution triple showing what changed
  - D: Tokens deleted from this column
  - R: Tokens retained in this column
  - N: New tokens added to this column
- **τ_c**: Current inclusion threshold for this column
- **ρ_c**: Current exclusion threshold for this column
- **BSS_τ(r, c)**: Current structural connection
- **BSS_ρ(r, c)**: Current structural divergence

**Strategy Selection**: These parameters enable multiple decision strategies:

1. **Conservative**: Weight R heavily, ignore D and N → predict similar to current
2. **Growth-oriented**: Weight N heavily → predict additions
3. **Decay-oriented**: Weight D heavily → predict removals
4. **Balanced**: Equal weights to D, R, N → steady evolution
5. **BSS-guided**: Use BSS_τ, BSS_ρ to modulate column contributions

### Projection ≠ W Iteration

**Wrong approach** (what we had before):

```text
Iterate W transformations on HLLSets directly
R₀ → C₀ → R₁ → C₁ ...
```

**Correct approach**:

```text
Predict next ingestion → Update AM → Get new W
Current W + History → Predicted tokens → New AM → New W
```

### Detailed Algorithm

```python
def project_from_row_basic(row_basic, current_state, strategy, tau_row, rho_row, kernel):
    """
    Project next ingestion from a single basic row HLLSet.
    
    Args:
        row_basic: Specific basic row HLLSet to project from
        current_state: Current state with W lattice, AM, and history
        strategy: 'conservative', 'growth', 'decay', 'balanced', 'bss_guided'
        tau_row: Inclusion threshold for this row
        rho_row: Exclusion threshold for this row
        kernel: Kernel for operations
    
    Returns:
        predicted_tokens: Set of tokens predicted for next ingestion
        confidence: Prediction confidence
    """
    
    # Step 1: Get column basics referenced by this row (via W)
    W = current_state.W_lattice
    referenced_columns = []
    
    for col_basic in W.col_basic:
        bss_tau = row_basic.bss_tau(col_basic, kernel)
        bss_rho = row_basic.bss_rho(col_basic, kernel)
        
        # This row references this column
        if bss_tau >= tau_row and bss_rho <= rho_row:
            # Get evolution triple for this column from history
            D_c, R_c, N_c = current_state.get_evolution_triple(col_basic)
            
            referenced_columns.append({
                'col_basic': col_basic,
                'bss_tau': bss_tau,
                'bss_rho': bss_rho,
                'D': D_c,  # Deleted tokens
                'R': R_c,  # Retained tokens
                'N': N_c,  # New tokens
            })
    
    # Step 2: Apply strategy to aggregate column contributions
    predicted_hllset = BasicHLLSet.empty(kernel)
    
    for col_info in referenced_columns:
        # Select tokens based on strategy
        if strategy == 'conservative':
            # Weight retained heavily
            contribution = col_info['R']
            
        elif strategy == 'growth':
            # Weight new heavily
            contribution = col_info['R'].union(col_info['N'])
            
        elif strategy == 'decay':
            # Predict some deletion
            contribution = col_info['R'] - sample(col_info['D'])
            
        elif strategy == 'balanced':
            # Equal consideration to D, R, N
            contribution = col_info['R']
            if random() > 0.5:
                contribution = contribution.union(sample(col_info['N']))
            else:
                contribution = contribution - sample(col_info['D'])
                
        elif strategy == 'bss_guided':
            # Use BSS strength to modulate
            strength = col_info['bss_tau'] * (1 - col_info['bss_rho'])
            if strength > 0.7:
                contribution = col_info['R'].union(col_info['N'])
            elif strength < 0.3:
                contribution = col_info['R'] - sample(col_info['D'])
            else:
                contribution = col_info['R']
        
        # Add to predicted HLLSet
        predicted_hllset = predicted_hllset.union(
            BasicHLLSet(contribution, kernel),
            kernel
        )
    
    # Step 3: Disambiguate HLLSet → tokens using AM + LUT
    predicted_tokens = disambiguate_using_am_lut(
        predicted_hllset,
        current_state.AM,
        current_state.LUT,
        kernel
    )
    
    # Confidence based on BSS strengths and strategy
    confidence = compute_confidence(referenced_columns, strategy)
    
    return predicted_tokens, confidence


def disambiguate_using_am_lut(hllset, AM, LUT, kernel):
    """
    Convert HLLSet fingerprint to actual tokens using AM and LUT.
    
    Key insights:
    1. No new tokens created (HLLSet ops preserve existing tokens)
    2. Only need AM subset covered by predicted HLLSet
    3. Don't need to restore order - just find matching tokens
    
    Args:
        hllset: Predicted HLLSet (fingerprint)
        AM: Current adjacency matrix (only query subset)
        LUT: Lookup table for token clarification
    
    Returns:
        tokens: Set of actual tokens matching the HLLSet
    """
    candidate_tokens = set()
    
    # Get HLL registers that are non-zero in predicted HLLSet
    # These tell us which hash buckets contain tokens
    active_registers = hllset.get_active_registers()
    
    # Only scan AM entries whose tokens hash to active registers
    # This is MUCH smaller than full AM
    for reg_idx in active_registers:
        # Get AM entries that hash to this register
        # This is the key optimization - indexed by register
        am_entries = AM.get_entries_by_register(reg_idx)
        
        for entry in am_entries:
            token = entry.get_token()  # Extract from (reg, zeros)
            
            # Verify token matches HLLSet fingerprint
            token_hllset = BasicHLLSet({token}, kernel)
            
            if token_hllset.bss_tau(hllset, kernel) > 0.5:
                # This token is part of prediction
                # Use LUT to resolve any ambiguity
                clarified = LUT.resolve(token)
                candidate_tokens.add(clarified)
    
    return candidate_tokens
    
    
def disambiguate_simple(hllset, column_sources, kernel):
    """
    Simplified disambiguation: tokens come from column (D, R, N).
    
    Even simpler: We already know candidate tokens from applying
    strategy to column evolution triples. No AM scan needed!
    
    Args:
        hllset: Predicted HLLSet (for verification)
        column_sources: The actual tokens from (D, R, N) aggregation
        kernel: Kernel
    
    Returns:
        tokens: Verified token set
    """
    # The tokens are already known from strategy application!
    # HLLSet is just for fingerprint/verification
    
    # Optional: verify that tokens match HLLSet
    tokens_hllset = BasicHLLSet(column_sources, kernel)
    assert tokens_hllset == hllset or tokens_hllset.bss_tau(hllset, kernel) > 0.9
    
    return column_sources

                    # Use LUT to resolve any ambiguity
                    clarified = LUT.resolve(token)
                    candidate_tokens.add(clarified)
    
    return candidate_tokens


def project_full_state(current_state, strategies, kernel):
    """
    Project from all row basics with different strategies.
    
    Returns multiple projected future states.
    """
    projected_states = []
    
    for row_basic in current_state.W_lattice.row_basic:
        for strategy in strategies:
            # Get projection from this row with this strategy
            predicted_tokens, confidence = project_from_row_basic(
                row_basic=row_basic,
                current_state=current_state,
                strategy=strategy,
                tau_row=current_state.config.tau_inclusion,
                rho_row=current_state.config.rho_exclusion,
                kernel=kernel
            )
            
            # Update AM with predicted tokens
            ephemeral_AM = current_state.AM.copy()
            ephemeral_AM.add_tokens(predicted_tokens, kernel)
            
            # Recompute basic HLLSets from updated AM
            new_basic_row = ephemeral_AM.get_row_basic(kernel)
            new_basic_col = ephemeral_AM.get_col_basic(kernel)
            
            # Build new W lattice from updated AM
            new_W = HLLSetLattice(
                row_basic=new_basic_row,
                col_basic=new_basic_col,
                kernel=kernel,
                tau=current_state.config.tau_inclusion,
                rho=current_state.config.rho_exclusion
            )
            
            projected_states.append({
                'source_row': row_basic,
                'strategy': strategy,
                'predicted_tokens': predicted_tokens,
                'confidence': confidence,
                'AM': ephemeral_AM,
                'W': new_W
            })
    
    return projected_states
```

### Parallelization by Basic Row HLLSets

**Key**: Each basic row HLLSet can be projected independently.

```python
def parallel_projection(current_state, strategies, kernel):
    """
    Run projections in parallel by basic row HLLSets.
    
    Safe because:
    - Each row basic is immutable
    - Projections don't interfere
    - Can cache by content hash
    - AM + LUT lookups are read-only during projection
    """
    from concurrent.futures import ProcessPoolExecutor
    
    all_row_basics = current_state.W_lattice.row_basic
    
    # Project each row basic independently, for each strategy
    with ProcessPoolExecutor() as executor:
        futures = []
        for row_basic in all_row_basics:
            for strategy in strategies:
                future = executor.submit(
                    project_from_row_basic,
                    row_basic,
                    current_state,
                    strategy,
                    current_state.config.tau_inclusion,
                    current_state.config.rho_exclusion,
                    kernel
                )
                futures.append((row_basic, strategy, future))
        
        # Collect all projections
        all_projections = []
        for row_basic, strategy, future in futures:
            predicted_tokens, confidence = future.result()
            all_projections.append({
                'source_row': row_basic,
                'strategy': strategy,
                'predicted_tokens': predicted_tokens,
                'confidence': confidence
            })
    
    # For each projection, update AM and build W (can also parallelize)
    projected_states = []
    for proj in all_projections:
        ephemeral_AM = current_state.AM.copy()
        ephemeral_AM.add_tokens(proj['predicted_tokens'], kernel)
        
        new_basic_row = ephemeral_AM.get_row_basic(kernel)
        new_basic_col = ephemeral_AM.get_col_basic(kernel)
        
        new_W = HLLSetLattice(
            row_basic=new_basic_row,
            col_basic=new_basic_col,
            kernel=kernel,
            tau=current_state.config.tau_inclusion,
            rho=current_state.config.rho_exclusion
        )
        
        projected_states.append({
            'source_row': proj['source_row'],
            'strategy': proj['strategy'],
            'predicted_tokens': proj['predicted_tokens'],
            'confidence': proj['confidence'],
            'AM': ephemeral_AM,
            'W': new_W
        })
    
    return projected_states
```

### Strategy Examples

**Conservative Strategy**: Predict minimal change

```python
# Heavily weight retained tokens from columns
contribution = col_info['R']
```

**Growth Strategy**: Predict addition of new tokens

```python
# Include retained + new tokens
contribution = col_info['R'].union(col_info['N'])
```

**Decay Strategy**: Predict removal of some tokens

```python
# Include retained but sample some deletions
contribution = col_info['R'] - sample(col_info['D'])
```

**BSS-Guided Strategy**: Use connection strength to decide

```python
strength = bss_tau * (1 - bss_rho)
if strength > 0.7:  # Strong connection
    contribution = col_info['R'].union(col_info['N'])  # Growth
elif strength < 0.3:  # Weak connection
    contribution = col_info['R'] - sample(col_info['D'])  # Decay
else:  # Moderate connection
    contribution = col_info['R']  # Conservative
'''
    all_basics.extend(ingestion['basic_decomposition'].values())
    
    # Project each basic independently
    with ProcessPoolExecutor() as executor:
        futures = []
        for basic in all_basics:
            for tau in tau_grid:
                for rho in rho_grid:
                    future = executor.submit(
                        project_single_basic,
                        basic,
                        current_state.W_lattice,
                        tau,
    contribution = col_info['R']  # Conservative
'''
```

### Non-Determinism Sources

Multiple sources of non-determinism enable exploration of possible futures:

1. **Strategy Selection**: Conservative, growth, decay, balanced, BSS-guided
2. **Row Basic Selection**: Different row basics → different column references
3. **(D, R, N) Sampling**: When using D or N, can sample different subsets
4. **τ/ρ Thresholds**: Different values → different column references included
5. **Disambiguation**: Same HLLSet → multiple possible token sets from AM/LUT

**Result**: Exponential space of possible projections to explore in parallel!

## Computational Complexity Analysis

### Single Row Basic Projection

For one row basic with one strategy:

#### **Step 1: Get Referenced Columns**

- Check BSS against all column basics: **O(|C| · k)**
  - |C| = number of column basics
  - k = HLL register count (typically 2^12 = 4096)
  - BSS computation: O(k) per pair

#### **Step 2: Load Evolution Triples (D, R, N)**

- For each referenced column: **O(|C_ref| · |T|)**
  - |C_ref| = number of referenced columns (usually |C_ref| ≪ |C|)
  - |T| = average tokens per (D, R, N) triple

#### **Step 3: Apply Strategy & Aggregate**

- Process (D, R, N) for each column: **O(|C_ref| · |T|)**
- Aggregate into HLLSet: **O(|C_ref| · |T| · k)**
  - Each token insertion: O(k)

#### **Step 4: Disambiguation** - **NO AM SCAN NEEDED!**

Two approaches:

**Approach A: Direct from (D, R, N)** - **O(1)**

- Tokens already known from strategy application on (D, R, N)
- HLLSet is just fingerprint for verification
- No disambiguation needed - tokens are the input!

**Approach B: If AM lookup required** - **O(|active_reg| · avg_entries)**

- Only scan AM entries in active HLL registers: **O(k_active · e)**
  - k_active = number of non-zero registers in predicted HLLSet (typically ≪ k)
  - e = average AM entries per register (typically ≪ n²/k)
  - Total: k_active · e ≪ n²
- LUT resolution: **O(|matches| · log|LUT|)**
  - |matches| = tokens found (same as |C_ref| · |T|)

**Key Optimization**: Don't scan full AM (n²), only subset covered by predicted HLLSet!

#### **Step 5: Update AM with Predicted Tokens**

- Add tokens to AM: **O(|pred| · log n)**
  - |pred| = number of predicted tokens (typically |C_ref| · |T|)
  - Using indexed/sparse AM: log n per insertion
  - Not O(|pred| · n) - don't need full scan

#### **Step 6: Recompute Basic HLLSets from AM**

- **Only recompute affected portions**:
- Incremental update: **O(|pred| · k)**
  - Only update rows/cols touched by new tokens
  - Not O(n · k · α) - don't recompute everything

#### **Step 7: Build New W Lattice**

- **Lazy evaluation**: Don't build immediately
- On-demand morphism computation: **O(d · k)** per morphism needed
- Initial access: **O(|R'| · |C'| · d · k)** if full lattice needed
- But typically only access small subset

**Total for Single Projection (Optimized)**:

```text
O(|C| · k + |C_ref| · |T| · k + |pred| · log n + |pred| · k + lazy_W)
```

Simplifies to: **O(|C| · k + |C_ref| · |T| · k)**

**Dominant Terms**:

- **|C| · k** from checking all columns for BSS references
- **|C_ref| · |T| · k** from aggregating (D, R, N) into HLLSet

**Eliminated**:

- ~~n² from full AM scan~~ → Not needed! Tokens from (D, R, N) directly
- ~~|pred| · n from full AM update~~ → Indexed insertion: |pred| · log n
- ~~|R'| · |C'| · d · k from full W rebuild~~ → Lazy evaluation

**Speedup vs Naive**:

- Was: O(n² + |R'| · |C'| · d · k) 
- Now: O(|C| · k + |C_ref| · |T| · k)
- For n=1000, |C|=20, |C_ref|=5, |T|=100, k=4096:
  - Was: ~10⁶ + ~10⁶ = 2·10⁶
  - Now: ~8·10⁴ + ~2·10⁶ = 2.1·10⁶
  - But avoids n² scan entirely when n grows!

### Full Parallel Projection

**Total Projections**: |R| × S

- |R| = number of row basics (typically 10-100)
- S = number of strategies (typically 3-5)

**Without Parallelization**: **O(|R| · S · T_single)**

- T_single = time for single projection

**With Parallelization**: **O(T_single)**

- All |R| × S projections run simultaneously
- Limited only by CPU cores available

### Space Complexity

**Per Projection State**:

- **HLLSets**: O(k) per set
  - Row basics: O(|R| · k)
  - Col basics: O(|C| · k)
- **AM**: O(n²)
  - Sparse representation can reduce to O(nnz) where nnz = non-zero entries
- **W Lattice**: O(|R| · |C| · d)
  - Morphisms: each pair stores d-depth connections
- **History (D, R, N)**: O(|C| · |T|)
  - Three sets per column

**Total Storage**: O(n² + |R| · |C| · d + |C| · |T| + k · (|R| + |C|))

**All Projections**: |R| × S × storage_per_projection

- With 20 rows, 5 strategies = 100 parallel states
- Content-addressable caching reduces duplicate storage

### Optimization Strategies

1. **Sparse AM Representation**
   - Use CSR/CSC format: O(nnz) instead of O(n²)
   - Typical sparsity: nnz ≪ n²

2. **HLLSet Caching**
   - Content-addressable: same tokens → same hash
   - Reuse across projections
   - Cache BSS computations

3. **Incremental AM Updates**
   - Don't copy entire AM for each projection
   - Track deltas: added/removed tokens
   - Apply deltas on read

4. **Lazy W Regeneration**
   - Don't build full W immediately
   - Compute morphisms on-demand
   - Cache frequently accessed paths

5. **Incremental AM Updates**
   - Don't copy entire AM for each projection
   - Track deltas: added/removed tokens
   - Use indexed insertions: O(log n) not O(n)

6. **Lazy W Regeneration**
   - Don't build full W immediately
   - Compute morphisms on-demand
   - Cache frequently accessed paths

7. **No Full AM Scan**
   - Tokens come directly from (D, R, N)
   - If AM lookup needed, only scan active registers
   - Register-indexed AM: O(k_active · e) not O(n²)

8. **Early Pruning**
   - Stop low-confidence projections early
   - Use BSS thresholds to filter
   - Reduces wasted computation

### Complexity Comparison

| Operation | Naive | Optimized | With Optimizations |
| ----------- | ------- | ----------- | ------------------- |
| Disambiguation | O(n²) | O(1) | Tokens from (D,R,N) |
| AM Update | O(pred·n) | O(pred·log n) | Indexed insertion |
| Basic extraction | O(n·k·α) | O(pred·k) | Incremental only |
| W rebuild | O(R'·C'·d·k) | O(0) initially | Lazy evaluation |
| **Single projection** | **O(n² + R'·C'·d·k)** | **O(C·k + C_ref·T·k)** | **~2·10⁶ ops** |
| **Full (R·S)** | **O(R·S·...)** | **O(...)** | **Parallel: same** |

**Practical Example**:

- n = 1000 (AM size)
- |R| = 20, |C| = 20 (basics)
- |C_ref| = 5 (referenced columns)
- |T| = 100 (tokens per triple)
- S = 5 (strategies)
- d = 2 (morphism depth)
- k = 4096 (HLL registers)

**Naive approach**: O(100 · (10⁶ + 1.6·10⁶)) ≈ 2.6·10⁸ operations

**Optimized approach**: O(20·4096 + 5·100·4096) ≈ 8·10⁴ + 2·10⁶ = 2.1·10⁶ operations

**Speedup: 130x** (per projection)

With parallelization (100 projections): **13,000x** total speedup!

### Bottleneck Analysis

**Most Expensive Operations** (optimized, in order):

1. **Aggregate (D, R, N) into HLLSet**: O(|C_ref| · |T| · k)
   - Mitigation: Already optimal, but can cache HLLSets

2. **Check all columns for BSS**: O(|C| · k)
   - Mitigation: Early termination, BSS caching

3. **AM Update** (if needed): O(|pred| · log n)
   - Mitigation: Batch insertions, already fast

**Eliminated Bottlenecks**:

- ~~AM Scan O(n²)~~ → Not needed! (O(1) disambiguation)
- ~~Full W Rebuild O(R'·C'·d·k)~~ → Lazy evaluation
- ~~Full Basic Extraction O(n·k·α)~~ → Incremental updates

**Least Expensive**:

- Strategy application: O(|C_ref| · |T|) - just set operations
- BSS computation: O(k) - constant time for fixed k
- Evolution triple lookup: O(1) with hash table

## Projection Storage: Separate from Evolution

### Key Distinction

**Evolution = Reality**:

- Committed states
- Actual ingested data
- Irreversible history
- Stored in persistent Git-like structure

**Projection = Hypothesis**:

- Exploratory exercises
- Predicted futures (not actual)
- Temporary, can be discarded
- Stored in separate projection workspace

### Why Separate Storage?

1. **Volume**: Projections generate many hypothetical states (R × S combinations)
2. **Lifetime**: Most projections are temporary exploration, not permanent
3. **Isolation**: Don't pollute committed history with "what-if" scenarios
4. **Comparison**: Need to compare projections against each other and reality
5. **Cleanup**: Can discard low-confidence projections without affecting evolution

### Projection Workspace Structure

```python
class ProjectionWorkspace:
    """
    Separate storage for projection exercises.
    Isolated from committed evolution history.
    """
    
    def __init__(self, base_state):
        self.base_state = base_state  # Reference to committed state
        self.projections = {}  # Projection ID → ProjectionState
        self.metadata = {}     # Projection metadata (params, confidence)
        
    def create_projection(self, row_basic, strategy, tau, rho):
        """Create new projection from base state."""
        proj_id = hash(row_basic, strategy, tau, rho)
        
        # Run projection algorithm
        predicted_tokens, confidence = project_from_row_basic(
            row_basic, self.base_state, strategy, tau, rho
        )
        
        # Store in workspace (not in committed history!)
        self.projections[proj_id] = ProjectionState(
            predicted_tokens=predicted_tokens,
            hypothetical_AM=None,  # Lazy: compute on demand
            hypothetical_W=None,   # Lazy: compute on demand
            confidence=confidence
        )
        
        self.metadata[proj_id] = {
            'source_row': row_basic,
            'strategy': strategy,
            'tau': tau,
            'rho': rho,
            'parent_commit': self.base_state.commit_hash
        }
        
        return proj_id
    
    def materialize_projection(self, proj_id):
        """
        Fully compute hypothetical AM and W for projection.
        Only done on-demand for high-confidence projections.
        """
        proj = self.projections[proj_id]
        if proj.hypothetical_AM is None:
            # Compute hypothetical AM
            proj.hypothetical_AM = self.base_state.AM.copy()
            proj.hypothetical_AM.add_tokens(proj.predicted_tokens)
            
            # Compute hypothetical W
            proj.hypothetical_W = build_W_from_AM(proj.hypothetical_AM)
    
    def compare_projections(self, proj_id1, proj_id2):
        """Compare two projections."""
        proj1 = self.projections[proj_id1]
        proj2 = self.projections[proj_id2]
        
        return {
            'token_overlap': len(proj1.predicted_tokens & proj2.predicted_tokens),
            'token_diff': len(proj1.predicted_tokens ^ proj2.predicted_tokens),
            'confidence_gap': abs(proj1.confidence - proj2.confidence)
        }
    
    def prune_low_confidence(self, threshold=0.3):
        """Remove projections below confidence threshold."""
        to_remove = [
            proj_id for proj_id, proj in self.projections.items()
            if proj.confidence < threshold
        ]
        for proj_id in to_remove:
            del self.projections[proj_id]
            del self.metadata[proj_id]
        return len(to_remove)
    
    def select_for_evolution(self, criteria='highest_confidence'):
        """
        Select which projection to actually execute (commit to evolution).
        This is the bridge from projection → reality.
        """
        if criteria == 'highest_confidence':
            best = max(self.projections.items(), 
                      key=lambda x: x[1].confidence)
            return best[0]
        # ... other selection criteria
```

### Storage Layout

```text
workspace/
├── committed/              # Actual evolution history (permanent)
│   ├── genesis.commit
│   ├── step_001.commit
│   ├── step_002.commit
│   └── ...
├── projections/           # Temporary projection workspace
│   ├── proj_001.json     # Lightweight: tokens + metadata only
│   ├── proj_002.json
│   ├── proj_003.json
│   └── ...
└── projection_index.db   # SQLite index for fast queries
```

**Committed files** (permanent):

- Full AM state
- W lattice basics
- Evolution triples (D, R, N)
- Ingestion history

**Projection files** (temporary):

- Predicted tokens (lightweight)
- Strategy parameters
- Confidence scores
- Parent reference (which commit projected from)

### Projection Lifecycle

1. **Create**: Generate many projections in parallel
2. **Store**: Save lightweight (just tokens + metadata)
3. **Compare**: Analyze differences between projections
4. **Prune**: Discard low-confidence projections
5. **Select**: Choose one to execute
6. **Commit**: Selected projection becomes actual evolution step
7. **Cleanup**: Discard remaining projections

### When Projection Becomes Evolution

```python
def execute_selected_projection(workspace, proj_id, evolution_manager):
    """
    Bridge from projection (hypothesis) to evolution (reality).
    Selected projection is actually ingested and becomes committed state.
    """
    proj = workspace.projections[proj_id]
    
    # These predicted tokens now become REAL tokens
    real_tokens = proj.predicted_tokens
    
    # Ingest into actual evolution
    evolution_manager.ingest(real_tokens)
    evolution_manager.evolve()
    
    # Commit to permanent storage
    commit_hash = evolution_manager.commit()
    
    # Clear projection workspace (or archive for analysis)
    workspace.clear()
    
    return commit_hash
```

### Benefits of Separate Storage

1. **Performance**: Don't persist every hypothetical state
2. **Clarity**: Clear distinction between reality and exploration
3. **Flexibility**: Can discard, compare, re-run projections freely
4. **Scalability**: Generate 1000s of projections without storage explosion
5. **Rollback**: Projections don't affect committed history
6. **Analysis**: Can study "roads not taken" after evolution

### Summary: Two Storage Layers

**Layer 1: Committed Evolution** (permanent)

- Actual states that happened
- Full AM + W lattice
- Irreversible history
- Content-addressed, immutable

**Layer 2: Projection Workspace** (temporary)

- Hypothetical futures being explored
- Lightweight (just tokens + metadata)
- Can be pruned, compared, selected
- Ephemeral, session-scoped

### Summary: Correct Projection Flow

```text
┌─────────────────────────────────────────────────────────┐
│  COMMITTED STATE (Current)                              │
│  - AM (adjacency matrix) with (reg, zeros) tokens       │
│  - LUT (lookup table) for token clarification           │
│  - W lattice (from AM basic HLLSets)                    │
│  - History: Evolution triples (D, R, N) per column      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  SELECT ROW BASIC + STRATEGY (parallel)                 │
│  - Choose row basic HLLSet from W                       │
│  - Choose strategy (conservative/growth/decay/etc)      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GET REFERENCED COLUMNS                                 │
│  - Via BSS_τ(row, col) and BSS_ρ(row, col)              │
│  - For each: load (D, R, N) from history                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  APPLY STRATEGY → AGGREGATE CONTRIBUTIONS               │
│  - Conservative: use R                                  │
│  - Growth: use R ∪ N                                    │
│  - Decay: use R \ sample(D)                             │
│  - Result: Predicted HLLSet fingerprint                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  DISAMBIGUATE HLLSET → TOKENS (using AM + LUT)          │
│  - HLLSet is fingerprint - need actual tokens           │
│  - Scan AM for matching (reg, zeros)                    │
│  - Use LUT to resolve ambiguities                       │
│  - No new tokens created (ops preserve existing)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  UPDATE AM WITH PREDICTED TOKENS                        │
│  - Add tokens to AM                                     │
│  - Recompute basic HLLSets from updated AM              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  REGENERATE W FROM UPDATED AM                           │
│  - Extract new row/col basic HLLSets                    │
│  - Build new W lattice                                  │
│  - THIS IS ONE PROJECTED FUTURE STATE                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  DISAMBIGUATE HLLSET → TOKENS                           │
│  - HLLSet is lossy - find candidate token sets          │
│  - Use token mappings from history                      │
│  - Generate multiple possibilities (non-deterministic)  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  UPDATE AM WITH PREDICTED TOKENS (per candidate)        │
│  - Add tokens to AM                                     │
│  - Recompute basic HLLSets                              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  REGENERATE W FROM UPDATED AM                           │
│  - Extract new row/col basic HLLSets                    │
│  - Build new W lattice                                  │
│  - THIS IS ONE PROJECTED FUTURE STATE                   │
└─────────────────────────────────────────────────────────┘
```

### Projection vs Evolution

**Projection**: What *might* be ingested (prediction)

- Select row basic + strategy
- Use column (D, R, N) to aggregate predicted tokens
- Disambiguate HLLSet using AM + LUT
- Hypothetical AM update → hypothetical W
- Non-deterministic (multiple possible futures)

**Evolution**: What *actually* is ingested (reality)

- Real tokens arrive from external source
- Real AM update with actual tokens
- Real W regeneration from actual AM
- Deterministic (one realized outcome)

Projection helps us **anticipate and prepare** for evolution before it occurs!

### Sub-Lattices and Parallel Exploration

**Critical Architectural Properties**:

1. **Immutability**: HLLSets never change after creation
2. **Idempotence**: Same operation → same result (can retry safely)
3. **Content-Addressability**: HLLSets identified by content hash
4. **AM + LUT Read-Only**: Disambiguation doesn't modify state

**Consequence**: We always work with **sub-lattices** (views of full lattice), and can explore **massively in parallel**.

```text
Full Lattice W_full = all possible (row, col) pairs
Sub-Lattice W_sub = subset of pairs (projection/view)

All operations on sub-lattices are:
- Safe (immutable - no side effects)
- Cacheable (content-addressed - reuse computations)
- Parallelizable (independent - no shared state)
```

#### Parallel Projection Exploration

```python
def explore_projections_parallel(initial_lattice, num_steps, tau_rho_grid):
    """
    Explore multiple projection trajectories in parallel.
    
    Safe because:
    - Immutability: No trajectory affects another
    - Idempotence: Can recompute any step
    - Content-addressability: Cache/reuse sub-lattice computations
    """
    from concurrent.futures import ProcessPoolExecutor
    
    # Each (τ, ρ) produces independent trajectory
    futures = []
    with ProcessPoolExecutor() as executor:
        for tau, rho in tau_rho_grid:
            future = executor.submit(
                project_forward,
                initial_lattice,
                num_steps,
                tau,
                rho
            )
            futures.append((tau, rho, future))
    
    # Collect all trajectories
    all_trajectories = []
    for tau, rho, future in futures:
        trajectory = future.result()
        all_trajectories.append({
            'tau': tau,
            'rho': rho,
            'trajectory': trajectory
        })
    
    return all_trajectories
```

#### Content-Addressed Caching

```python
def project_with_cache(lattice, tau, rho, cache):
    """
    Project using content-addressed cache.
    
    Key insight: Same (lattice_hash, tau, rho) → Same result
    Can cache at HLLSet level and lattice level
    """
    lattice_hash = hash_lattice(lattice)
    cache_key = (lattice_hash, tau, rho)
    
    if cache_key in cache:
        return cache[cache_key]  # Reuse computation
    
    # Compute projection
    result = compute_projection(lattice, tau, rho)
    
    # Cache for reuse
    cache[cache_key] = result
    
    return result
```

#### Sub-Lattice Operations

All operations work on **sub-lattices** without affecting full system:

- **Projection**: Explore possible futures without commitment
- **Branching**: Try multiple paths simultaneously
- **Rollback**: Discard projections safely (no cleanup needed)
- **Merge**: Combine sub-lattices when ready

**No locks, no race conditions, no cleanup** - functional architecture enables fearless parallelism!

## Noether Current as Sustainability Criterion

### Conservative Evolution

From Noether's theorem: **Symmetry → Conservation**

In our context:

- **Symmetry**: Structural preservation during evolution
- **Conservation**: Information balance (|D| ≈ |N|)

### Noether Current Φ

```text
Φ = (|N| - |D|) / (|R| + |N|)
```

**Interpretation**:

- Φ = 0: Perfectly conservative (|N| = |D|)
- Φ > 0: Information growth
- Φ < 0: Information decay

### Sustainable Evolution

**Target**: Keep Φ near zero over time.

- **Φ → 0**: System maintains steady state
- **|Φ| large**: System unstable (runaway growth or collapse)

### Management Strategy

Use τ/ρ thresholds to **steer** evolution toward Φ ≈ 0:

- If Φ > 0 (too much growth): Increase τ, decrease ρ (more selective, less ingestion)
- If Φ < 0 (too much decay): Decrease τ, increase ρ (less selective, more ingestion)

## Multiple Lattice Manifold

### Co-Evolution

Multiple W lattices evolve simultaneously:

```text
W_1(t+1) = evolve(W_1(t), ephemeral_1)
W_2(t+1) = evolve(W_2(t), ephemeral_2)
...
W_N(t+1) = evolve(W_N(t), ephemeral_N)
```

**Not synchronized**: Each lattice evolves at its own pace.

### Cross-Lattice Influence

Lattices influence each other through **structural coupling**:

```text
W_i(t+1) = evolve(W_i(t), ephemeral_i, context=[W_j for j ≠ i])
```

Structural similarity (ε-isomorphism) between lattices affects evolution:

- High similarity → Mutual reinforcement
- Low similarity → Independent evolution

### Manifold Coherence

**Global Noether Current**:

```text
Φ_manifold = Σ_i w_i · Φ_i
```

where w_i are lattice weights (importance/size).

**Sustainable manifold**: Φ_manifold ≈ 0

Different lattices may have Φ_i ≠ 0, but they balance globally.

## Practical Implementation

### Step 1: Single Lattice Projection

Start simple:

1. Given W_current
2. Analyze BSS structure
3. Project next state with different (τ, ρ)
4. Compare projections

### Step 2: Evolution Management

1. Monitor Noether current Φ
2. Adjust τ/ρ to steer toward Φ ≈ 0
3. Track evolution trajectory

### Step 3: Multi-Lattice Manifold

1. Multiple lattices evolving independently
2. Cross-lattice structural coupling
3. Global Φ_manifold monitoring
4. Coordinated management (swarm behavior)

## Mathematical Formalization

### Evolution Operator

```text
E: (W, τ, ρ) → W'
```

Non-deterministic: Same W with different (τ, ρ) → Different W'

### Trajectory Space

```text
T = {(W_0, W_1, W_2, ...)} subject to Φ(W_i → W_{i+1}) ≈ 0
```

Sustainable trajectories form a **manifold** in state space.

### Projection Operator

```text
P: (W, τ_proj, ρ_proj) → (W_projected, confidence)
```

Predicts next state without executing evolution.

## Applications

### 1. Adaptive AI Systems

- Monitor system Φ in real-time
- Adjust learning rates (τ/ρ) to prevent runaway growth or collapse
- Maintain sustainable knowledge accumulation

### 2. Multi-Agent Systems

- Each agent = W lattice
- Co-evolve without destroying each other
- Swarm intelligence through manifold coherence

### 3. Language Model Training

- W_train = Training data lattice
- W_model = Model knowledge lattice
- Sustainable training: Φ_model ≈ 0 over epochs

### 4. Knowledge Base Evolution

- W_KB = Current knowledge base
- W_new = Incoming information
- Controlled integration maintaining Φ ≈ 0

## References

- **Noether's Theorem**: Conservation laws from symmetry
- **PSO**: Particle Swarm Optimization (Kennedy & Eberhart, 1995)
- **Self-Reproduction**: von Neumann's Universal Constructor
- **DOCS/HRT_LATTICE_THEORY.md**: W lattice structure
- **DOCS/CONTEXTUAL_SELECTION_MECHANICS.md**: τ/ρ mechanics
