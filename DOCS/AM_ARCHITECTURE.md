# Adjacency Matrix - Two-Phase Architecture

## Overview

The Adjacency Matrix (AM) serves as a bridge between the ingestion and query phases, preserving token order that HLLSets cannot maintain.

## IICA Core Principles

AM construction must satisfy **IICA** properties:

- **Immutable**: Once built, AM never changes
- **Idempotent**: Re-ingesting same data produces identical AM
- **Content Addressable**: Cells identified by content hash

The specific tokenization strategy, window size, or overlap approach can vary,
as long as (within a single perceptron):

1. **Same hash morphism** used for all HLLSets (sha1 with SHARED_SEED)
2. **Same AM→W transformation** algorithm applied
3. **IICA properties** preserved throughout

**Note**: Each perceptron has its own AM and W lattices and can use its own
hash morphism. The consistency requirement is per-perceptron, not global.
This enables multi-modal systems where different perceptrons process
different data types with optimized hash functions.

This flexibility allows experimentation with different construction strategies
while maintaining structural topology preservation.

## Phase 1: Ingestion (Building the AM)

```text
Input Data: "The quick brown fox jumps over the lazy dog"
                    ↓
         ┌──────────────────────┐
         │   Tokenization       │
         └──────────────────────┘
                    ↓
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
                    ↓
         ┌──────────────────────┐
         │ Add START/END Markers│
         └──────────────────────┘
                    ↓
  ["START", "the", "quick", "brown", ..., "dog", "END"]
                    ↓
         ┌──────────────────────┐
         │ Sliding Window (3,1) │
         │                      │
         │ (START, the, quick)  │
         │ (the, quick, brown)  │
         │ (quick, brown, fox)  │
         │ ...                  │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  Compute (reg,zeros) │
         │  from token hashes   │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │   Update AM Cells    │
         │                      │
         │  Cell(id_a, id_b)++  │
         │  frequency++         │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Update Row/Col       │
         │ HLLSets              │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │   Commit Batch       │
         │   (driver→IDLE)      │
         └──────────────────────┘
                    ↓
          ✅ AM Ready for Query Phase
```

**Result**:

- Sparse matrix ~100K × 100K with transition frequencies
- Row/Column HLLSets for each identifier
- START/END boundaries preserved

## Phase 2: Query/Prompt Processing (Order Reconstruction)

```text
User Prompt: "Show me data about quick brown fox"
                    ↓
         ┌──────────────────────┐
         │ Create Prompt HLLSet │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Retrieve Relevant    │
         │ HLLSets from System  │
         └──────────────────────┘
                    ↓
  HLLSets: {distinct tokens, no order}
  Example: {"the", "quick", "brown", "fox", "jumps", ...}
  Cardinality: 9
                    ↓
         ┌──────────────────────┐
         │  ⚠️ PROBLEM:         │
         │  HLLSets have tokens │
         │  but NO ORDER!       │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  💡 SOLUTION:        │
         │  Traverse AM         │
         │  START → END         │
         └──────────────────────┘
                    ↓
    reconstruct_order(cardinality=9, threshold=0.9)
                    ↓
         ┌──────────────────────┐
         │ Traversal Algorithm: │
         │                      │
         │ 1. current = START   │
         │ 2. path = []         │
         │ 3. while not done:   │
         │    - Get cells from  │
         │      row=current     │
         │    - Skip END if     │
         │      len(path) < 8   │
         │    - Pick highest    │
         │      frequency next  │
         │    - path.append()   │
         │    - current = next  │
         └──────────────────────┘
                    ↓
  Ordered IDs: [(reg₁,z₁), (reg₂,z₂), ..., (reg₉,z₉)]
                    ↓
         ┌──────────────────────┐
         │ Map IDs to Tokens    │
         │ using LUT            │
         └──────────────────────┘
                    ↓
  Ordered Tokens: ["the", "quick", "brown", "fox", "jumps", ...]
                    ↓
         ┌──────────────────────┐
         │ Present Results in   │
         │ Original Order       │
         └──────────────────────┘
                    ↓
    ✅ "The quick brown fox jumps over the lazy dog"
```

## Key Components

### 1. Adjacency Matrix Structure

```text
       Col₀  Col₁  Col₂  Col₃  ...
Row₀  [freq  freq  0     freq  ...]
Row₁  [0     freq  freq  0     ...]
Row₂  [freq  0     freq  freq  ...]
Row₃  [0     0     0     freq  ...]
...

Where:
- Row/Col IDs are (reg, zeros) tuples
- freq = transition frequency (how often col follows row)
- Sparse storage: only non-zero cells kept
```

### 2. Special Tokens

```text
START_ID = (-1, 0)  ← Always first in traversal
END_ID = (-2, 0)    ← Stop condition with threshold

Example transitions:
START → (257, 0) → (465, 1) → (33, 0) → ... → END
```

### 3. Threshold Logic

```python
threshold = 0.9 × cardinality

# Example: cardinality = 15
# threshold = 0.9 × 15 = 13.5 ≈ 14 tokens

# During traversal at step 5:
if current_transitions contains END:
    if len(path) < 14:
        # Skip END, continue traversal
        next = highest_frequency_non_end()
    else:
        # Reached threshold, allow END
        next = END
```

**Why Needed**:

- Prevents premature termination
- END might appear as valid transition early in path
- Need most tokens before stopping
- Balances completeness vs infinite loops

## Data Flow Summary

```text
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PHASE                          │
│                                                             │
│  Raw Data → Tokens → AM Building → Row/Col HLLSets          │
│             ↓                                               │
│       START/END markers                                     │
│       Sliding window                                        │
│       Frequency tracking                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ AM persisted
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     QUERY PHASE                             │
│                                                             │
│  Prompt → HLLSets → Retrieve → AM Traversal → Ordered       │
│           (no order)            ↓                           │
│                          reconstruct_order()                │
│                          (START → END path)                 │
│                                 ↓                           │
│                          LUT mapping                        │
│                                 ↓                           │
│                          Original tokens                    │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Status

### ✅ Completed (Ingestion Phase)

- [x] IngestionAdjacencyMatrix dataclass
- [x] Sliding window processing (size 3, step 1)
- [x] (reg, zeros) identifier computation
- [x] Frequency tracking in cells
- [x] Row/Column HLLSet updates
- [x] START/END marker insertion
- [x] Commit operation for batch finalization
- [x] All 14 tests passing

### ⏳ Pending (Query Phase)

- [ ] reconstruct_order() implementation
- [ ] Traversal algorithm with threshold logic
- [ ] LUT integration for ID→token mapping
- [ ] Complete query processing pipeline
- [ ] Query phase tests

## Next Steps

1. **Implement reconstruct_order()**
   - Traverse AM from START to END
   - Follow highest frequency transitions
   - Apply threshold logic for END skipping
   - Return ordered list of identifiers

2. **Integrate with LUT**
   - Map (reg, zeros) back to tokens
   - Use n-token disambiguation
   - Handle ambiguous cases

3. **Build Query Pipeline**
   - Prompt processing
   - HLLSet retrieval
   - Order reconstruction
   - Result presentation

4. **Test Query Flow**
   - End-to-end query tests
   - Threshold logic validation
   - Order accuracy verification
