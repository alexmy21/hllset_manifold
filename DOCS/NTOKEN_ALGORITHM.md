# N-Token Ingestion Algorithm

## Overview

The **n-token ingestion algorithm** solves the fundamental problem of hash collisions in HLLSet-based systems by creating multiple representations of the same data using different n-token groupings. This enables disambiguation through intersection of candidate sets from Lookup Tables (LUTs).

## The Hash Collision Problem

### One-to-Many Mapping

In HLLSet systems, we face two levels of information loss:

1. **Hash Function**: Many tokens → one hash (collision)
2. **HLLSet Storage**: Many hashes → one bit position (register, zeros)

This makes it **impossible** to recover original tokens from HLLSet bits alone.

### Example of the Problem

```text
Tokens: {"apple", "apply", "application"}
       ↓ (hash function)
Hashes: {0x12AB...}  (all collide to same hash!)
       ↓ (HLLSet register allocation)
Bit: (reg=42, zeros=3)
```

**Question**: Which token produced this bit?  
**Answer**: We don't know! Could be any of them.

## The Solution: N-Token Representations

### Core Idea

Create **multiple representations** of the same token sequence using different n-token groupings:

```text
Original: ["the", "quick", "brown", "fox"]

1-tokens: [("the",), ("quick",), ("brown",), ("fox",)]
2-tokens: [("the","quick"), ("quick","brown"), ("brown","fox")]
3-tokens: [("the","quick","brown"), ("quick","brown","fox")]
```

### Why This Works

1. **Different HLLSets**: Each n-token group creates its own HLLSet
2. **Same Tokens**: All groups contain the same underlying tokens
3. **Natural Hash Differences**: Concatenating different numbers of tokens creates different hashes
   - 1-token: `hash("the")` ≠ 2-token: `hash("thequick")` ≠ 3-token: `hash("thequickbrown")`
   - No prefixes needed - natural concatenation distinguishes them (up to hash collision)
4. **Fully Content Addressable**: 
   - Compute hash from tokens → get (reg, zeros)
   - Look up (reg, zeros) in LUT
   - LUT tokens field reveals n-group: tuple length = n
   - Example: `[('the',)]` = 1-token, `[('the','quick')]` = 2-token
5. **Intersection Disambiguates**: Cross-referencing narrows candidates

### Mathematical Invariant

For any token set T:

```text
Union(1-tokens) = Union(2-tokens) = Union(3-tokens) = T
```

But the **structure** is different, enabling disambiguation!

## Algorithm Components

### 1. Tokenization (Order-Preserving)

```python
text = "the quick brown fox jumps"
tokens = ["the", "quick", "brown", "fox", "jumps"]  # List, not set!
```

**Key**: Order matters for sliding window n-token generation.

### 2. N-Token Generation (Sliding Window)

```python
def generate_n_tokens(tokens, n):
    result = []
    for i in range(len(tokens) - n + 1):
        result.append(tuple(tokens[i:i+n]))
    return result

# Example: n=2
# ["a", "b", "c", "d"] → [("a","b"), ("b","c"), ("c","d")]
```

**Properties**:

- Preserves sequential order
- Overlapping windows capture relationships
- Natural sliding window over text

### 3. HLLSet Creation (Per N-Group)

Each n-token group gets a separate HLLSet with hashes computed from natural concatenation:

```python
# 1-tokens: Hash from single token
token_1 = "the"
hash_1 = hash("the")  # Just the token

# 2-tokens: Hash from concatenation of 2 tokens
token_2 = ("the", "quick")
hash_2 = hash("thequick")  # Concatenated: "the" + "quick"

# 3-tokens: Hash from concatenation of 3 tokens  
token_3 = ("the", "quick", "brown")
hash_3 = hash("thequickbrown")  # Concatenated: "the" + "quick" + "brown"
```

**Why This Works**:
- Different n-token groups produce different hashes naturally (up to collision)
- `hash("the")` ≠ `hash("thequick")` ≠ `hash("thequickbrown")`
- **Content addressable**: Given tokens and knowing which n-group, hash is deterministic
- **No prefixes needed**: Natural concatenation distinguishes n-groups

**Lookup Example**:
```python
# To look up 1-token "the": compute hash("the"), check 1-token HLLSet
# To look up 2-token ("the", "quick"): compute hash("thequick"), check 2-token HLLSet
# You need to know the n-group, but hash computation is straightforward
```

### 4. LUT (Lookup Table) Construction

For each n-token group, build LUT mapping:

```text
(reg, zeros) → LUTRecord {
    reg: int
    zeros: int
    hashes: [hash1, hash2, ...]
    tokens: [(token_seq1,), (token_seq2,), ...]
}
```

**Purpose**: Track which token sequences contributed to each bit.

### 5. Token Disambiguation (Substring Search + Intersection)

```python
def find_tokens(query_substring):
    # Search across ALL LUT groups with substring query
    results_1 = LUT_1.search_substring(query_substring)
    results_2 = LUT_2.search_substring(query_substring)
    results_3 = LUT_3.search_substring(query_substring)
    
    # Each result has tokens field - split by array length
    tokens_1 = [seq for r in results_1 for seq in r.tokens if len(seq) == 1]
    tokens_2 = [seq for r in results_2 for seq in r.tokens if len(seq) == 2]
    tokens_3 = [seq for r in results_3 for seq in r.tokens if len(seq) == 3]
    
    return {
        '1-token': tokens_1,
        '2-token': tokens_2,
        '3-token': tokens_3
    }

# Intersection for disambiguation
def disambiguate(reg, zeros):
    candidates_1 = LUT_1[reg, zeros].tokens  # Get all token sequences
    candidates_2 = LUT_2[reg, zeros].tokens
    candidates_3 = LUT_3[reg, zeros].tokens
    
    # Split by array length to identify n-group
    # Then intersect to narrow down
    result = set()
    for seq in candidates_1:
        if len(seq) == 1:  # 1-token group
            result.add(seq[0])
    
    return result
```

**Key Insight**: Query with substring → get all matches → split by `len(tokens)` to determine n-group

## Implicit Order Preservation

The n-token structure implicitly preserves order:

```text
1-tokens: ("a",) < ("b",) < ("c",) < ("d",)
2-tokens: ("a","b") < ("b","c") < ("c","d")
3-tokens: ("a","b","c") < ("b","c","d")
```

**Implicit Order**:

```text
("a",) < ("a","b") < ("a","b","c") < ("b",) < ("b","c") < ("b","c","d") < ...
```

This creates a **partial order** on the token sequence that can be used to reconstruct sequential relationships.

## Data Structures

### NTokenRepresentation

```python
@dataclass
class NTokenRepresentation:
    original_tokens: List[str]                              # ["the", "quick", ...]
    n_token_groups: Dict[int, List[Tuple[str, ...]]]      # {1: [("the",), ...], 2: [...]}
    hllsets: Dict[int, HLLSet]                             # {1: HLLSet1, 2: HLLSet2, ...}
    luts: Dict[int, Dict[Tuple[int, int], LUTRecord]]     # {1: LUT1, 2: LUT2, ...}
```

### LUTRecord

```python
@dataclass
class LUTRecord:
    reg: int                                # Register number
    zeros: int                              # Run of trailing zeros
    hashes: List[int]                       # Hash values contributing to this bit
    tokens: List[Tuple[str, ...]]          # Token sequences for these hashes
```

### TokenizationConfig

```python
@dataclass
class TokenizationConfig:
    use_n_tokens: bool = True                         # Enable n-token algorithm
    n_token_groups: List[int] = [1, 2, 3]            # Which n-token groups to create
    maintain_order: bool = True                       # Use sliding window vs combinations
    min_token_length: int = 1
    max_token_length: int = 100
    lowercase: bool = True
    remove_punctuation: bool = False
    split_on: str = " "
```

## Complexity Analysis

### Space Complexity

For N tokens and n-token groups [1, 2, ..., k]:

- **1-tokens**: N tuples
- **2-tokens**: N-1 tuples
- **3-tokens**: N-2 tuples
- **...**
- **k-tokens**: N-k+1 tuples

**Total tuples**: ≈ k*N (linear in both k and N)

**HLLSets**: k HLLSets (one per group)

**LUTs**: k LUTs with ≈ N entries each

**Total space**: O(k * N)

### Time Complexity

**Ingestion**:

- Tokenization: O(N)
- N-token generation: O(k * N)
- HLLSet creation: O(k * N) (each token hashed once per group)
- LUT construction: O(k * N)

**Total**: O(k * N) where k is typically small (e.g., 3)

**Disambiguation**:

- Lookup in each LUT: O(k)
- Intersection of k sets: O(k * C) where C = avg candidates per set
- **Total**: O(k * C)

## Use Cases

### 1. Token Recovery from HLLSet Bits

Given a bit (reg, zeros) in multiple HLLSets:

```python
# Which tokens contributed to this bit?
candidates = representation.disambiguate_tokens(reg=42, zeros=3)
# Result: {'quick', 'brown'} (narrowed from many possibilities)
```

### 2. Order Reconstruction

```python
implicit_order = representation.get_implicit_order()
# [("a",), ("a","b"), ("a","b","c"), ("b",), ("b","c"), ...]
```

### 3. Adjacency Matrix Construction

The n-token structure naturally forms an **Adjacency Matrix (AM)** for HRT:

- **Nodes**: Tokens
- **Edges**: Co-occurrence in n-token groups
- **Weights**: Frequency of co-occurrence

```text
1-tokens: Self-loops (token exists)
2-tokens: Direct edges (sequential pairs)
3-tokens: Triangles (sequential triples)
```

### 4. Streaming Data Ingestion

Perfect for streaming:

```python
# Process stream chunk by chunk
for chunk in data_stream:
    representation = driver.process(chunk, kernel)
    # Immediately have n-token structure with LUTs
```

## Configuration Examples

### Example 1: Default (1, 2, 3-tokens)

```python
config = TokenizationConfig(
    use_n_tokens=True,
    n_token_groups=[1, 2, 3],
    maintain_order=True
)

driver = IngestDriver("default", config)
```

**Best for**: General text processing, moderate disambiguation

### Example 2: Deep N-Tokens (1-5)

```python
config = TokenizationConfig(
    use_n_tokens=True,
    n_token_groups=[1, 2, 3, 4, 5],
    maintain_order=True
)
```

**Best for**: Fine-grained disambiguation, phrase detection

**Trade-off**: More space, better disambiguation

### Example 3: Simple Mode (No N-Tokens)

```python
config = TokenizationConfig(
    use_n_tokens=False
)
```

**Best for**: Simple cardinality counting, no reconstruction needed

**Trade-off**: Minimal space, no disambiguation

### Example 4: Long Sequences

```python
config = TokenizationConfig(
    use_n_tokens=True,
    n_token_groups=[1, 2, 3, 5, 8, 13],  # Fibonacci-like
    maintain_order=True
)
```

**Best for**: Capturing multi-scale patterns, hierarchical structure

## Integration with HRT

### Adjacency Matrix Construction

```python
# From n-token representation
representation = os.ingest(text)

# Build AM from 2-token groups (edges)
am = AdjacencyMatrix()
for token_pair in representation.n_token_groups[2]:
    source, target = token_pair
    am.add_edge(source, target, weight=1.0)

# Now feed to HRT
hrt = HRT(adjacency_matrix=am, ...)
```

### Lattice Structure

The n-token groups form a **natural lattice**:

```text
                ("a","b","c")
               /              \
        ("a","b")            ("b","c")
           |                    |
         ("a")                ("b")
```

This is perfect for HRT's lattice-based operations!

## Advantages

1. **Disambiguation**: Intersection narrows candidates exponentially
2. **Order Preservation**: Sliding window maintains sequential structure
3. **Scalability**: Linear space/time complexity
4. **Streaming**: Can process data incrementally
5. **Flexibility**: Configurable n-token groups
6. **HRT Integration**: Natural adjacency matrix structure
7. **Parallel**: Different n-groups can be processed in parallel

## Disadvantages

1. **Space Overhead**: k HLLSets instead of 1
2. **Computation**: k hash operations per token
3. **Partial Disambiguation**: Intersection may still have multiple candidates
4. **Order Dependency**: Assumes sequential order in text (not bag-of-words)

## Implementation Notes

### Hash Computation via Natural Concatenation

**Correct Approach**: Hash computed from natural token concatenation:

```python
# 1-token group
for token in tokens:
    hash_val = hash(token)  # Single token
    hllset_1.add(hash_val)

# 2-token group  
for i in range(len(tokens) - 1):
    token_pair = tokens[i] + tokens[i+1]  # Concatenate
    hash_val = hash(token_pair)
    hllset_2.add(hash_val)

# 3-token group
for i in range(len(tokens) - 2):
    token_triple = tokens[i] + tokens[i+1] + tokens[i+2]  # Concatenate
    hash_val = hash(token_triple)
    hllset_3.add(hash_val)
```

**Properties**:
- **No prefixes needed**: Natural concatenation creates different hashes
- **Fully content addressable**: 
  - Compute hash from tokens → (reg, zeros)
  - Look up in LUT → get token sequences
  - Tuple length reveals n-group automatically
- **Efficient**: Single hash computation per n-token
- **Collision-resistant**: Different n-tokens rarely collide (up to hash function limits)

**Lookup Process**:
```python
# Query by substring - returns results from ALL n-groups
query = "token"
results = lut.search_substring(query)

# Results contain entries from all LUT groups (1-token, 2-token, 3-token...)
for lut_record in results:
    for token_seq in lut_record.tokens:
        n_group = len(token_seq)  # Array length = n-group!
        # token_seq = ('token',) → len=1 → 1-token group
        # token_seq = ('my', 'token') → len=2 → 2-token group  
        # token_seq = ('my', 'special', 'token') → len=3 → 3-token group
        print(f"Found in {n_group}-token group: {token_seq}")

# You get all matches across all n-groups in one query,
# then split by checking len(tokens_array)
```

### LUT Construction

Current implementation uses Python's `hash()`:

- Fast but not cryptographically secure
- Sufficient for disambiguation purposes
- Could be replaced with MurmurHash3 for consistency with HLLSet

### Precision Bits Handling

Gracefully handles both real and mock HLLSets:

```python
if hasattr(hllset, 'precision_bits'):
    p_bits = hllset.precision_bits
else:
    p_bits = 10  # Default
```

## Future Enhancements

### 1. Probabilistic Disambiguation

Instead of intersection, use **Bayesian inference**:

```python
P(token | bit) = P(bit | token) * P(token) / P(bit)
```

Combine evidence from all n-groups.

### 2. Dynamic N-Token Groups

Automatically determine optimal n-groups based on:

- Token sequence length
- Desired disambiguation accuracy
- Space constraints

### 3. Hierarchical Merging

Merge n-token groups hierarchically:

```text
1-tokens → 2-tokens → 4-tokens → 8-tokens → ...
```

Creates logarithmic structure instead of linear.

### 4. GPU Acceleration

- Hash computation: highly parallelizable
- LUT construction: map-reduce pattern
- Intersection: parallel set operations

## References

- [MANIFOLD_OS_DRIVERS.md](MANIFOLD_OS_DRIVERS.md) - Driver architecture
- [core/manifold_os.py](core/manifold_os.py) - Implementation
- [test_n_token_ingest.py](test_n_token_ingest.py) - Test suite
- HyperLogLog paper: Flajolet et al.
- Hash functions: MurmurHash3

## Example Session

```python
from core.manifold_os import ManifoldOS

# Create OS
os = ManifoldOS()

# Ingest data
text = "the quick brown fox jumps over the lazy dog"
representation = os.ingest(text)

# Examine structure
print(f"Original tokens: {representation.original_tokens}")
print(f"N-token groups: {list(representation.n_token_groups.keys())}")

# Check HLLSets
for n, hllset in representation.hllsets.items():
    print(f"{n}-token HLLSet: cardinality={hllset.cardinality()}")

# Disambiguate a specific bit
reg, zeros = 42, 3
candidates = representation.disambiguate_tokens(reg, zeros)
print(f"Candidates for (reg={reg}, zeros={zeros}): {candidates}")

# Get implicit order
order = representation.get_implicit_order()
print(f"Implicit order (first 10): {order[:10]}")
```

Output:

```text
Original tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
N-token groups: [1, 2, 3]
1-token HLLSet: cardinality=8.0
2-token HLLSet: cardinality=8.0
3-token HLLSet: cardinality=7.0
Candidates for (reg=42, zeros=3): {'quick', 'brown'}
Implicit order (first 10): [('the',), ('quick',), ('brown',), ('fox',), ...]
```
