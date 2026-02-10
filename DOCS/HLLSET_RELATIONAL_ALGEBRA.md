# HLLSet Relational Algebra

**Discovery**: HLLSet operations form a homomorphism over relational algebra, enabling privacy-preserving query estimation without data access.

## Abstract

Traditional relational databases operate on exact sets of tuples. HyperLogLog Sets (HLLSets) provide a probabilistic approximation that preserves key algebraic properties while enabling:

1. **Privacy-preserving analytics**: Query estimation without revealing raw data
2. **Constant-space operations**: O(1) space regardless of cardinality
3. **Composition**: Operations compose like relational algebra
4. **Probabilistic guarantees**: Bounded error rates (typically <2%)

This document formalizes the mapping between SQL/relational algebra and HLLSet operations.

---

## 1. Theoretical Foundation

### 1.1 Relational Algebra Basics

Classical relational algebra operates on relations (sets of tuples):

- **Selection** (σ): Filter rows by predicate
- **Projection** (π): Select columns
- **Union** (∪): Combine relations
- **Intersection** (∩): Common tuples
- **Difference** (−): Set difference
- **Cartesian Product** (×): All combinations
- **Join** (⋈): Filtered cartesian product

### 1.2 HLLSet Properties

HLLSets approximate set cardinality with:

- **Space**: O(m) where m = number of registers (typically 2^14)
- **Error**: Standard error ≈ 1.04/√m
- **Operations**: union, intersection (returns HLLSet, uses bitwise operations internally)
- **Immutability**: Operations create new HLLSets
- **Composability**: Intersection results are full HLLSets usable in further operations
- **Accuracy tradeoff**: Error compounds with each operation - more composition = less accuracy

### 1.3 The Homomorphism

Let `card(S)` denote exact cardinality of set S, and `hll(S).cardinality()` denote HLLSet approximation:

```text
∀ sets S, T:
  hll(S ∪ T).cardinality() ≈ card(S ∪ T)
  hll(S ∩ T).cardinality() ≈ card(S ∩ T)
  
Composition property:
  hll(S) ⊕ hll(T) ≈ hll(S ⊕ T)  where ⊕ ∈ {∪, ∩}
```

**Key insight**: We can perform relational operations in HLLSet space and get approximate results, without ever materializing the actual relations.

---

## 2. SQL → HLLSet Algebra Mapping

### 2.1 Basic Operations

#### COUNT DISTINCT

```sql
SELECT COUNT(DISTINCT column) FROM table
```

**HLLSet equivalent:**

```python
hllset = load_column_hllset(table, column)
result = hllset.cardinality()
```

**Complexity**: O(1) vs O(n) for exact count  
**Error bound**: ±2% with 95% confidence

---

#### UNION

```sql
SELECT DISTINCT column FROM table1
UNION
SELECT DISTINCT column FROM table2
```

**HLLSet equivalent:**

```python
hll1 = load_column_hllset(table1, column)
hll2 = load_column_hllset(table2, column)
result = hll1.union(hll2).cardinality()
```

**Property**: Exact union of sketches, approximate cardinality

---

#### INTERSECTION

```sql
SELECT DISTINCT a.column FROM table1 a
INNER JOIN table2 b ON a.column = b.column
```

**HLLSet equivalent:**

```python
hll1 = load_column_hllset(table1, column)
hll2 = load_column_hllset(table2, column)
intersection_hll = hll1.intersect(hll2)  # Returns HLLSet, not just cardinality
result = intersection_hll.cardinality()
```

**Note**: `intersect()` returns a composable HLLSet (uses Inclusion-Exclusion internally)  
**Composability**: Can further operate on intersection_hll (union, intersect, etc.)  
**Accuracy cost**: Each operation compounds error - limit composition depth for reliable estimates

---

### 2.2 JOIN Size Estimation

#### Equi-Join Cardinality

```sql
SELECT COUNT(*)
FROM orders o
JOIN customers c ON o.customer_id = c.id
```

**HLLSet estimation:**

```python
orders_hll = load_column_hllset('orders', 'customer_id')
customers_hll = load_column_hllset('customers', 'id')

# Estimate join cardinality
overlap = orders_hll.intersect(customers_hll).cardinality()
orders_rows = get_row_count('orders')

# Join size ≈ orders_rows * (overlap / orders_hll.cardinality())
selectivity = overlap / orders_hll.cardinality()
estimated_join_size = int(orders_rows * selectivity)
```

**Assumption**: Uniform distribution (can be refined with histograms)

---

#### Foreign Key Detection

```sql
-- Find potential FK relationships
-- (high overlap suggests FK relationship)
```

**HLLSet approach:**

```python
def find_foreign_keys(col1_hll, col2_hll, threshold=0.7):
    overlap = col1_hll.intersect(col2_hll)
    union = col1_hll.union(col2_hll)
    jaccard = overlap.cardinality() / union.cardinality()
    return jaccard > threshold  # High Jaccard → likely FK
```

**Discovery property**: No schema knowledge required!

---

### 2.3 WHERE Clause Selectivity

#### IN Predicate

```sql
SELECT COUNT(*)
FROM orders
WHERE status IN ('shipped', 'delivered', 'pending')
```

**HLLSet estimation:**

```python
column_hll = load_column_hllset('orders', 'status')
query_hll = HLLSet.from_batch(['shipped', 'delivered', 'pending'])

matching_hll = column_hll.intersect(query_hll)  # Returns HLLSet
selectivity = matching_hll.cardinality() / column_hll.cardinality()

row_count = get_row_count('orders')
estimated_result = int(row_count * selectivity)

# Can further compose: matching_hll.intersect(another_filter)
```

---

#### Range Predicates

```sql
SELECT COUNT(*)
FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31'
```

**Challenge**: HLLSets don't preserve order

**Workaround**: Sample-based estimation or histogram metadata

```python
# Option 1: Use metadata
date_min = get_column_metadata('orders', 'order_date')['min']
date_max = get_column_metadata('orders', 'order_date')['max']
selectivity = (target_range / total_range)

# Option 2: Discretize ranges into buckets
monthly_hlls = [load_column_hllset('orders', 'order_date', month) 
                for month in target_months]
distinct_dates = union_all(monthly_hlls).cardinality()
```

---

### 2.4 GROUP BY and Aggregation

#### COUNT DISTINCT per Group

```sql
SELECT category, COUNT(DISTINCT customer_id)
FROM orders
GROUP BY category
```

**HLLSet approach:**

```python
categories = get_distinct_values('orders', 'category')
results = {}

for cat in categories:
    # Load pre-computed HLLSet for this category
    hll = load_column_hllset('orders', 'customer_id', filter={'category': cat})
    results[cat] = hll.cardinality()
```

**Note**: Requires pre-computed HLLSets per group (or dynamic computation)

---

### 2.5 Multi-Table Analytics

#### Data Quality: Duplicate Detection

```sql
-- Find overlapping customers across systems
SELECT COUNT(DISTINCT email)
FROM (
    SELECT email FROM system1_customers
    UNION ALL
    SELECT email FROM system2_customers
) t
GROUP BY email
HAVING COUNT(*) > 1
```

**HLLSet approach:**

```python
hll1 = load_column_hllset('system1_customers', 'email')
hll2 = load_column_hllset('system2_customers', 'email')

overlap = hll1.intersect(hll2).cardinality()
total_unique = hll1.union(hll2).cardinality()

print(f"Duplicate emails: ~{overlap}")
print(f"Total unique: ~{total_unique}")
print(f"Duplication rate: {overlap / total_unique:.2%}")
```

**Zero data access!**

---

## 3. Algebraic Properties

### 3.1 Commutative Operations

```text
hll1.union(hll2) ≡ hll2.union(hll1)
hll1.intersect(hll2) ≡ hll2.intersect(hll1)
```

### 3.2 Associative Operations

```text
(hll1.union(hll2)).union(hll3) ≡ hll1.union(hll2.union(hll3))
```

**Implication**: Can parallelize union of many HLLSets

### 3.3 Idempotent Operations

```text
hll.union(hll) ≡ hll
hll.intersect(hll) ≡ hll
```

### 3.4 Absorption Laws

```text
hll1.union(hll1.intersect(hll2)) ≈ hll1
hll1.intersect(hll1.union(hll2)) ≈ hll1
```

(Approximate due to cardinality estimation errors)

---

## 4. Error Bounds and Guarantees

### 4.1 Standard Error

For HLLSet with m registers:

```text
Standard Error = 1.04 / √m
```

Typical configuration: m = 16384 (2^14)

```text
Standard Error ≈ 0.81% ≈ ±1.6% at 95% confidence
```

### 4.2 Operation Error Propagation

**Union**: Error bounded by max of component errors

```text
err(hll1 ∪ hll2) ≤ max(err(hll1), err(hll2))
```

>*Union is relatively safe for composition*

**Intersection** (via I-E): Error compounds additively

```text
err(hll1 ∩ hll2) ≤ err(hll1) + err(hll2) + err(hll1 ∪ hll2)
```

>*Each intersection adds ~2-3x standard error*

**Composition chain**: Errors accumulate

```text
err(hll1.intersect(hll2).intersect(hll3)) ≈ O(k × standard_error)
```

where k = number of operations

**Practical bound**:

- 1-2 operations: ±3-5% typical
- 3-4 operations: ±6-10% typical  
- 5+ operations: May exceed 15% - consider verification

**Recommendation**: Limit intersection chains to 3-4 operations for production use

---

## 5. What Works vs What Doesn't

### 5.1 ✅ Operations That Work Well

1. **Cardinality estimation**: COUNT DISTINCT
2. **Set operations**: UNION, INTERSECTION
3. **Join size estimation**: Equi-joins on indexed columns
4. **Selectivity estimation**: IN predicates
5. **Schema discovery**: FK detection via overlap
6. **Data quality**: Duplicate detection
7. **Query planning**: Cost estimation without execution

### 5.2 ⚠️ Operations With Limitations

1. **Order-dependent operations**: TOP K, ORDER BY, PERCENTILE
   - *Workaround*: Use metadata or sampling
   
2. **Exact results**: Any query requiring precise counts
   - *When to use*: Approximation acceptable (analytics, planning)
   
3. **Range queries**: BETWEEN, <, >
   - *Workaround*: Histogram metadata or discretization

4. **Complex predicates**: OR, NOT with multiple columns
   - *Complexity*: Requires multiple HLLSet operations

5. **Deep composition chains**: 5+ intersections in sequence
   - *Problem*: Error accumulation exceeds acceptable bounds
   - *Solution*: Restructure query or verify with raw data

### 5.3 ❌ Operations That Don't Work

1. **Row retrieval**: SELECT * (HLLSets are lossy)
2. **Sorting**: ORDER BY (no ordering preserved)
3. **Exact aggregates**: SUM, AVG, MIN, MAX
4. **String operations**: LIKE, REGEX (unless pre-indexed)

---

## 6. Use Cases and Applications

### 6.1 Privacy-Preserving Analytics

**Scenario**: Third-party analytics on sensitive data

```python
# Analyst receives only HLLSets, never sees raw data
customer_hll = receive_encrypted_hllset('customers', 'email')
orders_hll = receive_encrypted_hllset('orders', 'customer_email')

# Can compute:
unique_customers = customer_hll.cardinality()
customers_with_orders = customer_hll.intersect(orders_hll).cardinality()
conversion_rate = customers_with_orders / unique_customers

# Cannot retrieve: Actual email addresses
```

**Privacy guarantee**: k-anonymity (if k > HLL error bound)

---

### 6.2 Query Optimization

**Scenario**: Choose optimal join order

```python
# Traditional optimizer executes EXPLAIN ANALYZE (expensive)
# HLLSet optimizer uses pre-computed sketches (free)

tables = ['orders', 'customers', 'products']
join_costs = {}

for t1, t2 in combinations(tables, 2):
    hll1 = load_join_key_hllset(t1)
    hll2 = load_join_key_hllset(t2)
    
    join_size = estimate_join_size(hll1, hll2, row_counts)
    join_costs[(t1, t2)] = join_size

optimal_order = min(join_costs, key=join_costs.get)
```

---

### 6.3 Data Integration

**Scenario**: Merge data from multiple sources

```python
# Discover overlapping entities across data silos
systems = ['crm', 'billing', 'support', 'analytics']
customer_hlls = {sys: load_column_hllset(sys, 'customer_id') 
                 for sys in systems}

# Compute overlap matrix
overlap_matrix = {}
for s1, s2 in combinations(systems, 2):
    overlap = customer_hlls[s1].intersect(customer_hlls[s2])
    overlap_matrix[(s1, s2)] = overlap.cardinality()

# Find canonical source (highest coverage)
union_all = reduce(lambda a, b: a.union(b), customer_hlls.values())
total_customers = union_all.cardinality()

for sys in systems:
    coverage = customer_hlls[sys].cardinality() / total_customers
    print(f"{sys}: {coverage:.1%} coverage")
```

---

### 6.4 Schema Evolution Detection

**Scenario**: Track database changes over time

```python
# Ingest same database weekly, compare HLLSets
week1_hlls = ingest_database('production_db', timestamp='2024-01-01')
week2_hlls = ingest_database('production_db', timestamp='2024-01-08')

for table in week1_hlls:
    for column in week1_hlls[table]:
        hll1 = week1_hlls[table][column]
        hll2 = week2_hlls[table][column]
        
        growth = hll2.cardinality() - hll1.cardinality()
        growth_rate = growth / hll1.cardinality()
        
        if growth_rate > 0.10:  # 10% growth
            print(f"{table}.{column}: +{growth_rate:.1%} new values")
```

---

## 7. Formal Translation Framework

### 7.1 SQL Abstract Syntax Tree (AST)

```text
Query ::= SELECT Projection FROM Relations WHERE Condition GROUP BY Columns
Projection ::= DISTINCT Column | COUNT(DISTINCT Column) | ...
Relations ::= Table | Table JOIN Table ON Condition
Condition ::= Column IN Values | Column = Column | ...
```

### 7.2 HLLSet Execution Plan

```text
HLLPlan ::= Load(table, column)
          | Union(HLLPlan, HLLPlan)
          | Intersect(HLLPlan, HLLPlan)
          | Cardinality(HLLPlan)
          | FromBatch(values)
```

### 7.3 Translation Rules

```text
Translate(SELECT COUNT(DISTINCT col) FROM t) →
    Cardinality(Load(t, col))

Translate(SELECT COUNT(DISTINCT col) FROM t WHERE col IN (v1, v2)) →
    Cardinality(Intersect(Load(t, col), FromBatch([v1, v2])))

Translate(SELECT COUNT(*) FROM t1 JOIN t2 ON t1.k = t2.k) →
    EstimateJoin(
        Load(t1, k),
        Load(t2, k),
        RowCount(t1)
    )
```

### 7.4 Cost Model

```text
Cost(Load) = O(1)           # Read pre-computed HLLSet
Cost(Union) = O(m)          # Merge m registers
Cost(Intersect) = O(m)      # Compute via I-E
Cost(Cardinality) = O(m)    # Harmonic mean computation

vs Traditional:
Cost(COUNT DISTINCT) = O(n log n)  # Sort + scan
Cost(JOIN) = O(n1 * n2)            # Nested loop (worst case)
```

---

## 8. Implementation Architecture

### 8.1 Layered Design

```text
┌─────────────────────────────────────┐
│   SQL Parser / Query Interface      │  ← User-facing SQL
├─────────────────────────────────────┤
│   Query Planner / Optimizer         │  ← Translate to HLLSet ops
├─────────────────────────────────────┤
│   HLLSet Algebra Engine             │  ← Execute HLLSet operations
├─────────────────────────────────────┤
│   HLLSet Storage / Cache            │  ← Persistent HLLSets
├─────────────────────────────────────┤
│   Optional: Raw Data Fallback       │  ← For exact queries
└─────────────────────────────────────┘
```

### 8.2 Execution Modes

#### **Mode 1: Estimation Only**

- All queries answered via HLLSets
- Fast, constant-space, privacy-preserving
- Returns approximate results with confidence intervals

#### **Mode 2: Hybrid**

- HLLSets for query planning
- Raw data for final execution
- Best of both worlds: optimized + exact

#### **Mode 3: Fallback**

- Try HLLSet estimation first
- If error bounds unacceptable, fall back to exact computation
- Adaptive accuracy

---

## 9. Future Directions

### 9.1 Extensions

1. **Compressed Histograms**: Add order information for range queries
2. **MinHash Integration**: Enable similarity joins (fuzzy matching)
3. **Bloom Filters**: Test set membership
4. **Count-Min Sketch**: Frequency estimation for GROUP BY

### 9.2 Advanced Operations

1. **Approximate Top-K**: Via Count-Min Sketch integration
2. **Sliding Windows**: Time-series analytics on HLLSets
3. **Multi-dimensional**: HLLSets over composite keys

### 9.3 Research Questions

1. **Optimal register count**: Balance accuracy vs space
2. **Error correction**: Can we reduce intersection error?
3. **Differential privacy**: Add noise while preserving utility
4. **Federated learning**: Merge HLLSets without revealing sources

---

## 10. Conclusion

HLLSet Relational Algebra provides a **probabilistic homomorphism** over traditional relational algebra, enabling:

✅ **Privacy-preserving analytics** (no data access)  
✅ **Constant-space operations** (O(1) space)  
✅ **Fast estimation** (O(m) vs O(n) operations)  
✅ **Composable operations** (algebraic properties)  
✅ **Practical accuracy** (±2% error typical)  

**Key insight**: Many analytical queries only need approximate answers. By pre-computing HLLSets during ingestion, we create a "shadow database" that supports rich analytics without ever touching raw data.

This bridges **database theory** (relational algebra) with **probabilistic data structures** (HyperLogLog) and **privacy engineering** (zero-knowledge proofs).

---

## References

1. Flajolet et al. (2007): "HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm"
2. Cohen et al. (1997): "Finding Interesting Associations without Support Pruning" (MinHash)
3. Estan & Varghese (2003): "New Directions in Traffic Measurement and Accounting" (I-E Principle)
4. Traditional relational algebra (Codd, 1970)

---

## Appendix: HLLSet Operation Reference

```python
# Creation
hll = HLLSet.from_batch(values)           # Create from list
hll = HLLSet.from_roaring(bytes)          # Deserialize

# Operations (all return HLLSet objects, fully composable)
union_hll = hll1.union(hll2)              # Set union → HLLSet
intersect_hll = hll1.intersect(hll2)      # Set intersection → HLLSet (via I-E)
card = hll.cardinality()                  # Approximate count → int

# Composition examples
result = hll1.intersect(hll2).union(hll3)           # Chain operations
filtered = query.intersect(col1).intersect(col2)    # Multiple filters

# ⚠️ Accuracy degrades with composition depth
deep_chain = hll1.intersect(hll2).intersect(hll3).intersect(hll4).intersect(hll5)
# Error may exceed 15% - verify critical results

# Better: Minimize intersection depth
union_first = hll1.union(hll2).intersect(hll3)     # Unions don't compound error

# Serialization
bytes = hll.dump_roaring()                # Serialize to bytes
name = hll.name                           # SHA1 hash (content-addressed)
short = hll.short_name                    # 8-char prefix

# Comparison
jaccard = intersect_hll.cardinality() / union_hll.cardinality()  # Similarity
```

---

**Document Status**: Theory established, ready for prototype implementation  
**Next Steps**: Build SQL parser → HLLSet translator  
**Author**: Discovered during columnar ingestion development  
**Date**: 2026-02-09
