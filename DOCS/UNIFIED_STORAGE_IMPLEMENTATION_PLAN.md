# Unified Storage Extension Implementation Plan

## Current Progress

✅ **Phase 1 Complete**: Roaring Bitmap Compression (2026-02-08)
- Added `get_registers_roaring()` / `set_registers_roaring()` to Cython backend
- Added `dump_roaring()` / `from_roaring()` / `get_compression_stats()` to HLLSet
- Tested: 6-50x compression ratio on typical HLLSets
- Ready for DuckDB storage integration

✅ **Phase 2 Complete**: UnifiedStorageExtension Implementation (2026-02-08)
- Created `core/extensions/unified_storage.py` with full schema
- Implemented all 7 tables (perceptrons, lattices, nodes, edges, hllsets, entanglements, mappings)
- Added high-level methods: store_am_lattice(), store_w_lattice()
- Added query methods: get_lattice_nodes(), get_lattice_edges(), get_storage_stats()
- All tests passing (7/7 test cases in test_unified_storage.py)
- Created comprehensive demo notebook (demo_unified_storage.ipynb)

## Next: Phase 3 - Metadata Perceptron & Integration

### Steps

1. **Create new `UnifiedStorageExtension`** (in `core/extensions/unified_storage.py`)
   - Implements unified lattice schema from DUCKDB_UNIFIED_SCHEMA.md
   - Uses Roaring compression for HLLSets
   - Supports multi-perceptron storage

2. **Schema initialization**
   - Create tables: perceptrons, lattices, lattice_nodes, lattice_edges, hllsets, entanglements, entanglement_mappings
   - Add indexes for common query patterns

3. **Core operations**
   - `register_perceptron()` - Add new perceptron
   - `store_lattice()` - Store AM or W lattice
   - `store_hllset()` - Store with Roaring compression
   - `store_entanglement()` - Store cross-perceptron mappings
   - Query methods for retrieval

4. **Test with existing code**
   - Refactor demo notebooks to use unified schema
   - Update test cases

## Implementation Notes

### UnifiedStorageExtension Interface

```python
class UnifiedStorageExtension(StorageExtension):
    def register_perceptron(
        self,
        perceptron_id: str,
        perceptron_type: str,
        hash_function: str,
        hash_seed: int,
        config: dict
    ) -> None:
        """Register a new perceptron."""
        
    def store_lattice_am(
        self,
        perceptron_id: str,
        am_cells: List[Tuple[int, int, int]],  # (row_idx, col_idx, frequency)
        token_lut: Dict[Tuple[int, int], str]   # (reg, zeros) -> token
    ) -> str:  # Returns lattice_id
        """Store AM lattice with nodes (tokens) and edges (transitions)."""
        
    def store_lattice_w(
        self,
        perceptron_id: str,
        basic_hllsets: List[Tuple[int, str, HLLSet]],  # (idx, type, hllset)
        morphisms: List[Tuple[int, int, float, dict]]  # (src, tgt, weight, props)
    ) -> str:  # Returns lattice_id
        """Store W lattice with nodes (HLLSets) and edges (morphisms)."""
        
    def store_hllset(
        self,
        hllset: HLLSet
    ) -> None:
        """Store HLLSet with Roaring compression."""
        
    def retrieve_lattice(
        self,
        lattice_id: str
    ) -> dict:
        """Retrieve full lattice structure."""
        
    def query_entanglement(
        self,
        source_perceptron: str,
        target_perceptron: str,
        min_similarity: float = 0.7
    ) -> List[dict]:
        """Query cross-perceptron entanglements."""
```

### Storage Pattern

**AM Lattice Storage**:
1. Create lattice record (type='AM')
2. Store each token as lattice_node with properties JSON:
   ```json
   {
     "node_type": "am_token",
     "properties": {"token": "hello", "register": 42, "zeros": 3, "position": "row"}
   }
   ```
3. Store each transition as lattice_edge:
   ```json
   {
     "edge_type": "am_transition",
     "weight": 15.0,
     "properties": {"frequency": 15}
   }
   ```

**W Lattice Storage**:
1. Create lattice record (type='W')
2. Store each HLLSet in `hllsets` table with Roaring compression
3. Store each basic HLLSet as lattice_node:
   ```json
   {
     "node_type": "w_hllset",
     "properties": {"hllset_hash": "abc...", "p_bits": 14, "position": "row"}
   }
   ```
4. Store each morphism as lattice_edge:
   ```json
   {
     "edge_type": "w_morphism",
     "weight": 0.85,
     "properties": {"bss_tau": 0.75, "bss_rho": 0.20}
   }
   ```

## Refactoring Plan

### Test Files to Update

1. `tests/test_duckdb_storage.py` → Update for unified schema
2. `tests/test_manifold_drivers.py` → Update storage calls
3. `examples/demo_metadata_bridge.py` → Create new with metadata lattice

### Notebooks to Update

1. `demo_duckdb_metadata.ipynb` → Update for unified schema
2. Create new: `demo_unified_lattice_storage.ipynb`
3. Create new: `demo_metadata_perceptron.ipynb`

## Timeline

- [x] Phase 1: Roaring compression (DONE)
- [ ] Phase 2a: Create UnifiedStorageExtension class
- [ ] Phase 2b: Implement core storage methods
- [ ] Phase 2c: Add retrieval/query methods
- [ ] Phase 3: Refactor tests
- [ ] Phase 4: Update notebooks
- [ ] Phase 5: Create metadata perceptron example

## Next Immediate Step

Create `core/extensions/unified_storage.py` with UnifiedStorageExtension class.
