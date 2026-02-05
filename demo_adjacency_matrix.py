"""
Demonstration of Adjacency Matrix construction during ingestion.

This demo shows:
1. How tokens are processed through sliding windows
2. How (reg, zeros) identifiers are computed
3. How AM cells track transition frequencies
4. How row/column HLLSets are maintained
5. How batches update the same AM
"""

from core.manifold_os import ManifoldOS, IngestDriver, Kernel, TokenizationConfig


def demo_basic_am():
    """Demo basic AM construction with simple data."""
    print("="*70)
    print("Demo 1: Basic AM Construction")
    print("="*70)
    
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    driver = IngestDriver(driver_id="demo-driver", config=config)
    mos.register_driver(driver)
    driver.wake()
    
    # Process data
    data = "the quick brown fox"
    print(f"\nInput data: '{data}'")
    print(f"Tokens: {data.split()}")
    
    driver.process(data, kernel)
    am = driver.get_adjacency_matrix()
    
    # Show results
    print(f"\nAM Statistics:")
    print(f"  Total cells: {len(am.cells)}")
    print(f"  Row HLLSets: {len(am.row_hllsets)}")
    print(f"  Column HLLSets: {len(am.col_hllsets)}")
    
    # Show some cells
    print(f"\nSample cells (showing first 5):")
    for i, ((row_id, col_id), cell) in enumerate(am.cells.items()):
        if i >= 5:
            break
        print(f"  {row_id} → {col_id}: frequency={cell.frequency}")
    
    # Show START/END tokens
    START_ID = (-1, 0)
    END_ID = (-2, 0)
    start_cells = [(r, c) for (r, c) in am.cells.keys() if r == START_ID]
    end_cells = [(r, c) for (r, c) in am.cells.keys() if c == END_ID]
    print(f"\nBoundary markers:")
    print(f"  START token cells: {len(start_cells)}")
    print(f"  END token cells: {len(end_cells)}")
    
    return driver


def demo_repeated_patterns():
    """Demo frequency counting with repeated patterns."""
    print("\n" + "="*70)
    print("Demo 2: Frequency Counting with Repeated Patterns")
    print("="*70)
    
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    driver = IngestDriver(driver_id="demo-driver-2", config=config)
    mos.register_driver(driver)
    driver.wake()
    
    # Process data with repeated pattern
    data = "a b c a b c a b c"
    print(f"\nInput data: '{data}'")
    print(f"Pattern: 'a b c' repeated 3 times")
    
    driver.process(data, kernel)
    am = driver.get_adjacency_matrix()
    
    # Calculate statistics
    total_freq = sum(cell.frequency for cell in am.cells.values())
    max_freq = max(cell.frequency for cell in am.cells.values())
    avg_freq = total_freq / len(am.cells) if am.cells else 0
    
    print(f"\nFrequency Statistics:")
    print(f"  Total cells: {len(am.cells)}")
    print(f"  Total frequency: {total_freq}")
    print(f"  Max frequency: {max_freq}")
    print(f"  Average frequency: {avg_freq:.2f}")
    
    # Show cells sorted by frequency
    sorted_cells = sorted(am.cells.items(), key=lambda x: x[1].frequency, reverse=True)
    print(f"\nTop 5 cells by frequency:")
    for i, ((row_id, col_id), cell) in enumerate(sorted_cells[:5]):
        print(f"  {row_id} → {col_id}: frequency={cell.frequency}")
    
    return driver


def demo_batch_processing():
    """Demo how multiple batches update the same AM."""
    print("\n" + "="*70)
    print("Demo 3: Batch Processing with Cumulative Updates")
    print("="*70)
    
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    driver = IngestDriver(driver_id="demo-driver-3", config=config)
    mos.register_driver(driver)
    driver.wake()
    
    # Process first batch
    data1 = "the quick brown fox"
    print(f"\nBatch 1: '{data1}'")
    driver.process(data1, kernel)
    am = driver.get_adjacency_matrix()
    cells_after_batch1 = len(am.cells)
    freq_after_batch1 = sum(cell.frequency for cell in am.cells.values())
    print(f"  Cells: {cells_after_batch1}, Total frequency: {freq_after_batch1}")
    
    # Process second batch
    data2 = "the lazy dog"
    print(f"\nBatch 2: '{data2}'")
    driver.process(data2, kernel)
    cells_after_batch2 = len(am.cells)
    freq_after_batch2 = sum(cell.frequency for cell in am.cells.values())
    print(f"  Cells: {cells_after_batch2}, Total frequency: {freq_after_batch2}")
    
    # Process third batch (with repeated word "the")
    data3 = "the fast cat"
    print(f"\nBatch 3: '{data3}'")
    driver.process(data3, kernel)
    cells_after_batch3 = len(am.cells)
    freq_after_batch3 = sum(cell.frequency for cell in am.cells.values())
    print(f"  Cells: {cells_after_batch3}, Total frequency: {freq_after_batch3}")
    
    print(f"\nGrowth Analysis:")
    print(f"  Batch 1→2: +{cells_after_batch2 - cells_after_batch1} cells, +{freq_after_batch2 - freq_after_batch1} frequency")
    print(f"  Batch 2→3: +{cells_after_batch3 - cells_after_batch2} cells, +{freq_after_batch3 - freq_after_batch2} frequency")
    print(f"  Total: {cells_after_batch3} cells, {freq_after_batch3} total transitions")
    
    return driver


def demo_identifier_distribution():
    """Demo the distribution of (reg, zeros) identifiers."""
    print("\n" + "="*70)
    print("Demo 4: (reg, zeros) Identifier Distribution")
    print("="*70)
    
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    driver = IngestDriver(driver_id="demo-driver-4", config=config)
    mos.register_driver(driver)
    driver.wake()
    
    # Process longer data
    data = """
    The quick brown fox jumps over the lazy dog.
    The lazy dog sleeps under the old oak tree.
    The brown fox runs through the green forest.
    """
    print(f"\nInput: {len(data.split())} tokens from 3 sentences")
    
    driver.process(data, kernel)
    am = driver.get_adjacency_matrix()
    
    # Analyze identifier distribution
    all_ids = set()
    reg_values = set()
    zeros_values = set()
    
    for (row_id, col_id) in am.cells.keys():
        for id_tuple in [row_id, col_id]:
            if id_tuple not in [(-1, 0), (-2, 0)]:  # Skip START/END
                all_ids.add(id_tuple)
                reg_values.add(id_tuple[0])
                zeros_values.add(id_tuple[1])
    
    print(f"\nIdentifier Statistics:")
    print(f"  Unique identifiers: {len(all_ids)}")
    print(f"  Unique reg values: {len(reg_values)} (range: [{min(reg_values)}, {max(reg_values)}])")
    print(f"  Unique zeros values: {len(zeros_values)} (range: [{min(zeros_values)}, {max(zeros_values)}])")
    print(f"  Total cells: {len(am.cells)}")
    
    # Show HLLSet cardinalities
    row_cards = [hllset.cardinality() for hllset in am.row_hllsets.values()]
    col_cards = [hllset.cardinality() for hllset in am.col_hllsets.values()]
    
    print(f"\nHLLSet Statistics:")
    print(f"  Row HLLSets: {len(am.row_hllsets)}")
    print(f"  Column HLLSets: {len(am.col_hllsets)}")
    if row_cards:
        print(f"  Avg row cardinality: {sum(row_cards)/len(row_cards):.1f}")
    if col_cards:
        print(f"  Avg col cardinality: {sum(col_cards)/len(col_cards):.1f}")
    
    return driver


def demo_visualization():
    """Demo converting AM to dense array for visualization."""
    print("\n" + "="*70)
    print("Demo 5: AM Visualization (Dense Array Conversion)")
    print("="*70)
    
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    driver = IngestDriver(driver_id="demo-driver-5", config=config)
    mos.register_driver(driver)
    driver.wake()
    
    # Process small data for visualization
    data = "a b c d"
    print(f"\nInput data: '{data}'")
    
    driver.process(data, kernel)
    am = driver.get_adjacency_matrix()
    
    # Convert to dense
    dense = am.to_dense_array()
    
    print(f"\nDense Array Shape: {dense.shape}")
    print(f"Sparse cells: {len(am.cells)}")
    print(f"Dense size: {dense.shape[0] * dense.shape[1]} elements")
    print(f"Sparsity: {(1 - len(am.cells) / (dense.shape[0] * dense.shape[1])) * 100:.1f}%")
    
    print(f"\nDense Array (first 8x8 region):")
    import numpy as np
    np.set_printoptions(linewidth=100)
    print(dense[:min(8, dense.shape[0]), :min(8, dense.shape[1])])
    
    return driver


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADJACENCY MATRIX CONSTRUCTION DEMO")
    print("="*70)
    
    demo_basic_am()
    demo_repeated_patterns()
    demo_batch_processing()
    demo_identifier_distribution()
    demo_visualization()
    
    print("\n" + "="*70)
    print("All demos completed successfully!")
    print("="*70 + "\n")
