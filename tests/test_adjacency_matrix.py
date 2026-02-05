"""
Test Adjacency Matrix construction during ingestion.

Tests verify:
1. START/END token insertion
2. Sliding window processing (size 3, step 1)
3. (reg, zeros) identifier computation
4. Frequency counting in cells
5. Row/Column HLLSet updates
6. Batch processing with shared AM
"""

import pytest
import numpy as np
from core.manifold_os import (
    ManifoldOS, IngestDriver, Kernel,
    TokenizationConfig, IngestionAdjacencyMatrix
)


def test_start_end_token_insertion():
    """Test that START and END tokens are added to batch boundaries."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process simple data
    data = "hello world"
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # Check that START token appears as row ID
    START_ID = (-1, 0)
    found_start = False
    for cell_key in am.cells:
        if cell_key[0] == START_ID:
            found_start = True
            break
    
    assert found_start, "START token not found in AM rows"
    
    # Check that END token appears as column ID
    END_ID = (-2, 0)
    found_end = False
    for cell_key in am.cells:
        if cell_key[1] == END_ID:
            found_end = True
            break
    
    assert found_end, "END token not found in AM columns"
    
    print(f"✓ START/END tokens found. AM has {len(am.cells)} cells")


def test_sliding_window_processing():
    """Test that sliding window creates correct number of transitions."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process data with known tokens
    data = "a b c d"  # 4 tokens
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # With START and END:
    # ["START", "a", "b", "c", "d", "END"] = 6 tokens
    # Sliding window size 3, step 1 generates:
    # (START, a, b), (a, b, c), (b, c, d), (c, d, END) = 4 windows
    # Each window creates 3 AM updates (but may overlap)
    
    # We should have at least 4 cells (one per window minimum)
    assert len(am.cells) >= 4, f"Expected at least 4 cells, got {len(am.cells)}"
    
    print(f"✓ Sliding window processed. AM has {len(am.cells)} cells from 4 tokens")


def test_reg_zeros_identifiers():
    """Test that (reg, zeros) identifiers are computed correctly."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process data
    data = "hello world test"
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # Check that identifiers are tuples of (reg, zeros)
    # where reg is in range [0, 1024) for p_bits=10
    # and zeros is small (typically 0-10)
    
    for (row_id, col_id), cell in am.cells.items():
        # Check START/END special tokens
        if row_id == (-1, 0) or row_id == (-2, 0):
            continue
        if col_id == (-1, 0) or col_id == (-2, 0):
            continue
        
        # Check row_id
        assert isinstance(row_id, tuple) and len(row_id) == 2, f"Invalid row_id format: {row_id}"
        reg, zeros = row_id
        assert 0 <= reg < 1024, f"reg {reg} out of range [0, 1024)"
        assert 0 <= zeros < 64, f"zeros {zeros} out of reasonable range"
        
        # Check col_id
        assert isinstance(col_id, tuple) and len(col_id) == 2, f"Invalid col_id format: {col_id}"
        reg, zeros = col_id
        # Allow special token values
        if reg >= 0:
            assert reg < 1024, f"reg {reg} out of range [0, 1024)"
            assert 0 <= zeros < 64, f"zeros {zeros} out of reasonable range"
    
    print(f"✓ All {len(am.cells)} identifiers are valid (reg, zeros) tuples")


def test_frequency_counting():
    """Test that cell frequencies increment correctly."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process data with repeated patterns
    data = "a b c a b c"  # Should create repeated transitions
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # Check that frequencies are positive
    total_frequency = 0
    for cell in am.cells.values():
        assert cell.frequency > 0, f"Cell has non-positive frequency: {cell.frequency}"
        total_frequency += cell.frequency
    
    # With repeated pattern, total frequency should be substantial
    assert total_frequency > len(am.cells), "Expected higher total frequency with repeated patterns"
    
    print(f"✓ Total frequency: {total_frequency} across {len(am.cells)} cells")


def test_hllset_updates():
    """Test that row and column HLLSets are updated."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process data
    data = "hello world"
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # Check that HLLSets exist
    assert len(am.row_hllsets) > 0, "No row HLLSets created"
    assert len(am.col_hllsets) > 0, "No column HLLSets created"
    
    # Check that each HLLSet has non-zero cardinality
    for row_id, hllset in am.row_hllsets.items():
        cardinality = hllset.cardinality()
        assert cardinality > 0, f"Row {row_id} HLLSet has zero cardinality"
    
    for col_id, hllset in am.col_hllsets.items():
        cardinality = hllset.cardinality()
        assert cardinality > 0, f"Col {col_id} HLLSet has zero cardinality"
    
    print(f"✓ Row HLLSets: {len(am.row_hllsets)}, Col HLLSets: {len(am.col_hllsets)}")


def test_batch_processing():
    """Test that multiple batches update the same AM."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process first batch
    data1 = "hello world"
    driver.process(data1, kernel)
    
    am1_size = len(driver.get_adjacency_matrix().cells)
    
    # Process second batch
    data2 = "hello again"
    driver.process(data2, kernel)
    
    am2_size = len(driver.get_adjacency_matrix().cells)
    
    # AM should grow or stay same (if same transitions)
    assert am2_size >= am1_size, "AM shrank after second batch"
    
    print(f"✓ Batch 1: {am1_size} cells, Batch 2: {am2_size} cells")


def test_am_visualization():
    """Test that AM can be converted to dense array for visualization."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()  # Wake driver to IDLE state
    
    # Process data
    data = "a b c"
    driver.process(data, kernel)
    
    # Get AM
    am = driver.get_adjacency_matrix()
    
    # Convert to dense (should work without errors)
    dense_array = am.to_dense_array()
    
    # Check structure
    assert dense_array is not None
    assert isinstance(dense_array, np.ndarray), "Dense array should be numpy array"
    assert len(dense_array.shape) == 2, "Dense array should be 2D"
    assert dense_array.shape[0] > 0 and dense_array.shape[1] > 0, "Dense array should be non-empty"
    
    print(f"✓ Dense array: {dense_array.shape[0]}x{dense_array.shape[1]} matrix")


def test_commit_between_batches():
    """Test that commit is called between batches and driver state transitions correctly."""
    # Setup
    mos = ManifoldOS()
    kernel = Kernel(p_bits=10)
    
    config = TokenizationConfig(
        use_n_tokens=True,
        n_token_groups=[1, 2, 3],
        maintain_order=True
    )
    
    driver = IngestDriver(
        driver_id="test-driver",
        config=config
    )
    mos.register_driver(driver)
    driver.wake()
    
    # Process first batch
    from core.manifold_os import DriverState
    
    data1 = "hello world"
    driver.process(data1, kernel)
    
    # After process, driver should be in IDLE state (committed)
    assert driver.state == DriverState.IDLE, f"Expected IDLE state after commit, got {driver.state}"
    
    # AM should be preserved
    am1_size = len(driver.get_adjacency_matrix().cells)
    assert am1_size > 0, "AM should have cells after first batch"
    
    # Process second batch - should update same AM
    data2 = "hello again"
    driver.process(data2, kernel)
    
    # After second batch, still IDLE
    assert driver.state == DriverState.IDLE, f"Expected IDLE state after second batch, got {driver.state}"
    
    # AM should have accumulated
    am2_size = len(driver.get_adjacency_matrix().cells)
    assert am2_size >= am1_size, "AM should accumulate across batches"
    
    print(f"✓ Commit works: Batch 1→IDLE ({am1_size} cells), Batch 2→IDLE ({am2_size} cells)")


if __name__ == "__main__":
    print("Running Adjacency Matrix Tests\n")
    
    test_start_end_token_insertion()
    test_sliding_window_processing()
    test_reg_zeros_identifiers()
    test_frequency_counting()
    test_hllset_updates()
    test_batch_processing()
    test_am_visualization()
    test_commit_between_batches()
    
    print("\n✅ All Adjacency Matrix tests passed!")
