"""
Test Unified Storage Extension
"""

import pytest
from core.extensions.unified_storage import UnifiedStorageExtension, PerceptronConfig, LatticeNode, LatticeEdge
from core.hllset import HLLSet, compute_sha1


def test_unified_storage_initialization():
    """Test that schema initializes correctly."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Check that tables exist
    tables = storage.conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()
    
    table_names = [t[0] for t in tables]
    
    assert 'perceptrons' in table_names
    assert 'lattices' in table_names
    assert 'lattice_nodes' in table_names
    assert 'lattice_edges' in table_names
    assert 'hllsets' in table_names
    assert 'entanglements' in table_names
    assert 'entanglement_mappings' in table_names
    
    storage.close()


def test_perceptron_registration():
    """Test perceptron registration and retrieval."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Register a perceptron
    config = PerceptronConfig(
        perceptron_id="data_perceptron_1",
        perceptron_type="data",
        hash_function="sha1",
        hash_seed=42,
        config_dict={"n_tokens": 5, "p_bits": 14}
    )
    
    storage.register_perceptron(config, "Test data perceptron")
    
    # Retrieve it
    retrieved = storage.get_perceptron("data_perceptron_1")
    
    assert retrieved is not None
    assert retrieved['perceptron_id'] == "data_perceptron_1"
    assert retrieved['perceptron_type'] == "data"
    assert retrieved['hash_function'] == "sha1"
    assert retrieved['hash_seed'] == 42
    assert retrieved['config']['n_tokens'] == 5
    
    # List all perceptrons
    perceptrons = storage.list_perceptrons()
    assert len(perceptrons) == 1
    assert perceptrons[0]['perceptron_id'] == "data_perceptron_1"
    
    storage.close()


def test_hllset_storage_with_compression():
    """Test HLLSet storage with Roaring compression."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Create an HLLSet
    tokens = [f"token_{i}" for i in range(100)]
    hllset = HLLSet.from_batch(tokens, p_bits=14)
    
    # Store it
    storage.store_hllset(hllset)
    
    # Retrieve it
    retrieved = storage.retrieve_hllset(hllset.name)
    
    assert retrieved is not None
    assert retrieved.name == hllset.name
    assert abs(retrieved.cardinality() - hllset.cardinality()) < 1.0
    
    # Check compression stats
    stats = storage.get_hllset_stats(hllset.name)
    assert stats is not None
    assert stats['compression_ratio'] > 1.0  # Should be compressed
    assert stats['compressed_size'] < stats['original_size']
    
    storage.close()


def test_am_lattice_storage():
    """Test AM lattice storage."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Register perceptron
    config = PerceptronConfig(
        perceptron_id="am_test",
        perceptron_type="data",
        hash_function="sha1",
        hash_seed=42,
        config_dict={}
    )
    storage.register_perceptron(config)
    
    # Create simple AM data
    am_cells = [
        (0, 1, 10),  # token0 -> token1 (10 times)
        (1, 2, 5),   # token1 -> token2 (5 times)
        (0, 2, 3),   # token0 -> token2 (3 times)
    ]
    
    token_lut = {
        (0, 0): "hello",
        (1, 0): "world",
        (2, 0): "foo",
    }
    
    # Store AM lattice
    lattice_id = storage.store_am_lattice("am_test", am_cells, token_lut, dimension=3)
    
    # Verify lattice created
    info = storage.get_lattice_info(lattice_id)
    assert info is not None
    assert info['lattice_type'] == 'AM'
    assert info['perceptron_id'] == 'am_test'
    assert info['dimension'] == 3
    
    # Verify nodes
    nodes = storage.get_lattice_nodes(lattice_id, node_type="am_token")
    assert len(nodes) == 3
    
    # Check node properties
    tokens_found = [n['properties']['token'] for n in nodes]
    assert 'hello' in tokens_found
    assert 'world' in tokens_found
    assert 'foo' in tokens_found
    
    # Verify edges
    edges = storage.get_lattice_edges(lattice_id, edge_type="am_transition")
    assert len(edges) == 3
    
    # Check highest weight edge
    assert edges[0]['weight'] == 10.0  # Should be sorted by weight DESC
    
    storage.close()


def test_w_lattice_storage():
    """Test W lattice storage."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Register perceptron
    config = PerceptronConfig(
        perceptron_id="w_test",
        perceptron_type="data",
        hash_function="sha1",
        hash_seed=42,
        config_dict={}
    )
    storage.register_perceptron(config)
    
    # Create HLLSets
    hll1 = HLLSet.from_batch([f"token_{i}" for i in range(50)], p_bits=14)
    hll2 = HLLSet.from_batch([f"token_{i}" for i in range(25, 75)], p_bits=14)
    hll3 = HLLSet.from_batch([f"token_{i}" for i in range(50, 100)], p_bits=14)
    
    basic_hllsets = [
        (0, "start", hll1),
        (1, "middle", hll2),
        (2, "end", hll3),
    ]
    
    # Create morphisms
    morphisms = [
        (0, 1, 0.5, {"type": "subset"}),
        (1, 2, 0.6, {"type": "subset"}),
        (0, 2, 0.3, {"type": "overlap"}),
    ]
    
    # Store W lattice
    lattice_id = storage.store_w_lattice("w_test", basic_hllsets, morphisms, dimension=3)
    
    # Verify lattice
    info = storage.get_lattice_info(lattice_id)
    assert info is not None
    assert info['lattice_type'] == 'W'
    
    # Verify nodes (HLLSets)
    nodes = storage.get_lattice_nodes(lattice_id, node_type="w_hllset")
    assert len(nodes) == 3
    
    # Check that HLLSets were stored
    for node in nodes:
        hllset_hash = node['properties']['hllset_hash']
        retrieved_hll = storage.retrieve_hllset(hllset_hash)
        assert retrieved_hll is not None
    
    # Verify edges (morphisms)
    edges = storage.get_lattice_edges(lattice_id, edge_type="w_morphism")
    assert len(edges) == 3
    assert edges[0]['weight'] == 0.6  # Sorted by weight DESC
    
    # Filter by weight
    strong_edges = storage.get_lattice_edges(lattice_id, min_weight=0.5)
    assert len(strong_edges) == 2
    
    storage.close()


def test_storage_stats():
    """Test storage statistics."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Add some data
    config = PerceptronConfig(
        perceptron_id="test_perc",
        perceptron_type="data",
        hash_function="sha1",
        hash_seed=42,
        config_dict={}
    )
    storage.register_perceptron(config)
    
    # Add HLLSets
    for i in range(10):
        hllset = HLLSet.from_batch([f"token_{j}" for j in range(i*10, (i+1)*10)], p_bits=14)
        storage.store_hllset(hllset)
    
    # Get stats
    stats = storage.get_storage_stats()
    
    assert stats['perceptrons'] == 1
    assert stats['hllsets'] == 10
    assert stats['hllset_compression']['avg_compression_ratio'] > 1.0
    assert stats['hllset_compression']['total_compressed'] < stats['hllset_compression']['total_original']
    
    storage.close()


def test_multi_perceptron_storage():
    """Test storing lattices from multiple perceptrons."""
    storage = UnifiedStorageExtension(":memory:")
    
    # Register two perceptrons
    config1 = PerceptronConfig(
        perceptron_id="data_perc",
        perceptron_type="data",
        hash_function="sha1",
        hash_seed=42,
        config_dict={}
    )
    config2 = PerceptronConfig(
        perceptron_id="metadata_perc",
        perceptron_type="metadata",
        hash_function="sha1",
        hash_seed=99,
        config_dict={}
    )
    
    storage.register_perceptron(config1, "Data perceptron")
    storage.register_perceptron(config2, "Metadata perceptron")
    
    # Create lattices for each
    lattice1 = storage.create_lattice("data_perc", "AM", 100)
    lattice2 = storage.create_lattice("metadata_perc", "metadata", 50)
    
    # Verify both exist
    info1 = storage.get_lattice_info(lattice1)
    info2 = storage.get_lattice_info(lattice2)
    
    assert info1['perceptron_id'] == "data_perc"
    assert info1['perceptron_type'] == "data"
    assert info2['perceptron_id'] == "metadata_perc"
    assert info2['perceptron_type'] == "metadata"
    
    # List perceptrons
    perceptrons = storage.list_perceptrons()
    assert len(perceptrons) == 2
    
    storage.close()


if __name__ == "__main__":
    # Run tests
    test_unified_storage_initialization()
    print("✓ Schema initialization test passed")
    
    test_perceptron_registration()
    print("✓ Perceptron registration test passed")
    
    test_hllset_storage_with_compression()
    print("✓ HLLSet storage with compression test passed")
    
    test_am_lattice_storage()
    print("✓ AM lattice storage test passed")
    
    test_w_lattice_storage()
    print("✓ W lattice storage test passed")
    
    test_storage_stats()
    print("✓ Storage statistics test passed")
    
    test_multi_perceptron_storage()
    print("✓ Multi-perceptron storage test passed")
    
    print("\n✅ All unified storage tests passed!")
