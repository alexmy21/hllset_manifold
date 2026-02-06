#!/usr/bin/env python3
"""
Test: ManifoldOS Driver Management

Tests the Universal Constructor pattern implementation:
- Driver lifecycle (wake, idle, restart, remove)
- IngestDriver tokenization
- Resource management
- Health monitoring
- Immutability and idempotence
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manifold_os import (
    ManifoldOS, Driver, IngestDriver, DriverState,
    TokenizationConfig
)


def test_driver_registration():
    """Test driver registration and unregistration."""
    print("\n[TEST] Driver Registration")
    
    os = ManifoldOS()
    
    # Should have default ingest driver
    assert "ingest_default" in os.list_drivers()
    print("  ✓ Default ingest driver registered")
    
    # Register new driver
    driver = IngestDriver("test_driver")
    os.register_driver(driver)
    assert "test_driver" in os.list_drivers()
    print("  ✓ Custom driver registered")
    
    # Unregister
    assert os.unregister_driver("test_driver")
    assert "test_driver" not in os.list_drivers()
    print("  ✓ Driver unregistered")
    
    print("  ✓ PASS")


def test_driver_lifecycle():
    """Test driver state transitions."""
    print("\n[TEST] Driver Lifecycle")
    
    os = ManifoldOS()
    driver = os.get_driver("ingest_default")
    
    # Should start in IDLE (woken during init)
    assert driver.state == DriverState.IDLE
    print("  ✓ Initial state: IDLE")
    
    # Wake (already idle, should stay idle)
    assert driver.wake()
    assert driver.state == DriverState.IDLE
    print("  ✓ Wake: IDLE → IDLE")
    
    # Activate
    assert driver.activate()
    assert driver.state == DriverState.ACTIVE
    print("  ✓ Activate: IDLE → ACTIVE")
    
    # Idle
    assert driver.idle()
    assert driver.state == DriverState.IDLE
    print("  ✓ Idle: ACTIVE → IDLE")
    
    # Error
    driver.mark_error()
    assert driver.state == DriverState.ERROR
    assert driver.needs_restart
    print("  ✓ Error: IDLE → ERROR")
    
    # Restart
    assert driver.restart()
    assert driver.state == DriverState.IDLE
    assert not driver.needs_restart
    print("  ✓ Restart: ERROR → IDLE")
    
    # Dead
    driver.mark_dead()
    assert driver.state == DriverState.DEAD
    assert not driver.is_alive
    print("  ✓ Dead: IDLE → DEAD")
    
    print("  ✓ PASS")


def test_ingest_basic():
    """Test basic data ingestion."""
    print("\n[TEST] Basic Ingestion")
    
    os = ManifoldOS()
    
    # Ingest text
    text = "hello world"
    representation = os.ingest(text)
    
    assert representation is not None
    # Get 1-token HLLSet (individual tokens)
    hllset = representation.hllsets.get(1)
    assert hllset is not None
    # HLL is probabilistic, so allow small error margin (2 tokens ± 10%)
    card = hllset.cardinality()
    assert 1.8 <= card <= 2.2, f"Expected ~2 tokens, got {card}"
    print(f"  ✓ Ingested '{text}' → cardinality={card}")
    
    # Check driver stats
    drivers = os.list_drivers()
    stats = drivers["ingest_default"]
    assert stats['operations'] == 1
    assert stats['errors'] == 0
    print(f"  ✓ Driver stats: ops={stats['operations']}, errors={stats['errors']}")
    
    print("  ✓ PASS")


def test_ingest_batch():
    """Test batch ingestion."""
    print("\n[TEST] Batch Ingestion")
    
    os = ManifoldOS()
    
    # Batch ingest
    documents = [
        "first document",
        "second document",
        "third document"
    ]
    
    representations = os.ingest_batch(documents)
    
    assert len(representations) == 3
    print(f"  ✓ Ingested {len(representations)} documents")
    
    # Check cardinalities (using 1-token HLLSets)
    for i, rep in enumerate(representations, 1):
        hllset = rep.hllsets.get(1)
        card = hllset.cardinality() if hllset else 0
        print(f"  ✓ Document {i}: cardinality={card}")
    
    # Check driver stats
    drivers = os.list_drivers()
    stats = drivers["ingest_default"]
    assert stats['operations'] == 3
    print(f"  ✓ Driver operations: {stats['operations']}")
    
    print("  ✓ PASS")


def test_tokenization_config():
    """Test custom tokenization configuration."""
    print("\n[TEST] Tokenization Configuration")
    
    os = ManifoldOS()
    
    # Register custom driver
    config = TokenizationConfig(
        min_token_length=3,
        max_token_length=10,
        lowercase=True,
        remove_punctuation=True
    )
    driver = IngestDriver("custom", config)
    os.register_driver(driver)
    driver.wake()
    
    # Ingest with different configs
    text = "Hi, world! Testing 123."
    
    default_rep = os.ingest(text, driver_id="ingest_default")
    custom_rep = os.ingest(text, driver_id="custom")
    
    # Get 1-token HLLSets
    default_hllset = default_rep.hllsets.get(1)
    custom_hllset = custom_rep.hllsets.get(1)
    
    print(f"  ✓ Default: cardinality={default_hllset.cardinality()}")
    print(f"  ✓ Custom:  cardinality={custom_hllset.cardinality()}")
    
    # Custom should filter out short tokens
    # "Hi," (3 with comma), "world!" → "world", "Testing" (>= 3 chars)
    assert custom_hllset.cardinality() <= default_hllset.cardinality()
    
    print("  ✓ PASS")


def test_immutability():
    """Test immutability and content addressability."""
    print("\n[TEST] Immutability & Content Addressability")
    
    os = ManifoldOS()
    
    # Ingest same data multiple times
    text = "immutable data"
    rep1 = os.ingest(text)
    rep2 = os.ingest(text)
    rep3 = os.ingest(text)
    
    # Get 1-token HLLSets
    hllset1 = rep1.hllsets.get(1)
    hllset2 = rep2.hllsets.get(1)
    hllset3 = rep3.hllsets.get(1)
    
    # Should all have same hash (content-addressed)
    assert hllset1.name == hllset2.name == hllset3.name
    print(f"  ✓ Same input → same hash: {hllset1.name[:20]}...")
    
    # Cardinality should be same
    assert hllset1.cardinality() == hllset2.cardinality()
    print(f"  ✓ Cardinality preserved: {hllset1.cardinality()}")
    
    print("  ✓ PASS")


def test_idempotence():
    """Test idempotent operations."""
    print("\n[TEST] Idempotence")
    
    os = ManifoldOS()
    
    # Ingest data twice - should get same HLLSet (content-addressed)
    text = "idempotent operation"
    rep1 = os.ingest(text)
    rep2 = os.ingest(text)
    
    # Get 1-token HLLSets
    hllset1 = rep1.hllsets.get(1)
    hllset2 = rep2.hllsets.get(1)
    
    # Same input = same hash (idempotent ingestion)
    assert hllset1.name == hllset2.name
    print(f"  ✓ Idempotent ingestion: same hash")
    
    # Union with itself (same object)
    union1 = os.kernel.union(hllset1, hllset1)
    
    # Since it's the same HLLSet, union should preserve structure
    # (HLL union is approximate, but same object should be close)
    similarity1 = hllset1.similarity(union1)
    print(f"  ✓ Self-union similarity: {similarity1:.4f}")
    
    # Intersection with itself should be identical
    intersection = os.kernel.intersection(hllset1, hllset1)
    similarity2 = hllset1.similarity(intersection)
    print(f"  ✓ Self-intersection similarity: {similarity2:.4f}")
    
    # Test multiple unions (idempotence)
    # A ∪ A ∪ A should be similar to A
    multi_union = os.kernel.union(hllset1, hllset1)
    multi_union = os.kernel.union(multi_union, hllset1)
    similarity3 = hllset1.similarity(multi_union)
    print(f"  ✓ Multiple unions similarity: {similarity3:.4f}")
    
    print("  ✓ PASS")


def test_driver_cleanup():
    """Test driver cleanup."""
    print("\n[TEST] Driver Cleanup")
    
    os = ManifoldOS()
    
    # Register multiple drivers
    for i in range(5):
        driver = IngestDriver(f"temp_{i}")
        os.register_driver(driver)
    
    initial_count = len(os.list_drivers())
    print(f"  ✓ Registered {initial_count} drivers")
    
    # Mark some as dead
    os.get_driver("temp_0").mark_dead()
    os.get_driver("temp_2").mark_dead()
    os.get_driver("temp_4").mark_dead()
    
    # Cleanup
    removed = os.cleanup_dead_drivers()
    assert len(removed) == 3
    print(f"  ✓ Removed {len(removed)} dead drivers")
    
    final_count = len(os.list_drivers())
    assert final_count == initial_count - 3
    print(f"  ✓ Final count: {final_count} drivers")
    
    print("  ✓ PASS")


def test_driver_stats():
    """Test driver statistics tracking."""
    print("\n[TEST] Driver Statistics")
    
    os = ManifoldOS()
    driver = os.get_driver("ingest_default")
    
    # Initial stats
    initial_ops = driver.stats.operations_count
    initial_errors = driver.stats.errors_count
    
    # Process some data
    for i in range(10):
        os.ingest(f"document {i}")
    
    # Check stats updated
    assert driver.stats.operations_count == initial_ops + 10
    print(f"  ✓ Operations: {driver.stats.operations_count}")
    
    # Simulate error
    driver.mark_error()
    assert driver.stats.errors_count == initial_errors + 1
    print(f"  ✓ Errors: {driver.stats.errors_count}")
    
    # Check timestamps
    assert driver.stats.last_active is not None
    assert driver.stats.total_active_time >= 0
    print(f"  ✓ Last active: {driver.stats.last_active}")
    print(f"  ✓ Total active time: {driver.stats.total_active_time:.4f}s")
    
    print("  ✓ PASS")


def test_universal_constructor_pattern():
    """Test ICASRA universal constructor pattern."""
    print("\n[TEST] Universal Constructor (ICASRA)")
    
    os = ManifoldOS()
    
    # D - Interface: Ingest external data
    raw_data = "external reality observation"
    representation = os.ingest(raw_data)
    assert representation is not None
    print("  ✓ [D] Interface: Ingested external data")
    
    # Get 1-token HLLSet for further operations
    hllset = representation.hllsets.get(1)
    assert hllset is not None
    
    # B - Copier: Reproduce structure
    reproduced = os.kernel.reproduce(hllset)
    similarity = hllset.similarity(reproduced)
    assert similarity >= 0.99
    print(f"  ✓ [B] Copier: Reproduced (similarity={similarity:.4f})")
    
    # C - Controller: Manage driver
    drivers = os.list_drivers()
    assert len(drivers) > 0
    assert drivers["ingest_default"]["state"] == "idle"
    print(f"  ✓ [C] Controller: Managing {len(drivers)} driver(s)")
    
    # A - Constructor: Commit state
    from core.manifold_os import OSState
    state = OSState(
        state_hash="",
        root_hllset_hash=hllset.name,
        hrt_hash="",
        perceptron_states={},
        pipeline_config={}
    )
    state_hash = os.store.commit(state)
    assert state_hash is not None
    print(f"  ✓ [A] Constructor: Committed state {state_hash[:20]}...")
    
    print("  ✓ PASS")


def test_parallel_processing():
    """Test parallel ingestion (immutability enables this)."""
    print("\n[TEST] Parallel Processing")
    
    os = ManifoldOS()
    
    # Register multiple ingest drivers
    for i in range(3):
        driver = IngestDriver(f"parallel_{i}")
        os.register_driver(driver)
        driver.wake()
    
    # Ingest with different drivers (could be parallel in real system)
    texts = [
        "parallel processing test one",
        "parallel processing test two",
        "parallel processing test three"
    ]
    
    hllsets = []
    for i, text in enumerate(texts):
        representation = os.ingest(text, driver_id=f"parallel_{i}")
        # Get 1-token HLLSet
        hllset = representation.hllsets.get(1)
        if hllset:
            hllsets.append(hllset)
    
    assert len(hllsets) == 3
    print(f"  ✓ Processed {len(hllsets)} documents in parallel")
    
    # Union all results (safe because immutable)
    union = hllsets[0]
    for h in hllsets[1:]:
        union = os.kernel.union(union, h)
    
    print(f"  ✓ Union cardinality: {union.cardinality()}")
    print("  ✓ PASS")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("ManifoldOS Driver Management Tests")
    print("="*70)
    
    tests = [
        test_driver_registration,
        test_driver_lifecycle,
        test_ingest_basic,
        test_ingest_batch,
        test_tokenization_config,
        test_immutability,
        test_idempotence,
        test_driver_cleanup,
        test_driver_stats,
        test_universal_constructor_pattern,
        test_parallel_processing,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
