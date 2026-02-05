#!/usr/bin/env python3
"""
Demo: ManifoldOS with Driver Management

This demonstrates the Universal Constructor pattern with:
1. Driver lifecycle management (wake, idle, restart, remove)
2. Ingest driver for tokenizing external data
3. Resource management without scheduling
4. Immutable, idempotent operations

ICASRA Pattern:
- A (Constructor): commit() validates and persists states
- B (Copier): reproduce() via kernel operations
- C (Controller): Driver lifecycle management
- D (Interface): IngestDriver tokenizes external data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.manifold_os import (
    ManifoldOS, IngestDriver, TokenizationConfig,
    DriverState, Driver
)


def demo_basic_ingest():
    """Demo 1: Basic data ingestion."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Data Ingestion")
    print("="*70)
    
    # Create OS
    os = ManifoldOS()
    
    # Check default ingest driver
    drivers = os.list_drivers()
    print(f"\nRegistered drivers: {len(drivers)}")
    for driver_id, info in drivers.items():
        print(f"  {driver_id}: state={info['state']}, type={info['type']}")
    
    # Ingest some data
    raw_data = "The quick brown fox jumps over the lazy dog"
    print(f"\nIngesting: '{raw_data}'")
    
    hllset = os.ingest(raw_data)
    if hllset:
        print(f"✓ Created HLLSet: {hllset.name[:16]}...")
        print(f"  Cardinality: {hllset.cardinality()}")
    
    # Check driver stats after ingestion
    drivers = os.list_drivers()
    ingest_driver = drivers.get("ingest_default")
    if ingest_driver:
        print(f"\nIngest driver stats:")
        print(f"  Operations: {ingest_driver['operations']}")
        print(f"  Errors: {ingest_driver['errors']}")
        print(f"  Uptime: {ingest_driver['uptime']:.2f}s")


def demo_batch_ingest():
    """Demo 2: Batch ingestion."""
    print("\n" + "="*70)
    print("DEMO 2: Batch Ingestion")
    print("="*70)
    
    os = ManifoldOS()
    
    # Prepare batch data
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "HyperLogLog provides approximate cardinality estimation",
        "Immutability enables safe concurrent processing",
        "Content addressability ensures data integrity",
        "The universal constructor creates new patterns"
    ]
    
    print(f"\nIngesting {len(documents)} documents...")
    hllsets = os.ingest_batch(documents)
    
    print(f"✓ Created {len(hllsets)} HLLSets:")
    for i, hllset in enumerate(hllsets, 1):
        print(f"  {i}. {hllset.name[:16]}... (card: {hllset.cardinality()})")
    
    # Union all ingested HLLSets
    if hllsets:
        union = hllsets[0]
        for hllset in hllsets[1:]:
            union = os.kernel.union(union, hllset)
        print(f"\nUnion of all documents:")
        print(f"  Hash: {union.name[:16]}...")
        print(f"  Cardinality: {union.cardinality()}")


def demo_custom_tokenization():
    """Demo 3: Custom tokenization config."""
    print("\n" + "="*70)
    print("DEMO 3: Custom Tokenization")
    print("="*70)
    
    os = ManifoldOS()
    
    # Register custom ingest driver with different config
    config = TokenizationConfig(
        min_token_length=3,
        max_token_length=20,
        lowercase=True,
        remove_punctuation=True
    )
    custom_driver = IngestDriver("ingest_custom", config)
    os.register_driver(custom_driver)
    custom_driver.wake()
    
    print(f"\nRegistered drivers: {len(os.list_drivers())}")
    
    # Ingest with default driver
    text = "Hello, World! Testing 123."
    print(f"\nText: '{text}'")
    
    default_hllset = os.ingest(text, driver_id="ingest_default")
    custom_hllset = os.ingest(text, driver_id="ingest_custom")
    
    print(f"\nDefault tokenization:")
    print(f"  Cardinality: {default_hllset.cardinality()}")
    
    print(f"\nCustom tokenization (min_len=3, no punctuation):")
    print(f"  Cardinality: {custom_hllset.cardinality()}")


def demo_driver_lifecycle():
    """Demo 4: Driver lifecycle management."""
    print("\n" + "="*70)
    print("DEMO 4: Driver Lifecycle Management")
    print("="*70)
    
    os = ManifoldOS()
    
    # Create additional driver
    driver2 = IngestDriver("ingest_secondary")
    os.register_driver(driver2)
    
    print("\nInitial state:")
    for driver_id, info in os.list_drivers().items():
        print(f"  {driver_id}: {info['state']}")
    
    # Wake driver
    print("\nWaking secondary driver...")
    os.wake_driver("ingest_secondary")
    
    print("After wake:")
    for driver_id, info in os.list_drivers().items():
        print(f"  {driver_id}: {info['state']}")
    
    # Use driver, then idle it
    os.ingest("test data", driver_id="ingest_secondary")
    
    print("\nAfter processing:")
    for driver_id, info in os.list_drivers().items():
        print(f"  {driver_id}: {info['state']} (ops={info['operations']})")
    
    # Idle driver
    print("\nIdling default driver...")
    os.idle_driver("ingest_default")
    
    print("After idle:")
    for driver_id, info in os.list_drivers().items():
        print(f"  {driver_id}: {info['state']}")


def demo_driver_monitoring():
    """Demo 5: Driver health monitoring."""
    print("\n" + "="*70)
    print("DEMO 5: Driver Health Monitoring")
    print("="*70)
    
    os = ManifoldOS()
    
    # Start monitoring
    print("\nStarting driver monitoring...")
    os.start_driver_monitoring()
    
    # Simulate some errors
    print("\nSimulating driver errors...")
    driver = os.get_driver("ingest_default")
    driver.mark_error()
    
    print(f"Driver state: {driver.state.value}")
    print(f"Needs restart: {driver.needs_restart}")
    
    # Wait a bit for monitor to restart
    import time
    time.sleep(6)
    
    # Check if restarted
    print("\nAfter monitoring cycle:")
    drivers = os.list_drivers()
    for driver_id, info in drivers.items():
        print(f"  {driver_id}: {info['state']} (errors={info['errors']})")
    
    # Stop monitoring
    os.stop_driver_monitoring()
    print("\n✓ Monitoring stopped")


def demo_immutability_principle():
    """Demo 6: Immutability and idempotence."""
    print("\n" + "="*70)
    print("DEMO 6: Immutability & Idempotence")
    print("="*70)
    
    os = ManifoldOS()
    
    # Ingest same data multiple times
    text = "immutable data structure"
    print(f"\nIngesting '{text}' 3 times...")
    
    hllset1 = os.ingest(text)
    hllset2 = os.ingest(text)
    hllset3 = os.ingest(text)
    
    print(f"\nHLLSet hashes:")
    print(f"  1: {hllset1.name[:20]}...")
    print(f"  2: {hllset2.name[:20]}...")
    print(f"  3: {hllset3.name[:20]}...")
    
    # Same content -> same hash (content addressability)
    if hllset1.name == hllset2.name == hllset3.name:
        print("\n✓ Content addressability verified!")
        print("  Same input produces same hash (idempotent)")
    
    # Union with itself
    union = os.kernel.union(hllset1, hllset2)
    print(f"\nUnion of identical HLLSets:")
    print(f"  Original: card={hllset1.cardinality()}")
    print(f"  Union:    card={union.cardinality()}")
    
    if hllset1.cardinality() == union.cardinality():
        print("  ✓ Idempotence verified!")


def demo_universal_constructor_pattern():
    """Demo 7: Universal Constructor (ICASRA) pattern."""
    print("\n" + "="*70)
    print("DEMO 7: Universal Constructor (ICASRA)")
    print("="*70)
    
    os = ManifoldOS()
    
    print("\nICSARA Pattern Implementation:")
    print("  A (Constructor): commit() - validate and persist states")
    print("  B (Copier): reproduce() - copy with structure preservation")
    print("  C (Controller): driver lifecycle - wake, idle, restart, remove")
    print("  D (Interface): IngestDriver - tokenize external data")
    
    # D - Interface: Ingest external data
    print("\n[D - Interface] Ingesting external data...")
    raw_data = "universal constructor self replication pattern"
    hllset = os.ingest(raw_data)
    print(f"  ✓ Ingested: {hllset.name[:20]}...")
    
    # B - Copier: Reproduce structure
    print("\n[B - Copier] Reproducing structure...")
    copy_hllset = os.kernel.reproduce(hllset)
    print(f"  ✓ Reproduced: {copy_hllset.name[:20]}...")
    print(f"  Similarity: {hllset.similarity(copy_hllset):.4f}")
    
    # C - Controller: Manage driver
    print("\n[C - Controller] Managing driver lifecycle...")
    drivers = os.list_drivers()
    print(f"  Drivers: {len(drivers)}")
    print(f"  State: {drivers['ingest_default']['state']}")
    print(f"  Operations: {drivers['ingest_default']['operations']}")
    
    # A - Constructor: Commit state
    print("\n[A - Constructor] Committing state...")
    # Create OS state
    from core.manifold_os import OSState
    state = OSState(
        state_hash="",
        root_hllset_hash=hllset.name,
        hrt_hash="",
        perceptron_states={},
        pipeline_config={}
    )
    state_hash = os.store.commit(state)
    print(f"  ✓ Committed: {state_hash[:20]}...")
    
    print("\n✓ Universal Constructor pattern complete!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("ManifoldOS Driver Management Demos")
    print("Universal Constructor with ICASRA Pattern")
    print("="*70)
    
    demos = [
        demo_basic_ingest,
        demo_batch_ingest,
        demo_custom_tokenization,
        demo_driver_lifecycle,
        demo_immutability_principle,
        demo_universal_constructor_pattern,
        # demo_driver_monitoring,  # Skip for now (takes 6+ seconds)
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ All demos complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
