# run_tests.py
"""
Run all Phase 1 tests for HLLSet Theory implementation.
"""

import subprocess
import sys
import os


def run_test_suite(test_files=None):
    """Run the test suite."""
    
    if test_files is None:
        # Run all test files
        test_files = [
            "tests/test_anti_set.py",
            "tests/test_idempotence.py",
            # "tests/test_reality_absorber.py",  # Coming soon
            # "tests/test_selection_principle.py"  # Coming soon
        ]
    
    print("=" * 70)
    print("HLLSet Theory - Phase 1 Test Suite")
    print("Validating Contextual Anti-Sets and Selection Principle")
    print("=" * 70)
    
    results = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*60}")
            print(f"Running: {test_file}")
            print('='*60)
            
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True
            )
            
            results.append({
                'file': test_file,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        else:
            print(f"\n⚠️  Warning: Test file not found: {test_file}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for result in results:
        if result['returncode'] == 0:
            passed += 1
            status = "✅ PASSED"
        else:
            failed += 1
            status = "❌ FAILED"
        
        print(f"{status}: {result['file']}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All tests passed! Ready for Phase 2.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix before Phase 2.")
        return False


def quick_demo():
    """Run a quick demonstration of contextual selection."""
    print("\n" + "="*70)
    print("QUICK DEMO: Contextual Selection Principle")
    print("="*70)
    
    # Import here to avoid circular imports
    from core.anti_set import AntiSet
    
    # Create a measurement context
    measurement = AntiSet(
        name="schrodinger_experiment",
        tau=0.9,
        rho=0.1
    )
    
    # Experimental setup
    measurement.absorb("quantum_system")
    measurement.absorb("sealed_box")
    measurement.absorb("radioactive_source")
    measurement.absorb("geiger_counter")
    measurement.absorb("poison_vial")
    measurement.absorb("cat")
    
    print(f"\nExperimental Context: {measurement.name}")
    print(f"Setup: quantum system in sealed box with cat")
    print(f"Context selects compatible realities...\n")
    
    # What does this context select?
    possible_outcomes = [
        "cat_alive",
        "cat_dead",
        "cat_superposition",
        "box_unopened",
        "quantum_collapse",
        "multiple_worlds"
    ]
    
    for outcome in possible_outcomes:
        selected, metrics = measurement.select(outcome)
        status = "SELECTED" if selected else "rejected"
        print(f"  {outcome:25} → {status:12} "
              f"(compatibility: {metrics['bss_tau']:.3f})")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: The measurement context selects the outcome.")
    print("No 'collapse' - just contextual selection at work.")
    print("Schrödinger's cat is whatever the context selects.")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HLLSet tests")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    parser.add_argument("--tests", nargs="+", help="Specific test files to run")
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    else:
        success = run_test_suite(args.tests)
        sys.exit(0 if success else 1)