#!/usr/bin/env python3
"""
Time Travel Demo: Checkout Historical States and Create Branches

Key Concept:
- Evolution steps are irreversible (you can't "undo" a merge)
- But you can checkout any historical state and start a new branch
- This creates parallel timelines, not reversals

This is exactly like Git:
- Commits are immutable
- You can checkout old commits
- New commits from old state create branches
"""

import sys
sys.path.insert(0, '.')

from core import Kernel, HRTConfig, HRTEvolution


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def mock_commit(hrt):
    """Mock Git commit - would save to persistent store."""
    return hrt.name


def demo_linear_evolution():
    """Show linear evolution (main timeline)."""
    print_section("LINEAR EVOLUTION (Main Timeline)")
    
    kernel = Kernel(p_bits=8)
    config = HRTConfig(p_bits=8, h_bits=16)
    
    # Create evolution manager
    evolution = HRTEvolution(config)
    
    print("\nGenesis state:")
    print(f"  HRT₀: {evolution.get_current().name[:24]}...")
    
    # Evolution Cycle 1
    evolution.ingest({'camera': {'red', 'green', 'blue'}}, kernel)
    evolution.evolve(kernel, mock_commit)
    print(f"\nCycle 1: ingest camera → evolve")
    print(f"  HRT₁: {evolution.get_current().name[:24]}...")
    
    # Evolution Cycle 2
    evolution.ingest({'audio': {'low', 'mid', 'high'}}, kernel)
    evolution.evolve(kernel, mock_commit)
    print(f"\nCycle 2: ingest audio → evolve")
    print(f"  HRT₂: {evolution.get_current().name[:24]}...")
    
    # Evolution Cycle 3
    evolution.ingest({'touch': {'soft', 'hard', 'rough'}}, kernel)
    evolution.evolve(kernel, mock_commit)
    print(f"\nCycle 3: ingest touch → evolve")
    print(f"  HRT₃: {evolution.get_current().name[:24]}...")
    
    # Show lineage
    print(f"\nLineage (commit history):")
    lineage = evolution.get_lineage()
    for i, h in enumerate(lineage):
        marker = " ← HEAD" if i == len(lineage) - 1 else ""
        print(f"  [{i}] {h[:32]}...{marker}")
    
    return evolution, lineage


def demo_time_travel(evolution, lineage):
    """Demonstrate time travel by checking out historical state."""
    print_section("TIME TRAVEL: Checkout from History")
    
    kernel = Kernel(p_bits=8)
    config = HRTConfig(p_bits=8, h_bits=16)
    
    # Time travel: go back to HRT₁
    target_hash = lineage[1]  # HRT₁
    print(f"\nTime travel target: HRT₁ ({target_hash[:32]}...)")
    print("  (In real system: git checkout <hash>)")
    
    # Get the historical HRT (in real system, load from Git)
    # For demo, we'll use the current state to simulate
    # In practice: historical_hrt = git_storage.load(target_hash)
    historical_hrt = evolution.get_current()  # Simulating loaded state
    
    print(f"\nHistorical state loaded:")
    print(f"  Step: {historical_hrt.step_number}")
    print(f"  Hash: {historical_hrt.name[:32]}...")
    
    # Create NEW evolution manager from historical state
    # This creates a BRANCH
    print(f"\nCreating new branch from historical state...")
    branch_evolution = HRTEvolution(config, genesis_hrt=historical_hrt)
    
    print(f"  New branch created at step {branch_evolution.get_current().step_number}")
    
    return branch_evolution


def demo_branch_evolution(branch_evolution):
    """Continue evolution on the new branch."""
    print_section("BRANCH EVOLUTION (Parallel Timeline)")
    
    kernel = Kernel(p_bits=8)
    
    print("Continuing evolution on new branch...")
    
    # Branch Cycle 1
    branch_evolution.ingest({'alt_camera': {'cyan', 'magenta', 'yellow'}}, kernel)
    branch_evolution.evolve(kernel, mock_commit)
    print(f"\nBranch Cycle 1: ingest alt_camera → evolve")
    print(f"  HRT₂': {branch_evolution.get_current().name[:32]}...")
    print(f"  Step: {branch_evolution.get_current().step_number}")
    
    # Branch Cycle 2
    branch_evolution.ingest({'alt_audio': {'bass', 'treble'}}, kernel)
    branch_evolution.evolve(kernel, mock_commit)
    print(f"\nBranch Cycle 2: ingest alt_audio → evolve")
    print(f"  HRT₃': {branch_evolution.get_current().name[:32]}...")
    print(f"  Step: {branch_evolution.get_current().step_number}")
    
    # Show branch lineage
    print(f"\nBranch Lineage:")
    branch_lineage = branch_evolution.get_lineage()
    for i, h in enumerate(branch_lineage):
        marker = " ← BRANCH HEAD" if i == len(branch_lineage) - 1 else ""
        print(f"  [{i}] {h[:32]}...{marker}")
    
    return branch_evolution


def demo_parallel_timelines(main_evolution, branch_evolution):
    """Show that both timelines coexist."""
    print_section("PARALLEL TIMELINES (Both Exist)")
    
    print("Main Timeline (unchanged):")
    print(f"  Current: {main_evolution.get_current().name[:32]}...")
    print(f"  Step: {main_evolution.get_current().step_number}")
    print(f"  Parent: {main_evolution.get_current().parent_hrt[:24] if main_evolution.get_current().parent_hrt else 'genesis'}...")
    
    print("\nBranch Timeline (new):")
    print(f"  Current: {branch_evolution.get_current().name[:32]}...")
    print(f"  Step: {branch_evolution.get_current().step_number}")
    print(f"  Parent: {branch_evolution.get_current().parent_hrt[:24] if branch_evolution.get_current().parent_hrt else 'genesis'}...")
    
    print(f"\nDifferent hashes = different objects:")
    print(f"  Main HRT ≠ Branch HRT: {main_evolution.get_current().name != branch_evolution.get_current().name}")


def visualize_timelines():
    """Visual diagram of the timelines."""
    print_section("VISUAL DIAGRAM")
    
    print("""
    ORIGINAL TIMELINE (Linear):
    ──────────────────────────────
    
    genesis ──→ HRT₁ ──→ HRT₂ ──→ HRT₃ (HEAD)
    [hash0]     [hash1]   [hash2]   [hash3]
                ↑
                │
         Checkout this state
                │
                ▼
    
    AFTER TIME TRAVEL (Parallel Branches):
    ───────────────────────────────────────
    
    genesis ──→ HRT₁ ──→ HRT₂ ──→ HRT₃ (main branch HEAD)
                │           ↑
                │           │
                └──→ HRT₂' ─┘
                     [hash4]
                       │
                       ▼
                     HRT₃' (new branch HEAD)
                     [hash5]
    
    Key Observations:
    ─────────────────
    1. HRT₂ and HRT₂' are DIFFERENT objects (different content → different hashes)
    2. Both branches coexist in the system
    3. Git stores both branches with their full history
    4. Later, branches can be merged if needed
    
    Time Travel = Branch Creation
    ─────────────────────────────
    - You don't "go back" in time
    - You create a NEW timeline from a historical point
    - The original timeline continues to exist
    - This is exactly how Git branches work
    """)


def demo_key_insights():
    """Summarize key insights about time travel."""
    print_section("KEY INSIGHTS")
    
    print("""
    1. IRREVERSIBILITY ≠ IMMUTABILITY
    ───────────────────────────────────
    • Evolution steps cannot be undone
    • Commits cannot be modified
    • But you can always access historical states
    
    2. TIME TRAVEL = BRANCHING
    ───────────────────────────
    • Checking out old state creates a new branch
    • The new branch diverges from the original
    • Both timelines continue independently
    
    3. CONTENT ADDRESSING ENABLES THIS
    ──────────────────────────────────
    • Each state has a unique hash
    • You can reference any state by hash
    • Git stores all states efficiently
    
    4. PARALLEL TIMELINES ARE FIRST-CLASS
    ──────────────────────────────────────
    • Both branches are valid states
    • OS can work with multiple branches
    • Eventually, branches might converge (merge)
    """)


def main():
    """Run complete time travel demo."""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  TIME TRAVEL DEMO".center(68) + "█")
    print("█" + "  Checkout Historical States & Create Branches".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Main timeline
    main_evolution, lineage = demo_linear_evolution()
    
    # Time travel
    branch_evolution = demo_time_travel(main_evolution, lineage)
    
    # Branch evolution
    branch_evolution = demo_branch_evolution(branch_evolution)
    
    # Show parallel timelines
    demo_parallel_timelines(main_evolution, branch_evolution)
    
    # Visual diagram
    visualize_timelines()
    
    # Key insights
    demo_key_insights()
    
    print("\n" + "█" * 70)
    print("█" + "  TIME TRAVEL: History is Immutable but Navigable".center(68) + "█")
    print("█" * 70)
    print("\n")


if __name__ == "__main__":
    main()
