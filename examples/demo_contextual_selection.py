# File: examples/demo_contextual_selection.py
"""Demonstrate the Contextual Selection Principle."""

from core.anti_set import AntiSet
from core.reality_absorber import RealityAbsorber

def demo_quantum_measurement():
    """Show quantum measurement as contextual selection."""
    print("=== Quantum Measurement as Contextual Selection ===")
    
    # Create measurement context
    measurement_context = AntiSet("spin_measurement_up", tau=0.9, rho=0.1)
    
    # Add some "spin up" characteristics to the context
    measurement_context.absorb("definite_spin")
    measurement_context.absorb("quantum_collapse")
    measurement_context.absorb("measurement_basis_z")
    
    # Create possible particle states
    particle_states = [
        "spin_up_eigenstate",
        "spin_down_eigenstate", 
        "superposition_state"
    ]
    
    # Context selects compatible states
    print(f"\nMeasurement context: {measurement_context}")
    for state in particle_states:
        selected = measurement_context.absorb(state)
        print(f"  State '{state}': {'SELECTED' if selected else 'rejected'}")
    
    print("\nNo 'collapse' occurs - context was always selecting!")
    return measurement_context

def demo_biological_niche():
    """Show ecological niche as selecting context."""
    print("\n=== Biological Niche as Contextual Selection ===")
    
    # Create ecological context
    forest_context = AntiSet("temperate_forest", tau=0.7, rho=0.3)
    
    # Add forest characteristics
    for char in ["deciduous_trees", "moderate_rainfall", "seasonal_variation"]:
        forest_context.absorb(char)
    
    # Organisms that might inhabit
    organisms = [
        "squirrel",
        "oak_tree", 
        "cactus",  # Wrong context!
        "deer",
        "camel"    # Wrong context!
    ]
    
    print(f"\nEcological context: {forest_context}")
    for org in organisms:
        selected = forest_context.absorb(org)
        print(f"  Organism '{org}': {'SELECTED' if selected else 'rejected'}")
    
    return forest_context

def demo_entanglement():
    """Show entanglement between contexts."""
    print("\n=== Contextual Entanglement ===")
    
    absorber = RealityAbsorber()
    
    # Create related contexts
    context1 = AntiSet("quantum_context", tau=0.8, rho=0.2)
    context2 = AntiSet("quantum_context_alternative", tau=0.8, rho=0.2)
    
    # They absorb similar reality
    reality = {"superposition", "entanglement", "wave_function", "measurement"}
    
    absorber.register_context(context1)
    absorber.register_context(context2)
    
    selections = absorber.absorb_reality_slice(reality)
    
    print(f"\nContext 1 selected: {selections.get('quantum_context', [])}")
    print(f"Context 2 selected: {selections.get('quantum_context_alternative', [])}")
    
    # Detect entanglement
    entangled = absorber.detect_entanglement()
    print(f"\nEntangled pairs: {entangled}")
    
    return absorber

if __name__ == "__main__":
    demo_quantum_measurement()
    demo_biological_niche()
    demo_entanglement()