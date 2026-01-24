#!/usr/bin/env python
"""
Example demonstrating the hllset_manifold library.

This example shows all six requirements in action:
1. Manifolds
2. Idempotency
3. Morphisms without type separation
4. Disambiguation when needed
5. Entanglement as automorphism
6. Tangent vectors with time derivatives
"""

from hllset_manifold import Manifold, Morphism, Entanglement, TangentVector
from hllset_manifold.tangent_vector import MorphismTangentSpace


def main():
    print("=" * 70)
    print("hllset_manifold Example")
    print("=" * 70)
    
    # 1. Create manifolds (Requirement 1)
    print("\n1. Creating manifolds...")
    h1 = Manifold("H1")
    h2 = Manifold("H2")
    h3 = Manifold("H3")
    print(f"   Created: {h1}, {h2}, {h3}")
    
    # 2. Add elements with idempotency (Requirement 2)
    print("\n2. Adding elements with idempotency...")
    print("   Adding 'point1' to H1...")
    result1 = h1.add_element("point1")
    print(f"   First add: {result1} (True = new element)")
    result2 = h1.add_element("point1")
    print(f"   Second add: {result2} (False = idempotent, no change)")
    print(f"   H1 size: {len(h1)} element(s)")
    
    h1.add_element("point2")
    h1.add_element("point3")
    print(f"   Added more elements. H1 now has {len(h1)} elements")
    
    # 3. Morphisms without type separation (Requirement 3)
    print("\n3. Creating morphisms without type separation...")
    m1 = Morphism("H1", "H2", name="morph1")
    m2 = Morphism("H2", "H3", name="morph2")
    print(f"   {m1}")
    print(f"   {m2}")
    print(f"   No type labels needed - morphisms work independently")
    
    # Apply morphisms
    print("\n   Applying morphisms...")
    for element in h1.get_elements():
        transformed = m1(element)
        h2.add_element(transformed)
        print(f"   {element} -> {transformed[:16]}...")
    print(f"   H2 now has {len(h2)} elements")
    
    # Compose morphisms
    print("\n   Composing morphisms: m1 ∘ m2")
    m_composed = m1.compose(m2)
    print(f"   {m_composed}")
    result = m_composed("test_value")
    print(f"   Composed result: {result[:32]}...")
    
    # 4. Disambiguation when needed (Requirement 4)
    print("\n4. Disambiguation only when needed...")
    m3 = Morphism("H1", "H2", name="special_morph")
    print(f"   Created {m3}")
    print(f"   Needs disambiguation? {m3.needs_disambiguation()}")
    
    print("   Setting disambiguation type...")
    m3.set_disambiguation_type("special_type")
    print(f"   Disambiguation type: {m3.get_disambiguation_type()}")
    print(f"   {m3}")
    
    # 5. Entanglement as automorphism (Requirement 5)
    print("\n5. Entanglement as automorphism in manifold...")
    entanglement = Entanglement(h1, name="H1_Entanglement")
    print(f"   {entanglement}")
    
    # Verify it's an automorphism (maps manifold to itself)
    auto = entanglement.get_automorphism()
    print(f"   Automorphism: {auto}")
    print(f"   Maps {auto.source_id} -> {auto.target_id} (same manifold)")
    
    # Entangle elements
    print("\n   Entangling elements...")
    entanglement.entangle("point1", "point2")
    entanglement.entangle("point2", "point3")
    print(f"   Entangled pairs: {entanglement.get_entanglement_count()}")
    print(f"   Is point1 entangled with point2? {entanglement.is_entangled('point1', 'point2')}")
    
    entangled_with_point2 = entanglement.get_entangled_with("point2")
    print(f"   Elements entangled with point2: {entangled_with_point2}")
    
    # Apply automorphism
    print("\n   Applying automorphism...")
    for element in ["point1", "point2"]:
        result = entanglement.apply_automorphism(element)
        print(f"   auto({element}) = {result[:24]}...")
    
    # 6. Tangent vectors with derivatives (Requirement 6)
    print("\n6. Tangent vectors for morphisms m: H₁ → H₂...")
    print("   For morphism m1, define m' = {D, R, N}")
    
    # Create tangent space for morphism
    tangent_space = MorphismTangentSpace(m1.name)
    
    # Set tangent vector m' = {D, R, N}
    m_prime = TangentVector(D=1.5, R=2.5, N=3.5)
    tangent_space.set_tangent_vector(m_prime)
    print(f"   m' = {m_prime}")
    
    # Calculate magnitude
    print(f"   |m'| = {m_prime.magnitude():.4f}")
    
    # Define time derivative m'ₜ = (dD/dt, dR/dt, dN/dt)
    print("\n   Defining time derivative m'ₜ = (dD/dt, dR/dt, dN/dt)...")
    
    def derivative_func(t):
        """Time derivative function"""
        return TangentVector(D=0.5*t, R=1.0*t, N=1.5*t)
    
    tangent_space.set_derivative_function(derivative_func)
    
    # Get derivatives at different times
    print("   Time derivatives:")
    for t in [0.0, 1.0, 2.0, 3.0]:
        m_prime_t = tangent_space.get_derivative_at(t)
        print(f"   t={t}: m'ₜ = {m_prime_t}")
    
    # Vector operations
    print("\n   Vector operations:")
    v1 = TangentVector(D=1.0, R=0.0, N=0.0)
    v2 = TangentVector(D=0.0, R=1.0, N=0.0)
    
    print(f"   v1 = {v1}")
    print(f"   v2 = {v2}")
    print(f"   v1 · v2 = {v1.dot(v2)} (dot product)")
    print(f"   v1 × v2 = {v1.cross(v2)} (cross product)")
    print(f"   v1 + v2 = {v1.add(v2)} (addition)")
    print(f"   2 * v1 = {v1.scale(2.0)} (scaling)")
    
    print("\n" + "=" * 70)
    print("All six requirements demonstrated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
