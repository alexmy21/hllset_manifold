"""
Integration tests for the hllset_manifold package.

Tests the interaction between different components.
"""

import pytest
from hllset_manifold import Manifold, Morphism, Entanglement, TangentVector
from hllset_manifold.tangent_vector import MorphismTangentSpace


def test_full_system_integration():
    """Test all components working together."""
    # Create two manifolds
    h1 = Manifold("H1")
    h2 = Manifold("H2")
    
    # Add elements
    h1.add_element("element1")
    h1.add_element("element2")
    
    # Create a morphism between them
    m = Morphism("H1", "H2", name="test_morphism")
    
    # Apply morphism
    result1 = m("element1")
    result2 = m("element2")
    
    # Add transformed elements to target manifold
    h2.add_element(result1)
    h2.add_element(result2)
    
    assert len(h1) == 2
    assert len(h2) == 2


def test_entanglement_with_morphism():
    """Test entanglement with morphism application."""
    # Create manifold
    m = Manifold("TestManifold")
    m.add_element("a")
    m.add_element("b")
    
    # Create entanglement (automorphism)
    e = Entanglement(m)
    e.entangle("a", "b")
    
    # Apply automorphism
    auto = e.get_automorphism()
    result_a = auto("a")
    result_b = auto("b")
    
    # Results should be hashes
    assert isinstance(result_a, str)
    assert isinstance(result_b, str)


def test_tangent_vectors_with_morphisms():
    """Test tangent vectors associated with morphisms (requirement 6)."""
    # Create morphism m: H1 -> H2
    m = Morphism("H1", "H2", name="test_morph")
    
    # Create tangent space for morphism
    tangent_space = MorphismTangentSpace(m.name)
    
    # Set tangent vector m' = {D, R, N}
    m_prime = TangentVector(D=1.0, R=2.0, N=3.0)
    tangent_space.set_tangent_vector(m_prime)
    
    # Define time derivative function m'_t = (d_D/d_t, d_R/d_t, d_N/d_t)
    def derivative_func(t):
        return TangentVector(D=0.5*t, R=1.0*t, N=1.5*t)
    
    tangent_space.set_derivative_function(derivative_func)
    
    # Get derivative at t=2
    m_prime_t = tangent_space.get_derivative_at(2.0)
    
    assert m_prime_t.D == 1.0
    assert m_prime_t.R == 2.0
    assert m_prime_t.N == 3.0


def test_morphism_composition_chain():
    """Test chaining multiple morphisms."""
    m1 = Morphism("H1", "H2")
    m2 = Morphism("H2", "H3")
    m3 = Morphism("H3", "H4")
    
    # Compose m1 -> m2 -> m3
    m12 = m1.compose(m2)
    m123 = m12.compose(m3)
    
    assert m123.source_id == "H1"
    assert m123.target_id == "H4"
    
    # Test application
    result = m123("test")
    assert isinstance(result, str)


def test_disambiguation_workflow():
    """Test disambiguation workflow (requirement 4)."""
    # Create multiple morphisms without type separation (requirement 3)
    m1 = Morphism("H1", "H2", name="morph1")
    m2 = Morphism("H1", "H2", name="morph2")
    m3 = Morphism("H1", "H2", name="morph3")
    
    # They work without type labels
    assert not m1.needs_disambiguation()
    assert not m2.needs_disambiguation()
    assert not m3.needs_disambiguation()
    
    # When disambiguation is needed (requirement 4)
    m1.set_disambiguation_type("type_A")
    m2.set_disambiguation_type("type_B")
    
    assert m1.get_disambiguation_type() == "type_A"
    assert m2.get_disambiguation_type() == "type_B"
    assert m3.get_disambiguation_type() is None


def test_idempotency_across_system():
    """Test idempotency as the only restriction (requirement 2)."""
    # Manifold is idempotent
    manifold = Manifold("Test")
    manifold.add_element("x")
    manifold.add_element("x")
    manifold.add_element("x")
    assert len(manifold) == 1
    
    # Entanglement pairs are idempotent
    manifold.add_element("y")
    e = Entanglement(manifold)
    e.entangle("x", "y")
    e.entangle("x", "y")
    e.entangle("y", "x")
    assert e.get_entanglement_count() == 1


def test_manifold_like_structure():
    """Test that the structure behaves like manifolds (requirement 1)."""
    # Create a manifold with structure
    m = Manifold("TestManifold")
    
    # Set manifold properties
    m.set_structure("dimension", 3)
    m.set_structure("topology", "compact")
    
    # Add elements (points on the manifold)
    for i in range(10):
        m.add_element(f"point_{i}")
    
    # Create automorphism (self-mapping)
    e = Entanglement(m)
    auto = e.get_automorphism()
    
    # Automorphism maps manifold to itself
    assert auto.source_id == auto.target_id
    
    # Can apply transformations preserving manifold structure
    for elem in m.get_elements():
        transformed = e.apply_automorphism(elem)
        assert isinstance(transformed, str)


def test_package_imports():
    """Test that all main classes are importable from package."""
    from hllset_manifold import Manifold, Morphism, Entanglement, TangentVector
    
    # Should be able to instantiate all
    m = Manifold("test")
    morph = Morphism("A", "B")
    e = Entanglement(m)
    tv = TangentVector(1, 2, 3)
    
    assert isinstance(m, Manifold)
    assert isinstance(morph, Morphism)
    assert isinstance(e, Entanglement)
    assert isinstance(tv, TangentVector)
