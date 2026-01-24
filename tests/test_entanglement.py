"""
Tests for the Entanglement class.
"""

import pytest
from hllset_manifold.manifold import Manifold
from hllset_manifold.entanglement import Entanglement
from hllset_manifold.morphism import Morphism


def test_entanglement_creation():
    """Test creating an entanglement."""
    m = Manifold("TestManifold")
    e = Entanglement(m)
    
    assert e.manifold == m
    assert "Entanglement" in e.name
    assert e.get_entanglement_count() == 0


def test_entanglement_automorphism():
    """Test that entanglement is an automorphism (requirement 5)."""
    m = Manifold("TestManifold")
    e = Entanglement(m)
    
    # Get the automorphism
    auto = e.get_automorphism()
    
    # Verify it's a morphism from manifold to itself
    assert auto.source_id == m.name
    assert auto.target_id == m.name


def test_entanglement_entangle_elements():
    """Test entangling elements."""
    m = Manifold("TestManifold")
    m.add_element("a")
    m.add_element("b")
    
    e = Entanglement(m)
    e.entangle("a", "b")
    
    assert e.is_entangled("a", "b")
    assert e.is_entangled("b", "a")  # Symmetric
    assert e.get_entanglement_count() == 1


def test_entanglement_multiple_pairs():
    """Test multiple entangled pairs."""
    m = Manifold("TestManifold")
    elements = ["a", "b", "c", "d"]
    for elem in elements:
        m.add_element(elem)
    
    e = Entanglement(m)
    e.entangle("a", "b")
    e.entangle("c", "d")
    
    assert e.get_entanglement_count() == 2
    assert e.is_entangled("a", "b")
    assert e.is_entangled("c", "d")
    assert not e.is_entangled("a", "c")


def test_entanglement_get_entangled_with():
    """Test getting elements entangled with a given element."""
    m = Manifold("TestManifold")
    for elem in ["a", "b", "c", "d"]:
        m.add_element(elem)
    
    e = Entanglement(m)
    e.entangle("a", "b")
    e.entangle("a", "c")
    
    entangled_with_a = e.get_entangled_with("a")
    assert "b" in entangled_with_a
    assert "c" in entangled_with_a
    assert len(entangled_with_a) == 2


def test_entanglement_element_not_in_manifold():
    """Test that entangling non-manifold elements raises error."""
    m = Manifold("TestManifold")
    m.add_element("a")
    
    e = Entanglement(m)
    
    with pytest.raises(ValueError):
        e.entangle("a", "b")  # "b" not in manifold


def test_entanglement_apply_automorphism():
    """Test applying automorphism to element."""
    m = Manifold("TestManifold")
    m.add_element("test")
    
    e = Entanglement(m)
    result = e.apply_automorphism("test")
    
    # Should return a hash
    assert isinstance(result, str)
    assert len(result) > 0


def test_entanglement_custom_automorphism():
    """Test setting custom automorphism function."""
    m = Manifold("TestManifold")
    e = Entanglement(m)
    
    def custom_auto(x):
        return f"transformed_{x}"
    
    e.set_custom_automorphism(custom_auto)
    result = e.apply_automorphism("test")
    assert result == "transformed_test"


def test_entanglement_verify_automorphism():
    """Test verifying automorphism property."""
    m = Manifold("TestManifold")
    m.add_element("test")
    
    e = Entanglement(m)
    assert e.verify_automorphism_property("test")


def test_entanglement_idempotent_pairing():
    """Test that entangling same pair multiple times is idempotent."""
    m = Manifold("TestManifold")
    m.add_element("a")
    m.add_element("b")
    
    e = Entanglement(m)
    e.entangle("a", "b")
    e.entangle("a", "b")
    e.entangle("b", "a")  # Order shouldn't matter
    
    # Should still have only 1 pair
    assert e.get_entanglement_count() == 1


def test_entanglement_repr():
    """Test string representation."""
    m = Manifold("TestManifold")
    e = Entanglement(m, name="TestEntanglement")
    
    repr_str = repr(e)
    assert "TestEntanglement" in repr_str
    assert "TestManifold" in repr_str
    assert "pairs=0" in repr_str


def test_entanglement_as_automorphism_in_manifold():
    """Test that entanglement is truly automorphism in manifold (requirement 5)."""
    # Create a manifold
    m = Manifold("TestManifold")
    m.add_element("x")
    m.add_element("y")
    
    # Create entanglement (automorphism)
    e = Entanglement(m)
    
    # Automorphism should map manifold to itself
    auto = e.get_automorphism()
    assert isinstance(auto, Morphism)
    assert auto.source_id == auto.target_id == m.name
    
    # Applying automorphism should preserve structure
    result_x = e.apply_automorphism("x")
    result_y = e.apply_automorphism("y")
    
    # Results should be deterministic
    assert e.apply_automorphism("x") == result_x
    assert e.apply_automorphism("y") == result_y
