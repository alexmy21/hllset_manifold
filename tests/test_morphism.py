"""
Tests for the Morphism class.
"""

import pytest
from hllset_manifold.morphism import Morphism


def test_morphism_creation():
    """Test basic morphism creation."""
    m = Morphism("H1", "H2")
    assert m.source_id == "H1"
    assert m.target_id == "H2"
    assert m.name == "m_H1_to_H2"


def test_morphism_apply():
    """Test applying morphism to values."""
    m = Morphism("H1", "H2")
    
    # Hash should be consistent
    result1 = m.apply("test")
    result2 = m.apply("test")
    assert result1 == result2
    
    # Different inputs should give different outputs
    result3 = m.apply("different")
    assert result1 != result3


def test_morphism_callable():
    """Test that morphism is callable."""
    m = Morphism("H1", "H2")
    
    result1 = m("test")
    result2 = m.apply("test")
    assert result1 == result2


def test_morphism_disambiguation():
    """Test morphism disambiguation (requirement 4)."""
    m = Morphism("H1", "H2")
    
    # Initially no disambiguation
    assert not m.needs_disambiguation()
    assert m.get_disambiguation_type() is None
    
    # Set disambiguation type
    m.set_disambiguation_type("type_A")
    assert m.needs_disambiguation()
    assert m.get_disambiguation_type() == "type_A"


def test_morphism_no_separation():
    """Test that morphisms are not separated by type (requirement 3)."""
    # Multiple morphisms without type separation
    m1 = Morphism("H1", "H2", name="morph1")
    m2 = Morphism("H1", "H2", name="morph2")
    m3 = Morphism("H1", "H2", name="morph3")
    
    # They should all work without needing type labels
    assert not m1.needs_disambiguation()
    assert not m2.needs_disambiguation()
    assert not m3.needs_disambiguation()
    
    # All can apply transformations
    result1 = m1("test")
    result2 = m2("test")
    result3 = m3("test")
    
    # They're using same default hash, so results should be same
    assert result1 == result2 == result3


def test_morphism_custom_hash():
    """Test morphism with custom hash function."""
    def custom_hash(value):
        return f"custom_{value}"
    
    m = Morphism("H1", "H2", hash_func=custom_hash)
    result = m("test")
    assert result == "custom_test"


def test_morphism_composition():
    """Test composing morphisms."""
    m1 = Morphism("H1", "H2")
    m2 = Morphism("H2", "H3")
    
    # Compose m1 then m2
    m_composed = m1.compose(m2)
    
    assert m_composed.source_id == "H1"
    assert m_composed.target_id == "H3"
    
    # Test that composition works
    value = "test"
    intermediate = m1(value)
    final = m2(intermediate)
    composed_result = m_composed(value)
    assert composed_result == final


def test_morphism_composition_invalid():
    """Test that invalid composition raises error."""
    m1 = Morphism("H1", "H2")
    m2 = Morphism("H3", "H4")  # H3 != H2
    
    with pytest.raises(ValueError):
        m1.compose(m2)


def test_morphism_repr():
    """Test string representation."""
    m = Morphism("H1", "H2", name="test_morph")
    repr_str = repr(m)
    assert "test_morph" in repr_str
    assert "H1→H2" in repr_str
    
    m.set_disambiguation_type("type_A")
    repr_str = repr(m)
    assert "type_A" in repr_str
