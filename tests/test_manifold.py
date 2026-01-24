"""
Tests for the Manifold class.
"""

import pytest
from hllset_manifold.manifold import Manifold


def test_manifold_creation():
    """Test basic manifold creation."""
    m = Manifold("TestManifold")
    assert m.name == "TestManifold"
    assert len(m) == 0


def test_manifold_add_element():
    """Test adding elements to manifold."""
    m = Manifold("TestManifold")
    
    # First addition should return True
    assert m.add_element("element1") is True
    assert len(m) == 1
    assert m.contains("element1")
    
    # Second addition of same element should return False (idempotent)
    assert m.add_element("element1") is False
    assert len(m) == 1


def test_manifold_idempotency():
    """Test that manifold operations are idempotent."""
    m = Manifold("TestManifold")
    
    # Add same element multiple times
    m.add_element("x")
    m.add_element("x")
    m.add_element("x")
    
    # Should only have one element
    assert len(m) == 1
    assert "x" in m.get_elements()


def test_manifold_multiple_elements():
    """Test manifold with multiple elements."""
    m = Manifold("TestManifold")
    
    elements = ["a", "b", "c", "d", "e"]
    for elem in elements:
        m.add_element(elem)
    
    assert len(m) == 5
    for elem in elements:
        assert m.contains(elem)


def test_manifold_structure():
    """Test manifold structural properties."""
    m = Manifold("TestManifold")
    
    m.set_structure("dimension", 3)
    m.set_structure("metric", "euclidean")
    
    assert m.get_structure("dimension") == 3
    assert m.get_structure("metric") == "euclidean"
    assert m.get_structure("nonexistent") is None


def test_manifold_get_elements():
    """Test getting elements returns a copy."""
    m = Manifold("TestManifold")
    m.add_element("x")
    
    elements = m.get_elements()
    elements.add("y")
    
    # Original manifold should not be affected
    assert len(m) == 1
    assert not m.contains("y")


def test_manifold_repr():
    """Test string representation."""
    m = Manifold("TestManifold")
    m.add_element("x")
    m.add_element("y")
    
    repr_str = repr(m)
    assert "TestManifold" in repr_str
    assert "elements=2" in repr_str
