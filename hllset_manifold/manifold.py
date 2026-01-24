"""
Manifold implementation with idempotency restriction.

The manifold is the fundamental structure that enforces idempotency
as its only restriction.
"""

from typing import Any, Set, Dict, Optional


class Manifold:
    """
    A manifold structure with idempotency as the only restriction.
    
    This class represents a mathematical manifold where operations are idempotent,
    meaning applying the same operation multiple times has the same effect as
    applying it once.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a manifold.
        
        Args:
            name: Optional name for the manifold
        """
        self.name = name or f"Manifold_{id(self)}"
        self._elements: Set[Any] = set()
        self._structure: Dict[str, Any] = {}
    
    def add_element(self, element: Any) -> bool:
        """
        Add an element to the manifold (idempotent operation).
        
        Args:
            element: The element to add
            
        Returns:
            True if element was newly added, False if it already existed
        """
        size_before = len(self._elements)
        self._elements.add(element)
        return len(self._elements) > size_before
    
    def contains(self, element: Any) -> bool:
        """
        Check if an element exists in the manifold.
        
        Args:
            element: The element to check
            
        Returns:
            True if element exists in the manifold
        """
        return element in self._elements
    
    def get_elements(self) -> Set[Any]:
        """
        Get all elements in the manifold.
        
        Returns:
            Set of all elements
        """
        return self._elements.copy()
    
    def set_structure(self, key: str, value: Any) -> None:
        """
        Set a structural property of the manifold.
        
        Args:
            key: The property key
            value: The property value
        """
        self._structure[key] = value
    
    def get_structure(self, key: str) -> Optional[Any]:
        """
        Get a structural property of the manifold.
        
        Args:
            key: The property key
            
        Returns:
            The property value or None if not found
        """
        return self._structure.get(key)
    
    def __len__(self) -> int:
        """Return the number of elements in the manifold."""
        return len(self._elements)
    
    def __repr__(self) -> str:
        """Return string representation of the manifold."""
        return f"Manifold(name='{self.name}', elements={len(self._elements)})"
