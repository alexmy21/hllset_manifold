"""
Morphism implementation representing hash functions.

Morphisms are hash functions that map between manifolds. They are not separated
by type except during disambiguation.
"""

from typing import Any, Callable, Optional, Dict
import hashlib


class Morphism:
    """
    A morphism (hash function) that maps between manifolds.
    
    Morphisms are not separated by type, except when performing disambiguation.
    They represent transformations between different spaces in the manifold structure.
    """
    
    def __init__(self, 
                 source_id: str,
                 target_id: str,
                 hash_func: Optional[Callable[[Any], str]] = None,
                 name: Optional[str] = None):
        """
        Initialize a morphism.
        
        Args:
            source_id: Identifier for the source manifold (H_1)
            target_id: Identifier for the target manifold (H_2)
            hash_func: Optional custom hash function, defaults to SHA256
            name: Optional name for the morphism
        """
        self.source_id = source_id
        self.target_id = target_id
        self.name = name or f"m_{source_id}_to_{target_id}"
        self._hash_func = hash_func or self._default_hash
        self._disambiguation_type: Optional[str] = None
    
    @staticmethod
    def _default_hash(value: Any) -> str:
        """
        Default hash function using SHA256.
        
        Args:
            value: The value to hash
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(value, str):
            data = value.encode('utf-8')
        elif isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def apply(self, value: Any) -> str:
        """
        Apply the morphism to a value.
        
        Args:
            value: The value to transform
            
        Returns:
            The hashed/transformed value
        """
        return self._hash_func(value)
    
    def set_disambiguation_type(self, type_label: str) -> None:
        """
        Set the disambiguation type for this morphism.
        
        This is only used when disambiguation is needed, as per requirement (4).
        
        Args:
            type_label: The type label for disambiguation
        """
        self._disambiguation_type = type_label
    
    def get_disambiguation_type(self) -> Optional[str]:
        """
        Get the disambiguation type if set.
        
        Returns:
            The disambiguation type or None
        """
        return self._disambiguation_type
    
    def needs_disambiguation(self) -> bool:
        """
        Check if this morphism has a disambiguation type set.
        
        Returns:
            True if disambiguation type is set
        """
        return self._disambiguation_type is not None
    
    def compose(self, other: 'Morphism') -> 'Morphism':
        """
        Compose this morphism with another morphism.
        
        Creates a new morphism that represents m2 ∘ m1 where m1 is self.
        
        Args:
            other: The morphism to compose with
            
        Returns:
            A new composed morphism
            
        Raises:
            ValueError: If target of self doesn't match source of other
        """
        if self.target_id != other.source_id:
            raise ValueError(
                f"Cannot compose: target {self.target_id} != source {other.source_id}"
            )
        
        def composed_func(value: Any) -> str:
            intermediate = self.apply(value)
            return other.apply(intermediate)
        
        return Morphism(
            source_id=self.source_id,
            target_id=other.target_id,
            hash_func=composed_func,
            name=f"{self.name}_comp_{other.name}"
        )
    
    def __call__(self, value: Any) -> str:
        """
        Make the morphism callable.
        
        Args:
            value: The value to transform
            
        Returns:
            The hashed/transformed value
        """
        return self.apply(value)
    
    def __repr__(self) -> str:
        """Return string representation of the morphism."""
        disambig = f", type={self._disambiguation_type}" if self._disambiguation_type else ""
        return f"Morphism(name='{self.name}', {self.source_id}→{self.target_id}{disambig})"
