"""
hllset_manifold - Generalization of SGS.ai, HLLSet, and Entanglement as foundation for AI models

This package implements a mathematical framework for manifolds with morphisms and entanglement,
focusing on idempotency as the primary restriction.
"""

from .manifold import Manifold
from .morphism import Morphism
from .entanglement import Entanglement
from .tangent_vector import TangentVector

__version__ = "0.1.0"
__all__ = ["Manifold", "Morphism", "Entanglement", "TangentVector"]
