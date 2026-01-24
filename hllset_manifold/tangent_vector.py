"""
Tangent vector implementation for morphisms.

For any morphism m: H_1 -> H_2, we can define tangent vectors
m' = {D, R, N} and m'_t = (d_D/d_t, d_R/d_t, d_N/d_t).
"""

from typing import Tuple, Callable, Optional


class TangentVector:
    """
    A tangent vector for a morphism with components {D, R, N}.
    
    This represents the rate of change of the morphism components:
    - D: Domain component
    - R: Range component  
    - N: Normal component
    
    The time derivative is m'_t = (d_D/d_t, d_R/d_t, d_N/d_t)
    """
    
    def __init__(self, D: float = 0.0, R: float = 0.0, N: float = 0.0):
        """
        Initialize a tangent vector.
        
        Args:
            D: Domain component
            R: Range component
            N: Normal component
        """
        self.D = D
        self.R = R
        self.N = N
    
    def derivative(self, 
                   d_D_dt: float = 0.0, 
                   d_R_dt: float = 0.0, 
                   d_N_dt: float = 0.0) -> 'TangentVector':
        """
        Calculate the time derivative of the tangent vector.
        
        Returns a new TangentVector representing m'_t = (d_D/d_t, d_R/d_t, d_N/d_t)
        
        Args:
            d_D_dt: Time derivative of D component
            d_R_dt: Time derivative of R component
            d_N_dt: Time derivative of N component
            
        Returns:
            A new TangentVector representing the time derivative
        """
        return TangentVector(D=d_D_dt, R=d_R_dt, N=d_N_dt)
    
    def magnitude(self) -> float:
        """
        Calculate the magnitude of the tangent vector.
        
        Returns:
            The Euclidean norm of the vector
        """
        return (self.D**2 + self.R**2 + self.N**2)**0.5
    
    def normalize(self) -> 'TangentVector':
        """
        Normalize the tangent vector to unit length.
        
        Returns:
            A new normalized TangentVector
            
        Raises:
            ValueError: If the magnitude is zero
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return TangentVector(D=self.D/mag, R=self.R/mag, N=self.N/mag)
    
    def dot(self, other: 'TangentVector') -> float:
        """
        Calculate the dot product with another tangent vector.
        
        Args:
            other: Another TangentVector
            
        Returns:
            The dot product
        """
        return self.D * other.D + self.R * other.R + self.N * other.N
    
    def cross(self, other: 'TangentVector') -> 'TangentVector':
        """
        Calculate the cross product with another tangent vector.
        
        Args:
            other: Another TangentVector
            
        Returns:
            A new TangentVector representing the cross product
        """
        return TangentVector(
            D=self.R * other.N - self.N * other.R,
            R=self.N * other.D - self.D * other.N,
            N=self.D * other.R - self.R * other.D
        )
    
    def scale(self, scalar: float) -> 'TangentVector':
        """
        Scale the tangent vector by a scalar.
        
        Args:
            scalar: The scaling factor
            
        Returns:
            A new scaled TangentVector
        """
        return TangentVector(D=self.D * scalar, R=self.R * scalar, N=self.N * scalar)
    
    def add(self, other: 'TangentVector') -> 'TangentVector':
        """
        Add another tangent vector.
        
        Args:
            other: Another TangentVector
            
        Returns:
            A new TangentVector representing the sum
        """
        return TangentVector(D=self.D + other.D, R=self.R + other.R, N=self.N + other.N)
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """
        Get the components as a tuple.
        
        Returns:
            Tuple of (D, R, N)
        """
        return (self.D, self.R, self.N)
    
    def __repr__(self) -> str:
        """Return string representation of the tangent vector."""
        return f"TangentVector(D={self.D:.4f}, R={self.R:.4f}, N={self.N:.4f})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another tangent vector."""
        if not isinstance(other, TangentVector):
            return NotImplemented
        return (abs(self.D - other.D) < 1e-10 and 
                abs(self.R - other.R) < 1e-10 and 
                abs(self.N - other.N) < 1e-10)


class MorphismTangentSpace:
    """
    Manages tangent vectors for a morphism.
    
    This class associates tangent vectors with morphisms and can compute
    time derivatives.
    """
    
    def __init__(self, morphism_name: str):
        """
        Initialize a tangent space for a morphism.
        
        Args:
            morphism_name: Name of the associated morphism
        """
        self.morphism_name = morphism_name
        self.tangent_vector: Optional[TangentVector] = None
        self._derivative_func: Optional[Callable[[float], TangentVector]] = None
    
    def set_tangent_vector(self, vector: TangentVector) -> None:
        """
        Set the current tangent vector.
        
        Args:
            vector: The tangent vector to set
        """
        self.tangent_vector = vector
    
    def set_derivative_function(self, func: Callable[[float], TangentVector]) -> None:
        """
        Set a function that computes the time derivative.
        
        Args:
            func: A function that takes time t and returns the derivative TangentVector
        """
        self._derivative_func = func
    
    def get_derivative_at(self, t: float) -> Optional[TangentVector]:
        """
        Get the time derivative at a specific time.
        
        Args:
            t: The time parameter
            
        Returns:
            The derivative TangentVector at time t, or None if no function is set
        """
        if self._derivative_func is None:
            return None
        return self._derivative_func(t)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"MorphismTangentSpace(morphism='{self.morphism_name}', vector={self.tangent_vector})"
