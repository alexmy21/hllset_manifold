"""
Tests for the TangentVector class.
"""

import pytest
import math
from hllset_manifold.tangent_vector import TangentVector, MorphismTangentSpace


def test_tangent_vector_creation():
    """Test creating a tangent vector."""
    tv = TangentVector(1.0, 2.0, 3.0)
    assert tv.D == 1.0
    assert tv.R == 2.0
    assert tv.N == 3.0


def test_tangent_vector_default():
    """Test creating tangent vector with defaults."""
    tv = TangentVector()
    assert tv.D == 0.0
    assert tv.R == 0.0
    assert tv.N == 0.0


def test_tangent_vector_derivative():
    """Test time derivative of tangent vector (requirement 6)."""
    tv = TangentVector(1.0, 2.0, 3.0)
    
    # Calculate derivative m'_t = (d_D/d_t, d_R/d_t, d_N/d_t)
    derivative = tv.derivative(d_D_dt=0.5, d_R_dt=1.0, d_N_dt=1.5)
    
    assert derivative.D == 0.5
    assert derivative.R == 1.0
    assert derivative.N == 1.5


def test_tangent_vector_magnitude():
    """Test magnitude calculation."""
    tv = TangentVector(3.0, 4.0, 0.0)
    assert tv.magnitude() == 5.0
    
    tv2 = TangentVector(1.0, 1.0, 1.0)
    assert abs(tv2.magnitude() - math.sqrt(3)) < 1e-10


def test_tangent_vector_normalize():
    """Test vector normalization."""
    tv = TangentVector(3.0, 4.0, 0.0)
    normalized = tv.normalize()
    
    assert abs(normalized.magnitude() - 1.0) < 1e-10
    assert abs(normalized.D - 0.6) < 1e-10
    assert abs(normalized.R - 0.8) < 1e-10


def test_tangent_vector_normalize_zero():
    """Test that normalizing zero vector raises error."""
    tv = TangentVector(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        tv.normalize()


def test_tangent_vector_dot_product():
    """Test dot product."""
    tv1 = TangentVector(1.0, 2.0, 3.0)
    tv2 = TangentVector(4.0, 5.0, 6.0)
    
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert tv1.dot(tv2) == 32.0


def test_tangent_vector_cross_product():
    """Test cross product."""
    tv1 = TangentVector(1.0, 0.0, 0.0)
    tv2 = TangentVector(0.0, 1.0, 0.0)
    
    cross = tv1.cross(tv2)
    # i × j = k, so result should be (0, 0, 1)
    assert cross.D == 0.0
    assert cross.R == 0.0
    assert cross.N == 1.0


def test_tangent_vector_scale():
    """Test scaling vector."""
    tv = TangentVector(1.0, 2.0, 3.0)
    scaled = tv.scale(2.0)
    
    assert scaled.D == 2.0
    assert scaled.R == 4.0
    assert scaled.N == 6.0


def test_tangent_vector_add():
    """Test vector addition."""
    tv1 = TangentVector(1.0, 2.0, 3.0)
    tv2 = TangentVector(4.0, 5.0, 6.0)
    
    result = tv1.add(tv2)
    assert result.D == 5.0
    assert result.R == 7.0
    assert result.N == 9.0


def test_tangent_vector_as_tuple():
    """Test converting to tuple."""
    tv = TangentVector(1.0, 2.0, 3.0)
    assert tv.as_tuple() == (1.0, 2.0, 3.0)


def test_tangent_vector_equality():
    """Test vector equality."""
    tv1 = TangentVector(1.0, 2.0, 3.0)
    tv2 = TangentVector(1.0, 2.0, 3.0)
    tv3 = TangentVector(1.0, 2.0, 3.1)
    
    assert tv1 == tv2
    assert tv1 != tv3


def test_tangent_vector_repr():
    """Test string representation."""
    tv = TangentVector(1.5, 2.5, 3.5)
    repr_str = repr(tv)
    assert "TangentVector" in repr_str
    assert "1.5" in repr_str or "1.50" in repr_str


def test_morphism_tangent_space():
    """Test MorphismTangentSpace class."""
    mts = MorphismTangentSpace("test_morphism")
    assert mts.morphism_name == "test_morphism"
    assert mts.tangent_vector is None


def test_morphism_tangent_space_set_vector():
    """Test setting tangent vector."""
    mts = MorphismTangentSpace("test_morphism")
    tv = TangentVector(1.0, 2.0, 3.0)
    
    mts.set_tangent_vector(tv)
    assert mts.tangent_vector == tv


def test_morphism_tangent_space_derivative():
    """Test derivative function in tangent space."""
    mts = MorphismTangentSpace("test_morphism")
    
    # Define a derivative function
    def deriv_func(t):
        return TangentVector(D=t, R=2*t, N=3*t)
    
    mts.set_derivative_function(deriv_func)
    
    # Get derivative at t=2
    deriv = mts.get_derivative_at(2.0)
    assert deriv.D == 2.0
    assert deriv.R == 4.0
    assert deriv.N == 6.0


def test_morphism_tangent_space_no_derivative():
    """Test that without derivative function, returns None."""
    mts = MorphismTangentSpace("test_morphism")
    assert mts.get_derivative_at(1.0) is None


def test_tangent_vector_drn_components():
    """Test that D, R, N components work as specified in requirement 6."""
    # For morphism m: H_1 -> H_2
    # m' = {D, R, N} represents the tangent vector components
    tv = TangentVector(D=1.0, R=2.0, N=3.0)
    
    # Verify components are accessible
    assert tv.D == 1.0  # Domain component
    assert tv.R == 2.0  # Range component
    assert tv.N == 3.0  # Normal component
    
    # m'_t = (d_D/d_t, d_R/d_t, d_N/d_t)
    tv_t = tv.derivative(d_D_dt=0.1, d_R_dt=0.2, d_N_dt=0.3)
    
    assert tv_t.D == 0.1
    assert tv_t.R == 0.2
    assert tv_t.N == 0.3
