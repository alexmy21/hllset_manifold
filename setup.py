"""
Setup script for building Cython extensions.

Build with:
    python setup.py build_ext --inplace
    
Or with pip:
    pip install -e .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        name="core.hll_core",
        sources=["core/hll_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Optimize for speed
        extra_link_args=[],
        language="c"
    )
]

setup(
    name="hllset_manifold",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True,
        },
        annotate=True,  # Generate HTML annotation files
    ),
    zip_safe=False,
)
