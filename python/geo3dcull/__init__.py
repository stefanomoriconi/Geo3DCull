"""
geo3dcull — GPU/CPU-accelerated 3D mesh visibility culling.

This is the main public API of the geo3dcull package.

Quick start
-----------
>>> import geo3dcull
>>> wrapper = geo3dcull.Geo3DCullDLL(use_cuda=False)
>>> isFvis, isVvis = wrapper.getVisibleFcsVtsCull(...)

See README.md for full documentation, COMPILING.md for build instructions.
"""

from geo3dcull._core import (
    Geo3DCullDLL,
    Geometry,
    g3c_uvect,
    g3c_projct,
)

__version__ = "1.0.0"
__all__ = [
    "Geo3DCullDLL",
    "Geometry",
    "g3c_uvect",
    "g3c_projct",
    "__version__",
]
