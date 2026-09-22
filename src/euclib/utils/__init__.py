# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/__init__.py
'''Low-level utility functions for the ``euclib`` library.

The functions in this subpackage are the kernels on which the rest of
``euclib`` is built. They are pure with respect to the data they are given and
they do not depend on any of the geometric types. Each kernel is implemented in
pure Python in ``euclib.utils._pycore`` and, where it pays to do so, again in
the optional C extension ``euclib._c``; ``euclib.utils._core`` chooses between
them and records its choice in ``using_c_extension`` and ``backend_error``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from ._core import (
    using_c_extension,
    backend_error,
    is_pointdata,
    unique_columns,
    unique_coords,
    simplex_measures,
    nearest_vertices,
    project_onto_face,
    closest_simplex,
    closest_prism,
    barycentric_coords)
from ._hash import content_hash, values_equal


# Exports ####################################################################

__all__ = (
    'using_c_extension', 'backend_error',
    'is_pointdata', 'unique_columns', 'unique_coords',
    'simplex_measures',
    'nearest_vertices', 'project_onto_face', 'closest_simplex',
    'closest_prism', 'barycentric_coords',
    'content_hash', 'values_equal')
