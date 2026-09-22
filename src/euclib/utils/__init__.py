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
    bounds_of, simplex_boxes, split_cells, octree_split, quadtree_split,
    nearest_vertices,
    project_onto_face,
    closest_simplex,
    closest_prism,
    barycentric_coords,
    cross3,
    closest_segment_params,
    segments_intersect,
    barycentric_in_triangle,
    segments_triangles_intersect,
    triangles_segments_intersect,
    tetrahedron_box_intersection)
from ._hash import content_hash, values_equal
from ._spatial import SpatialTree


# Exports ####################################################################

__all__ = (
    'using_c_extension', 'backend_error',
    'is_pointdata', 'unique_columns', 'unique_coords',
    'simplex_measures',
    'bounds_of', 'simplex_boxes', 'split_cells',
    'octree_split', 'quadtree_split',
    'nearest_vertices', 'project_onto_face', 'closest_simplex',
    'closest_prism', 'barycentric_coords',
    'cross3', 'closest_segment_params', 'segments_intersect',
    'barycentric_in_triangle', 'segments_triangles_intersect',
    'triangles_segments_intersect',
    'tetrahedron_box_intersection',
    'content_hash', 'values_equal',
    'SpatialTree')
