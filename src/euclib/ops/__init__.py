# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/__init__.py
'''Geometric operations over ``euclib`` objects.

The functions in this subpackage take geometries and answer questions about
them --- how far apart they are, where they meet, what line or plane their
points lie nearest --- without adding to the types themselves. They are
exported as functions rather than as methods so that an operation can be
extended to a new pair of types without changing either type.
'''

# Dependencies ###############################################################

from __future__ import annotations

from ._axes import oriented_bounding_box, principal_axes
from ._cross import positions_of, sample, transfer
from ._intersect import (
    contains, integrate, mesh_intersections, path_crossings,
    path_intersections, ray_crossings, tolerance_of, voxel_intersections,
    voxel_segment_intersections, voxel_surface_intersections)
from ._distance import distance, nearest, separation
from ._fit import fit_line, fit_plane
from ._geodesic import geodesic
from ._smooth import smooth


# Exports ####################################################################

__all__ = ('distance', 'nearest', 'separation', 'fit_line', 'fit_plane',
           'principal_axes', 'oriented_bounding_box',
           'geodesic', 'smooth',
           'positions_of', 'sample', 'transfer',
           'path_crossings', 'path_intersections', 'contains',
           'tolerance_of', 'voxel_intersections',
           'voxel_surface_intersections', 'voxel_segment_intersections',
           'integrate', 'ray_crossings', 'mesh_intersections')
