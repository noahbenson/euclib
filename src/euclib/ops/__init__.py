# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/__init__.py
'''Geometric operations over ``euclib`` objects.

The functions in this subpackage take geometries and answer questions about
them --- how far apart they are, where they meet --- without adding to the types
themselves. They are exported as functions rather than as methods so that an
operation can be extended to a new pair of types without changing either type.
'''

# Dependencies ###############################################################

from __future__ import annotations

from ._cross import positions_of, sample, transfer
from ._intersect import (
    contains, mesh_intersections, path_crossings, path_intersections,
    tolerance_of, voxel_intersections)
from ._distance import distance, nearest, separation
from ._geodesic import geodesic


# Exports ####################################################################

__all__ = ('distance', 'nearest', 'separation', 'geodesic',
           'positions_of', 'sample', 'transfer',
           'path_crossings', 'path_intersections', 'contains',
           'tolerance_of', 'voxel_intersections', 'mesh_intersections')
