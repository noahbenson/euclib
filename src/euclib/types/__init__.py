# -*- coding: utf-8 -*-
###############################################################################
# euclib/types/__init__.py
'''The concrete types of the ``euclib`` library.

This subpackage holds the types that a user constructs directly: the topologies
that describe how geometric objects are connected, and the geometries that
place them in space. It depends on ``euclib.abc``, which defines their abstract
bases, and on ``euclib.utils``, which provides its kernels; nothing in
``euclib.abc`` or ``euclib.utils`` depends on this subpackage.
'''

# Dependencies ###############################################################

from __future__ import annotations

from ._topo import *
from ._geom import *
from ._transform import *


# Exports ####################################################################

__all__ = (
    # from euclib.types._topo
    'VertexLoc', 'SegLoc', 'TriLoc', 'TetLoc',
    'PrismLoc',
    'GridLoc1', 'GridLoc2', 'GridLoc3', 'GRID_LOCS',
    'VertexTopology', 'SegTopology', 'TriTopology', 'TetTopology',
    'PrismTopology', 'GridTopology',
    # from euclib.types._geom
    'VertexSet', 'SegPath', 'TriMesh', 'TetMesh', 'PrismMesh', 'Grid',
    'points', 'segpath', 'trimesh', 'tetmesh', 'prismmesh', 'grid',
    # from euclib.types._transform
    'Transform', 'Affine',
    'affine_identity', 'affine_translation', 'affine_scaling')
