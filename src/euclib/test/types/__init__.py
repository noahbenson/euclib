# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/__init__.py
'''Tests for the ``euclib.types`` subpackage.'''

# Dependencies ###############################################################

from __future__ import annotations

from .test_topo import *
from .test_geom import *
from .test_transform import *
from .test_grid import *
from .test_prism import *
from .test_interp import *


# Exports ####################################################################

__all__ = (
    'TestConcreteTopologies', 'TestConcreteLocs',
    'TestVertexSet', 'TestSegPath', 'TestTriMesh', 'TestTetMesh',
    'TestMeasures', 'TestLocalRoundTrip', 'TestTorchBackend',
    'TestAffine', 'TestTransformed', 'TestBBox',
    'TestPrismTopology', 'TestPrismMesh',
    'TestGridTopology', 'TestGrid',
    'TestLinearInterpolation', 'TestNearestInterpolation',
    'TestLocalCoordinates', 'TestCoordinateNames', 'TestExtrapolation',
    'TestMasks',
    'TestGridInterpolation', 'TestUnimplementedOrders',
    'TestPointCloudInterpolation', 'TestSpatialIndex')
