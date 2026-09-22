# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/__init__.py
'''Tests for the ``euclib.ops`` subpackage.'''

# Dependencies ###############################################################

from __future__ import annotations

from .test_cross import *
from .test_intersect import *
from .test_distance import *
from .test_geodesic import *


# Exports ####################################################################

__all__ = ('TestPositions', 'TestTransfer',
           'TestDistance', 'TestNearest', 'TestSeparation',
           'TestGeometriesThatAreNotCoordinateMatrices',
           'TestAlongAPath', 'TestAlongASurface', 'TestEdgesOfTheDefinition',
           'TestAgainstAnIndependentSearch',
           'TestTolerance', 'TestPathCrossings', 'TestPathIntersections',
           'TestContains', 'TestVoxelIntersections', 'TestMeshIntersections')
