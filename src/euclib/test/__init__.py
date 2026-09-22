# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/__init__.py
'''The unit tests for the ``euclib`` library.

The tests use the standard library's ``unittest`` framework and mirror the
structure of the package itself: ``euclib.test.abc`` tests ``euclib.abc``, and
so on. They are run with ``python -m euclib.test``.

Each test module's ``TestCase`` classes are re-exported here so that a single
``unittest`` load of this module finds every test. Test discovery is
deliberately not used, because combining it with these re-exports causes each
test to be collected twice.
'''

# Dependencies ###############################################################

from __future__ import annotations

from .abc import *
from .utils import *
from .types import *
from .ops import *


# Exports ####################################################################

__all__ = (
    # from euclib.test.abc
    'TestPlantypeABC', 'TestNormalizers', 'TestProperty',
    'TestLoc', 'TestTopologyAbstractness', 'TestSimplexTopology',
    'TestTopologyMetadata',
    'TestGeometryBasics', 'TestProperties', 'TestSimplexProperties',
    'TestEquality', 'TestNameSplitting', 'TestSpatialIndex',
    # from euclib.test.types
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
    'TestPointCloudInterpolation',
    # from euclib.test.ops
    'TestPositions', 'TestTransfer',
    'TestDistance', 'TestNearest', 'TestSeparation',
    'TestGeometriesThatAreNotCoordinateMatrices',
    'TestAlongAPath', 'TestAlongASurface', 'TestEdgesOfTheDefinition',
    'TestAgainstAnIndependentSearch',
    'TestTolerance', 'TestPathCrossings', 'TestPathIntersections',
    'TestContains', 'TestVoxelIntersections', 'TestMeshIntersections',
    # from euclib.test.utils
    'TestIsPointdata', 'TestUniqueColumns', 'TestUniqueCoords',
    'TestContentHash', 'TestSubdivision', 'TestSpatialTree',
    'TestSegments', 'TestSegmentsAndTriangles', 'TestBarycentric',
    'TestTetrahedronAndBox', 'TestTriangles')
