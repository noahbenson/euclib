# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/__init__.py
'''Tests for the ``euclib.abc`` subpackage.'''

# Dependencies ###############################################################

from __future__ import annotations

from .test_core import *
from .test_property import *
from .test_topo import *
from .test_geom import *


# Exports ####################################################################

__all__ = ('TestPlantypeABC', 'TestNormalizers', 'TestProperty',
           'TestLoc', 'TestTopologyAbstractness', 'TestSimplexTopology',
           'TestTopologyMetadata',
           'TestGeometryBasics', 'TestProperties', 'TestSimplexProperties',
           'TestEquality', 'TestNameSplitting')
