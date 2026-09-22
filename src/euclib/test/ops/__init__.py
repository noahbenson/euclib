# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/__init__.py
'''Tests for the ``euclib.ops`` subpackage.'''

# Dependencies ###############################################################

from __future__ import annotations

from .test_cross import *
from .test_distance import *


# Exports ####################################################################

__all__ = ('TestPositions', 'TestTransfer',
           'TestDistance', 'TestNearest', 'TestSeparation')
