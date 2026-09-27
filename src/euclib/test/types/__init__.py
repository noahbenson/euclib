# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/__init__.py
'''Tests for the ``euclib.types`` subpackage.The modules are re-exported wholesale: this package deliberately carries no
``__all__``, because a hand-maintained list of test cases drifts, and the drift
is silent --- a class left out of it is a class the ``python -m euclib.test``
entry point never loads, however it is written. ``test_suite`` checks that every
case defined in this package is reachable.
'''

# Tests are collected by ``euclib.test.load_tests``, which discovers
# the modules under this package; nothing is re-exported here.
