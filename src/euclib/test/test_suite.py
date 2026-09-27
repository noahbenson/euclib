# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/test_suite.py
'''Tests for the test suite itself.

``python -m euclib.test`` loads the test *package* rather than discovering
modules, because a discovery pass combined with this package's re-exports
collects every test twice. The re-exports are therefore load-bearing: a module
whose cases are not re-exported by its package's ``__init__`` is never run by the
command the workflow uses, however it is written. It is a quiet failure --- the
suite is smaller than it looks, and nothing says so --- and it happened here: the
Clough-Tocher, Powell-Sabin, parity and backend tests all ran under
``unittest discover`` and not under ``python -m euclib.test``.

This module is the guard. It walks the test package and checks that every test
module on disk is re-exported, so a new one that is not is a failure rather than
a silent omission.
'''

# Dependencies ###############################################################

from __future__ import annotations

import pathlib
import unittest
from unittest import TestCase, TestLoader


# Tests ######################################################################

class TestTheTestSuite(TestCase):
    '''That the suite the workflow runs is the whole suite.'''

    def test_the_loader_finds_exactly_the_test_modules_on_disk(self):
        # The outcome that matters: every ``test_*.py`` file under this package
        # must be collected by ``python -m euclib.test``. A module that is not
        # is a smaller suite with nothing to say so, which is how the
        # Clough-Tocher, Powell-Sabin, parity and backend tests came to run
        # under ``unittest discover`` and not under the command the workflow
        # uses.
        import euclib.test as suite
        root = pathlib.Path(suite.__file__).parent

        def walked(node, /):
            for one in node:
                if isinstance(one, unittest.TestSuite):
                    yield from walked(one)
                else:
                    yield one

        collected = list(walked(suite.load_tests(TestLoader(),
                                                 unittest.TestSuite(), None)))
        found = {type(one).__module__ for one in collected}
        on_disk = {
            '.'.join(('euclib',) + path.relative_to(root.parent).with_suffix('').parts)
            for path in root.rglob('test_*.py')}
        self.assertEqual(sorted(on_disk - found), [],
                         "these modules are on disk but not collected")
        self.assertEqual(sorted(found - on_disk), [],
                         "these modules were collected but are not on disk")
        self.assertGreater(len(collected), 500)

    def test_a_case_from_a_module_that_was_missing_is_reachable(self):
        # The specific regression: these modules were loaded by discovery and
        # not by the entry point, so a case from each must be in the suite.
        import euclib.test as suite
        from unittest import TestLoader
        names = set()
        def walked(node, /):
            for one in node:
                if isinstance(one, unittest.TestSuite):
                    yield from walked(one)
                else:
                    yield one
        for one in walked(suite.load_tests(TestLoader(), unittest.TestSuite(), None)):
            names.add(type(one).__name__)
        for name in ('TestTheReferenceElement', 'TestTheSplit', 'TestRegionParity',
                     'TestRequiringTheExtension', 'TestGridCubic',
                     'TestGridSpline', 'TestGridLanczos'):
            with self.subTest(case=name):
                self.assertIn(name, names,
                              f"{name} is not collected by euclib.test")
