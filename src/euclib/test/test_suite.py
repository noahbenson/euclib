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

import ast
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


class TestTheShapeOfACalc(TestCase):
    '''A calc is a function in a class body, and may not reach for `self`.

    A `@calc` is not a method: it is called with the plan\'s fields, and there is
    no instance for `self` to be. Reaching for one raises `NameError` --- but
    only when the field is *evaluated*, and most fields are lazy, so a calc that
    is broken this way looks exactly like one that works until something reads
    it. That is how one got past the whole suite: `proc_tetmesh` referenced
    `self` inside a `lazy`, every test passed, and `PrismMesh.tetmesh` was
    broken.

    Read from the source rather than from the runtime objects, so that it holds
    for every calc however `immlib` wraps it.
    '''

    def _calcs(self, /):
        '''Every calc function in the library, as (path, node) pairs.'''
        import euclib
        root = pathlib.Path(euclib.__file__).parent
        for path in sorted(root.rglob('*.py')):
            if pathlib.Path('test') in path.relative_to(root).parents:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                     # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any((isinstance(d, ast.Call)
                        and getattr(d.func, 'id', None) == 'calc')
                       or getattr(d, 'id', None) == 'calc'
                       for d in node.decorator_list):
                    yield (path.relative_to(root), node)

    def test_no_calc_reaches_for_self(self):
        found = list(self._calcs())
        self.assertGreater(len(found), 50,
                           "the walk found almost no calcs, so it is not looking"
                           " where they are")
        offenders = []
        for (path, node) in found:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and inner.id == 'self':
                    offenders.append(f"{path}:{inner.lineno} in {node.name}")
                    break
        self.assertEqual(offenders, [],
                         f"these calcs reach for `self`, which a calc has not")
