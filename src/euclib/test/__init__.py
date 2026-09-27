# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/__init__.py
'''The unit tests for the ``euclib`` library.

The tests use the standard library's ``unittest`` framework and mirror the
structure of the package itself: ``euclib.test.abc`` tests ``euclib.abc``, and
so on. They are run with ``python -m euclib.test``, which is what the workflow
uses.

**How they are collected.** This package defines ``load_tests``, which discovers
the test modules under it. That is deliberate and it is the third design tried
here. The first re-exported every test case into this module's namespace, so that
a single load of ``euclib.test`` found them all; that flattens the suite, so two
modules defining a class of the same name collide and one of them is lost, and
the ``__all__`` lists that filtered the re-exports had to be maintained by hand
--- which they were not, and the classes they left out were never loaded, however
they were written. The second ran discovery alongside the re-exports, which
collects every test twice.

``load_tests`` has neither fault: the modules are imported under their own names,
so nothing collides, and each is collected once. ``test_suite`` checks that the
modules the loader finds are exactly the ones on disk, so a new test file that is
somehow not collected is a failure rather than a smaller suite.
'''

# Dependencies ###############################################################

from __future__ import annotations

import pathlib
from unittest import TestLoader


# Collection #################################################################

HERE = pathlib.Path(__file__).parent


def load_tests(loader, tests, pattern, /):
    '''Collects every test module under this package, once.

    Parameters
    ----------
    loader : unittest.TestLoader
        The loader doing the collecting.
    tests : unittest.TestSuite
        The tests collected so far.
    pattern : str or None
        The filename pattern to discover by.

    Returns
    -------
    unittest.TestSuite
        The whole suite.
    '''
    # The top level is the directory the *package* sits in, so that modules are
    # named ``euclib.test.types.test_ct`` rather than ``test.types.test_ct`` --
    # a module's name is its identity, and two spellings of one module are two
    # modules to the loader.
    return loader.discover(str(HERE), pattern=(pattern or 'test_*.py'),
                           top_level_dir=str(HERE.parent.parent))


# Exports ####################################################################

__all__ = ('load_tests',)
