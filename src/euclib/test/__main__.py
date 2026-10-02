# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/__main__.py
'''Runs the ``euclib`` unit tests via ``python -m euclib.test``.

This runs the *whole* suite, which is what it is for. To run part of it, use
``unittest``, which resolves a dotted name properly::

    python -m unittest euclib.test.test_suite
    python -m unittest euclib.test.test_suite.TestTheShapeOfACalc
    python -m unittest euclib.test.test_suite.TestTheShapeOfACalc.test_a_case

Passing a name here does not work and does not say so. The runner below hands
``unittest.main`` the package as its *module*, so a name it is given is looked
for *inside* that package: ``euclib.test.test_suite`` asks for an attribute
``euclib`` of ``euclib.test``, which fails with ``module 'euclib.test' has no
attribute 'euclib'`` and runs a single failing test. That reads as a failure of
whatever was being checked rather than of the way it was asked for, which is
exactly how it misled once.
'''

# Dependencies ###############################################################

from __future__ import annotations


# Testing ####################################################################

def run_tests(verbosity=2, **kwargs):
    '''Runs every ``euclib`` unit test.

    Parameters
    ----------
    verbosity : int, optional
        The verbosity passed to the test runner. The default is ``2``.
    **kwargs
        Additional keyword arguments passed to ``unittest.main``.

    Returns
    -------
    unittest.result.TestResult
        The result of the test run.
    '''
    from unittest import main
    return main('euclib.test', verbosity=verbosity, exit=False, **kwargs)


if __name__ == '__main__':
    run_tests()
